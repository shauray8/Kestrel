import json, sys
import torch
import torch.nn.functional as F
from kestrel import ops

ext = ops.fused_mod()
if "--build-only" in sys.argv:
    sys.exit(0)
E4, BF, EPS, C = torch.float8_e4m3fn, torch.bfloat16, 1e-6, 1536

_flush = None

def bench(fn, it=50, cold=False):
    global _flush
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    if not cold:
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            fn()
        e.record(); e.synchronize()
        return s.elapsed_time(e) / it * 1e3
    if _flush is None:
        _flush = torch.zeros(64 << 20, device="cuda")
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
    for s, e in ev:
        _flush.sum()
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in ev) / it * 1e3

def bench_graph(fn, n=20, reps=10):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(n):
            fn()
    gr.replay(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        gr.replay()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / (n * reps) * 1e3

def ref_modnorm32(x, sh, sc):
    return F.layer_norm(x, (x.shape[-1],), eps=EPS) * (1 + sc) + sh

def ref_modnorm_bf16(x, sh, sc):
    return ref_modnorm32(x, sh, sc).to(BF)

def ref_fp8(v, s):
    return (v / s).clamp(-448.0, 448.0).to(E4)

def ulp(t, mant):
    return torch.exp2(torch.floor(torch.log2(t.abs().clamp(min=1e-30))) - mant)

def make(M, R, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, C, device="cuda", generator=g) * 3 + torch.randn(M, 1, device="cuda", generator=g) * 20
    x[:, :8] *= 30
    y = (torch.randn(M, C, device="cuda", generator=g) * 2).to(BF)
    e = torch.randn(R, 6, C, device="cuda", generator=g) * 0.5
    shift, scale, gate = (t.squeeze(1) for t in e.chunk(6, dim=1)[:3])
    return x, y, shift, scale, gate

def fp8_ulp_err(a, b):
    a, b = a.float(), b.float()
    u = torch.maximum(ulp(torch.maximum(a.abs(), b.abs()), 3), torch.full_like(a, 2.0 ** -9))
    return float(((a - b).abs() / u).max())

def check(M, R):
    x, y, sh, sc, g = make(M, R)
    rep = lambda t: t.repeat_interleave(M // R, 0) if R > 1 else t
    shr, scr, gr = rep(sh), rep(sc), rep(g)
    res = {"kind": "correctness", "M": M, "R": R}

    def truth(xx):
        n = F.layer_norm(xx.double(), (C,), eps=EPS) * (1 + scr.double())
        return n + shr.double(), n.abs() + shr.double().abs()

    def bf16_err(o, r64, T):
        return float(((o.double() - r64).abs() / ulp(T, 7)).max())

    r32 = ref_modnorm32(x, shr, scr)
    r64, T = truth(x)
    o32 = ext.mod_norm(x, sh, sc, EPS, torch.float32)
    res["modnorm_f32_maxrel_vs_torch"] = float((o32 - r32).abs().max() / r32.abs().max())
    res["modnorm_f32_maxrel_vs_fp64"] = float((o32.double() - r64).abs().max() / r64.abs().max())
    res["torch_f32_maxrel_vs_fp64"] = float((r32.double() - r64).abs().max() / r64.abs().max())
    ob = ext.mod_norm(x, sh, sc, EPS)
    rb = r32.to(BF)
    res["modnorm_bf16_maxabs_vs_torch"] = float((ob.float() - rb.float()).abs().max())
    res["modnorm_bf16_bitexact_frac"] = float((ob.view(torch.int16) == rb.view(torch.int16)).float().mean())
    res["modnorm_bf16_err_termulp_ext"] = bf16_err(ob, r64, T)
    res["modnorm_bf16_err_termulp_torch"] = bf16_err(rb, r64, T)
    xr = x + y * gr
    xk = x.clone(); ext.gated_residual(xk, y, g)
    res["gres_bitexact"] = bool(torch.equal(xk, xr))
    xk = x.clone(); ext.gated_residual(xk, y, None)
    res["res_nogate_bitexact"] = bool(torch.equal(xk, x + y))
    xk = x.clone()
    of = ext.residual_then_mod_norm(xk, y, g, sh, sc, EPS)
    rf32 = ref_modnorm32(xr, shr, scr)
    rf64, Tf = truth(xr)
    res["fused_x_bitexact"] = bool(torch.equal(xk, xr))
    res["fused_bf16_bitexact_frac"] = float((of.view(torch.int16) == rf32.to(BF).view(torch.int16)).float().mean())
    res["fused_bf16_err_termulp_ext"] = bf16_err(of, rf64, Tf)
    res["fused_bf16_err_termulp_torch"] = bf16_err(rf32.to(BF), rf64, Tf)
    s = float(rf32.abs().amax()) / 448.0 * 0.5
    xk = x.clone()
    o8 = ext.residual_then_mod_norm_fp8(xk, y, g, sh, sc, EPS, s)
    r8 = ref_fp8(rf32, s)
    res["fused_fp8_byteexact_frac"] = float((o8.view(torch.uint8) == r8.view(torch.uint8)).float().mean())
    res["fused_fp8_max_ulp_diff"] = fp8_ulp_err(o8, r8)
    res["fused_fp8_saturated_frac"] = float((o8.float().abs() == 448).float().mean())
    o8b = ext.mod_norm_fp8(x, sh, sc, EPS, s)
    r8b = ref_fp8(r32, s)
    res["modnorm_fp8_byteexact_frac"] = float((o8b.view(torch.uint8) == r8b.view(torch.uint8)).float().mean())
    res["modnorm_fp8_max_ulp_diff"] = fp8_ulp_err(o8b, r8b)
    inv_s = (torch.ones((), device="cuda") / s).item()
    res["torch_scalar_div_eq_recip_mul"] = bool(torch.equal(rf32 / s, rf32 * inv_s))
    res["modnorm_fp8_vs_quant_of_ext_f32_bitexact"] = bool(torch.equal(o8b.view(torch.uint8), ref_fp8(o32, s).view(torch.uint8)))
    print(json.dumps({k: (float(f"{v:.7g}") if isinstance(v, float) else v) for k, v in res.items()}), flush=True)

def timing(M, use_compile):
    x, y, sh, sc, g = make(M, 1)
    shc, scc, gc = sh.contiguous(), sc.contiguous(), g.contiguous()
    s = 0.05
    xw = x.clone()
    MB = M * C
    comp = {}
    if use_compile:
        comp["mn"] = torch.compile(lambda x, sh, sc: ref_modnorm_bf16(x, sh, sc))
        comp["gr"] = torch.compile(lambda x, y, g: x + y * g)
        comp["fu"] = torch.compile(lambda x, y, g, sh, sc: (lambda xn: (xn, ref_modnorm_bf16(xn, sh, sc)))(x + y * g))
        comp["f8"] = torch.compile(lambda x, y, g, sh, sc: (lambda xn: (xn, ref_fp8(ref_modnorm32(xn, sh, sc), s)))(x + y * g))

    def row(op, f_torch, f_ext, bytes_moved, ck=None, cargs=()):
        r = {"kind": "timing", "M": M, "op": op, "MB_moved": round(bytes_moved / 1e6, 1)}
        for tag, cold in (("hot", False), ("cold", True)):
            tt, te = bench(f_torch, cold=cold), bench(f_ext, cold=cold)
            r[f"us_torch_{tag}"], r[f"us_ext_{tag}"] = round(tt, 1), round(te, 1)
            r[f"speedup_{tag}"], r[f"ext_GBps_{tag}"] = round(tt / te, 2), round(bytes_moved / (te * 1e-6) / 1e9)
            if ck in comp:
                r[f"us_compile_{tag}"] = round(bench(lambda: comp[ck](*cargs), cold=cold), 1)
        tt, te = bench_graph(f_torch), bench_graph(f_ext)
        r["us_torch_graph"], r["us_ext_graph"], r["speedup_graph"] = round(tt, 1), round(te, 1), round(tt / te, 2)
        print(json.dumps(r), flush=True)

    row("mod_norm->bf16", lambda: ref_modnorm_bf16(x, shc, scc), lambda: ext.mod_norm(x, sh, sc, EPS),
        MB * 4 + MB * 2, "mn", (x, shc, scc))
    row("gated_residual (ext in place)", lambda: x + y * gc, lambda: ext.gated_residual(xw, y, g),
        MB * 4 * 2 + MB * 2, "gr", (x, y, gc))

    def torch_fused():
        xn = x + y * gc
        return xn, ref_modnorm_bf16(xn, shc, scc)
    row("residual+mod_norm->bf16", torch_fused, lambda: ext.residual_then_mod_norm(xw, y, g, sh, sc, EPS),
        MB * 4 * 2 + MB * 2 + MB * 2, "fu", (x, y, gc, shc, scc))

    def torch_fused8():
        xn = x + y * gc
        return xn, ref_fp8(ref_modnorm32(xn, shc, scc), s)
    row("residual+mod_norm->fp8", torch_fused8, lambda: ext.residual_then_mod_norm_fp8(xw, y, g, sh, sc, EPS, s),
        MB * 4 * 2 + MB * 2 + MB, "f8", (x, y, gc, shc, scc))
    row("mod_norm->fp8", lambda: ref_fp8(ref_modnorm32(x, shc, scc), s), lambda: ext.mod_norm_fp8(x, sh, sc, EPS, s),
        MB * 4 + MB)

if __name__ == "__main__":
    if "--no-check" not in sys.argv:
        for M, R in [(3120, 1), (4680, 1), (4680, 3), (3120, 3120)]:
            check(M, R)
    for M in (3120, 4680):
        timing(M, "--compile" in sys.argv)
