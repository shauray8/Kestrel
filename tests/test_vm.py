import json, math, sys, time
import torch
import torch.nn.functional as F
from kestrel import ops, vm

n1, n2, fm, ext, rope = ops.n1(), ops.n2(), ops.fused_mod(), ops.vm(), ops.rope_kv()
if "--build-only" in sys.argv:
    sys.exit(0)
E4, BF, D, FF, FRAME, TARGET, EPS = torch.float8_e4m3fn, torch.bfloat16, 1536, 8960, 1560, 16.0, 1e-6

def pow2(v):
    return 2.0 ** math.ceil(math.log2(float(v)))

def qw(w):
    sw = (w.abs().amax(1) / TARGET).clamp(min=1e-12)
    return (w / sw[:, None]).to(E4).contiguous(), sw.float().contiguous()

def weights(g):
    w0 = torch.randn(FF, D, device="cuda", generator=g) * D ** -0.5
    w2 = torch.randn(D, FF, device="cuda", generator=g) * FF ** -0.5
    b0 = (torch.randn(FF, device="cuda", generator=g) * 0.1).to(BF)
    b2 = (torch.randn(D, device="cuda", generator=g) * 0.1).to(BF)
    (w0q, sw0), (w2q, sw2) = qw(w0), qw(w2)
    return w0q, sw0, b0, w2q, sw2, b2

def ref_up(a8, sa, W):
    return F.gelu((a8.float() * sa) @ (W[0].float() * W[1][:, None]).T + W[2].float(), approximate="tanh")

def ref_dn(h, s_mid, W):
    h8 = (h / s_mid).clamp(-448, 448).to(E4)
    return h8, (h8.float() * s_mid) @ (W[3].float() * W[4][:, None]).T + W[5].float()

def diff(a, b):
    return (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).float().mean().item()

def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()

def emit(d):
    print(json.dumps({k: (float(f"{v:.5g}") if isinstance(v, float) else v) for k, v in d.items()}), flush=True)

def ffn_case(M, window, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = weights(g)
    x = torch.randn(M, D, device="cuda", generator=g).to(BF)
    sa = pow2(x.abs().max() / TARGET)
    x8 = (x.float() / sa).to(E4)
    hr = ref_up(x8, sa, W)
    s_mid = pow2(hr.abs().max() / TARGET)
    h8r, yr = ref_dn(hr, s_mid, W)
    h8, y = torch.empty(M, FF, dtype=E4, device="cuda"), torch.empty(M, D, dtype=BF, device="cuda")
    p = vm.ffn_program(x8, sa, W[0], W[1], W[2], h8, s_mid, W[3], W[4], W[5], y, window)
    p.run()
    torch.cuda.synchronize()
    st = p.status()
    h8o = n1.n1_linear_gelu_f8((x8.float() * sa).to(BF), W[0], W[1], W[2], sa, s_mid)
    yo = n1.n1_linear_f8in(h8o, W[3], W[4], W[5], s_mid)
    hv, yv = h8.clone(), y.clone()
    h8.zero_(); y.zero_()
    p.run_tiles()
    torch.cuda.synchronize()
    r = {"test": "vm_ffn", "M": M, "window": window, **st, "h_mismatch_vs_n1": diff(hv, h8o), "y_mismatch_vs_n1": diff(yv, yo),
         "tiles_mismatch_vs_vm": diff(h8, hv) + diff(y, yv), "h_mismatch_vs_ref": diff(hv, h8r), "y_rel_vs_ref": rel(yv, yr),
         "n1_y_rel_vs_ref": rel(yo, yr)}
    r["pass"] = st["err"] == "ok" and st["done"] == st["n"] and r["h_mismatch_vs_n1"] == 0 and r["y_mismatch_vs_n1"] == 0 \
        and r["tiles_mismatch_vs_vm"] == 0 and r["y_rel_vs_ref"] < 2e-2
    emit(r)
    return r["pass"]

def glue_case(M, window, seed=1, bcast=False, rebind=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = weights(g)
    R = 1 if bcast else M // FRAME
    x = torch.randn(M, D, device="cuda", generator=g)
    ca = torch.randn(M, D, device="cuda", generator=g).to(BF)
    shift, scale, gate = (torch.randn(R, D, device="cuda", generator=g) * 0.1 for _ in range(3))
    rows = lambda t: t.repeat_interleave(M // t.size(0), 0)
    x1 = x + ca.float()
    nr = F.layer_norm(x1, (D,), eps=EPS) * (1 + rows(scale)) + rows(shift)
    sa = pow2(nr.abs().max() / TARGET)
    h0r = (nr / sa).clamp(-448, 448).to(E4)
    hr = ref_up(h0r, sa, W)
    s_mid = pow2(hr.abs().max() / TARGET)
    _, yr = ref_dn(hr, s_mid, W)
    xr = x1 + yr * rows(gate)

    xv = x.clone()
    h0, h8 = torch.empty(M, D, dtype=E4, device="cuda"), torch.empty(M, FF, dtype=E4, device="cuda")
    if rebind:
        e = torch.empty_like
        p = vm.ffn_glue_program(e(xv), e(ca), e(shift), e(scale), e(gate), e(h0), 1.0, e(W[0]), e(W[1]), e(W[2]), e(h8), 1.0, e(W[3]),
                                e(W[4]), e(W[5]), EPS, window)
        p.keep = []
        for k, t in dict(x=xv, ca=ca, shift=shift, scale=scale, gate=gate, b0=W[2].view(1, -1), b2=W[5].view(1, -1)).items():
            p.bind(k, t)
        p.bind("h0", h0, sa); p.bind("h8", h8, s_mid); p.bind("w0", W[0], vscale=W[1]); p.bind("w2", W[3], vscale=W[4])
    else:
        p = vm.ffn_glue_program(xv, ca, shift, scale, gate, h0, sa, W[0], W[1], W[2], h8, s_mid, W[3], W[4], W[5], EPS, window)
    p.run()
    torch.cuda.synchronize()
    st = p.status()

    xo = x.clone()
    h0o = fm.residual_then_mod_norm_fp8(xo, ca, None, shift, scale, EPS, sa)
    h8o = n1.n1_linear_gelu_f8((h0o.float() * sa).to(BF), W[0], W[1], W[2], sa, s_mid)
    yo = n1.n1_linear_f8in(h8o, W[3], W[4], W[5], s_mid)
    fm.gated_residual(xo, yo, gate)

    xb = x.clone()
    hin = fm.residual_then_mod_norm(xb, ca, None, shift, scale, EPS)
    yb = n1.n1_linear_f8in(n1.n1_linear_gelu_f8(hin, W[0], W[1], W[2], sa, s_mid), W[3], W[4], W[5], s_mid)
    fm.gated_residual(xb, yb, gate)

    xt = x.clone()
    p2 = vm.ffn_glue_program(xt, ca, shift, scale, gate, torch.empty_like(h0), sa, W[0], W[1], W[2], torch.empty_like(h8), s_mid,
                             W[3], W[4], W[5], EPS, window)
    p2.run_tiles()
    torch.cuda.synchronize()
    r = {"test": "vm_ffn_glue", "M": M, "window": window, "bcast": bcast, "rebind": rebind, **st, "h0_mismatch_vs_fused_mod": diff(h0, h0o),
         "h_mismatch_vs_n1": diff(h8, h8o),
         "x_mismatch_vs_ops": diff(xv, xo), "tiles_x_mismatch_vs_vm": diff(xt, xv),
         "x_rel_vs_ref": rel(xv, xr), "dx_rel_vs_ref": rel(xv - x1, xr - x1), "ops_bf16_dx_rel_vs_ref": rel(xb - x1, xr - x1),
         "dx_rel_vs_ops_bf16": rel(xv - x1, xb - x1)}
    r["pass"] = st["err"] == "ok" and st["done"] == st["n"] and r["h0_mismatch_vs_fused_mod"] == 0 and r["h_mismatch_vs_n1"] == 0 \
        and r["x_mismatch_vs_ops"] == 0 and r["tiles_x_mismatch_vs_vm"] == 0 and r["dx_rel_vs_ref"] < 2e-2
    emit(r)
    return r["pass"]

def rope_kv_case(L=257, H=12, HD=128, C=800, offset=137):
    g = torch.Generator(device="cuda").manual_seed(4)
    q, k, v = (torch.randn(1, L, H, HD, device="cuda", dtype=BF, generator=g) for _ in range(3))
    pos = torch.arange(L, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(HD // 2, device="cuda", dtype=torch.float32) / (HD // 2)))
    theta = pos[:, None] * inv[None, :]
    cos, sin = theta.cos()[:, None].contiguous(), theta.sin()[:, None].contiguous()
    cs = torch.stack((cos[:, 0], sin[:, 0]), -1).flatten(1).contiguous()
    kc0 = torch.randn(1, C, H, HD, device="cuda", dtype=BF, generator=g)
    vc0 = torch.randn(1, C, H, HD, device="cuda", dtype=BF, generator=g)
    kc_ref, vc_ref, kc_vm, vc_vm = kc0.clone(), vc0.clone(), kc0.clone(), vc0.clone()
    qr = rope.rope_qk_append(q, k, v, cos, sin, kc_ref, vc_ref, offset)
    qv = q.clone().view(L, H * HD)
    p = vm.Program()
    ids = [p.tensor(t) for t in (qv, k.view(L, H * HD), v.view(L, H * HD), cs,
                                  kc_vm.view(C, H * HD), vc_vm.view(C, H * HD))]
    p.rope_kv_append(*ids, offset)
    p.build()
    p.run()
    torch.cuda.synchronize()
    st = p.status()
    r = {"test": "vm_rope_kv_append", "L": L, "offset": offset, **st, "q_mismatch_vs_rope_kv": diff(qv.view_as(qr), qr),
         "k_mismatch_vs_rope_kv": diff(kc_vm, kc_ref), "v_mismatch_vs_rope_kv": diff(vc_vm, vc_ref)}
    r["pass"] = st["err"] == "ok" and st["done"] == st["n"] and r["q_mismatch_vs_rope_kv"] == 0 \
        and r["k_mismatch_vs_rope_kv"] == 0 and r["v_mismatch_vs_rope_kv"] == 0
    emit(r)
    return r["pass"]

def kv_advance_case(rows=400, width=1536, sink=32, drop=32):
    g = torch.Generator(device="cuda").manual_seed(5)
    k, v = (torch.randn(rows, width, device="cuda", dtype=BF, generator=g) for _ in range(2))
    kr, vr = k.clone(), v.clone()
    kr[sink:rows - drop] = kr[sink + drop:rows].clone()
    vr[sink:rows - drop] = vr[sink + drop:rows].clone()
    p = vm.Program()
    ki, vi = p.tensor(k), p.tensor(v)
    p.kv_advance(ki, vi, sink, drop, rows)
    p.build()
    p.run()
    torch.cuda.synchronize()
    st = p.status()
    r = {"test": "vm_kv_advance", "rows": rows, "sink": sink, "drop": drop, **st,
         "k_mismatch": diff(k, kr), "v_mismatch": diff(v, vr)}
    r["pass"] = st["err"] == "ok" and st["done"] == st["n"] and r["k_mismatch"] == 0 and r["v_mismatch"] == 0
    emit(r)
    return r["pass"]
def attn_window_case(Lq=127, Lk=193, H=4):
    g = torch.Generator(device="cuda").manual_seed(6)
    q = torch.randn(Lq, H * 128, device="cuda", dtype=BF, generator=g)
    k, v = (torch.randn(Lk, H * 128, device="cuda", dtype=BF, generator=g) for _ in range(2))
    ref = n2.n2_attention(q.view(1, Lq, H, 128).half(), k.view(1, Lk, H, 128).half(),
                          v.view(1, Lk, H, 128).half()).to(BF).view_as(q)
    out = torch.empty_like(q)
    p = vm.Program()
    qi, ki, vi, oi = (p.tensor(t) for t in (q, k, v, out))
    p.attn_window(qi, ki, vi, oi)
    p.build()
    p.run()
    torch.cuda.synchronize()
    st = p.status()
    r = {"test": "vm_attn_window", "Lq": Lq, "Lk": Lk, "H": H, **st,
         "mismatch_vs_n2": diff(out, ref), "rel_vs_n2": rel(out, ref)}
    r["pass"] = st["err"] == "ok" and st["done"] == st["n"] and r["mismatch_vs_n2"] == 0
    emit(r)
    return r["pass"]


def block_ops_case(M=3120, L=512):
    g = torch.Generator(device="cuda").manual_seed(7)
    def linear(N, K):
        w = torch.randn(N, K, device="cuda", generator=g) * K ** -0.5
        b = (torch.randn(N, device="cuda", generator=g) * 0.1).to(BF)
        wq, sw = qw(w)
        return wq, sw, b
    W = [linear(D, D) for _ in range(4)]
    x = torch.randn(M, D, device="cuda", generator=g)
    shift, scale, gate = (torch.randn(1, D, device="cuda", generator=g) * 0.1 for _ in range(3))
    sa = 0.25
    mod = fm.mod_norm(x.clone(), shift, scale, EPS)
    refs = [n1.n1_linear(mod, w[0], w[1], w[2], sa) for w in W[:3]]
    outs = [torch.empty(M, D, dtype=BF, device="cuda") for _ in range(3)]
    p = vm.self_qkv_program(x, shift, scale, EPS, sa, *W[0], *W[1], *W[2], *outs)
    p.run(); torch.cuda.synchronize()
    st = p.status()
    qkv_diff = sum(diff(a, b) for a, b in zip(outs, refs))
    attn = torch.randn(M, D, device="cuda", dtype=BF, generator=g)
    xr = x.clone()
    yr = n1.n1_linear(attn, *W[3], sa)
    fm.gated_residual(xr, yr, gate)
    xo = x.clone()
    y = torch.empty_like(attn)
    po = vm.o_residual_program(attn, sa, *W[3], y, xo, gate)
    po.run(); torch.cuda.synchronize()
    residual_diff = diff(xo, xr)
    context = torch.randn(L, D, device="cuda", dtype=BF, generator=g)
    qc = torch.empty(M, D, dtype=BF, device="cuda")
    kc, vc = (torch.empty(L, D, dtype=BF, device="cuda") for _ in range(2))
    pc = vm.cross_qkv_program(x, context, sa, sa, *W[0], *W[1], *W[2], qc, kc, vc)
    pc.run(); torch.cuda.synchronize()
    cross_diff = diff(qc, refs[0]) + sum(diff(a, n1.n1_linear(context, w[0], w[1], w[2], sa)) for a, w in zip((kc, vc), W[1:3]))
    nq, nk = (torch.randn(1, D, device="cuda", dtype=BF, generator=g) for _ in range(2))
    pa, ca = vm.cross_attention_program(qc, kc, vc, nq, nk, EPS)
    def rms(a, w):
        return (a.float() * torch.rsqrt(a.float().pow(2).mean(-1, keepdim=True) + EPS)).to(BF) * w
    qr, kr = rms(qc, nq), rms(kc, nk)
    H = D // 128
    ar = n2.n2_attention(qr.view(1, M, H, 128).half(), kr.view(1, L, H, 128).half(), vc.view(1, L, H, 128).half()).to(BF).view(M, D)
    pa.run(); torch.cuda.synchronize()
    attn_rel, attn_diff = rel(ca, ar), diff(ca, ar)
    H, offset = D // 128, 64
    pos = torch.arange(M, device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(64, device="cuda", dtype=torch.float32) / 64))
    theta = pos[:, None] * inv[None, :]
    cos, sin = theta.cos()[:, None].contiguous(), theta.sin()[:, None].contiguous()
    cs = torch.stack((cos[:, 0], sin[:, 0]), -1).flatten(1).contiguous()
    kc0 = torch.randn(offset + M, D, device="cuda", dtype=BF, generator=g)
    vc0 = torch.randn(offset + M, D, device="cuda", dtype=BF, generator=g)
    kcr, vcr, kcv, vcv = kc0.clone(), vc0.clone(), kc0.clone(), vc0.clone()
    q4, k4, v4 = (t.view(1, M, H, 128) for t in outs)
    qr4, kr4 = rms(outs[0], nq).view(1, M, H, 128), rms(outs[1], nk).view(1, M, H, 128)
    qr4 = rope.rope_qk_append(qr4, kr4, v4, cos, sin, kcr.view(1, offset + M, H, 128), vcr.view(1, offset + M, H, 128), offset)
    ar4 = n2.n2_attention(qr4.half(), kcr[:offset + M].view(1, offset + M, H, 128).half(),
                          vcr[:offset + M].view(1, offset + M, H, 128).half()).to(BF)
    ps, sa4 = vm.self_attention_program(q4, k4, v4, cs, kcv, vcv, 0, offset + M, offset, None, nq, nk, EPS)
    ps.run(); torch.cuda.synchronize()
    self_rel = rel(sa4, ar4)
    statuses = [p.status(), po.status(), pc.status(), pa.status(), ps.status()]
    r = {"test": "vm_block_ops", "M": M, "L": L, **st, "qkv_mismatch": qkv_diff, "cross_qkv_mismatch": cross_diff,
         "residual_mismatch": residual_diff, "cross_attn_mismatch": attn_diff, "cross_attn_rel": attn_rel, "self_attn_rel": self_rel}
    r["pass"] = all(s["err"] == "ok" and s["done"] == s["n"] for s in statuses) and qkv_diff == 0 and cross_diff == 0 \
        and residual_diff == 0 and attn_rel < 1e-3 and self_rel < 1e-3
    emit(r)
    return r["pass"]

def cycle_case(M=3120, timeout_ms=50.0):
    g = torch.Generator(device="cuda").manual_seed(2)
    W = weights(g)
    x8 = (torch.randn(M, D, device="cuda", generator=g) * 4).to(E4)
    h8, y = torch.empty(M, FF, dtype=E4, device="cuda"), torch.empty(M, D, dtype=BF, device="cuda")
    p = vm.Program()
    up, dn = vm.ffn(p, p.tensor(x8, 1 / 16), W[0], W[1], W[2], h8, 1.0, W[3], W[4], W[5], p.tensor(y))
    p.edge(dn[0][0], up[0][0])
    p.build()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    p.run(timeout_ms=timeout_ms)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    st = p.status()
    ok = ffn_case(M, None, seed=3)
    r = {"test": "vm_cycle_watchdog", "M": M, "timeout_ms": timeout_ms, "wall_ms": wall, **st, "expected_done": st["n"] - 2 - len(dn[0]),
         "healthy_after": ok}
    r["pass"] = st["err"] == "timeout" and st["done"] == r["expected_done"] and ok
    emit(r)
    return r["pass"]

if __name__ == "__main__":
    res = [ffn_case(M, w) for M in (4680, 3120) for w in (None, 8)]
    res += [glue_case(M, w) for M in (4680, 3120) for w in (None, 8)]
    res += [glue_case(4680, None, rebind=True), glue_case(3120, None, bcast=True, rebind=True), rope_kv_case(), kv_advance_case(),
            attn_window_case(), block_ops_case()]
    res.append(cycle_case())
    emit({"test": "vm_all", "pass": all(res), "n": len(res)})
    sys.exit(0 if all(res) else 1)
