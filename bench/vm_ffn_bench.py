import json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_vm import BF, D, E4, EPS, FF, FRAME, TARGET, fm, n1, pow2, weights
from kestrel import vm

WINDOWS = (None, 4, 8, 12, 16, 24)

def bench(fn, it=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / it

def bench_graph(fn, n=10, reps=5):
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
    return s.elapsed_time(e) / (n * reps)

def report(kind, M, flops, res):
    base = res["tiles"]
    for k, ms in res.items():
        print(json.dumps({"kind": kind, "M": M, "variant": k, "ms": round(ms, 4), "tf": round(flops / ms / 1e9, 1),
                          "speedup_vs_tiles": round(base / ms, 3)}), flush=True)

def vm_sweep(make, res):
    best = None
    for w in WINDOWS:
        p = make(w)
        ms = bench(p.run)
        assert p.status()["err"] == "ok"
        res[f"vm_w{w}"] = ms
        best = best if best and best[0] <= ms else (ms, p, w)
    res[f"vm_graph_w{best[2]}"] = bench_graph(best[1].run)
    return best[1], best[2]

def prof(kind, M, p, w):
    for op, d in vm.profile(p).items():
        print(json.dumps({"kind": kind + "_prof", "M": M, "window": w, "op": op, **{k: round(v, 3) for k, v in d.items()}}), flush=True)

def ffn(M, g):
    W = weights(g)
    x = torch.randn(M, D, device="cuda", generator=g).to(BF)
    sa, s_mid = pow2(x.abs().max() / TARGET), 2.0 ** -2
    x8 = (x.float() / sa).to(E4)
    h8, y = torch.empty(M, FF, dtype=E4, device="cuda"), torch.empty(M, D, dtype=BF, device="cuda")
    ops_path = lambda: n1.n1_linear_f8in(n1.n1_linear_gelu_f8(x, W[0], W[1], W[2], sa, s_mid), W[3], W[4], W[5], s_mid)
    res = {}
    p, w = vm_sweep(lambda w: vm.ffn_program(x8, sa, W[0], W[1], W[2], h8, s_mid, W[3], W[4], W[5], y, w), res)
    res["tiles"], res["tiles_graph"] = bench(p.run_tiles), bench_graph(p.run_tiles)
    res["ops_n1"], res["ops_n1_graph"] = bench(ops_path), bench_graph(ops_path)
    report("vm_ffn", M, 4.0 * M * D * FF, res)
    prof("vm_ffn", M, p, w)

def glue(M, g):
    W = weights(g)
    R = M // FRAME
    x = torch.randn(M, D, device="cuda", generator=g)
    ca = torch.randn(M, D, device="cuda", generator=g).to(BF)
    shift, scale, gate = (torch.randn(R, D, device="cuda", generator=g) * 0.1 for _ in range(3))
    sa, s_mid = 2.0 ** -3, 2.0 ** -2
    h0, h8 = torch.empty(M, D, dtype=E4, device="cuda"), torch.empty(M, FF, dtype=E4, device="cuda")

    def ops_path():
        hin = fm.residual_then_mod_norm(x, ca, None, shift, scale, EPS)
        fm.gated_residual(x, n1.n1_linear_f8in(n1.n1_linear_gelu_f8(hin, W[0], W[1], W[2], sa, s_mid), W[3], W[4], W[5], s_mid), gate)

    res = {}
    p, w = vm_sweep(lambda w: vm.ffn_glue_program(x, ca, shift, scale, gate, h0, sa, W[0], W[1], W[2], h8, s_mid, W[3], W[4], W[5], EPS, w), res)
    res["tiles"], res["tiles_graph"] = bench(p.run_tiles), bench_graph(p.run_tiles)
    res["ops_n1_fused_mod"], res["ops_n1_fused_mod_graph"] = bench(ops_path), bench_graph(ops_path)
    report("vm_ffn_glue", M, 4.0 * M * D * FF, res)
    prof("vm_ffn_glue", M, p, w)

if __name__ == "__main__":
    g = torch.Generator(device="cuda").manual_seed(0)
    print(json.dumps({"kind": "vm_grid", "ctas": vm.grid()}), flush=True)
    for M in (4680, 3120):
        ffn(M, g)
        glue(M, g)
