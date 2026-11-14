import torch
from kestrel import ops
ext = ops.n3()
torch.manual_seed(0)
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
for name, m, k, n in [("dec q/o", 15600, 4096, 4096), ("dec k/v", 15600, 4096, 1024), ("dec gate+up", 15600, 4096, 24576),
                      ("dec down", 15600, 12288, 4096), ("dit qkv", 11700, 2048, 6144), ("dit fc1", 11700, 2048, 8192), ("dit fc2", 11700, 8192, 2048)]:
    x = (torch.randn(m, k, device="cuda") * 2).bfloat16()
    wf = torch.randn(n, k, device="cuda") * 0.02
    sw = (wf.abs().amax(1) / 448).float().contiguous()
    w = (wf / sw[:, None]).to(torch.float8_e4m3fn)
    sa = 0.25
    x8 = ext.n3_quantize(x, sa)
    ref = (x8.float() * sa) @ (w.float() * sw[:, None]).t()
    yn = ext.n3_linear_f8in(x8, w, sw, None, sa)
    sA = torch.full((m, 1), sa, device="cuda"); sB = sw.view(1, n)
    run = lambda: torch._scaled_mm(x8, w.t(), scale_a=sA, scale_b=sB, out_dtype=torch.bfloat16)
    try:
        yc = run(); tc = bench(run); ec = ((yc.float() - ref).norm() / ref.norm()).item()
    except Exception as e:
        tc, ec = float("nan"), float("nan"); print("rowwise err", str(e)[:200])
    tn = bench(lambda: ext.n3_linear_f8in(x8, w, sw, None, sa))
    en = ((yn.float() - ref).norm() / ref.norm()).item()
    print(f"{name:12s} n3 {tn:.3f} ms relL2 {en:.2e} | cuBLAS rowwise {tc:.3f} ms relL2 {ec:.2e} | speedup {tn/tc:.2f}x", flush=True)
