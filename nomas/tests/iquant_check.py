import glob, os, sys, torch, triton
from nomas import glue
from kestrel import ops
torch.set_grad_enabled(False)
ext = ops.iquant()
def timed(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
def case(name, args, grid):
    res = {}
    outs = {}
    for mode in ("triton", "0", "1"):
        out, so = torch.full_like(args[5], 7), torch.full_like(args[6], -1.)
        a = list(args); a[5], a[6] = out, so
        if mode == "triton":
            f = lambda: glue._indexed_quant[grid](*a, enable_fp_fusion=False)
        else:
            f = lambda: ext.indexed_quant(*a[:8], grid[1], *a[8:18], a[18], mode == "1")
        f(); outs[mode] = (out.clone(), so.clone()); res[mode] = timed(f)
    t = outs["triton"]
    for mode in ("0", "1"):
        o, s = outs[mode]
        print(f"{name} cuda{'fast' if mode=='1' else 'ieee'}: bytes_equal={torch.equal(o, t[0])} mismatched={(o != t[0]).sum().item()}/{o.numel()} "
              f"scales_equal={torch.equal(s, t[1])} ms triton {res['triton']:.3f} cuda {res[mode]:.3f} ({res['triton']/res[mode]:.2f}x)", flush=True)
for f in sorted(glob.glob(sys.argv[1] + "/dec_*.pt"))[:2]:
    d = torch.load(f, map_location="cuda")
    q, k, v, paged = d["q"][0], d["k"][0], d["v"][0], d["paged"]
    heads, kvh, nb = q.shape[1], k.shape[1], paged["blocks"]
    idx = paged["index"]
    mean = (k.float().mean(0)).contiguous()
    for name, src, value, center in (("K", k, False, mean), ("V", v, True, mean)):
        out = torch.empty((heads, nb, 128, 64) if value else (heads, nb, 64, 128), dtype=torch.uint8, device="cuda")
        so = torch.empty((heads, nb), device="cuda")
        case(f"{os.path.basename(f)} {name}", (src, src, src, src, idx, out, so, center, kvh, 0, src.shape[0], src.stride(0), 0, False,
              idx.numel(), nb, 0, value, heads // kvh), (nb, heads))
    qi = paged["groups"][0][0]; nq = triton.cdiv(qi.numel(), 64)
    out = torch.empty((heads, nq, 64, 128), dtype=torch.uint8, device="cuda"); so = torch.empty((heads, nq), device="cuda")
    case(f"{os.path.basename(f)} Q", (q, q, q, q, qi, out, so, torch.zeros(heads, 128, device="cuda"), heads, 0, q.shape[0], q.stride(0), 0, False,
          qi.numel(), nq, 0, False, 1), (nq, heads))
# DiT-like: int8 history cache + bf16 img/txt
H, hist, ni, nt = 16, 8300, 11700, 512
cache = torch.randint(-127, 128, (1, 9000, H, 128), dtype=torch.int8, device="cuda")[0]
cs = (torch.rand(9000, H, device="cuda") * 0.02)
img = (torch.randn(ni, H, 128, device="cuda") * 2).bfloat16(); txt = (torch.randn(nt, H, 128, device="cuda") * 2).bfloat16()
ki = torch.cat((torch.arange(hist, device="cuda"), hist + torch.arange(ni + nt, device="cuda")))
nb, first = triton.cdiv(ki.numel(), 64), hist // 64
mean = torch.randn(H, 128, device="cuda") * 0.1
for name, value in (("K", False), ("V", True)):
    out = torch.empty((H, nb, 128, 64) if value else (H, nb, 64, 128), dtype=torch.uint8, device="cuda"); so = torch.empty((H, nb), device="cuda")
    case(f"dit-int8 {name}", (cache, cs, img, txt, ki, out, so, mean, H, hist, ni, cache.stride(0), cs.stride(0), True, ki.numel(), nb, first, value, 1), (nb - first, H))
