import glob, os, sys, torch
os.environ.setdefault("NOMAS_GLUE", "1")
from nomas import glue, attn
torch.set_grad_enabled(False)
def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
eng = attn._engine()
for f in sorted(glob.glob(sys.argv[1] + "/dec_*.pt"))[:3]:
    d = torch.load(f, map_location="cuda")
    q, k, v = d["q"], d["k"], d["v"]
    res = {}
    for qt in (128, 64, 128):
        eng.q_tile = qt
        o = glue._decoder_paged(q, k, v, d["paged"])
        res.setdefault(qt, []).append((bench(lambda: glue._decoder_paged(q, k, v, d["paged"])), o))
    same = torch.equal(res[128][0][1], res[64][0][1])
    t128 = min(t for t, _ in res[128]); t64 = res[64][0][0]
    print(f"{os.path.basename(f)} q_tile128 {t128:.3f} ms q_tile64 {t64:.3f} ms ({t128/t64:.2f}x) bitexact={same}", flush=True)
