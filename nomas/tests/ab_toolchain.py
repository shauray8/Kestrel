import glob, os, sys, torch
os.environ.setdefault("NOMAS_GLUE", "1")
from kestrel import ops
from nomas import glue, attn
torch.set_grad_enabled(False); torch.manual_seed(0)
tag = sys.argv[1]
ext = ops.n3()
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
outs, times = {}, {}
for name, m, k, n in [("dec_q", 15600, 4096, 4096), ("dec_kv", 15600, 4096, 1024), ("dit_qkv", 11700, 2048, 6144),
                      ("dit_fc1", 11700, 2048, 8192), ("dit_fc2", 11700, 8192, 2048)]:
    x = (torch.randn(m, k, device="cuda")).bfloat16()
    wf = torch.randn(n, k, device="cuda") * k ** -.5
    sw = (wf.abs().amax(1) / 448).float().contiguous(); w = (wf / sw[:, None]).to(torch.float8_e4m3fn)
    sa = float(x.float().abs().amax()) / 16
    x8 = ext.n3_quantize(x, sa)
    outs[name] = ext.n3_linear_f8in(x8, w, sw, None, sa); times[name] = bench(lambda: ext.n3_linear_f8in(x8, w, sw, None, sa))
    if name == "dit_fc1":
        mid = 0.05
        outs["fc1_gelu"] = ext.n3_linear_gelu_f8(x, w, sw, None, sa, mid).view(torch.uint8)
        times["fc1_gelu"] = bench(lambda: ext.n3_linear_gelu_f8(x, w, sw, None, sa, mid))
for f in sorted(glob.glob("/workspace/prof/dec/dec_*.pt"))[:2]:
    d = torch.load(f, map_location="cuda"); key = "fa_paged_" + os.path.basename(f)[:-3]
    outs[key] = glue._decoder_paged(d["q"], d["k"], d["v"], d["paged"])
    times[key] = bench(lambda: glue._decoder_paged(d["q"], d["k"], d["v"], d["paged"]))
torch.save({k: v.cpu() for k, v in outs.items()}, f"/workspace/prof/ab_{tag}.pt")
print(tag, " ".join(f"{k}={v:.4f}" for k, v in times.items()), flush=True)
