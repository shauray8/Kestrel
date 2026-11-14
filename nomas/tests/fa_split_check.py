import os, torch, triton
from kestrel import ops
torch.set_grad_enabled(False); torch.manual_seed(0)
os.environ["NOMAS_GLUE"] = "1"
from nomas import glue, attn
ext = ops.fa()
eng = attn._engine()
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
H = 16
worst = 0
for lq, lk in ((100, 20500), (300, 20500), (2400, 20500), (7168, 20500), (12212, 20500)):
    q = (torch.randn(1, lq, H, 128, device="cuda") * 1.5).bfloat16(); k = (torch.randn(1, lk, H, 128, device="cuda") * 1.5).bfloat16()
    v = torch.randn(1, lk, H, 128, device="cuda").bfloat16()
    st = glue._buffers(q, lk, lq); nb = st["k8"].shape[1]
    mean = k[0].float().mean(0)
    qi, ki = torch.arange(lq, device="cuda"), torch.arange(lk, device="cuda")
    for src, idx, name, sc, val, ctr in ((q[0], qi, "q8", "qs", False, st["zero"]), (k[0], ki, "k8", "ks", False, mean), (v[0], ki, "v8", "vs", True, mean)):
        glue._indexed_quant[(st[name].shape[1], H)](src, src, src, src, idx, st[name], st[sc], ctr, H, 0, src.shape[0], src.stride(0), 0, False, idx.numel(), st[name].shape[1], 0, val, 1, enable_fp_fusion=False)
    base = lambda: ext.fa_attention(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"], lq, lk, 128 ** -.5, eng.load_mode, eng.exp_mode, eng.rescale_threshold, st["plan"], 128, True, False, True, True)
    ref = torch.softmax(q[0].float().transpose(0, 1) @ k[0].float().transpose(0, 1).transpose(1, 2) * 128 ** -.5, -1) @ v[0].float().transpose(0, 1)
    ref = ref.transpose(0, 1)
    s = glue._splits(triton.cdiv(lq, 128) * H, nb)
    tab = torch.arange(nb, device="cuda", dtype=torch.int32); val = torch.full((nb,), 64, device="cuda", dtype=torch.int32); val[-1] = lk - (nb - 1) * 64
    for S in sorted({2, 4, s}):
        ws = torch.empty(S * H * lq * 130, device="cuda")
        split = lambda: ext.fa_attention_split(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"], tab, val, lq, S, 128 ** -.5, eng.exp_mode, eng.rescale_threshold, st["plan"], ws)
        ob, os_ = base(), split()
        eb = ((ob.float() - ref).norm() / ref.norm()).item(); es = ((os_.float() - ref).norm() / ref.norm()).item()
        tb, ts = bench(base), bench(split)
        worst = max(worst, es / eb)
        print(f"Lq{lq} Lk{lk} S={S}{'*' if S == s else ' '} base {tb:.3f} ms relL2 {eb:.4e} | split {ts:.3f} ms relL2 {es:.4e} | {tb/ts:.2f}x", flush=True)
print("CONTRACT", "PASS" if worst <= 1.1 else "FAIL", f"worst={worst:.3f}")
