import os, torch
os.environ["NOMAS_DEC_FUSE"] = "1"
from torch import nn
from nomas import gemm

torch.manual_seed(0)
hidden, inter, heads, kv = 4096, 12288, 32, 8
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = (gemm.N3Linear(nn.Linear(a, b, bias=False).cuda().bfloat16())
            for a, b in ((hidden, inter), (hidden, inter), (inter, hidden)))
class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = (gemm.N3Linear(nn.Linear(hidden, n * 128, bias=False).cuda().bfloat16())
            for n in (heads, kv, kv))
def timed(fn, n=10):
    for _ in range(2): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
ok = True
with torch.no_grad():
    for m in (15600, 6240):
        x = (torch.randn(1, m, hidden, device="cuda") * 3).bfloat16()
        mlp, att = MLP().eval(), Attn().eval()
        os.environ["NOMAS_DEC_FUSE"] = "0"
        ref = mlp.down_proj(nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
        refqkv = [mod(x) for mod in (att.q_proj, att.k_proj, att.v_proj)]
        t0 = timed(lambda: gemm.dec_mlp(mlp, x)); tq0 = timed(lambda: gemm.dec_qkv(att, x))
        os.environ["NOMAS_DEC_FUSE"] = "1"
        gemm.dec_mlp(mlp, x)
        out = gemm.dec_mlp(mlp, x); qkv = gemm.dec_qkv(att, x)
        e1 = torch.equal(out, ref); e2 = all(torch.equal(a, b) for a, b in zip(qkv, refqkv))
        t1 = timed(lambda: gemm.dec_mlp(mlp, x)); tq1 = timed(lambda: gemm.dec_qkv(att, x))
        os.environ["NOMAS_DEC_FUSE"] = "0"; t0b = timed(lambda: gemm.dec_mlp(mlp, x)); os.environ["NOMAS_DEC_FUSE"] = "1"
        ok &= e1 and e2
        print(f"M={m} mlp bitexact={e1} maxdiff={(out.float()-ref.float()).abs().max().item():.3g} ms {min(t0,t0b):.3f}->{t1:.3f} | "
              f"qkv bitexact={e2} ms {tq0:.3f}->{tq1:.3f}", flush=True)
print("FUSE", "PASS" if ok else "FAIL")
