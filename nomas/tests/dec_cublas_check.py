import os, torch
from torch import nn
from nomas import gemm
torch.set_grad_enabled(False); torch.manual_seed(0)
hidden, inter = 4096, 12288
def mk(a, b):
    l = nn.Linear(a, b, bias=False).cuda(); nn.init.normal_(l.weight, std=a ** -.5); return gemm.N3Linear(l.bfloat16())
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); self.gate_proj, self.up_proj, self.down_proj = mk(hidden, inter), mk(hidden, inter), mk(inter, hidden)
def timed(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
mlp = MLP().eval()
x = (torch.randn(1, 15600, hidden, device="cuda")).bfloat16()
g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
wg = g.wq.float() * g.sw[:, None]; wu = u.wq.float() * u.sw[:, None]; wd = d.wq.float() * d.sw[:, None]
os.environ["NOMAS_DEC_FUSE"] = "1"; os.environ["NOMAS_DEC_CUBLAS"] = "0"
gemm.dec_mlp(mlp, x); fused = gemm.dec_mlp(mlp, x)
xf = x.float()[0]
ref = (nn.functional.silu(xf @ wg.t()) * (xf @ wu.t())) @ wd.t()
t_f = timed(lambda: gemm.dec_mlp(mlp, x))
os.environ["NOMAS_DEC_CUBLAS"] = "1"
cub = gemm.dec_mlp(mlp, x); t_c = timed(lambda: gemm.dec_mlp(mlp, x))
os.environ["NOMAS_DEC_CUBLAS"] = "0"; t_f2 = timed(lambda: gemm.dec_mlp(mlp, x))
ef = ((fused[0].float() - ref).norm() / ref.norm()).item(); ec = ((cub[0].float() - ref).norm() / ref.norm()).item()
print(f"mlp M=15600 fused-n3 {min(t_f,t_f2):.3f} ms relL2 {ef:.4e} | cublas {t_c:.3f} ms relL2 {ec:.4e} | speedup {min(t_f,t_f2)/t_c:.2f}x ratio {ec/ef:.3f}")
print("CONTRACT", "PASS" if ec <= 1.1 * ef else "FAIL")
