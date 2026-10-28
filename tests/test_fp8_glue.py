import json
import torch
from kestrel import ops, vm

ext = ops.n1()
E4 = torch.float8_e4m3fn
K, N = 1536, 8960

def bench(fn, it=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / it

torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)
w = torch.randn(N, K, device="cuda") * 0.02
b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
sw = (w.abs().amax(1) / 16.0).float()
wq = (w / sw[:, None]).to(E4).contiguous()
w2 = torch.randn(K, N, device="cuda") * 0.02
b2 = (torch.randn(K, device="cuda") * 0.1).bfloat16()
sw2 = (w2.abs().amax(1) / 16.0).float()
wq2 = (w2 / sw2[:, None]).to(E4).contiguous()
wd = wq.float() * sw[:, None]
w2d = wq2.float() * sw2[:, None]
rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())
for M in (3120, 4680):
    x = (torch.randn(M, K, device="cuda") * 1.3).bfloat16()
    sa = float(x.float().abs().max()) / 16.0
    x8 = (x.float() / sa).to(E4).contiguous()
    xb = (x8.float() * sa).bfloat16()
    ref_h = torch.nn.functional.gelu((x8.float() * sa) @ wd.t() + b.float(), approximate="tanh")
    s_mid = float(ref_h.abs().max()) / 16.0
    old = ext.n1_linear_gelu_f8(xb, wq, sw, b, sa, s_mid)
    new = ext.n1_linear_f8in_gelu_f8(x8, wq, sw, b, sa, s_mid)
    h8_vm = torch.empty_like(old)
    y_vm = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")
    p = vm.ffn_program(x8, sa, wq, sw, b, h8_vm, s_mid, wq2, sw2, b2, y_vm)
    p.run_tiles()
    torch.cuda.synchronize()
    ob, nb = old.view(torch.uint8), new.view(torch.uint8)
    same = torch.equal(ob, nb)
    of, nf = old.float(), new.float()
    ulp = torch.exp2(torch.floor(torch.log2(torch.maximum(of.abs(), nf.abs()).clamp(min=1e-30))) - 3).clamp(min=2.0 ** -9)
    max_ulp = float(((of - nf).abs() / ulp).max())
    mismatch = float((ob != nb).float().mean())
    vm_mismatch = float((nb != h8_vm.view(torch.uint8)).float().mean())
    old_y = ext.n1_linear_f8in(old, wq2, sw2, b2, s_mid)
    new_y = ext.n1_linear_f8in(new, wq2, sw2, b2, s_mid)
    ref_y = ref_h @ w2d.t() + b2.float()
    rel_old_new = rel(new_y, old_y)
    rel_old_ref = rel(old_y, ref_y)
    rel_new_ref = rel(new_y, ref_y)
    t_old = bench(lambda: ext.n1_linear_gelu_f8(xb, wq, sw, b, sa, s_mid))
    t_new = bench(lambda: ext.n1_linear_f8in_gelu_f8(x8, wq, sw, b, sa, s_mid))
    print(json.dumps({"kind": "fp8_glue", "M": M, "K": K, "N": N, "h8_bit_identical": same,
                      "h8_mismatch_frac": mismatch, "h8_max_ulp_diff": max_ulp, "h8_mismatch_vs_vm": vm_mismatch,
                      "ffn_rel_new_vs_old": rel_old_new, "ffn_rel_old_vs_fp32": rel_old_ref, "ffn_rel_new_vs_fp32": rel_new_ref,
                      "speedup": round(t_old / t_new, 3)}), flush=True)
    assert rel_old_new < 0.05 and rel_new_ref < 0.1, f"unexpected FFN drift at M={M}"
