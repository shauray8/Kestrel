import json
import torch
import torch.nn.functional as F
from kestrel import ops

fp4 = ops.nvfp4()
n1 = ops.n1()
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")

def fake_nvfp4(t, g):
    x = t.float().reshape(-1, 16)
    s = (x.abs().amax(1, keepdim=True) / g / 6.0).to(torch.float8_e4m3fn).float()
    y = torch.where(s > 0, x / (s * g), torch.zeros_like(x)).clamp(-6, 6)
    q = GRID[(y.abs().unsqueeze(-1) - GRID).abs().argmin(-1)] * y.sign()
    return (q * s * g).reshape(t.shape)

def mm_nvfp4(xq, xs, wq, ws, ga, gb, bias):
    y = torch._scaled_mm(xq.view(torch.float4_e2m1fn_x2), wq.view(torch.float4_e2m1fn_x2).t(),
                         scale_a=xs.view(torch.float8_e4m3fn), scale_b=ws.view(torch.float8_e4m3fn), out_dtype=torch.bfloat16)
    return torch.add(bias, y, alpha=ga * gb)

def bench(fn, it=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / it

for name, M, N, K in [("qkvo", 4680, 1536, 1536), ("ffn_up", 4680, 8960, 1536), ("ffn_down", 4680, 1536, 8960),
                      ("nomas_fc1", 11700, 8192, 2048), ("dec_down", 15600, 4096, 12288)]:
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda") * 1.3).bfloat16()
    w = (torch.randn(N, K, device="cuda") * 0.03)
    b = (torch.randn(N, device="cuda") * 0.1).bfloat16()
    ga = float(x.float().abs().max()) / (6 * 448)
    gb = float(w.abs().max()) / (6 * 448)
    xq, xs = fp4.nvfp4_quant(x, ga)
    wq, ws = fp4.nvfp4_quant(w.bfloat16().contiguous(), gb)
    y = mm_nvfp4(xq, xs, wq, ws, ga, gb, b)
    ref_fake = fake_nvfp4(x, ga) @ fake_nvfp4(w.bfloat16(), gb).t() + b.float()
    ref = x.float() @ w.t() + b.float()
    err_layout = float((y.float() - ref_fake).norm() / ref_fake.norm())
    err_true = float((y.float() - ref).norm() / ref.norm())
    wb = w.bfloat16()
    sw8 = (w.abs().amax(1) / 16).clamp(min=1e-12)
    w8 = (w / sw8[:, None]).to(torch.float8_e4m3fn).contiguous()
    sa8 = float(x.float().abs().max()) / 16
    t_bf = bench(lambda: F.linear(x, wb, b))
    t_n1 = bench(lambda: n1.n1_linear(x, w8, sw8.float(), b, sa8))
    t_q = bench(lambda: fp4.nvfp4_quant(x, ga))
    t_mm = bench(lambda: mm_nvfp4(xq, xs, wq, ws, ga, gb, b))
    t_fp4 = bench(lambda: mm_nvfp4(*fp4.nvfp4_quant(x, ga), wq, ws, ga, gb, b))
    tf = lambda t: round(2 * M * N * K / (t * 1e-3) / 1e12, 1)
    print(json.dumps({"kind": "nvfp4_linear", "shape": name, "err_vs_fake_ref": round(err_layout, 5), "err_vs_fp32": round(err_true, 4),
                      "ms_bf16": round(t_bf, 4), "ms_n1_fp8": round(t_n1, 4), "ms_quant": round(t_q, 4), "ms_gemm_only": round(t_mm, 4),
                      "ms_nvfp4_total": round(t_fp4, 4), "tf_gemm_only": tf(t_mm), "tf_total": tf(t_fp4),
                      "speedup_vs_bf16": round(t_bf / t_fp4, 2), "speedup_vs_n1": round(t_n1 / t_fp4, 2)}), flush=True)
