import json
import torch
from kestrel import ops

ext = ops.n1()
E4 = torch.float8_e4m3fn

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
    sw = (w.abs().amax(1) / 448.0).float()
    wq = (w / sw[:, None]).to(E4).contiguous()
    sa = float(x.float().abs().max()) / 16.0
    y = ext.n1_linear(x, wq, sw, b, sa)
    ref = (x.float() @ w.t() + b.float())
    err = float((y.float() - ref).norm() / ref.norm())
    ref_bf16 = (x @ w.bfloat16().t() + b)
    err_bf16 = float((ref_bf16.float() - ref).norm() / ref.norm())
    wb = w.bfloat16()
    t_bf16 = bench(lambda: torch.nn.functional.linear(x, wb, b))
    t_n1 = bench(lambda: ext.n1_linear(x, wq, sw, b, sa))

    def naive():
        xs = x.abs().amax(1, keepdim=True).float().clamp(min=1e-12) / 448.0
        xq = (x.float() / xs).to(E4)
        return torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=sw[None, :], out_dtype=torch.bfloat16) + b
    t_naive = bench(naive)
    tf = lambda t: round(2 * M * N * K / (t * 1e-3) / 1e12, 1)
    print(json.dumps({"kind": "n1_linear", "shape": name, "M": M, "N": N, "K": K, "rel_err_vs_fp32": round(err, 5),
                      "rel_err_bf16_vs_fp32": round(err_bf16, 5), "ms_bf16": round(t_bf16, 4), "ms_n1": round(t_n1, 4),
                      "ms_naive_fp8": round(t_naive, 4), "tf_bf16": tf(t_bf16), "tf_n1": tf(t_n1), "tf_naive_fp8": tf(t_naive),
                      "speedup_vs_bf16": round(t_bf16 / t_n1, 2)}), flush=True)
