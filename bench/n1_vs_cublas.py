import json
import torch
from kestrel import ops

n1 = ops.n1()
E4 = torch.float8_e4m3fn
flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)

def timed(fn, it=20):
    ts = []
    for i in range(it + 3):
        flush.fill_(i)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        if i >= 3:
            ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]

for name, M, N, K in [("lingbot_qkvo", 4680, 1536, 1536), ("lingbot_ffn_up", 4680, 8960, 1536), ("lingbot_ffn_down", 4680, 1536, 8960),
                      ("nomas_fc1", 11700, 8192, 2048), ("dec_down", 15600, 4096, 12288), ("square_8192", 8192, 8192, 8192)]:
    torch.manual_seed(0)
    a = (torch.randn(M, K, device="cuda") * 2.0).clamp(-16, 16).to(E4)
    w = (torch.randn(N, K, device="cuda") * 2.0).clamp(-16, 16).to(E4).contiguous()
    sw = torch.ones(N, device="cuda")
    one = torch.tensor(1.0, device="cuda")
    t_n1 = timed(lambda: n1.n1_linear_f8in(a, w, sw, None, 1.0))
    t_cb = min(timed(lambda: torch._scaled_mm(a, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=f)) for f in (False, True))
    y = n1.n1_linear_f8in(a, w, sw, None, 1.0).float()
    ref = a.float() @ w.float().t()
    tf = lambda t: round(2 * M * N * K / (t * 1e-3) / 1e12, 1)
    print(json.dumps({"kind": "n1_random_l2flush", "shape": name, "ms_n1": round(t_n1, 4), "ms_cublas_fp8": round(t_cb, 4),
                      "tf_n1": tf(t_n1), "tf_cublas": tf(t_cb), "n1_over_cublas": round(t_cb / t_n1, 3),
                      "n1_rel_err": round(float((y - ref).norm() / ref.norm()), 5), "n1_nonfinite": int((~torch.isfinite(y)).sum())}), flush=True)
