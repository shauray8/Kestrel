import glob, json, os
import torch
import torch.nn.functional as F
from kestrel import ops

ext = ops.n2()

def bench(fn, it=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record(); e.synchronize()
    return s.elapsed_time(e) / it

def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())

def sdpa(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None])[0].transpose(0, 1)

caps = sorted(glob.glob("/workspace/captures/lingbot/call*.pt"))
for path in caps[1:2]:
    cap = torch.load(path, weights_only=False)
    for blk, d in sorted(cap["attn"].items()):
        q, k, v = (d[t].cuda()[0] for t in ("q", "k", "v"))
        qs = q[:1024].contiguous()
        ref = sdpa(qs.float(), k.float(), v.float())
        o_n2 = ext.n2_attention(qs.half().contiguous(), k.half().contiguous(), v.half().contiguous())
        o_v0 = ext.n2_attention_v0(qs.half().contiguous(), k.half().contiguous(), v.half().contiguous())
        o_v2 = ext.n2_attention_v2(qs.half().contiguous(), k.half().contiguous(), v.half().contiguous())
        o_bf = sdpa(qs, k, v)
        print(json.dumps({"kind": "n2_check", "cap": os.path.basename(path), "block": blk, "Lk": k.shape[0],
                          "err_n2_hsm": round(rel(o_n2, ref), 5), "err_n2_v0": round(rel(o_v0, ref), 5), "err_n2_v2": round(rel(o_v2, ref), 5),
                          "err_sdpa_bf16": round(rel(o_bf, ref), 5), "nonfinite_v2": int((~torch.isfinite(o_v2)).sum())}), flush=True)

for name, Lq, Lk, H in [("lingbot_chunk3_window18", 4680, 28080, 12), ("lingbot_chunk2_window18", 3120, 28080, 12),
                        ("nomas_seg_like", 11700, 23400, 16), ("square_8k", 8192, 8192, 16)]:
    torch.manual_seed(0)
    q = torch.randn(Lq, H, 128, device="cuda", dtype=torch.half)
    k = torch.randn(Lk, H, 128, device="cuda", dtype=torch.half)
    v = torch.randn(Lk, H, 128, device="cuda", dtype=torch.half)
    qb, kb, vb = q.bfloat16(), k.bfloat16(), v.bfloat16()
    flops = 4 * Lq * Lk * H * 128
    t_n2 = bench(lambda: ext.n2_attention(q, k, v))
    t_v0 = bench(lambda: ext.n2_attention_v2(q, k, v))
    t_bf = bench(lambda: sdpa(qb, kb, vb))
    print(json.dumps({"kind": "n2_speed", "shape": name, "Lq": Lq, "Lk": Lk, "H": H,
                      "ms_n2_v1": round(t_n2, 3), "ms_n2_v2": round(t_v0, 3), "ms_sdpa_bf16": round(t_bf, 3),
                      "tf_n2_v1": round(flops / t_n2 / 1e9, 1), "tf_n2_v2": round(flops / t_v0 / 1e9, 1),
                      "tf_sdpa_bf16": round(flops / t_bf / 1e9, 1), "best_speedup_vs_bf16": round(t_bf / min(t_n2, t_v0), 2)}), flush=True)
