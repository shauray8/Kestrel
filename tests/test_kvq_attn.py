import json
import torch
import torch.nn.functional as F
from kestrel.kvq import KVQAttention
from sageattention import sageattn

def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())

caps = [torch.load(f"/workspace/captures/lingbot/call{i}.pt", weights_only=False) for i in range(5)]
blocks = sorted(caps[0]["attn"].keys())
kv = KVQAttention()
for c, cap in enumerate(caps):
    for bi, blk in enumerate(blocks):
        q, k, v = (cap["attn"][blk][t].cuda() for t in ("q", "k", "v"))
        ref = F.scaled_dot_product_attention(q[:, :1024].float().transpose(1, 2), k.float().transpose(1, 2), v.float().transpose(1, 2)).transpose(1, 2)
        o_kvq = kv(bi, q, k, v)[:, :1024]
        o_sage = sageattn(q, k, v, tensor_layout="NHD")[:, :1024]
        hist_same = c == 0 or torch.equal(caps[0]["attn"][blk]["k"][:, : k.shape[1] - q.shape[1]], cap["attn"][blk]["k"][:, : k.shape[1] - q.shape[1]])
        print(json.dumps({"kind": "kvq_acc", "call": c, "block": blk, "Lq": q.shape[1], "L": k.shape[1], "history_identical_to_call0": bool(hist_same),
                          "err_kvq": round(rel(o_kvq, ref), 5), "err_sage2": round(rel(o_sage, ref), 5)}), flush=True)
print(json.dumps({"kind": "kvq_stats", **kv.stats}))

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

for name, Lq, L in [("chunk3", 4680, 28080), ("chunk2", 3120, 28080)]:
    H = 12
    q = torch.randn(1, Lq, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, L, H, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, L, H, 128, device="cuda", dtype=torch.bfloat16)
    kq = KVQAttention()

    def chunk_kvq():
        kq.new_epoch()
        for _ in range(5):
            kq(0, q, k, v)

    def chunk_sage():
        for _ in range(5):
            sageattn(q, k, v, tensor_layout="NHD")
    t_k, t_s = bench(chunk_kvq), bench(chunk_sage)
    print(json.dumps({"kind": "kvq_speed", "shape": name, "ms_per_call_kvq": round(t_k / 5, 3), "ms_per_call_sage2": round(t_s / 5, 3),
                      "speedup": round(t_s / t_k, 3)}), flush=True)
