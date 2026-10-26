import json
import torch
import torch.nn.functional as F
from kestrel import ops

ext = ops.n2()
impls = {}
try:
    from sageattn3 import sageattn3_blackwell
    impls["sage3_fp4"] = lambda q, k, v: sageattn3_blackwell(q, k, v, is_causal=False)
except Exception as e:
    print(json.dumps({"kind": "import", "sage3": repr(e)[:300]}))
try:
    from sageattention import sageattn
    impls["sage2"] = lambda q, k, v: sageattn(q, k, v, tensor_layout="HND", is_causal=False)
except Exception as e:
    print(json.dumps({"kind": "import", "sage2": repr(e)[:300]}))
impls["sdpa_bf16"] = lambda q, k, v: F.scaled_dot_product_attention(q, k, v)
impls["n2_fp16"] = lambda q, k, v: ext.n2_attention(q[0].transpose(0, 1).half().contiguous(), k[0].transpose(0, 1).half().contiguous(),
                                                   v[0].transpose(0, 1).half().contiguous()).transpose(0, 1)[None]

def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())

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

cap = torch.load("/workspace/captures/lingbot/call1.pt", weights_only=False)
for blk, d in sorted(cap["attn"].items()):
    q, k, v = (d[t].cuda()[0].transpose(0, 1)[None].contiguous() for t in ("q", "k", "v"))
    qs = q[:, :, :1024].contiguous()
    ref = F.scaled_dot_product_attention(qs.float(), k.float(), v.float())
    row = {"kind": "attn_acc", "block": blk, "Lk": k.shape[2]}
    for name, fn in impls.items():
        try:
            row[name] = round(rel(fn(qs, k, v), ref), 5)
        except Exception as e:
            row[name] = repr(e)[:120]
    print(json.dumps(row), flush=True)

for name_s, Lq, Lk, H in [("lingbot_chunk3", 4680, 28080, 12), ("lingbot_chunk2", 3120, 28080, 12), ("nomas_like", 11700, 23400, 16)]:
    q = torch.randn(1, H, Lq, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, H, Lk, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, H, Lk, 128, device="cuda", dtype=torch.bfloat16)
    flops = 4 * Lq * Lk * H * 128
    row = {"kind": "attn_speed", "shape": name_s}
    for name, fn in impls.items():
        if name == "n2_fp16":
            qh, kh, vh = (t[0].transpose(0, 1).half().contiguous() for t in (q, k, v))
            ms = bench(lambda: ext.n2_attention(qh, kh, vh))
        else:
            try:
                ms = bench(lambda: fn(q, k, v))
            except Exception as e:
                row[name] = repr(e)[:120]; continue
        row[name + "_ms"] = round(ms, 3)
        row[name + "_tf"] = round(flops / ms / 1e9, 1)
    print(json.dumps(row), flush=True)
