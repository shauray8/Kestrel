import json
import sys
import torch
from kestrel import ops

ext = ops.rope_kv()
if "--build-only" in sys.argv:
    sys.exit(0)

B, L, H, D, C = 1, 3120, 12, 128, 28080
flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

def tables(start_frame):
    f, h, w, c = 2, 30, 52, D // 2
    cuts = [c - 2 * (c // 3), c // 3, c // 3]
    pos = torch.arange(max(start_frame + f, h, w), device="cuda", dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(c, device="cuda", dtype=torch.float32).double() / c))
    freqs = torch.polar(torch.ones_like(inv), pos.double()[:, None] * inv[None, :])
    fs = freqs.split(cuts, dim=1)
    fi = torch.cat([fs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(L, 1, -1)
    return fi.real.float().contiguous(), fi.imag.float().contiguous()

def rope32(x, cos, sin):
    xr = x.float().reshape(B, L, H, D // 2, 2)
    a, b = xr[..., 0], xr[..., 1]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(3).type_as(x)

def reference(q, k, v, cos, sin, kc, vc, offset):
    rq, rk = rope32(q, cos, sin), rope32(k, cos, sin)
    kc[:, offset:offset + L] = rk
    vc[:, offset:offset + L] = v
    return rq

def errors(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "max_rel": float((d / b.float().abs().clamp_min(1e-6)).max()),
            "rel_l2": float(d.norm() / b.float().norm()), "mismatches": int((a != b).sum())}

def check(start_frame, offset):
    torch.manual_seed(1000 + start_frame)
    q, k, v = (torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    kc0 = torch.randn(B, C, H, D, device="cuda", dtype=torch.bfloat16)
    vc0 = torch.randn(B, C, H, D, device="cuda", dtype=torch.bfloat16)
    kc1, vc1, kc2, vc2 = kc0.clone(), vc0.clone(), kc0.clone(), vc0.clone()
    cos, sin = tables(start_frame)
    qr = reference(q, k, v, cos, sin, kc1, vc1, offset)
    qo = ext.rope_qk_append(q, k, v, cos, sin, kc2, vc2, offset)
    guard_k = torch.cat([kc1[:, :offset], kc1[:, offset + L:]], 1)
    guard_v = torch.cat([vc1[:, :offset], vc1[:, offset + L:]], 1)
    guard_ko = torch.cat([kc2[:, :offset], kc2[:, offset + L:]], 1)
    guard_vo = torch.cat([vc2[:, :offset], vc2[:, offset + L:]], 1)
    print(json.dumps({"kind": "correctness", "start_frame": start_frame, "write_offset": offset,
                      "q": errors(qo, qr), "k": errors(kc2, kc1), "v": errors(vc2, vc1),
                      "untouched": bool(torch.equal(guard_k, guard_ko) and torch.equal(guard_v, guard_vo))}), flush=True)

def bench(fn, iterations, cold):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    total = 0.0
    for _ in range(iterations):
        if cold:
            flush.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        total += s.elapsed_time(e)
    return total * 1000 / iterations

def timing():
    torch.manual_seed(7)
    q, k, v = (torch.randn(B, L, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    kc1 = torch.empty(B, C, H, D, device="cuda", dtype=torch.bfloat16)
    vc1 = torch.empty_like(kc1)
    kc2, vc2 = kc1.clone(), vc1.clone()
    cos, sin = tables(18)
    for cold in (False, True):
        ref_us = bench(lambda: reference(q, k, v, cos, sin, kc1, vc1, 24960), 20 if cold else 100, cold)
        fused_us = bench(lambda: ext.rope_qk_append(q, k, v, cos, sin, kc2, vc2, 24960), 20 if cold else 100, cold)
        print(json.dumps({"kind": "timing", "cache": "l2_flushed" if cold else "hot", "torch_us": ref_us,
                          "fused_us": fused_us, "speedup": ref_us / fused_us}), flush=True)

for sf, off in ((0, 0), (6, 9360), (18, 24960), (18, 18720)):
    check(sf, off)
timing()
