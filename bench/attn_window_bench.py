import argparse
import json
import os

import torch
from kestrel import ops, vm


def bench(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ctas", type=int)
    args = ap.parse_args()
    assert args.warmup >= 0 and args.iters > 0
    ext = ops.n2()
    g = torch.Generator(device="cuda").manual_seed(42)
    Lq, H, D = 3120, 12, 128
    ctas = args.ctas or vm.grid()
    resources = ops.vm().resources()
    for Lk in (9360, 18720, 28080):
        q = torch.randn(Lq, H * D, device="cuda", dtype=torch.bfloat16, generator=g)
        k, v = (torch.randn(Lk, H * D, device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(2))
        qh, kh, vh = (t.view(1, t.size(0), H, D).half() for t in (q, k, v))
        out = torch.empty_like(q)
        p = vm.Program()
        p.attn_window(*(p.tensor(t) for t in (q, k, v, out)))
        p.build()
        os.environ["N2_ATTN_SHARED"] = "0"
        ref = ext.n2_attention(qh, kh, vh)
        n2_ms = bench(lambda: ext.n2_attention(qh, kh, vh), args.warmup, args.iters)
        os.environ["N2_ATTN_SHARED"] = "1"
        shared = ext.n2_attention(qh, kh, vh)
        shared_ms = bench(lambda: ext.n2_attention(qh, kh, vh), args.warmup, args.iters)
        os.environ["N2_ATTN_SHARED"] = "0"
        vm_ms = bench(lambda: p.run(ctas), args.warmup, args.iters)
        mismatch = (out != ref.to(torch.bfloat16).view_as(out)).sum().item()
        tiles_ms = bench(p.run_tiles, args.warmup, args.iters)
        tiles_mismatch = (out != ref.to(torch.bfloat16).view_as(out)).sum().item()
        shared_mismatch = (shared != ref).sum().item()
        profile = vm.profile(p, it=args.iters, ctas=ctas)
        status = p.status()
        result = {"Lq": Lq, "Lk": Lk, "H": H, "D": D, "warmup": args.warmup, "iters": args.iters,
                  "mode": int(os.environ.get("VM_ATTN_PERF", "0")), "AQ": vm.AQ, "ctas": ctas,
                  "instructions": p.n, "smem_kib": 96 if os.environ.get("VM_ATTN_PERF") == "96" else 48,
                  "n2_ms": n2_ms, "n2_shared_ms": shared_ms, "vm_ms": vm_ms, "tiles_ms": tiles_ms,
                  "vm_over_n2": vm_ms / n2_ms, "mismatch_vs_n2": mismatch,
                  "resources": resources,
                  "shared_mismatch_vs_n2": shared_mismatch, "tiles_mismatch_vs_n2": tiles_mismatch, "status": status, "profile": profile}
        print(json.dumps(result), flush=True)
        assert status["err"] == "ok" and status["done"] == status["n"] and mismatch == tiles_mismatch == shared_mismatch == 0


if __name__ == "__main__":
    main()
