import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_ffn_cluster import BF, D, FF, inputs, program
from kestrel import ops


def bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def case(M, gated, variants, warmup, iters):
    x8, sa, s_mid, W, residual, gate = inputs(M)
    gate = gate if gated else None
    result = {}
    for variant in variants:
        y = residual.clone() if gated else torch.empty(M, D, device="cuda", dtype=BF)
        if variant == "cluster":
            ext = ops.ffn_cluster()
            fn = lambda: ext.run(x8, sa, *W[:3], s_mid, *W[3:], y, gate)
        elif variant == "vm":
            p = program(x8, sa, s_mid, W, y, gate)
            fn = p.run
        else:
            n1 = ops.n1()
            fm = ops.fused_mod() if gated else None

            def fn():
                h8 = n1.n1_linear_f8in_gelu_f8(x8, *W[:3], sa, s_mid)
                out = n1.n1_linear_f8in(h8, *W[3:], s_mid)
                if gated:
                    fm.gated_residual(y, out, gate)
                return out

        ms = bench(fn, warmup, iters)
        if variant == "vm":
            status = p.status()
            assert status["err"] == "ok" and status["done"] == status["n"], status
        result[variant] = ms
        print(json.dumps({"kind": "ffn_cluster", "M": M, "gated": gated, "variant": variant,
                          "warmup": warmup, "iters": iters, "ms": ms, "tflops": 4 * M * D * FF / ms / 1e9,
                          "h8_logical_roundtrip_mib": 13 * M * FF / 2 ** 20,
                          "baseline_m_tiles": (M + 127) // 128, "cluster_m_tiles": (M + 63) // 64}), flush=True)
    if len(result) > 1:
        print(json.dumps({"kind": "ffn_cluster_speedup", "M": M, "gated": gated,
                          "vs_vm": result["vm"] / result["cluster"], "vs_n1": result["n1"] / result["cluster"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("all", "cluster", "vm", "n1"), default="all")
    parser.add_argument("--m", type=int, nargs="+", default=(3120, 4680))
    parser.add_argument("--gated", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    assert args.warmup >= 0 and args.iters > 0 and all(m > 0 for m in args.m)
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(json.dumps({"kind": "device", "name": device.name, "sms": device.multi_processor_count,
                      "cluster_blocks": 8, "cluster_rows": 64, "smem_per_block": 96 * 1024}), flush=True)
    variants = ("cluster", "vm", "n1") if args.variant == "all" else (args.variant,)
    for M in args.m:
        for gated in (False, True) if args.gated is None else (args.gated,):
            case(M, gated, variants, args.warmup, args.iters)


if __name__ == "__main__":
    main()
