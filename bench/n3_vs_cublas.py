import argparse
import itertools
import json
import os

import torch

from kestrel import ops


def timed(fn, iterations, flush):
    for _ in range(3):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for i in range(iterations + 3):
        if flush is not None:
            flush.fill_(i)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        if i >= 3:
            samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cold", action="store_true")
    args = parser.parse_args()
    n1, n3 = ops.n1(), ops.n3()
    if args.build_only:
        return
    torch.manual_seed(13)
    flush = torch.empty(256 * 1024 * 1024 // 4, device="cuda") if args.cold else None
    for m, (n, k) in itertools.product((3120, 4680, 129, 512), ((1536, 1536), (8960, 1536), (1536, 8960))):
        a = (torch.randn(m, k, device="cuda") * 1.3).bfloat16()
        aq = a.to(torch.float8_e4m3fn)
        w = (torch.randn(n, k, device="cuda") * 0.03).to(torch.float8_e4m3fn).contiguous()
        sw = torch.ones(n, device="cuda")
        one = torch.ones((), device="cuda")
        cases = {
            "n1_f8in": lambda: n1.n1_linear_f8in(aq, w, sw, None, 1.0),
            "n3_f8in": lambda: n3.n3_linear_f8in(aq, w, sw, None, 1.0),
            "n1_bf16in": lambda: n1.n1_linear(a, w, sw, None, 1.0),
            "n3_bf16in": lambda: n3.n3_linear(a, w, sw, None, 1.0),
            "n1_gelu_f8": lambda: n1.n1_linear_gelu_f8(a, w, sw, None, 1.0, 1.0),
            "n3_gelu_f8": lambda: n3.n3_linear_gelu_f8(a, w, sw, None, 1.0, 1.0),
        }
        for fast in (False, True):
            cases[f"cublas_f8in_fast{int(fast)}"] = lambda fast=fast: torch._scaled_mm(
                aq, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=fast)
            cases[f"cublas_bf16in_fast{int(fast)}"] = lambda fast=fast: torch._scaled_mm(
                a.to(torch.float8_e4m3fn), w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16, use_fast_accum=fast)
        for name, fn in cases.items():
            ms = timed(fn, args.iterations, flush)
            tflops = 2 * m * n * k / (ms * 1e9)
            print(json.dumps({"kind": name, "shape": [m, n, k], "cold": args.cold,
                              "stages": int(os.environ.get("N3_STAGES", "5")), "cont": os.environ.get("N3_CONT", "0") == "1", "ms": ms,
                              "tflops": tflops, "pct_1016_peak": 100 * tflops / 1016}), flush=True)


if __name__ == "__main__":
    main()
