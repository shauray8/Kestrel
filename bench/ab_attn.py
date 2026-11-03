import argparse
import json
import math
import time

import torch


def kernel_ms(fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def wall_ms(fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def token_sparsity(plan, m, n, h, selection_block):
    indices = plan.indices.to(torch.int64)
    lengths = (n - indices * selection_block).clamp(min=0, max=selection_block)
    lengths = torch.where(indices >= 0, lengths, 0)
    queries = (m - torch.arange(math.ceil(m / 128), device=indices.device) * 128).clamp(max=128)
    kept_pairs = (lengths.sum(-1) * queries[None]).sum().item()
    return 1 - kept_pairs / (m * n * h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=3120)
    parser.add_argument("--n", type=int, default=28080)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    parser.add_argument("--selection-block", type=int, choices=[4, 64], default=4)
    parser.add_argument("--sink-tokens", type=int, default=9360)
    parser.add_argument("--current-tokens", type=int)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--selector-iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.m, args.n, args.heads, args.iters, args.selector_iters) < 1 or args.warmup < 0:
        parser.error("lengths, heads, and iteration counts must be positive; warmup must be nonnegative")
    if any(not 0 <= sparsity <= 1 for sparsity in args.sparsities):
        parser.error("sparsities must be fractions in [0, 1]")
    current = args.m if args.current_tokens is None else args.current_tokens
    if not 0 <= args.sink_tokens <= args.n or not 0 <= current <= args.n:
        parser.error("sink and current token counts must be in [0, n]")
    if not torch.cuda.is_available():
        print(json.dumps({"kind": "skip", "reason": "CUDA is unavailable"}), flush=True)
        return
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("N2 extension targets sm_120a")

    from kestrel import ops
    from kestrel.ab import make_plan

    ext = ops.n2()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    q, k, v = (torch.randn(length, args.heads, 128, device="cuda", dtype=torch.float16,
                           generator=generator) for length in (args.m, args.n, args.n))
    shape = {"m": args.m, "n": args.n, "heads": args.heads, "head_dim": 128,
             "selection_block": args.selection_block, "sink_tokens": args.sink_tokens,
             "current_tokens": current, "iters": args.iters, "selector_iters": args.selector_iters,
             "warmup": args.warmup, "seed": args.seed, "device": torch.cuda.get_device_name()}
    dense = lambda: ext.n2_attention(q, k, v)
    dense_kernel = kernel_ms(dense, args.iters, args.warmup)
    dense_wall = wall_ms(dense, args.iters, args.warmup)
    print(json.dumps({"kind": "ab_dense", **shape, "dense_kernel_ms": dense_kernel,
                      "dense_wall_ms": dense_wall}), flush=True)

    for keep_sink, keep_current in ((True, True), (True, False), (False, True), (False, False)):
        sink_tokens, current_tokens = args.sink_tokens if keep_sink else 0, current if keep_current else 0
        for sparsity in args.sparsities:
            plan_generator = torch.Generator(device="cuda").manual_seed(args.seed)

            def select():
                return make_plan(q, k, sparsity=sparsity, sink_tokens=sink_tokens,
                                 current_tokens=current_tokens, clusters=args.clusters,
                                 selection_block=args.selection_block, generator=plan_generator)

            plan = select()
            torch.cuda.synchronize()
            sparse = lambda: ext.n2_attention_sparse(q, k, v, plan.indices, plan.log_bias, args.selection_block)
            sparse_kernel = kernel_ms(sparse, args.iters, args.warmup)
            selector_wall = wall_ms(select, args.selector_iters, args.warmup)

            def end_to_end():
                fresh = select()
                return ext.n2_attention_sparse(q, k, v, fresh.indices, fresh.log_bias, args.selection_block)

            end_to_end_wall = wall_ms(end_to_end, args.selector_iters, args.warmup)
            block = args.selection_block
            blocks = math.ceil(args.n / block)
            starts = torch.arange(blocks, device="cuda") * block
            mandatory = (starts < sink_tokens) | ((starts + block > args.n - current_tokens) & (current_tokens > 0))
            mandatory_tokens = (args.n - starts[mandatory]).clamp(max=block).sum().item()
            print(json.dumps({"kind": "ab_sparse", **shape, "clusters": args.clusters,
                              "keep_sink": keep_sink, "keep_current": keep_current,
                              "sink_tokens": sink_tokens, "current_tokens": current_tokens,
                              "requested_sparsity": sparsity, "achieved_block_sparsity": 1 - plan.kept / plan.total,
                              "achieved_token_sparsity": token_sparsity(plan, args.m, args.n, args.heads, block),
                              "mandatory_token_fraction": mandatory_tokens / args.n,
                              "kept_blocks": plan.kept, "total_blocks": plan.total,
                              "slots_per_query_block": plan.indices.shape[-1],
                              "dense_kernel_ms": dense_kernel, "dense_wall_ms": dense_wall,
                              "sparse_kernel_ms": sparse_kernel, "selector_wall_ms": selector_wall,
                              "selector_plus_kernel_wall_ms": end_to_end_wall}), flush=True)


if __name__ == "__main__":
    main()
