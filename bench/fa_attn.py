import argparse, itertools, json, math, sys
from pathlib import Path
import torch
from kestrel import ops
from kestrel.fa import FAttention

SHAPES = list(itertools.product((3120, 6240), (9360, 18720, 28080)))

def numbers(value, cast, allowed=None):
    values = [cast(v) for v in value.split(",")]
    if not values or any(not math.isfinite(v) or v < 0 or (allowed is not None and v not in allowed) for v in values):
        raise ValueError(f"invalid sweep: {value}")
    return values

def relative(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))

def reference(q, k, v, chunk):
    k, v = k.float().transpose(0, 1), v.float().transpose(0, 1)
    return torch.cat([(torch.softmax((part.float().transpose(0, 1) @ k.transpose(1, 2)) * (128 ** -0.5), dim=-1) @ v).transpose(0, 1)
                      for part in q.split(chunk)], dim=0)

def packed_buffers(x, transpose=False):
    l, h, d = x.shape
    n = (l + 63) // 64
    return (torch.empty((h, n, d, 64) if transpose else (h, n, 64, d), dtype=torch.uint8, device=x.device),
            torch.empty((h, n), dtype=torch.float32, device=x.device))

def timing(fn, warmup, iters, graph):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            output = fn()
        fn = captured.replay
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iters

def inputs(args):
    if args.capture:
        for filename in args.capture:
            cap = torch.load(filename, map_location="cpu", weights_only=False)
            for block, values in sorted(cap["attn"].items()):
                if args.block and str(block) not in args.block:
                    continue
                q, k, v = (values[name] for name in ("q", "k", "v"))
                q, k, v = (x[0] if x.ndim == 4 and x.shape[0] == 1 else x for x in (q, k, v))
                if any(x.ndim != 3 or x.shape[-1] != 128 for x in (q, k, v)):
                    raise ValueError(f"capture {filename} block {block} must contain [1,L,H,128] or [L,H,128]")
                yield f"{Path(filename).name}:{block}", tuple(x.cuda().half().contiguous() for x in (q, k, v))
    else:
        shapes = SHAPES if not args.shape or "all" in args.shape else [tuple(map(int, s.lower().replace("x", ",").split(","))) for s in args.shape]
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        for lq, lk in shapes:
            yield f"q{lq}_k{lk}", tuple(torch.randn(l, 12, 128, dtype=torch.float16, device="cuda", generator=gen) for l in (lq, lk, lk))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--shape", action="append", help="Lq,Lk or all; H=12 D=128")
    ap.add_argument("--load-mode", default="0", help="comma-separated 0,1,3; 2 is unsupported on RTX50")
    ap.add_argument("--exp-mode", default="0", help="comma-separated 0,1,2")
    ap.add_argument("--rescale-threshold", default="0", help="comma-separated nonnegative thresholds; 0 is exact")
    ap.add_argument("--q-tile", default="128", help="comma-separated 64,128 query rows per CTA")
    ap.add_argument("--capture", action="append", help="capture .pt path; may repeat")
    ap.add_argument("--block", action="append", help="captured block key; may repeat")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reference-queries", type=int, default=128, help="0 disables FP32 checks")
    ap.add_argument("--reference-chunk", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--allow-missing-sage", action="store_true")
    ap.add_argument("--json", type=Path, help="also write all result records as a JSON array")
    args = ap.parse_args()
    try:
        modes = list(itertools.product(numbers(args.load_mode, int, (0, 1, 3)), numbers(args.exp_mode, int, (0, 1, 2)),
                                       numbers(args.rescale_threshold, float), numbers(args.q_tile, int, (64, 128))))
    except ValueError as exc:
        ap.error(str(exc))
    if any(mode[2] > 0.5 for mode in modes):
        ap.error("rescale-threshold must be in [0,0.5] log2 units")
    if args.iters < 1 or args.warmup < 1 or args.reference_chunk < 1 or args.reference_queries < 0:
        ap.error("iters, warmup, reference-chunk must be positive; reference-queries must be nonnegative")
    records = []
    def emit(row):
        records.append(row)
        print(json.dumps(row), flush=True)
    ext = ops.fa()
    emit({"kind": "fa_build", "extension": ext.__name__, "build_only": args.build_only})
    if args.build_only:
        if args.json:
            args.json.write_text(json.dumps(records, indent=2) + "\n")
        return 0
    if not torch.cuda.is_available():
        ap.error("CUDA required")
    from_sage = None
    try:
        from sageattention import sageattn
        from_sage = sageattn
    except ImportError as exc:
        if not args.allow_missing_sage:
            raise
        emit({"kind": "fa_import", "sage2": str(exc)})
    n2 = ops.n2()
    torch.backends.cuda.matmul.allow_tf32 = False
    emit({"kind": "fa_device", "name": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
          "torch": torch.__version__, "cuda": torch.version.cuda, "graph": args.graph,
          "fp8_peak_tops": 1016, "fp16_peak_tops": 510, "peak_basis": "dense 4*Lq*Lk*H*D operations"})
    count = 0
    with torch.inference_mode():
        for name, (q, k, v) in inputs(args):
            count += 1
            lq, h, d = q.shape
            lk = k.shape[0]
            if k.shape != v.shape or k.shape[1:] != q.shape[1:] or lk < lq:
                raise ValueError(f"invalid q/k/v shapes for {name}")
            flops = 4 * lq * lk * h * d
            base = {"shape": name, "Lq": lq, "Lk": lk, "H": h, "D": d, "graph": args.graph}
            nr = min(lq, args.reference_queries)
            ref = reference(q[:nr], k, v, args.reference_chunk) if nr else None
            baselines = {"n2_fp16": lambda: n2.n2_attention(q, k, v)}
            if from_sage is not None:
                baselines["sage2"] = lambda: from_sage(q[None], k[None], v[None], tensor_layout="NHD", is_causal=False)[0]
            baseline_ms, baseline_out = {}, {}
            for method, fn in baselines.items():
                output = fn()
                torch.cuda.synchronize()
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"{method} produced nonfinite output for {name}")
                baseline_out[method] = output[:nr].clone() if nr else None
                ms = timing(fn, args.warmup, args.iters, args.graph)
                baseline_ms[method] = ms
                tops = flops / ms / 1e9
                row = {"kind": "fa_speed", **base, "method": method, "ms": ms, "tops": tops,
                       "percent_peak": 100 * tops / (510 if method == "n2_fp16" else 1016)}
                if nr:
                    row.update(reference_queries=nr, rel_fp32=relative(output[:nr], ref))
                emit(row)
            buffers = (packed_buffers(q), packed_buffers(k), packed_buffers(v, True))
            def quantize():
                for i, (x, (packed, scales)) in enumerate(zip((q, k, v), buffers)):
                    ext.fa_quant(x, packed, scales, 0, (x.shape[0] + 63) // 64, i == 2, 1 if i == 2 else 8)
            quantize()
            (q8, qs), (k8, ks), (v8, vs) = buffers
            for mode in modes:
                plan = ext.fa_plan(k8, v8) if mode[0] in (1, 3) else None
                def cached():
                    return ext.fa_attention(q8, k8, v8, qs, ks, vs, lq, lk, 128 ** -0.5, *mode[:3], plan, mode[3])
                def inclusive():
                    quantize()
                    return cached()
                fa = FAttention(mode="fp8", load_mode=mode[0], exp_mode=mode[1], rescale_threshold=mode[2], q_tile=mode[3])
                fa(0, q[None], k[None], v[None])
                methods = {"fa_cached": cached, "fa_quant_inclusive": inclusive,
                           "fa_history_cached": lambda: fa(0, q[None], k[None], v[None])[0]}
                out = cached()
                if not torch.isfinite(out).all():
                    raise RuntimeError(f"FA produced nonfinite output for {name}, mode {mode}")
                accuracy = {"reference_queries": nr, "rel_fp32": relative(out[:nr], ref)} if nr else {}
                if nr and "sage2" in baseline_out:
                    accuracy["rel_sage2"] = relative(out[:nr], baseline_out["sage2"])
                for method, fn in methods.items():
                    ms = timing(fn, args.warmup, args.iters, args.graph)
                    tops = flops / ms / 1e9
                    emit({"kind": "fa_speed", **base, "method": method, "load_mode": mode[0], "exp_mode": mode[1],
                          "rescale_threshold": mode[2], "q_tile": mode[3], "ms": ms, "tops": tops, "percent_peak": 100 * tops / 1016,
                          "speedup_n2": baseline_ms["n2_fp16"] / ms,
                          **({"speedup_sage2": baseline_ms["sage2"] / ms} if "sage2" in baseline_ms else {}), **accuracy})
    if not count:
        ap.error("no input shapes or matching captured blocks")
    if args.json:
        args.json.write_text(json.dumps(records, indent=2) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
