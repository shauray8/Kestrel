import argparse
import importlib
import importlib.metadata
import inspect
import json
import math
import statistics
from pathlib import Path

import torch
from kestrel import ops


REV = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
SOURCE = f"https://github.com/thu-ml/SageAttention/blob/{REV}/"
SCALE = 128 ** -0.5
LKS = (3120, 9360, 18720, 28080)


def location(fn):
    try:
        return f"{inspect.getsourcefile(fn)}:{inspect.getsourcelines(fn)[1]}"
    except (OSError, TypeError):
        return None


def package(name, symbol):
    try:
        module = importlib.import_module(name)
        fn = getattr(module, symbol)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        return fn, {"package_version": version, "installed_source": location(fn)}
    except Exception as exc:
        return None, {"import_error": repr(exc)}


def reference(q, k, v, chunk):
    k, v = k.float().transpose(0, 1), v.float().transpose(0, 1)
    return torch.cat([(torch.softmax((part.float().transpose(0, 1) @ k.transpose(1, 2)) * SCALE, dim=-1) @ v).transpose(0, 1)
                      for part in q[:512].split(chunk)], dim=0)


def timing(fn, prepare, warmup, iters):
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    output = None
    for _ in range(iters):
        if prepare is not None:
            prepare()
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), output


def inputs(args):
    captured = None
    if args.capture:
        captured = torch.load(args.capture, map_location="cpu", weights_only=True)
        if not isinstance(captured, dict) or any(name not in captured for name in ("q", "k", "v")):
            raise ValueError("--capture must contain a dictionary with q, k, v tensors [L,12,128]")
        captured = tuple(captured[name] for name in ("q", "k", "v"))
        if any(not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1:] != (12, 128) for x in captured):
            raise ValueError("captured q, k, v must each have shape [L,12,128]")
        if captured[0].shape[0] < 3120 or captured[1].shape != captured[2].shape or captured[1].shape[0] < max(args.lk):
            raise ValueError("capture must supply at least 3120 queries and max(--lk) matching keys/values")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for lk in args.lk:
        lengths = (3120, lk, lk)
        if captured is None:
            tensors = tuple(torch.randn(length, 12, 128, dtype=torch.bfloat16, device="cuda", generator=generator) for length in lengths)
        else:
            tensors = tuple(x[:length].to(device="cuda", dtype=torch.bfloat16).contiguous() for x, length in zip(captured, lengths))
        yield tensors


def sage2_kernel(q, k, v):
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("Sage2 kernel-only preprocessing mirrors the sm120 dispatch; requires sm120")
    from sageattention.quant import per_warp_int8, per_channel_fp8
    from sageattention import _qattn_sm89
    kernel = _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
    km = k.mean(dim=1, keepdim=True)
    qi, qs, ki, ks = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
    vi, vs, _ = per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25, smooth_v=False)
    output = torch.empty_like(q)

    def run():
        kernel(qi, ki, vi, output, qs, ks, vs, 0, 0, 2, SCALE, 0)
        return output

    details = {"kernel": "sageattention._qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf",
               "kernel_arguments": ["q_int8", "k_int8", "v_fp8", "output_bf16", "q_scale", "k_scale", "v_scale", 0, 0, 2, SCALE, 0],
               "scalar_arguments": {"tensor_layout": 0, "is_causal": 0, "qk_quant_gran": 2, "sm_scale": SCALE, "return_lse": 0},
               "preprocessing": {"k_mean": "k.mean(dim=1, keepdim=True)",
                                 "qk": "sageattention.quant.per_warp_int8", "BLKQ": 128, "WARPQ": 32, "BLKK": 64,
                                 "v": "sageattention.quant.per_channel_fp8", "scale_max": 2.25, "smooth_v": False,
                                 "tensor_layout": "NHD", "timed": False},
               "sources": [SOURCE + "sageattention/core.py#L149-L150", SOURCE + "sageattention/core.py#L766-L812",
                           SOURCE + "sageattention/quant.py#L105-L180", SOURCE + "sageattention/quant.py#L224-L293",
                           SOURCE + "sageattention/sm89_compile.py#L45-L63", SOURCE + "setup.py#L186-L205"]}
    return run, lambda out: out[0], None, details


def sage3_end_to_end(fn, q, k, v):
    scratch_k = torch.empty_like(k)

    def prepare():
        scratch_k.copy_(k)

    def run():
        return fn(q, scratch_k, v, is_causal=False)

    return run, lambda out: out[0].transpose(0, 1), prepare, {
        "api": "sageattn3.sageattn3_blackwell", "tensor_layout": "HND", "is_causal": False,
        "k_restore": "copy_ before each sample, outside CUDA events; upstream preprocess_qkv mutates K",
        "sources": [SOURCE + "sageattention3_blackwell/sageattn3/api.py#L72-L89",
                    SOURCE + "sageattention3_blackwell/sageattn3/api.py#L127-L148"]}


def sage3_kernel(q, k, v):
    from sageattn3.api import preprocess_qkv, scale_and_quant_fp4, scale_and_quant_fp4_permute, scale_and_quant_fp4_transpose
    import fp4attn_cuda
    lq, lk = q.shape[2], k.shape[2]
    pq, pk, pv, delta_s = preprocess_qkv(q, k.clone(), v, True)
    qi, qs = scale_and_quant_fp4(pq)
    ki, ks = scale_and_quant_fp4_permute(pk)
    vi, vs = scale_and_quant_fp4_transpose(pv)

    def run():
        return fp4attn_cuda.fwd(qi, ki, vi, qs, ks, vs, delta_s, lk, None, SCALE, False, True, True)

    details = {"kernel": "fp4attn_cuda.fwd",
               "kernel_arguments": ["q_fp4", "k_fp4", "v_fp4", "q_scale", "k_scale", "v_scale", "delta_s", lk, None, SCALE, False, True, True],
               "scalar_arguments": {"KL": lk, "out": None, "softmax_scale": SCALE, "is_causal": False,
                                    "per_block_mean": True, "is_bf16": True},
               "preprocessing": {"functions": ["sageattn3.api.preprocess_qkv", "sageattn3.api.scale_and_quant_fp4",
                                                "sageattn3.api.scale_and_quant_fp4_permute", "sageattn3.api.scale_and_quant_fp4_transpose"],
                                 "per_block_mean": True, "k_clone": True, "timed": False},
               "postprocessing": "output[0][0, :, :Lq, :].transpose(0, 1), outside timing",
               "sources": [SOURCE + "sageattention3_blackwell/sageattn3/api.py#L72-L148"]}
    return run, lambda out: out[0][0, :, :lq, :].transpose(0, 1), None, details


def packed_buffers(x, transpose=False):
    length, heads, dim = x.shape
    blocks = (length + 63) // 64
    return (torch.empty((heads, blocks, dim, 64) if transpose else (heads, blocks, 64, dim), dtype=torch.uint8, device=x.device),
            torch.empty((heads, blocks), dtype=torch.float32, device=x.device))


def ours(ext, q, k, v, mx, bal, inclusive, i8=False):
    (qi, qs), (ki, ks), (vi, vs) = packed_buffers(q), packed_buffers(k), packed_buffers(v, True)

    def quantize():
        mean = k.float().mean(0).contiguous() if i8 else None
        for i, (x, packed, scales) in enumerate(((q, qi, qs), (k, ki, ks), (v, vi, vs))):
            target = (448 if mx else 1) if i == 2 else (127 if i8 else 8)
            ext.fa_quant(x, packed, scales, 0, (x.shape[0] + 63) // 64, i == 2, target, mx and i == 2, i8 and i < 2, mean if i == 1 else None)

    quantize()
    plan = ext.fa_plan(ki, vi)

    def run():
        if inclusive:
            quantize()
        return ext.fa_attention(qi, ki, vi, qs, ks, vs, q.shape[0], k.shape[0], SCALE, 3, 0, 0.5, plan, 128, mx, bal, i8)

    return run, lambda out: out, None, {"kernel": "kestrel.ops.fa().fa_attention", "mx": mx, "bal": bal, "i8": i8,
                                      "load_mode": 3, "exp_mode": 0, "rescale_threshold": 0.5, "q_tile": 128,
                                      "qk_target": 127 if i8 else 8, "v_target": 448 if mx else 1, "v_pow2": mx,
                                      "quantization_timed": inclusive, "plan_timed": False}


def main():
    parser = argparse.ArgumentParser(description="BF16 attention head-to-head: CUDA-event median; FP32 accuracy on first 512 queries")
    parser.add_argument("--capture", type=Path, help="dictionary containing q,k,v [L,12,128]; prefixes selected for each shape")
    parser.add_argument("--lk", type=int, nargs="+", choices=LKS, default=list(LKS))
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reference-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 1 or args.reference_chunk < 1:
        parser.error("iters, warmup, and reference-chunk must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sage2, sage2_info = package("sageattention", "sageattn")
    sage3, sage3_info = package("sageattn3", "sageattn3_blackwell")
    ext = ops.fa()
    rows = []
    device = {"device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "torch": torch.__version__, "cuda": torch.version.cuda}
    with torch.inference_mode():
        for q, k, v in inputs(args):
            lq, heads, dim = q.shape
            lk = k.shape[0]
            ref = reference(q, k, v, args.reference_chunk)
            nhd = tuple(x[None] for x in (q, k, v))
            hnd = tuple(x.transpose(0, 1)[None].contiguous() for x in (q, k, v))
            base = {"Lq": lq, "Lk": lk, "H": heads, "D": dim, "dtype": "bfloat16", "seed": args.seed,
                    "capture": str(args.capture) if args.capture else None, "iters": args.iters, "warmup": args.warmup,
                    "timing": "median_cuda_events", "reference_queries": 512, "reference_dtype": "float32", **device}

            def measure(method, factory, info=None):
                row = {**base, "method": method, **(info or {})}
                try:
                    if "import_error" in row:
                        row["status"] = "unavailable"
                    else:
                        fn, normalize, prepare, details = factory()
                        row.update(details)
                        ms, output = timing(fn, prepare, args.warmup, args.iters)
                        checked = normalize(output)[:512].float()
                        if checked.shape != ref.shape or not torch.isfinite(checked).all():
                            raise RuntimeError(f"invalid output: shape={tuple(checked.shape)}, expected={tuple(ref.shape)} or nonfinite values")
                        delta = checked - ref
                        rel = float(delta.norm() / ref.norm().clamp_min(1e-12))
                        maximum = float(delta.abs().max())
                        if not all(math.isfinite(value) for value in (ms, rel, maximum)) or ms <= 0:
                            raise RuntimeError("nonfinite accuracy/timing or nonpositive elapsed time")
                        row.update(status="ok", median_ms=ms, tops=4 * lq * lk * heads * dim / ms / 1e9,
                                   relative_l2=rel, max_abs_error=maximum)
                except Exception as exc:
                    row.update(status="error", error=repr(exc))
                rows.append(row)
                print(json.dumps(row, allow_nan=False), flush=True)

            measure("sage2_e2e", lambda: (lambda: sage2(*nhd, tensor_layout="NHD", is_causal=False), lambda out: out[0], None,
                                        {"api": "sageattention.sageattn", "tensor_layout": "NHD", "is_causal": False}), sage2_info)
            measure("sage2_kernel", lambda: sage2_kernel(*nhd), sage2_info)
            measure("sage3_e2e", lambda: sage3_end_to_end(sage3, *hnd), sage3_info)
            measure("sage3_kernel", lambda: sage3_kernel(*hnd), sage3_info)
            for mx, bal, i8 in ((False, False, False), (True, False, False), (True, True, False), (True, True, True)):
                for inclusive in (False, True):
                    name = f"ours_mx{int(mx)}_bal{int(bal)}{'_i8' if i8 else ''}_{'quant_inclusive' if inclusive else 'kernel'}"
                    measure(name, lambda: ours(ext, q, k, v, mx, bal, inclusive, i8))
    print("\n Lq    Lk  method                                  median ms     TOPS    rel L2   max abs")
    for row in rows:
        prefix = f"{row['Lq']:4d} {row['Lk']:5d}  {row['method']:<38}"
        if row["status"] == "ok":
            print(f"{prefix} {row['median_ms']:9.4f} {row['tops']:8.1f} {row['relative_l2']:9.5f} {row['max_abs_error']:9.5f}")
        else:
            print(f"{prefix} {row['status']}: {row.get('import_error', row.get('error'))}")
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
