import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas"))

from kestrel import ops
from nomas.dec4 import NVFP4Linear, _mm, _pack, _weight, mlp_forward
from nomas.gemm import N3Linear


class StoredFP8(nn.Module):
    def __init__(self, q, s):
        super().__init__()
        self.register_buffer("fp8_weight", q)
        self.register_buffer("fp8_scale", s)
        self.bias = None
        self.eval()


def source(n, k, key=None, checkpoint=None):
    if checkpoint is not None:
        from safetensors import safe_open
        with safe_open(checkpoint, framework="pt", device="cuda") as saved:
            if key + ".fp8_weight" in saved.keys():
                return StoredFP8(saved.get_tensor(key + ".fp8_weight"), saved.get_tensor(key + ".fp8_scale"))
            weight = saved.get_tensor(key + ".weight").bfloat16()
    else:
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
    n, k = weight.shape
    blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
    scales = (blocks.abs().amax((1, 3)) / 448).clamp_min(1e-12)
    q = (blocks / scales[:, None, :, None]).to(torch.float8_e4m3fn).reshape(n, k)
    return StoredFP8(q.contiguous(), scales.contiguous())


def error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return {"relative_l2": float((actual - expected).norm() / expected.norm().clamp_min(1e-12)),
            "max_absolute": float((actual - expected).abs().amax()),
            "finite": bool(torch.isfinite(actual).all())}


def timing(fn, repeats):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def baseline(layer, backend):
    if backend == "n3":
        return N3Linear(layer)
    from hymm.models.vae.pixelumm_decoder.cutlass_fp8 import CutlassFP8Linear
    return CutlassFP8Linear.from_quantized(layer.fp8_weight, layer.fp8_scale, layer.fp8_weight.shape)


def require(record, field, limit):
    comparison = record[field]
    if not comparison["finite"] or comparison["relative_l2"] > limit:
        raise AssertionError(record)


def capture_check(fn, expected):
    for _ in range(3):
        fn()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fn()
    graph.replay()
    torch.cuda.synchronize()
    record = error(result, expected)
    if not record["finite"] or record["relative_l2"] > 0.001:
        raise AssertionError(record)
    return record


def linear_check(m, n, k, name, args):
    layer = source(n, k, f"decoder.layers.{args.layer}.{name}", args.checkpoint)
    fp8, fp4 = baseline(layer, args.baseline), NVFP4Linear(layer)
    n, k = layer.fp8_weight.shape
    x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
    expected = fp8(x)
    fp4(x)
    actual = fp4(x)
    reference = F.linear(x, _weight(layer).bfloat16())
    record = {"kind": "decoder_linear", "name": name, "shape_mnk": [m, n, k],
              "weight_source": args.checkpoint or "synthetic_128x128_fp8", "baseline": args.baseline,
              "fp4_vs_fp8": error(actual, expected), "fp4_vs_dequant_bf16": error(actual, reference),
              "fp8_ms": timing(lambda: fp8(x), args.repeats),
              "fp4_ms": timing(lambda: fp4(x), args.repeats)}
    record["speedup"] = record["fp8_ms"] / record["fp4_ms"]
    print(json.dumps(record), flush=True)
    require(record, "fp4_vs_fp8", args.max_relative_error)
    if args.graph:
        print(json.dumps({"kind": "linear_graph", "name": name,
                          "comparison": capture_check(lambda: fp4(x), actual)}), flush=True)
    if args.compile:
        compiled = torch.compile(fp4, fullgraph=True)
        result = compiled(x)
        record = {"kind": "linear_compile", "name": name, "comparison": error(result, actual),
                  "compiled_ms": timing(lambda: compiled(x), args.repeats)}
        print(json.dumps(record), flush=True)
        require(record, "comparison", 0.001)


def mlp_check(m, hidden, args):
    from hymm.models.vae.pixelumm_decoder.model import GenExpertArch, GenMLP
    arch = GenExpertArch(intermediate_size=hidden)
    with torch.device("meta"):
        fp8, fp4 = GenMLP(arch).eval(), GenMLP(arch).eval()
    for name, n, k in (("gate_proj", hidden, 4096), ("up_proj", hidden, 4096), ("down_proj", 4096, hidden)):
        original = source(n, k, f"decoder.layers.{args.layer}.mlp.{name}", args.checkpoint)
        setattr(fp8, name, baseline(original, args.baseline))
        setattr(fp4, name, NVFP4Linear(original))
    x = torch.randn(1, m, 4096, device="cuda", dtype=torch.bfloat16)
    expected = fp8(x)
    mlp_forward(fp4, x)
    actual = mlp_forward(fp4, x)
    def unfused():
        return fp4.down_proj(F.silu(fp4.gate_proj(x)) * fp4.up_proj(x))
    record = {"kind": "decoder_swiglu", "shape_mnk": [m, hidden, 4096],
              "baseline": args.baseline, "fp4_vs_fp8": error(actual, expected),
              "fused_vs_unfused": error(actual, unfused()),
              "fp8_ms": timing(lambda: fp8(x), args.repeats),
              "fp4_fused_ms": timing(lambda: mlp_forward(fp4, x), args.repeats),
              "fp4_unfused_ms": timing(unfused, args.repeats)}
    record["speedup"] = record["fp8_ms"] / record["fp4_fused_ms"]
    record["mlp_36_layers_8_windows_seconds"] = record["fp4_fused_ms"] * 36 * 8 / 1000
    print(json.dumps(record), flush=True)
    require(record, "fp4_vs_fp8", args.max_mlp_relative_error)
    require(record, "fused_vs_unfused", 0.02)
    if args.graph:
        print(json.dumps({"kind": "mlp_graph", "comparison": capture_check(lambda: mlp_forward(fp4, x), actual)}), flush=True)
    if args.compile:
        compiled = torch.compile(lambda value: mlp_forward(fp4, value), fullgraph=True)
        result = compiled(x)
        record = {"kind": "mlp_compile", "comparison": error(result, actual),
                  "compiled_ms": timing(lambda: compiled(x), args.repeats)}
        print(json.dumps(record), flush=True)
        require(record, "comparison", 0.001)


def packing_check():
    x = torch.randn(259, 4096, device="cuda", dtype=torch.bfloat16)
    scale = float(x.abs().amax()) / (6 * 448)
    q, s = _pack(x, scale)
    ref_q, ref_s = ops.nvfp4().nvfp4_quant(x, scale)
    w = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
    wg = float(w.abs().amax()) / (6 * 448)
    wq, ws = ops.nvfp4().nvfp4_quant(w, wg)
    actual, expected = _mm(q, s, wq, ws, scale * wg), _mm(ref_q, ref_s, wq, ws, scale * wg)
    left = torch.stack((q & 15, q >> 4), -1).reshape_as(x).long()
    right = torch.stack((ref_q & 15, ref_q >> 4), -1).reshape_as(x).long()
    row = torch.arange(x.shape[0], device=x.device)[:, None]
    col = torch.arange(x.shape[1] // 16, device=x.device)[None, :]
    offset = ((row // 128 * (x.shape[1] // 64) + col // 4) * 512
              + row % 32 * 16 + row % 128 // 32 * 4 + col % 4)
    block_scale = ref_s.flatten().view(torch.float8_e4m3fn).float()[offset]
    inverse = torch.tensor(1.0 / scale, device=x.device, dtype=torch.float32) / block_scale
    normalized = (x.float().reshape(x.shape[0], -1, 16) * inverse[:, :, None]).reshape_as(x).abs()
    grid = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.], device=x.device)
    a, b = left & 7, right & 7
    midpoint = (grid[a] + grid[b]) * 0.5
    tolerance = 8 * torch.finfo(torch.float32).eps * midpoint.clamp_min(1)
    tie = ((left >> 3) == (right >> 3)) & ((a - b).abs() == 1) & ((normalized - midpoint).abs() <= tolerance)
    signed_zero = (a == 0) & (b == 0)
    mismatch = left != right
    unexpected = mismatch & ~(tie | signed_zero)
    record = {"kind": "packing_vs_kestrel", "comparison": error(actual, expected),
              "packed_byte_agreement": float((q == ref_q).float().mean()),
              "scale_byte_agreement": float((s == ref_s).float().mean()),
              "rounding_boundary_nibbles": int((mismatch & tie).sum()),
              "unexpected_nibbles": int(unexpected.sum())}
    print(json.dumps(record), flush=True)
    if record["unexpected_nibbles"] or not torch.equal(s, ref_s) or not record["comparison"]["finite"]:
        raise AssertionError(record)


def lifecycle_check():
    import logging
    from types import SimpleNamespace
    from hymm.models.vae.pixelumm_decoder import loader
    from hymm.models.vae.pixelumm_decoder.model import DCAEVPixelUMMDecoder, GenExpertArch
    from nomas.dec4 import PaddedN3Linear, convert_decoder, install

    previous = os.environ.get("NOMAS_DEC_GEMM")
    os.environ["NOMAS_DEC_GEMM"] = "nvfp4"
    try:
        install()
    finally:
        if previous is None:
            os.environ.pop("NOMAS_DEC_GEMM")
        else:
            os.environ["NOMAS_DEC_GEMM"] = previous
    arch = GenExpertArch(hidden_size=128, intermediate_size=128, num_layers=1, num_heads=1, num_kv_heads=1)
    decoder = DCAEVPixelUMMDecoder(arch).bfloat16().eval().requires_grad_(False)
    config = SimpleNamespace(cutlass_fp8=False, deepgemm_fp8=False, cute_attention=False, compile_decoder=False)
    loader._install_inference_optimizations(decoder, config, logging.getLogger("dec4_check"))
    if any(isinstance(layer, NVFP4Linear) for layer in decoder.modules()):
        raise AssertionError("CPU decoder converted before residency")
    decoder.cuda()
    latent = torch.randn(1, 32, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    noise = torch.randn(1, 3, 4, 32, 32, device="cuda", dtype=torch.bfloat16)
    actual = decoder(latent, noise)
    if not isinstance(decoder.layers[0].mlp.gate_proj, NVFP4Linear) or decoder._forward_pre_hooks:
        raise AssertionError("First CUDA forward did not complete one-shot conversion")
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError("Deferred decoder returned nonfinite pixels")
    arch = GenExpertArch(hidden_size=128, intermediate_size=193, num_layers=1, num_heads=1, num_kv_heads=1)
    fallback = DCAEVPixelUMMDecoder(arch).cuda().bfloat16().eval().requires_grad_(False)
    x = torch.randn(1, 259, 128, device="cuda", dtype=torch.bfloat16)
    expected = fallback.layers[0].mlp(x)
    convert_decoder(fallback)
    if not all(isinstance(getattr(fallback.layers[0].mlp, name), PaddedN3Linear)
               for name in ("gate_proj", "up_proj", "down_proj")):
        raise AssertionError("Unaligned MLP did not use N3 fallback")
    record = {"kind": "cpu_to_cuda_lifecycle", "deferred_conversion": True,
              "unaligned_ffn": 193, "fallback_comparison": error(fallback.layers[0].mlp(x), expected)}
    print(json.dumps(record), flush=True)
    require(record, "fallback_comparison", 0.2)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="NVFP4 decoder shapes against the current N3 FP8 path; synthetic or checkpoint weights")
    parser.add_argument("--checkpoint")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--m", type=int, nargs="+", default=[15600])
    parser.add_argument("--ffn", type=int, nargs="+", default=[12288, 14336])
    parser.add_argument("--baseline", choices=("n3", "cutlass"), default="n3")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-relative-error", type=float, default=0.25)
    parser.add_argument("--max-mlp-relative-error", type=float, default=0.35)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.checkpoint:
        from safetensors import safe_open
        with safe_open(args.checkpoint, framework="pt", device="cpu") as saved:
            key = f"decoder.layers.{args.layer}.mlp.gate_proj."
            key += "fp8_weight" if key + "fp8_weight" in saved.keys() else "weight"
            args.ffn = [saved.get_slice(key).get_shape()[0]]
    os.environ.setdefault("N3_CONT", "1")
    torch.manual_seed(7)
    ops.nvfp4()
    if args.baseline == "n3":
        ops.n3()
    packing_check()
    lifecycle_check()
    gc.collect()
    torch.cuda.empty_cache()
    for m in args.m:
        for hidden in args.ffn:
            for name, n, k in (("mlp.gate_proj", hidden, 4096), ("mlp.up_proj", hidden, 4096),
                               ("mlp.down_proj", 4096, hidden)):
                linear_check(m, n, k, name, args)
                gc.collect()
                torch.cuda.empty_cache()
            mlp_check(m, hidden, args)
            gc.collect()
            torch.cuda.empty_cache()
        if args.attention:
            for name, n in (("q_proj", 4096), ("k_proj", 1024), ("v_proj", 1024), ("o_proj", 4096)):
                linear_check(m, n, 4096, "attn." + name, args)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
