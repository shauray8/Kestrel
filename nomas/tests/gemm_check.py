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
from nomas.gemm import N3Linear, _mlp, _mlp_residual, _project_residual


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


def source_linear(n, k, bias):
    layer = nn.Linear(k, n, bias=bias, device="cuda", dtype=torch.bfloat16).eval()
    layer.weight.normal_(0, k ** -0.5)
    if layer.bias is not None:
        layer.bias.normal_(0, 0.01)
    return layer


def int8_linear(layer):
    from hymm.models.dit.modules.int8_stored_bf16_linear import Int8StoredBF16Linear, quantize_weight
    result = Int8StoredBF16Linear(layer)
    result.weight_q, result.weight_scale = quantize_weight(layer.weight)
    return result.eval()


def linear_check(m, n, k, decoder, args):
    source = source_linear(n, k, not decoder)
    if decoder and args.decoder_backend == "cutlass":
        from hymm.models.vae.pixelumm_decoder.cutlass_fp8 import CutlassFP8Linear
        original = CutlassFP8Linear(source)
    elif decoder:
        original = source
    else:
        original = int8_linear(source)
    accelerated = N3Linear(original)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    expected = F.linear(x, source.weight, source.bias)
    baseline = original(x)
    accelerated(x)
    actual = accelerated(x)
    record = {"kind": "decoder" if decoder else "dit", "shape_mnk": [m, n, k],
              "baseline": args.decoder_backend if decoder else "int8_storage_bf16",
              "n3_vs_bf16": error(actual, expected), "n3_vs_original": error(actual, baseline),
              "original_vs_bf16": error(baseline, expected),
              "original_ms": timing(lambda: original(x), args.repeats),
              "n3_ms": timing(lambda: accelerated(x), args.repeats)}
    record["speedup"] = record["original_ms"] / record["n3_ms"]
    print(json.dumps(record), flush=True)
    if not record["n3_vs_bf16"]["finite"] or record["n3_vs_bf16"]["relative_l2"] > args.max_relative_error:
        raise AssertionError(record)
    if args.compile:
        frozen_scale = accelerated.activation_scale.clone()
        compiled = torch.compile(accelerated, fullgraph=True)
        compiled_result = compiled(x)
        if not torch.equal(accelerated.activation_scale, frozen_scale):
            raise AssertionError("Compilation changed frozen activation calibration")
        if not torch.equal(compiled(x), compiled_result):
            raise AssertionError("Repeated compiled execution changed output")
        print(json.dumps({"kind": "compile", "shape_mnk": [m, n, k],
                          "comparison": error(compiled_result, actual),
                          "compiled_ms": timing(lambda: compiled(x), args.repeats)}), flush=True)


def mlp_check(m, args):
    a, b = source_linear(8192, 2048, True), source_linear(2048, 8192, True)
    original_a, original_b = int8_linear(a), int8_linear(b)
    fast_a, fast_b = N3Linear(original_a), N3Linear(original_b)
    x = torch.randn(1, m, 2048, device="cuda", dtype=torch.bfloat16)
    packed = (x, fast_a.wq, fast_a.sw, fast_a.bias, fast_b.wq, fast_b.sw,
              fast_b.bias, fast_a.activation_scale, fast_b.activation_scale)
    reference = lambda: F.linear(F.gelu(F.linear(x, a.weight, a.bias), approximate="tanh"), b.weight, b.bias)
    baseline = lambda: original_b(F.gelu(original_a(x), approximate="tanh"))
    _mlp(*packed)
    expected, actual = reference(), _mlp(*packed)
    record = {"kind": "fused_gelu_mlp", "shape_mnk": [m, 8192, 2048],
              "n3_vs_bf16": error(actual, expected),
              "original_ms": timing(baseline, args.repeats),
              "n3_ms": timing(lambda: _mlp(*packed), args.repeats)}
    print(json.dumps(record), flush=True)
    if not record["n3_vs_bf16"]["finite"] or record["n3_vs_bf16"]["relative_l2"] > args.max_relative_error:
        raise AssertionError(record)
    residual = torch.randn(1, m, 2048, device="cuda", dtype=torch.float32)
    for rows in (1, 30):
        if m % rows:
            continue
        gate = torch.randn(1, rows, 2048, device="cuda", dtype=torch.float32)
        expanded = gate.repeat_interleave(m // rows, dim=1)
        def fused():
            return _mlp_residual(*packed, residual, gate)
        expected = residual + _mlp(*packed).float() * expanded
        result = fused()
        print(json.dumps({"kind": "fused_residual", "gate_rows": rows,
                          "comparison": error(result, expected),
                          "fused_ms": timing(fused, args.repeats),
                          "unfused_ms": timing(lambda: residual + _mlp(*packed).float() * expanded,
                                               args.repeats)}), flush=True)
        if not torch.allclose(result, expected, atol=0.05, rtol=0.05):
            raise AssertionError("Fused residual differs from unfused N3")
    gate = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    bf16_residual = residual.bfloat16()
    expected = bf16_residual + fast_b(F.gelu(fast_a(x), approximate="tanh")) * gate.unsqueeze(1)
    result = _project_residual(fast_b, F.gelu(fast_a(x), approximate="tanh"), bf16_residual, gate)
    if not torch.equal(result, expected):
        raise AssertionError("BF16 residual/gate fallback changed rounding")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument("--decoder-m", type=int, nargs="+", default=[15600, 46800])
    parser.add_argument("--decoder-backend", choices=("cutlass", "bf16"), default="cutlass")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-relative-error", type=float, default=0.15)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dit-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("N3_CONT", "1")
    torch.manual_seed(7)
    ops.n3()
    for m in (11700 + args.text_tokens, args.text_tokens):
        for n, k in ((2048, 2048), (6144, 2048), (8192, 2048), (2048, 8192)):
            linear_check(m, n, k, False, args)
            gc.collect()
            torch.cuda.empty_cache()
    mlp_check(11700, args)
    gc.collect()
    torch.cuda.empty_cache()
    if not args.dit_only:
        for m in args.decoder_m:
            for n, k in ((4096, 4096), (1024, 4096), (12288, 4096),
                         (4096, 12288), (14336, 4096), (4096, 14336)):
                linear_check(m, n, k, True, args)
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
