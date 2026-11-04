import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kestrel import ops
from nomas import glue


def emit(**record):
    print(json.dumps(record), flush=True)


def clone(x):
    return tuple(clone(t) for t in x) if isinstance(x, tuple) else x.clone()


def ordered(x):
    if x.dtype == torch.float32:
        bits, sign = x.view(torch.int32).long(), 1 << 31
    elif x.dtype in (torch.bfloat16, torch.float16):
        bits, sign = x.view(torch.int16).long(), 1 << 15
    else:
        bits, sign = x.view(torch.uint8).long(), 1 << 7
    magnitude = bits & (sign - 1)
    return torch.where((bits & sign) != 0, sign - magnitude, sign + magnitude)


def compare(name, expected, actual, stats=False, exact=False):
    if isinstance(expected, tuple):
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(f"{name}/{i}", a, b, stats, exact)
        return
    assert expected.shape == actual.shape and expected.dtype == actual.dtype, name
    if exact or not expected.is_floating_point():
        different = expected != actual
        count = int(different.sum())
        emit(case=name, dtype=str(expected.dtype), shape=list(expected.shape), mismatches=count, exact=True)
        if count:
            raise AssertionError(f"{name}: {count} nonidentical values")
        return
    a, b = expected.float(), actual.float()
    finite = bool(torch.isfinite(a).all() & torch.isfinite(b).all())
    ulps = (ordered(expected) - ordered(actual)).abs()
    different = ulps != 0
    count = int(different.sum())
    maximum = int(ulps.max()) if ulps.numel() else 0
    relative = float((a - b).norm() / a.norm().clamp_min(1.e-30))
    fraction = count / max(a.numel(), 1)
    fp8 = expected.dtype == torch.float8_e4m3fn
    accepted = finite and (relative < 1.e-6 if stats else maximum <= 1)
    if fp8:
        accepted = finite and maximum <= 1 and fraction < 1.e-4
    emit(case=name, dtype=str(expected.dtype), shape=list(expected.shape), finite=finite,
         mismatches=count, mismatch_fraction=fraction, max_ulps=maximum, relative_l2=relative,
         max_absolute=float((a - b).abs().max()) if a.numel() else 0., stats=stats, accepted=accepted)
    if not accepted:
        raise AssertionError(f"{name}: numerical parity failed")


def bench(name, fn, repeats):
    record = dict(case=name, launches=repeats)
    for cuda, label in ((False, "triton"), (True, "cuda")):
        glue._CUDA = cuda
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(repeats):
            fn()
        submitted = time.perf_counter()
        end.record()
        end.synchronize()
        record[label + "_gpu_ms"] = start.elapsed_time(end) / repeats
        record[label + "_submit_us"] = (submitted - wall) * 1.e6 / repeats
        record[label + "_wall_ms"] = (time.perf_counter() - wall) * 1000 / repeats
    record["wall_speedup"] = record["triton_wall_ms"] / record["cuda_wall_ms"]
    emit(**record)


def run(name, fn, args, stats=False, exact=False):
    glue._CUDA = False
    expected = clone(fn())
    glue._CUDA = True
    actual = fn()
    compare(name, expected, actual, stats, exact)
    if args.graphs:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn()
        graph.replay()
        compare(name + "/graph", expected, captured, stats, exact)
    if args.compile and not stats and not name.startswith(("indexed_quant/", "decoder_quant/")):
        compiled = torch.compile(fn, fullgraph=True)
        compare(name + "/compile", expected, compiled(), stats, exact)
    del actual, expected
    if args.repeats:
        bench(name, fn, args.repeats)


def modulation(args):
    for tokens in (11700, 12212):
        for xd, md in ((torch.float32, torch.float32), (torch.float32, torch.bfloat16),
                       (torch.bfloat16, torch.bfloat16)):
            x = torch.randn(1, tokens, 2048, device="cuda", dtype=xd)
            packed = torch.randn(1, 12288, device="cuda", dtype=md) * .1
            shift, scale = packed.chunk(6, -1)[:2]
            for activation_scale in (0., .25, .3):
                run(f"mod/{tokens}/{xd}/{md}/{activation_scale}",
                    lambda: glue.mod_norm(x, shift, scale, 1.e-6, activation_scale), args)
    x = torch.randn(2, 65, 2048, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(2, 12288, device="cuda", dtype=torch.bfloat16) * .1
    shift, scale = packed.chunk(6, -1)[:2]
    run("mod/batched_strided_modulation", lambda: glue.mod_norm(x, shift, scale, 1.e-6, .25), args)


def rms(args):
    for xd, wd in ((torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32),
                   (torch.float32, torch.float32)):
        x = torch.randn(1, 15600, 4096, device="cuda", dtype=xd)
        residual = torch.randn_like(x)
        weight = torch.empty(4096, device="cuda", dtype=wd).uniform_(.8, 1.2)
        run(f"rms/15600/{xd}/{wd}", lambda: glue.rms_norm(x, weight, 1.e-6), args)
        run(f"add_rms/15600/{xd}/{wd}", lambda: glue.add_rms(x, residual, weight, 1.e-6), args)


def rotary(args):
    for decoder, tokens, heads in ((False, 11700, 16), (False, 12212, 16),
                                   (True, 15600, 32), (True, 15600, 8)):
        x = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        angle = torch.randn(tokens, 128, device="cuda")
        cos, sin = angle.cos(), angle.sin()
        for wd in (torch.bfloat16, torch.float32):
            size = 64 if decoder else 128
            weight = torch.empty(size, device="cuda", dtype=wd).uniform_(.8, 1.2)
            other = torch.empty_like(weight).uniform_(.8, 1.2)
            for dtype in (torch.bfloat16, torch.float32):
                for rope in (False, True):
                    run(f"rope/{decoder}/{tokens}/{heads}/{wd}/{dtype}/{rope}",
                        lambda: glue.norm_rope(x, weight, other, cos, sin, 1.e-6, decoder, rope, dtype), args)


def quantization(args):
    for integer in (False, True):
        h, hist, ni, nt = 16, 7020, 11700, 512
        if integer:
            cache = torch.randint(-127, 128, (hist * 2, h, 128), device="cuda", dtype=torch.int8)[::2]
        else:
            cache = torch.randn(hist * 2, h, 128, device="cuda", dtype=torch.bfloat16)[::2]
        scale = torch.empty(hist * 2, h, device="cuda").uniform_(.001, .03)[::2]
        image = torch.randn(ni, h, 128, device="cuda", dtype=torch.bfloat16)
        text = torch.randn(nt, h, 128, device="cuda", dtype=torch.bfloat16)
        hi = torch.arange(0, hist, 2, device="cuda")
        n = hi.numel()
        parts = torch.empty((n + 127) // 128, h, 128, device="cuda")
        def history():
            glue._mean((h, parts.shape[0]), cache, scale, hi, parts, h, hist,
                       cache.stride(0), scale.stride(0), integer, n, enable_fp_fusion=False)
            return parts
        run(f"history_mean/{integer}", history, args, stats=True)
        glue._CUDA = False
        center = history().sum(0)
        indices = torch.cat((hi, hist + torch.arange(ni, device="cuda"),
                             hist + ni + torch.arange(0, nt, 3, device="cuda")))
        nk, nb = indices.numel(), (indices.numel() + 63) // 64
        for value in (False, True):
            out = torch.full((h, nb, 128, 64) if value else (h, nb, 64, 128), 113, device="cuda", dtype=torch.uint8)
            scales = torch.full((h, nb), -1., device="cuda")
            for first in (0, n // 64, nb):
                out.fill_(113)
                scales.fill_(-1.)
                def pack():
                    glue._quant((nb - first, h), cache, scale, image, text, indices,
                                out, scales, center, h, hist, ni, cache.stride(0), scale.stride(0),
                                integer, nk, nb, first, value, enable_fp_fusion=False)
                    return out, scales
                run(f"indexed_quant/{integer}/{value}/first={first}", pack, args, exact=True)
    for h, repeat in ((32, 1), (8, 4)):
        x = torch.randn(15600, h, 128, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(15598, -1, -3, device="cuda")
        nk, nb = indices.numel(), (indices.numel() + 63) // 64
        center = torch.randn(h, 128, device="cuda") * .01
        for value in (False, True):
            out = torch.empty((h * repeat, nb, 128, 64) if value else (h * repeat, nb, 64, 128), device="cuda", dtype=torch.uint8)
            scales = torch.empty(h * repeat, nb, device="cuda")
            def pack_decoder():
                glue._quant((nb, h * repeat), x, x, x, x, indices, out, scales, center, h, 0,
                            15600, x.stride(0), 0, False, nk, nb, 0, value, repeat, enable_fp_fusion=False)
                return out, scales
            run(f"decoder_quant/15600/{h}/repeat={repeat}/{value}", pack_decoder, args, exact=True)


def camera(args):
    for xd, fd in ((torch.bfloat16, torch.bfloat16), (torch.float32, torch.bfloat16),
                   (torch.float32, torch.float32)):
        x = torch.randn(1, 11700, 2048, device="cuda", dtype=xd)
        frames = torch.randn(1, 30, 2048, device="cuda", dtype=fd)
        for masked in (False, True):
            mask = torch.rand(1, 30, device="cuda") if masked else None
            run(f"camera/{xd}/{fd}/{masked}", lambda: glue.camera_add(x, frames, mask, 390), args, exact=True)
    x = torch.randn(2, 12, 128, device="cuda", dtype=torch.bfloat16)
    frames = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[0., .5, 1.], [1., 0., .25]], device="cuda")
    run("camera/batched_fractional_mask", lambda: glue.camera_add(x, frames, mask, 4), args, exact=True)
    mask = mask.bool()
    run("camera/batched_bool_mask", lambda: glue.camera_add(x, frames, mask, 4), args, exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--only", nargs="+", choices=("mod", "rms", "rope", "quant", "camera"), default=None)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    ops.nomas_glue()
    emit(build_seconds=time.perf_counter() - start, torch=torch.__version__, backend_env=os.environ.get("NOMAS_GLUE_CUDA", "0"))
    if args.build_only:
        return
    if args.repeats < 0:
        parser.error("--repeats must be nonnegative")
    torch.manual_seed(20261104)
    with torch.inference_mode():
        for name, check in (("mod", modulation), ("rms", rms), ("rope", rotary),
                            ("quant", quantization), ("camera", camera)):
            if args.only is None or name in args.only:
                check(args)
    emit(result="passed")


if __name__ == "__main__":
    main()
