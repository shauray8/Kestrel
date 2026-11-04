import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas"))


def load_latents(path):
    value = torch.from_numpy(np.load(path, allow_pickle=False))
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[1] != 32:
        raise ValueError("Expected saved DCAE latents [32,L,H,W] or [B,32,L,H,W]")
    return value.cuda().bfloat16().contiguous()


def load_decoder(args, fp4):
    from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
    from hymm.models.vae.pixelumm_decoder.model import GenExpertArch
    decoder = PixelUMMDCAELatentDecoder.from_checkpoint(
        args.checkpoint, device="cuda", dtype=torch.bfloat16,
        arch=GenExpertArch(intermediate_size=args.ffn), fp8_backend="cutlass",
        seed=args.seed, temporal_window_latents=args.window,
        temporal_overlap_latents=args.overlap, local_attention_tile_pixels=args.local_attention_tile,
        decoder_tile_pixels=args.tile_pixels, decoder_tile_overlap_pixels=args.tile_overlap,
        decode_output_cpu=True, num_steps=args.steps,
    )
    if fp4:
        from nomas.dec4 import convert_decoder
        counts = convert_decoder(decoder.decoder, args.fp4_attention)
        print(json.dumps({"kind": "conversion", "path": "nvfp4", "nvfp4_linears": counts[0],
                          "n3_linears": counts[1]}), flush=True)
    else:
        from nomas.gemm import _replace_linears
        count = _replace_linears(decoder.decoder)
        print(json.dumps({"kind": "conversion", "path": "n3_fp8", "n3_linears": count}), flush=True)
    return decoder


def decode(args, latent, fp4):
    decoder = load_decoder(args, fp4)
    name = "nvfp4" if fp4 else "n3_fp8"
    for _ in range(args.warmup):
        decoder.decode(latent, pixel_frames=args.pixel_frames)
    times = []
    video = None
    for _ in range(args.repeats):
        video = None
        torch.cuda.synchronize()
        start = time.perf_counter()
        video = decoder.decode(latent, pixel_frames=args.pixel_frames).sample
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    del decoder
    gc.collect()
    torch.cuda.empty_cache()
    record = {"kind": "decode_timing", "path": name, "seconds": times,
              "median_seconds": statistics.median(times), "output_shape": list(video.shape)}
    print(json.dumps(record), flush=True)
    return video, record["median_seconds"]


def ssim_frame(actual, expected, kernel):
    mu_a = F.conv2d(actual, kernel, groups=3)
    mu_b = F.conv2d(expected, kernel, groups=3)
    var_a = F.conv2d(actual * actual, kernel, groups=3) - mu_a.square()
    var_b = F.conv2d(expected * expected, kernel, groups=3) - mu_b.square()
    covariance = F.conv2d(actual * expected, kernel, groups=3) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
             / ((mu_a.square() + mu_b.square() + 0.01 ** 2) * (var_a + var_b + 0.03 ** 2)))
    return float(score.mean())


def metrics(args, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Decode shapes differ: {actual.shape} vs {expected.shape}")
    try:
        lpips = importlib.import_module("lpips")
    except ModuleNotFoundError as exc:
        if exc.name != "lpips":
            raise
        lpips = None
    metric = None if lpips is None else lpips.LPIPS(net="alex").cuda().eval()
    coords = torch.arange(11, device="cuda", dtype=torch.float32) - 5
    gaussian = (-coords.square() / (2 * 1.5 ** 2)).exp()
    gaussian /= gaussian.sum()
    kernel = (gaussian[:, None] * gaussian[None, :]).expand(3, 1, 11, 11).contiguous()
    squared_error, elements, perceptual, frame_psnr = 0.0, 0, [], []
    for batch in range(actual.shape[0]):
        for frame in range(0, actual.shape[2], args.metric_stride):
            a = actual[batch:batch + 1, :, frame].cuda().float()
            b = expected[batch:batch + 1, :, frame].cuda().float()
            if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                raise ValueError("Nonfinite decoded pixels")
            a.mul_(0.5).add_(0.5).clamp_(0, 1)
            b.mul_(0.5).add_(0.5).clamp_(0, 1)
            error = float((a - b).square().sum())
            squared_error += error
            elements += a.numel()
            frame_psnr.append(float("inf") if error == 0 else 10 * math.log10(a.numel() / error))
            perceptual.append(float(metric(a * 2 - 1, b * 2 - 1).mean()) if metric is not None
                              else ssim_frame(a, b, kernel))
    psnr = float("inf") if squared_error == 0 else 10 * math.log10(elements / squared_error)
    score = statistics.mean(perceptual)
    passed = psnr >= args.min_psnr and (score <= args.max_lpips if metric is not None else score >= args.min_ssim)
    return {"kind": "quality", "reference": "n3_fp8_same_checkpoint_same_latents_same_seed",
            "frames_measured": len(perceptual), "metric_stride": args.metric_stride,
            "psnr_db": psnr, "minimum_frame_psnr_db": min(frame_psnr),
            "lpips" if metric is not None else "ssim": score,
            "gate": {"minimum_psnr_db": args.min_psnr,
                     "maximum_lpips" if metric is not None else "minimum_ssim":
                     args.max_lpips if metric is not None else args.min_ssim}, "passed": passed}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Compare N3 FP8 vs NVFP4 decode of inference.save_latents=true .npy; exit 1 if the quality gate fails")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--pixel-frames", type=int)
    parser.add_argument("--ffn", type=int, default=12288)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--local-attention-tile", type=int, default=128)
    parser.add_argument("--tile-pixels", type=int, default=0)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fp4-attention", action="store_true")
    parser.add_argument("--metric-stride", type=int, default=1)
    parser.add_argument("--min-psnr", type=float, default=30)
    parser.add_argument("--max-lpips", type=float, default=0.05)
    parser.add_argument("--min-ssim", type=float, default=0.95)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or args.metric_stride < 1:
        parser.error("repeats, warmup and metric-stride must be positive")
    os.environ.setdefault("N3_CONT", "1")
    if os.environ.get("NOMAS_DEC_ATTN") == "fa":
        from nomas import attn
        attn.install()
    if os.environ.get("NOMAS_GLUE") == "1":
        from nomas import glue
        glue.install()
    latent = load_latents(args.latents)
    print(json.dumps({"kind": "configuration", "checkpoint": args.checkpoint,
                      "latent_shape": list(latent.shape), "seed": args.seed,
                      "window": args.window, "overlap": args.overlap,
                      "local_attention_tile": args.local_attention_tile, "pixel_frames": args.pixel_frames,
                      "attention": os.environ.get("NOMAS_DEC_ATTN", "upstream"),
                      "glue": os.environ.get("NOMAS_GLUE", "0"),
                      "fp4_attention": args.fp4_attention}), flush=True)
    reference, fp8_seconds = decode(args, latent, False)
    actual, fp4_seconds = decode(args, latent, True)
    record = metrics(args, actual, reference)
    record.update(fp8_seconds=fp8_seconds, fp4_seconds=fp4_seconds, speedup=fp8_seconds / fp4_seconds)
    print(json.dumps(record), flush=True)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
