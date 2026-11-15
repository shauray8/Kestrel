import argparse
import contextlib
import glob
import json
import math
import sys

import numpy as np
import torch

from model import Decoder
from nomas_io import dcae_decode, load_dcae


def load_student(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
    args = checkpoint.get("args", {})
    size = args.get("size", "M")
    gauss = "gauss_head.net.3.bias" in weights
    model = Decoder(size, gauss=gauss, gauss_rgb=gauss and weights["gauss_head.net.3.bias"].numel() == 12,
                    lookahead=args.get("lookahead", 0), extra_tail=args.get("extra_tail", 0))
    model.load_state_dict(weights)
    if model.lookahead:
        raise ValueError("the student causality sanity check requires a --lookahead 0 checkpoint")
    return model.to(device=device).eval().requires_grad_(False)


def load_latent(path, device, dtype):
    z = np.load(path)
    if z.shape != (32, 60, 15, 26):
        raise ValueError(f"expected [32,60,15,26]: {path}: {z.shape}")
    return torch.from_numpy(z)[None].to(device=device, dtype=dtype)


def add_stats(total, baseline, changed, masks):
    delta = changed - baseline
    absolute = delta.abs().sum(1)
    squared = delta.square().sum(1)
    for name, mask in masks.items():
        selected = mask[:, :delta.shape[2]]
        count = 3 * int(selected.sum())
        row = total.setdefault(name, [0.0, 0.0, 0, 0.0])
        row[0] += float(absolute.masked_select(selected).sum())
        row[1] += float(squared.masked_select(selected).sum())
        row[2] += count
        if count:
            row[3] = max(row[3], float(delta.abs().amax(1).masked_select(selected).max()))
    per_frame = delta.abs().mean((0, 1, 3, 4)).tolist()
    previous = total.setdefault("per_frame_mae_sum", [0.0] * len(per_frame))
    for i, value in enumerate(per_frame):
        previous[i] += value
    total["samples"] = total.get("samples", 0) + 1


def summarize(total):
    result = {}
    for name in ("all", "motion", "static"):
        absolute, squared, count, maximum = total[name]
        result[name] = {"mae": absolute / count if count else None,
                        "psnr_db": 10 * math.log10(4 * count / squared) if squared else None,
                        "max_abs": maximum if count else None, "rgb_values": count}
    result["per_frame_mae"] = [v / total["samples"] for v in total["per_frame_mae_sum"]]
    return result


@torch.inference_mode()
def probe(a):
    files = sorted(glob.glob(a.latents))
    if len(files) < 2:
        raise ValueError("at least two held-out latent clips are required for cross-clip replacement")
    if a.tile not in range(6) or a.clips < 0 or a.motion_threshold < 0:
        raise ValueError("tile must be 0..5; clips and motion threshold must be nonnegative")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    device = torch.device(a.device)
    student = load_student(a.student, device)
    teacher = load_dcae(device=a.device, dtype=dtype)
    start, stop = 10 * a.tile, 10 * (a.tile + 1)
    selected = files[:a.clips] if a.clips else files
    totals = {}
    for index, path in enumerate(selected):
        z = load_latent(path, device, dtype)[:, :, :stop]
        donor = load_latent(files[(index + 1) % len(files)], device, dtype)[:, :, :stop]
        def decode(model, value):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
                y = dcae_decode(model, value) if model is teacher else model.decode_stream(value, chunk=10)
            if tuple(y.shape[:3]) != (1, 3, 4 * stop):
                raise ValueError(f"expected 4 RGB frames per latent, got {tuple(y.shape)}")
            return y[:, :, 4 * start:4 * stop].float().cpu()
        baselines = {"teacher": decode(teacher, z), "student": decode(student, z)}
        reference = baselines["teacher"]
        motion = (reference[:, :, 1:] - reference[:, :, :-1]).abs().mean(1)
        motion = torch.cat((motion[:, :1], motion), 1) > a.motion_threshold
        masks = {"all": torch.ones_like(motion), "motion": motion, "static": ~motion}
        for replacement in ("other_clip", "zeros"):
            for j in range(10):
                modified = z.clone()
                future = slice(start + j + 1, stop)
                modified[:, :, future] = donor[:, :, future] if replacement == "other_clip" else 0
                frames = 4 * (j + 1)
                for name, model in (("teacher", teacher), ("student", student)):
                    changed = decode(model, modified) if j < 9 else baselines[name]
                    total = totals.setdefault((name, replacement, j), {})
                    add_stats(total, baselines[name][:, :, :frames], changed[:, :, :frames], masks)
        print(f"probed {index + 1}/{len(selected)}: {path}", file=sys.stderr, flush=True)
    rows = [{"model": name, "replacement": replacement, "j": j, "rgb_frames": 4 * (j + 1), **summarize(total)}
            for (name, replacement, j), total in sorted(totals.items())]
    return {"clips": selected, "donors": [files[(i + 1) % len(files)] for i in range(len(selected))],
            "student": a.student, "tile": a.tile, "latent_start": start,
            "prefix_latents": start, "dtype": a.dtype, "motion_threshold_rgb_minus1_1": a.motion_threshold,
            "psnr_peak": 2, "zero_mse_psnr": "+infinity represented by null; empty regions also use null",
            "motion_mask": "teacher baseline adjacent-frame mean RGB absolute difference above threshold",
            "table": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="/workspace/runs/distill_dcae2/ema_0002000.pt")
    p.add_argument("--latents", default="/workspace/data/gen/showcase/*/*.npy")
    p.add_argument("--tile", type=int, default=1, help="Global 10-latent teacher tile; decode its unmodified full prefix to preserve history")
    p.add_argument("--clips", type=int, default=0, help="Number of held-out clips; 0 uses all")
    p.add_argument("--motion_threshold", type=float, default=0.02)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    with contextlib.redirect_stdout(sys.stderr):
        result = probe(p.parse_args())
    print(json.dumps(result, indent=2, allow_nan=False))
