import argparse
import json
import math
import statistics
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def video_writer(path, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "24", "-i", "-",
                             "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)
    try:
        yield lambda frame: proc.stdin.write(frame.contiguous().numpy().tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited with status {code}: {path}")


def load_student(path):
    import torch
    from model import Decoder
    sd = torch.load(path, map_location="cpu", weights_only=False)
    weights = sd["ema"]
    gauss = any(k.startswith("gauss_head.") for k in weights)
    rgb = gauss and weights["gauss_head.net.3.bias"].numel() == 12
    model = Decoder(sd["args"]["size"], gauss=gauss, gauss_rgb=rgb)
    model.load_state_dict(weights)
    return model.cuda().eval().requires_grad_(False)


def load_latent(path):
    import numpy as np
    import torch
    z = np.load(path)
    if z.ndim != 4 or z.shape[0] != 32 or tuple(z.shape[-2:]) != (15, 26) or not z.shape[1]:
        raise ValueError(f"Expected [32,T,15,26] latents: {path}: {z.shape}")
    return torch.from_numpy(z)[None].cuda().to(torch.bfloat16)


def streamed(model, z, chunk):
    import torch
    state, parts = {}, []
    with torch.autocast("cuda", torch.bfloat16):
        for i in range(0, z.shape[2], chunk):
            parts.append(model(z[:, :, i:i + chunk], state=state).to("cpu", dtype=torch.bfloat16))
    return torch.cat(parts, 2)


def aligned(video, frames, name):
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 3) or tuple(video.shape[-2:]) != (480, 832):
        raise ValueError(f"Expected [1,3,T,480,832] from {name}: {tuple(video.shape)}")
    if video.shape[2] < frames:
        raise ValueError(f"{name} has {video.shape[2]} frames, expected at least {frames}")
    return video[:, :, :frames]


def pixels(video):
    return video[0].float().clamp(-1, 1).add_(1).mul_(127.5).round_().byte().permute(1, 2, 3, 0).cpu()


def score(pred, ref, batch):
    import torch
    from metrics import _lpips, grain
    pred, ref = pred.cuda(), ref.cuda()
    result = grain(pred, ref, chunk=batch)
    psnr, lpips = [], []
    for i in range(0, pred.shape[2], batch):
        p = pred[0, :, i:i + batch].transpose(0, 1).float().clamp(-1, 1)
        r = ref[0, :, i:i + batch].transpose(0, 1).float().clamp(-1, 1)
        mse = (p - r).square().mean((1, 2, 3))
        psnr.extend((10 * torch.log10(4 / mse.clamp_min(1e-10))).tolist())
        lpips.extend(_lpips()(p, r).flatten().tolist())
    result.update(lpips=statistics.mean(lpips), psnr=statistics.mean(psnr))
    return result


def heat_frame(frame, scale, alpha, lut):
    import torch
    import torch.nn.functional as F
    rgb = frame.permute(2, 0, 1)[None].float().div(255)
    gray = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    kernel = gray.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    energy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode="reflect"), kernel).abs()
    energy = F.avg_pool2d(energy, 16, stride=16)
    energy = F.interpolate(energy, size=frame.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    hot = (energy / scale).clamp(0, 1)
    color = lut[(hot * 255).long()]
    opacity = (alpha * hot)[..., None]
    return (frame.float() * (1 - opacity) + color * opacity).round().byte()


def visuals(out, clip, videos, chosen, anchor, xview, a):
    import torch
    from PIL import Image, ImageDraw
    from matplotlib import colormaps
    left, right = videos["dcae"], videos[chosen]
    n, h, w, _ = left.shape
    lut = torch.tensor(colormaps["magma"]([i / 255 for i in range(256)])[:, :3]).float().mul_(255)
    with video_writer(out / f"wipe_{clip}.mp4", w, h) as wipe, video_writer(out / f"heat_{clip}.mp4", w, h) as heat:
        for i in range(n):
            split = round((0.5 - 0.5 * math.cos(2 * math.pi * i / max(n - 1, 1))) * w)
            frame = torch.cat((left[i, :, :split], right[i, :, split:]), 1)
            hot = torch.cat((heat_frame(left[i], a.heat_scale, a.heat_alpha, lut)[:, :split],
                             heat_frame(right[i], a.heat_scale, a.heat_alpha, lut)[:, split:]), 1)
            if 0 < split < w:
                frame[:, max(0, split - 1):min(w, split + 1)] = 255
                hot[:, max(0, split - 1):min(w, split + 1)] = 255
            wipe(frame)
            heat(hot)
    index = n // 2 if a.grid_frame < 0 else min(a.grid_frame, n - 1)
    grid = Image.new("RGB", (2 * w, 2 * h))
    for j, (name, label) in enumerate((("pixelumm", "PixelUMM"), ("dcae", "DCAE-V"),
                                      (anchor, anchor), (xview, xview))):
        image = Image.fromarray(videos[name][index].numpy())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, w, 34), fill="black")
        draw.text((12, 8), label, fill="white")
        grid.paste(image, ((j % 2) * w, (j // 2) * h))
    grid.save(out / f"grid_{clip}.jpg", quality=95)


def write_metrics(out, rows, a):
    names = ["dcae", *a.names]
    keys = ["hf_ratio", "temporal_ratio", "temporal_hf_ratio", "flat_hf_ratio", "lpips", "psnr"]
    medians = {name: {key: statistics.median(row["metrics"][name][key] for row in rows) for key in keys}
               for name in names}
    payload = {"reference": "PixelUMM", "checkpoints": dict(zip(a.names, a.ckpts)), "clips": rows,
               "medians": medians, "heat_scale": a.heat_scale, "heat_alpha": a.heat_alpha}
    (out / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["| Clip | Arm | Frames | " + " | ".join(keys) + " |",
             "| --- | --- | ---: | " + " | ".join(["---:"] * len(keys)) + " |"]
    for row in rows:
        for name in names:
            values = " | ".join(f"{row['metrics'][name][key]:.5f}" for key in keys)
            lines.append(f"| {row['clip']} | {name} | {row['frames']} | {values} |")
    for name in names:
        values = " | ".join(f"{medians[name][key]:.5f}" for key in keys)
        lines.append(f"| **Median** | **{name}** | — | {values} |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")


def evaluate(a):
    import torch
    from torchcodec.decoders import VideoDecoder
    from nomas_io import dcae_decode, load_dcae
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = a.arm or a.names[-1]
    anchor = a.anchor or ("anchor" if "anchor" in a.names else a.names[0])
    xview = a.xview or ("xview" if "xview" in a.names else a.names[-1])
    if any(name not in a.names for name in (chosen, anchor, xview)):
        raise ValueError("--arm, --anchor and --xview must be entries in --names")
    files = sorted(Path(a.gen).rglob("*.npy"))
    if not files:
        raise ValueError(f"No generated latents under {a.gen}")
    ids = [p.relative_to(a.gen).with_suffix("").as_posix().replace("/", "__") for p in files]
    ids = [p.stem if sum(q.stem == p.stem for q in files) == 1 else ident for p, ident in zip(files, ids)]
    if len(set(ids)) != len(ids):
        raise ValueError("Generated paths produce duplicate output identifiers")
    vae, rows = load_dcae(), []
    with torch.inference_mode():
        for npy, clip in zip(files, ids):
            mp4 = npy.with_suffix(".mp4")
            if not mp4.is_file():
                raise FileNotFoundError(mp4)
            z = load_latent(npy)
            decoder = VideoDecoder(str(mp4), device="cuda")
            n = min(4 * z.shape[2], decoder.metadata.num_frames)
            if n < 2:
                raise ValueError(f"Need at least two aligned frames: {mp4}")
            teacher = decoder.get_frames_in_range(0, n).data.permute(1, 0, 2, 3)[None].to(torch.bfloat16).div_(127.5).sub_(1).cpu()
            teacher = aligned(teacher, n, "PixelUMM")
            del decoder
            videos = {"pixelumm": pixels(teacher)}
            with torch.autocast("cuda", torch.bfloat16):
                reference = aligned(dcae_decode(vae, z), n, "DCAE-V").cpu()
            metrics = {"dcae": score(reference, teacher, a.metric_batch)}
            videos["dcae"] = pixels(reference)
            del reference
            for name, ckpt in zip(a.names, a.ckpts):
                model = load_student(ckpt)
                pred = aligned(streamed(model, z, a.chunk), n, name)
                del model
                metrics[name] = score(pred, teacher, a.metric_batch)
                if name in (chosen, anchor, xview):
                    videos[name] = pixels(pred)
                del pred
            rows.append({"clip": clip, "path": str(npy), "frames": n, "metrics": metrics})
            write_metrics(out, rows, a)
            visuals(out, clip, videos, chosen, anchor, xview, a)
            print(json.dumps(rows[-1]), flush=True)


def orbit_pose(center, position, up):
    import torch
    import torch.nn.functional as F
    forward = F.normalize(center - position, dim=0)
    right = F.normalize(torch.linalg.cross(forward, up), dim=0)
    down = torch.linalg.cross(forward, right)
    pose = torch.eye(4, device=center.device)
    pose[:3, :3] = torch.stack((right, down, forward), 1)
    pose[:3, 3] = position
    return pose


def splat(a):
    import torch
    import torch.nn.functional as F
    from data import load_poses
    from gauss import render, unproject
    candidate = Path(a.clip)
    matches = [candidate] if candidate.is_file() else sorted(Path(a.gen).rglob(f"{a.clip}.npy"))
    if len(matches) != 1:
        raise ValueError(f"Expected one latent file for {a.clip}, found {len(matches)}")
    npy = matches[0]
    labels = load_poses(a.labels)
    path = str(npy.with_suffix(".mp4").resolve())
    keys = [key for key in labels.windows if str(Path(key).resolve()) == path]
    found = labels.get(keys[0], a.start, 40) if len(keys) == 1 else None
    if found is None:
        raise ValueError(f"No DA3 label window covering {path}@{a.start} for 40 frames")
    label, offset = found
    c2w = label["c2w"][offset:offset + 40].cuda().float()
    K = label["K"][offset:offset + 40].cuda().float().clone()
    height, width = label["hw"]
    K[:, 0] *= 832 / width
    K[:, 1] *= 480 / height
    model = load_student(a.ckpt)
    if not hasattr(model, "gauss_head"):
        raise ValueError("splat requires a checkpoint with a Gaussian head")
    z = load_latent(npy)
    if a.start + 40 > 4 * z.shape[2]:
        raise ValueError("Requested 40-frame window exceeds latent rollout")
    state, cloud, decoded = {}, None, []
    with torch.inference_mode():
        for i in range((a.start + 43) // 4):
            with torch.autocast("cuda", torch.bfloat16):
                result = model(z[:, :, i:i + 1], state=state, gauss=4 * i + 4 > a.start)
            if 4 * i + 4 <= a.start:
                continue
            rgb, g = result
            rgb = aligned(rgb, 4, "Gaussian decoder")
            h, w = g["depth"].shape[-2:]
            if cloud is None:
                cloud = tuple(torch.empty((40 * h * w, d), device="cuda") if d > 1 else
                              torch.empty(40 * h * w, device="cuda") for d in (3, 4, 3, 1, 3))
            small = F.interpolate(rgb[0].transpose(0, 1).float(), (h, w), mode="area")
            for j in range(4):
                frame = 4 * i + j - a.start
                if not 0 <= frame < 40:
                    continue
                k = K[frame].clone()
                k[0] *= w / 832
                k[1] *= h / 480
                points, _ = unproject({key: value[0, :, j].float() for key, value in g.items()},
                                      c2w[frame], k, small[j])
                for dest, source in zip(cloud, points):
                    dest[frame * h * w:(frame + 1) * h * w].copy_(source)
                decoded.append(pixels(rgb[:, :, j:j + 1])[0])
            del result, rgb, g, small, points
        del model, state, z
        center = cloud[0].mean(0)
        base = c2w[0, :3, 3] - center
        depth = torch.linalg.vector_norm(cloud[0] - c2w[0, :3, 3], dim=-1).median().clamp_min(1e-3)
        up = F.normalize(-c2w[0, :3, 1], dim=0)
        if base.norm() < 1e-4:
            base = -c2w[0, :3, 2] * depth
        direction = F.normalize(base, dim=0)
        out = Path(a.out)
        side = out.with_name(out.stem + "_side_by_side" + out.suffix)
        print(json.dumps({"gaussians": cloud[0].shape[0], "cloud_bytes": sum(x.numel() * x.element_size() for x in cloud),
                          "centroid": center.tolist(), "scene_depth": depth.item(), "side_by_side": str(side)}), flush=True)
        with video_writer(out, 832, 480) as write, video_writer(side, 1664, 480) as write_side:
            for i in range(a.frames):
                t = i / max(1, a.frames - 1)
                dolly = min(2 * t, 1)
                dolly = dolly * dolly * (3 - 2 * dolly)
                angle = max(2 * t - 1, 0)
                angle = math.radians(a.degrees) * angle * angle * (3 - 2 * angle)
                vector = base + direction * (2 * depth * dolly)
                vector = vector * math.cos(angle) + torch.linalg.cross(up, vector) * math.sin(angle) + up * torch.dot(up, vector) * (1 - math.cos(angle))
                pose = orbit_pose(center, center + vector, up)
                image, _, _ = render(cloud, torch.linalg.inv(pose)[None], K[:1], 832, 480)
                frame = pixels(image.unsqueeze(2))[0]
                write(frame)
                write_side(torch.cat((decoded[round(t * 39)], frame), 1))


def parser():
    p = argparse.ArgumentParser(description="Compare decoder arms and orbit an accumulated 40-frame Gaussian cloud.",
                                epilog="Requires CUDA, ffmpeg, torchcodec, LPIPS, Pillow and matplotlib; splat also requires gsplat and DA3 labels.")
    commands = p.add_subparsers(dest="command", required=True)
    e = commands.add_parser("eval", help="Full-resolution rollout metrics, wipe/heat videos and labeled grids")
    e.add_argument("--ckpts", nargs="+", required=True)
    e.add_argument("--names", nargs="+", required=True)
    e.add_argument("--gen", default="/workspace/data/gen/showcase")
    e.add_argument("--out", required=True)
    e.add_argument("--arm", default="", help="Wipe/heat arm; defaults to the last checkpoint")
    e.add_argument("--anchor", default="", help="Bottom-left grid arm; defaults to anchor, otherwise first")
    e.add_argument("--xview", default="", help="Bottom-right grid arm; defaults to xview, otherwise last")
    e.add_argument("--chunk", type=int, default=1, help="Latents per stateful student decode chunk")
    e.add_argument("--metric-batch", type=int, default=4)
    e.add_argument("--grid-frame", type=int, default=-1, help="Frame index; negative selects the midpoint")
    e.add_argument("--heat-scale", type=float, default=0.08, help="Fixed Laplacian magnitude scale in RGB [0,1], shared across clips and arms")
    e.add_argument("--heat-alpha", type=float, default=0.65)
    s = commands.add_parser("splat", help="Accumulate all Gaussians from 40 frames; dolly then orbit")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--clip", required=True, help="Rollout ID or latent .npy path")
    s.add_argument("--start", type=int, default=0, help="Decoded-frame start, not latent index")
    s.add_argument("--gen", default="/workspace/data/gen/showcase")
    s.add_argument("--labels", default="/workspace/data/gen_labels")
    s.add_argument("--out", required=True)
    s.add_argument("--frames", type=int, default=240)
    s.add_argument("--degrees", type=float, default=60)
    return p


def main():
    p = parser()
    a = p.parse_args()
    if a.command == "eval":
        if len(a.ckpts) != len(a.names) or len(set(a.names)) != len(a.names) or set(a.names) & {"dcae", "pixelumm"}:
            p.error("--names must be unique, match --ckpts, and exclude dcae/pixelumm")
        if a.chunk < 1 or a.metric_batch < 1 or a.heat_scale <= 0 or not 0 <= a.heat_alpha <= 1:
            p.error("chunk, metric-batch and heat-scale must be positive; heat-alpha must be in [0,1]")
        evaluate(a)
    else:
        if a.start < 0 or a.frames < 2:
            p.error("start must be nonnegative and frames must be at least 2")
        splat(a)


if __name__ == "__main__":
    main()
