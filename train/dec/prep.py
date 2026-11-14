import argparse
import json
import os
import random
import shutil
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

H, W, FPS, FRAMES, GOP = 480, 832, 24, 120, 20
VF = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},fps={FPS}"


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", path],
                       capture_output=True, text=True, timeout=60)
    j = json.loads(r.stdout or "{}")
    s = (j.get("streams") or [{}])[0]
    return int(s.get("width", 0)), int(s.get("height", 0)), float(j.get("format", {}).get("duration", 0) or 0)


def cut(args):
    src, out_dir, max_segs = args
    try:
        w, h, dur = probe(src)
    except Exception:
        return []
    if min(w, h) < H or dur < FRAMES / FPS + 0.5:
        return []
    n = min(max_segs, int((dur - 0.5) // (FRAMES / FPS)))
    outs = []
    for k in range(n):
        t0 = 0.25 + k * FRAMES / FPS
        dst = Path(out_dir) / f"{Path(src).stem}_{k}.mp4"
        tmp = dst.with_suffix(".tmp")
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-i", src, "-vf", VF, "-frames:v", str(FRAMES),
               "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p",
               "-g", str(GOP), "-keyint_min", str(GOP), "-sc_threshold", "0", "-bf", "0",
               "-force_key_frames", f"expr:eq(mod(n,{GOP}),0)", "-threads", "1", "-f", "mp4", str(tmp)]
        if subprocess.run(cmd, timeout=300).returncode == 0 and tmp.exists():
            r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", str(tmp)], capture_output=True, text=True)
            if r.stdout.strip() == str(FRAMES):
                tmp.rename(dst)
                outs.append(str(dst))
                continue
        tmp.unlink(missing_ok=True)
    return outs


def clips(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp)
    for z in a.zips:
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with zipfile.ZipFile(z) as f:
            names = [n for n in f.namelist() if n.lower().endswith((".mp4", ".mkv", ".webm", ".mov"))]
            if a.limit:
                names = names[: a.limit]
            for n in names:
                f.extract(n, tmp)
        vids = [str(p) for p in tmp.rglob("*") if p.is_file()]
        done = []
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(cut, [(v, str(out), a.segs) for v in vids], chunksize=4):
                done += r
        shutil.rmtree(tmp, ignore_errors=True)
        if a.delete_zip:
            os.remove(z)
        print(json.dumps({"zip": z, "videos": len(vids), "clips": len(done), "total": len(list(out.glob("*.mp4")))}), flush=True)


def encode(a):
    import torch
    from torchcodec.decoders import VideoDecoder

    from data import is_holdout
    from nomas_io import dcae_encode, load_dcae
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(out / "index.jsonl")] if (out / "index.jsonl").exists() else []
    seen = {p for r in rows for p in r["paths"]}
    paths = sorted(str(p) for p in Path(a.clips).glob("*.mp4") if str(p) not in seen and is_holdout(p) == a.holdout)
    random.Random(len(rows)).shuffle(paths)
    if a.max_total:
        paths = paths[: max(0, a.max_total - len(seen))]
    if len(paths) < a.min_new:
        print(json.dumps({"new": len(paths), "skip": True}))
        return
    vae = load_dcae()
    for k, si in enumerate(range(0, len(paths), a.shard)):
        name = f"lat_{len(rows) + k:05d}.pt"
        group, lats = paths[si: si + a.shard], []
        s1 = torch.zeros(32, dtype=torch.float64, device="cuda")
        s2 = torch.zeros_like(s1)
        for p in group:
            v = VideoDecoder(p, device="cuda").get_frames_in_range(0, a.frames).data
            x = v.permute(1, 0, 2, 3)[None].to(torch.bfloat16).div_(127.5).sub_(1)
            z = dcae_encode(vae, x)[0]
            zf = z.double()
            s1 += zf.sum((1, 2, 3))
            s2 += (zf * zf).sum((1, 2, 3))
            lats.append(z.to(torch.float16).cpu())
        cnt = len(group) * lats[0][0].numel()
        torch.save({"latents": torch.stack(lats), "paths": group, "s1": s1.cpu(), "s2": s2.cpu(), "cnt": cnt}, out / name)
        with open(out / "index.jsonl", "a") as f:
            f.write(json.dumps({"shard": name, "paths": group}) + "\n")
        print(json.dumps({"shard": name, "done": si + len(group), "of": len(paths)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("clips")
    c.add_argument("zips", nargs="+")
    c.add_argument("--out", default="/workspace/data/clips")
    c.add_argument("--tmp", default="/workspace/data/tmp")
    c.add_argument("--workers", type=int, default=18)
    c.add_argument("--segs", type=int, default=2)
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--delete_zip", action="store_true")
    e = sp.add_parser("encode")
    e.add_argument("--clips", default="/workspace/data/clips")
    e.add_argument("--out", default="/workspace/data/latents")
    e.add_argument("--shard", type=int, default=512)
    e.add_argument("--min_new", type=int, default=1)
    e.add_argument("--frames", type=int, default=40)
    e.add_argument("--holdout", action="store_true")
    e.add_argument("--max_total", type=int, default=0)
    a = p.parse_args()
    {"clips": clips, "encode": encode}[a.cmd](a)
