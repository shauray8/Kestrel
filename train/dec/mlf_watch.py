import argparse
import json
import time
from pathlib import Path

import mlflow
import torch

import mlf
from arms import load_latent, load_student, pixels, streamed, video_writer
from data import read_frames
from metrics import grain


def records(path, offset):
    if not path.exists():
        return [], offset
    with open(path) as f:
        f.seek(offset)
        lines = f.readlines()
        offset = f.tell()
    return [json.loads(l) for l in lines if l.strip()], offset


@torch.no_grad()
def probe(ckpt, step, a, out):
    model = load_student(ckpt)
    stats = {}
    for clip in a.clips:
        root = Path(a.gen) / clip
        z = load_latent(root / f"{clip}.npy")[:, :, :a.latents]
        y = streamed(model, z, 10)
        n = y.shape[2]
        ref = read_frames(str(root / f"{clip}.mp4"), 0, n).permute(1, 0, 2, 3)[None].to(torch.bfloat16).div(127.5).sub(1)
        for k, v in grain(y.cuda(), ref.cuda()).items():
            stats.setdefault(f"gen/{k}", []).append(float(v))
        ours, theirs = pixels(y), pixels(ref.cpu())
        path = out / f"{clip}_{step:07d}.mp4"
        with video_writer(path, 1664, 480) as write:
            for p, r in zip(ours, theirs):
                write(torch.cat([r, p], 1))
        mlflow.log_artifact(str(path), f"videos/{clip}")
        idx = [int(i * (n - 1) / 3) for i in range(4)]
        strip = torch.cat([torch.cat([theirs[i] for i in idx], 1), torch.cat([ours[i] for i in idx], 1)], 0)
        mlflow.log_image(strip[::2, ::2].numpy(), key=f"frames/{clip}", step=step)
    mlflow.log_metrics({k: sum(v) / len(v) for k, v in stats.items()}, step=step)
    del model
    torch.cuda.empty_cache()


def main(a):
    run = Path(a.run)
    out = run / "mlf_videos"
    out.mkdir(exist_ok=True)
    cfg = json.load(open(run / "args.json"))
    mlf.start(a.experiment, run.name, cfg)
    offset, seen, last_step = 0, set(), 0
    while True:
        recs, offset = records(run / "log.jsonl", offset)
        for r in recs:
            if "step" in r:
                last_step = max(last_step, r["step"])
                mlf.log(r, r["step"])
        for ck in sorted(run.glob("ema_*.pt")):
            if ck.name not in seen and time.time() - ck.stat().st_mtime > 20:
                try:
                    probe(ck, int(ck.stem.split("_")[1]), a, out)
                except Exception as e:
                    print(f"probe {ck.name} failed: {e}", flush=True)
        if last_step >= cfg["steps"] and not recs and all(c.name in seen for c in run.glob("ema_*.pt")):
            break
        time.sleep(a.every)
    mlflow.end_run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--experiment", default="kestrel-decoder")
    p.add_argument("--gen", default="/workspace/data/gen/showcase")
    p.add_argument("--clips", nargs="+", default=["neon_robot_rainbreak", "autumn_fox_watercolor"])
    p.add_argument("--latents", type=int, default=24)
    p.add_argument("--every", type=int, default=60)
    main(p.parse_args())
