import argparse
import json
from pathlib import Path

F16 = 65504.0


def load_da3(model_dir):
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(model_dir).to("cuda").eval()


def to_c2w(ext):
    import torch
    e = torch.as_tensor(ext, dtype=torch.float64)
    if e.shape[-2] == 3:
        e = torch.cat([e, e.new_tensor([0, 0, 0, 1]).expand(len(e), 1, 4)], 1)
    return (e[0] @ torch.linalg.inv(e)).float()


def label(model, v, a, start=0):
    import torch
    H, W = v.shape[-2:]
    pr = model.inference(list(v.permute(0, 2, 3, 1).cpu().numpy()), process_res=a.res, ref_view_strategy=a.ref)
    d = torch.as_tensor(pr.depth).float()
    h, w = d.shape[-2:]
    k = torch.as_tensor(pr.intrinsics, dtype=torch.float64).clone()
    k[:, 0] *= W / w
    k[:, 1] *= H / h
    r = {"depth": d.clamp(-F16, F16).half(), "c2w": to_c2w(pr.extrinsics), "K": k.float(),
         "scale": (H / h, W / w), "hw": (H, W), "start": start}
    if pr.conf is not None:
        r["conf"] = torch.as_tensor(pr.conf).float().clamp(-F16, F16).half()
    return r


def jobs(a, seen):
    from torchcodec.decoders import VideoDecoder
    if a.gen:
        out = []
        for p in sorted(str(p) for p in Path(a.gen).rglob("*.mp4")):
            n = VideoDecoder(p).metadata.num_frames
            out += [(p, s) for s in range(0, n - a.frames + 1, a.stride) if f"{p}@{s}" not in seen]
        return out
    paths = [p for l in open(Path(a.latents) / "index.jsonl") for p in json.loads(l)["paths"]]
    return [(p, 0) for p in paths if p not in seen]


def run(a):
    import torch
    from torchcodec.decoders import VideoDecoder
    out = Path(a.out or ("/workspace/data/gen_labels" if a.gen else "/workspace/data/labels"))
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(out / "index.jsonl")] if (out / "index.jsonl").exists() else []
    todo = jobs(a, {p for r in rows for p in r["paths"]})
    if not todo:
        print(json.dumps({"new": 0, "skip": True}))
        return
    model = load_da3(a.model_dir)
    cur, frames = None, None
    for k, si in enumerate(range(0, len(todo), a.shard)):
        name = f"lab_{len(rows) + k:05d}.pt"
        group, labs = todo[si: si + a.shard], {}
        for p, s in group:
            if a.gen:
                if cur != p:
                    dec = VideoDecoder(p, device="cuda")
                    cur, frames = p, dec.get_frames_in_range(0, dec.metadata.num_frames).data
                v, key = frames[s: s + a.frames], f"{p}@{s}"
            else:
                v, key = VideoDecoder(p, device="cuda").get_frames_in_range(0, a.frames).data, p
            labs[key] = label(model, v, a, s)
        torch.save(labs, out / name)
        with open(out / "index.jsonl", "a") as f:
            f.write(json.dumps({"shard": name, "paths": list(labs)}) + "\n")
        print(json.dumps({"shard": name, "done": si + len(group), "of": len(todo)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--latents", default="/workspace/data/latents")
    p.add_argument("--out", default="")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--shard", type=int, default=64)
    p.add_argument("--model_dir", default="depth-anything/DA3-BASE")
    p.add_argument("--gen", nargs="?", const="/workspace/data/gen", default="")
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--ref", default="saddle_balanced")
    run(p.parse_args())
