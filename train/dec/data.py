import glob
import json
import queue
import random
import threading
import hashlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchcodec.decoders import VideoDecoder

LAT_H, LAT_W, WIN = 15, 26, 5


def load_latents(root, limit=0):
    root = Path(root)
    shards = [json.loads(l)["shard"] for l in open(root / "index.jsonl")]
    lats, paths = [], []
    s1 = s2 = 0
    cnt = 0
    for s in shards:
        d = torch.load(root / s, weights_only=False)
        lats.append(d["latents"])
        paths += d["paths"]
        s1, s2, cnt = s1 + d["s1"], s2 + d["s2"], cnt + d["cnt"]
        if limit and len(paths) >= limit:
            break
    lat = torch.cat(lats).pin_memory()
    mean = s1 / cnt
    st = {"mean": mean.float(), "std": (s2 / cnt - mean * mean).clamp_min(1e-12).sqrt().float()}
    return lat, paths, st


def is_holdout(path):
    return int(hashlib.md5(Path(path).name.encode()).hexdigest(), 16) % 100 == 0


def read_frames(path, start, n):
    return VideoDecoder(path, device="cuda").get_frames_in_range(start, start + n).data


class PoseLabels:
    def __init__(self, root):
        root = Path(root)
        self.root = root if root.is_dir() else root.parent
        index = root / "index.jsonl" if root.is_dir() else root
        self.index, self.windows, self.cache = {}, {}, OrderedDict()
        with open(index) as f:
            for line in f:
                row = json.loads(line)
                for key in row["paths"]:
                    self.index[key] = row["shard"]
                    path, sep, start = key.rpartition("@")
                    if sep and start.isdigit():
                        self.windows.setdefault(path, []).append((int(start), key))
        for windows in self.windows.values():
            windows.sort(reverse=True)

    def get(self, path, start, n):
        keys = ([path] if path in self.index else []) + [k for s, k in self.windows.get(path, []) if s <= start]
        for key in keys:
            shard = self.index[key]
            if shard not in self.cache:
                self.cache[shard] = torch.load(self.root / shard, map_location="cpu", weights_only=False)
                if len(self.cache) > 2:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(shard)
            label = self.cache[shard][key]
            offset = start - label.get("start", int(key.rpartition("@")[2]) if key != path else 0)
            if offset >= 0 and offset + n <= len(label["c2w"]) and offset + n <= len(label["K"]):
                return label, offset
        return None


def load_poses(root):
    return PoseLabels(root)


class Batcher:
    def __init__(self, lat, paths, idx, batch, crop=8, ctx=2, depth=4, seed=0, threads=8, poses=None, win=WIN, anchor=True):
        self.lat, self.paths, self.idx, self.batch, self.crop, self.ctx = lat, paths, idx, batch, crop, ctx
        self.win, self.anchor = win, anchor
        self.poses = poses
        self.full = crop <= 0
        self.q = queue.Queue(depth)
        self.rng = random.Random(seed)
        self.pool = ThreadPoolExecutor(threads)
        self.stream = torch.cuda.Stream()
        self.err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _spec(self):
        i = self.rng.choice(self.idx)
        s = self.rng.randrange(0, self.lat.shape[2] // self.win) * self.win
        ch, cw = (LAT_H, LAT_W) if self.full else (self.crop, self.crop)
        return i, s, ch, cw, self.rng.randrange(0, LAT_H - ch + 1), self.rng.randrange(0, LAT_W - cw + 1)

    def _latent(self, i, s, ch, cw, cy, cx):
        lo = s - self.ctx
        z = self.lat[i, :, max(lo, 0): s + self.win, cy: cy + ch, cx: cx + cw]
        if lo < 0:
            z = torch.cat([z[:, :1].expand(-1, -lo, -1, -1), z], 1)
        return z

    def _frames(self, spec):
        i, s, ch, cw, cy, cx = spec
        with torch.cuda.stream(self.stream):
            return read_frames(self.paths[i], 4 * s, 4 * self.win)[:, :, cy * 32: (cy + ch) * 32, cx * 32: (cx + cw) * 32]

    def _poses(self, spec):
        i, s, ch, cw, cy, cx = spec
        found = self.poses.get(self.paths[i], 4 * s, 4 * self.win)
        if found is None:
            return None
        label, offset = found
        sl = slice(offset, offset + 4 * self.win)
        K = label["K"][sl].float().clone()
        K[:, 0, 2] -= cx * 32
        K[:, 1, 2] -= cy * 32
        pose = {"c2w": label["c2w"][sl].float(), "K": K}
        if "depth" in label:
            H, W = label["hw"]
            yy, xx = torch.meshgrid((torch.arange(ch * 16).float() + 0.5) * 2 + cy * 32,
                                    (torch.arange(cw * 16).float() + 0.5) * 2 + cx * 32, indexing="ij")
            grid = torch.stack((2 * xx / W - 1, 2 * yy / H - 1), -1)
            depth = label["depth"][sl, None].float()
            pose["depth"] = F.grid_sample(depth, grid[None].expand(len(depth), -1, -1, -1),
                                           mode="nearest", padding_mode="border", align_corners=False)
        return {k: v.to("cuda", non_blocking=True) for k, v in pose.items()}

    def _run(self):
        try:
            with torch.cuda.stream(self.stream):
                while True:
                    specs = [self._spec() for _ in range(self.batch)]
                    vs = list(self.pool.map(self._frames, specs)) if self.anchor else None
                    zs = [self._latent(*sp) for sp in specs]
                    z = torch.stack(zs).to("cuda", non_blocking=True).to(torch.bfloat16)
                    y = torch.stack(vs).permute(0, 2, 1, 3, 4).to(torch.bfloat16).div_(127.5).sub_(1) if vs is not None else None
                    z = z.contiguous(memory_format=torch.channels_last_3d)
                    if y is not None:
                        y = y.contiguous(memory_format=torch.channels_last_3d)
                    poses = [self._poses(sp) for sp in specs] if self.poses is not None else None
                    ev = torch.cuda.Event()
                    ev.record(self.stream)
                    self.q.put((z, y, poses, ev))
        except Exception as e:
            self.err = e
            self.q.put(None)

    def next(self):
        item = self.q.get()
        if item is None:
            raise RuntimeError(f"batcher failed: {self.err!r}")
        z, y, poses, ev = item
        torch.cuda.current_stream().wait_event(ev)
        z.record_stream(torch.cuda.current_stream())
        if y is not None:
            y.record_stream(torch.cuda.current_stream())
        if self.poses is not None:
            for pose in poses:
                if pose is not None:
                    for tensor in pose.values():
                        tensor.record_stream(torch.cuda.current_stream())
            return z, y, poses
        return z, y


class GeneratedBatcher(Batcher):
    def __init__(self, root, poses, batch, crop=8, ctx=2, seed=0, anchor=True, anchor_suffix=""):
        self.anchor_suffix = anchor_suffix
        candidates = glob.glob(str(Path(root).expanduser()), recursive=True)
        files = set()
        for candidate in candidates:
            p = Path(candidate)
            files.update(p.rglob("*.npy") if p.is_dir() else [p.with_suffix(".npy")])
        self.arrays, self.windows = {}, []
        paths = []
        labelled = {str(Path(path).resolve()): path for path in poses.windows}
        for p in sorted(files):
            video = str(p.with_suffix(".mp4"))
            key = labelled.get(str(Path(video).resolve()), video)
            windows = poses.windows.get(key, [])
            if not p.is_file() or not windows:
                continue
            if anchor and not Path(video[:-4] + anchor_suffix + ".mp4").is_file():
                raise FileNotFoundError(video[:-4] + anchor_suffix + ".mp4")
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[0] != 32 or tuple(arr.shape[2:]) != (LAT_H, LAT_W):
                raise ValueError(f"invalid generated latent shape: {p}: {arr.shape}")
            self.arrays[str(p)] = arr
            for start, _ in windows:
                if start % 4:
                    raise ValueError(f"generated label start must be divisible by 4: {key}@{start}")
                if start // 4 + 10 <= arr.shape[1]:
                    self.windows.append((str(p), start // 4))
                    paths.append(key)
        if not self.windows:
            raise ValueError(f"no labelled 40-frame generated windows under {root}")
        super().__init__(None, paths, list(range(len(paths))), batch, crop, ctx, seed=seed,
                         poses=poses, win=10, anchor=anchor)

    def _spec(self):
        i = self.rng.choice(self.idx)
        ch, cw = (LAT_H, LAT_W) if self.full else (self.crop, self.crop)
        return i, self.windows[i][1], ch, cw, self.rng.randrange(0, LAT_H - ch + 1), self.rng.randrange(0, LAT_W - cw + 1)

    def _latent(self, i, s, ch, cw, cy, cx):
        arr = self.arrays[self.windows[i][0]][:, max(s - self.ctx, 0):s + self.win, cy:cy + ch, cx:cx + cw]
        z = torch.from_numpy(arr.copy())
        if s < self.ctx:
            z = torch.cat([z[:, :1].expand(-1, self.ctx - s, -1, -1), z], 1)
        return z

    def _frames(self, spec):
        i, s, ch, cw, cy, cx = spec
        with torch.cuda.stream(self.stream):
            return read_frames(self.paths[i][:-4] + self.anchor_suffix + ".mp4", 4 * s, 4 * self.win)[:, :, cy * 32: (cy + ch) * 32, cx * 32: (cx + cw) * 32]

    def _poses(self, spec):
        pose = super()._poses(spec)
        if pose is None:
            raise ValueError(f"missing generated labels: {self.paths[spec[0]]}@{4 * spec[1]}")
        return pose
