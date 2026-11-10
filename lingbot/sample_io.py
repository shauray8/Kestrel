import json, os, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from PIL import Image

FLAGS = ("MODEL", "ATTN", "N3", "FUSED_GLUE", "KESTREL_E_BCAST", "CAM_HOIST", "FUSED_FFN", "ROPE32", "ROPE_KV", "XATTN", "RMS_ROPE",
         "GLUE2", "QKV_FUSE", "FA_LOAD_MODE", "FA_EXP_MODE", "FA_Q_TILE", "FA_RESCALE_THRESHOLD", "FA_MX", "FA_BAL", "N3_CONT", "FA_QK",
         "CHUNK", "WINDOW", "SINK", "FRAMES", "SEED", "AMAX_TARGET", "ONLY", "CALIB_CACHE", "CONFIG", "GIT_HASH", "GIT_DIRTY")


def gpu_state():
    q = "clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active,name,driver_version"
    try:
        v = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        return dict(zip(q.split(","), [x.strip() for x in v.split(",")]))
    except Exception as e:
        return {"error": str(e)}


class ClockSampler:
    def __init__(self):
        self.samples, self.stop = [], threading.Event()
        self.t = threading.Thread(target=self.run, daemon=True)
        self.t.start()

    def run(self):
        while not self.stop.wait(0.5):
            s = gpu_state()
            try:
                self.samples.append((float(s["clocks.sm"]), float(s["power.draw"])))
            except Exception:
                pass

    def finish(self):
        self.stop.set()
        self.t.join()
        if not self.samples:
            return {}
        a = np.array(self.samples)
        return {"sm_clock_mhz_mean": round(float(a[:, 0].mean()), 1), "sm_clock_mhz_min": float(a[:, 0].min()),
                "power_w_mean": round(float(a[:, 1].mean()), 1), "power_w_max": float(a[:, 1].max()), "n": len(a)}


class Recorder:
    def __init__(self, pipe, root, name, prompt, image, action, attn, fps):
        self.pipe, self.root, self.name, self.prompt, self.image, self.action, self.attn, self.fps = pipe, root, name, prompt, image, action, attn, fps
        self.ends, self.t0, self.clock, self.first = [], None, None, None
        self.forwards = 0
        orig = pipe.model.forward

        def forward(*a, **kw):
            if self.t0 is not None and self.first is None:
                torch.cuda.synchronize()
                self.first = time.perf_counter()
            out = orig(*a, **kw)
            self.forwards += 1
            t = kw.get("t")
            if self.t0 is not None and t is not None and bool((t == 0).all()):
                torch.cuda.synchronize()
                self.ends.append(time.perf_counter())
            return out
        pipe.model.forward = forward

    def start(self, tag, t0):
        self.ends, self.t0, self.forwards, self.first = [], t0, 0, None
        self.gpu_start = gpu_state()
        self.clock = ClockSampler()

    def save(self, tag, video, t1):
        clocks = self.clock.finish()
        cfg = os.environ.get("CONFIG", tag)
        d = os.path.join(self.root, self.name, cfg)
        fd = os.path.join(d, "frames")
        os.makedirs(fd, exist_ok=True)
        v = ((video.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda i: Image.fromarray(v[i]).save(os.path.join(fd, f"{i:05d}.png"), compress_level=1), range(len(v))))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i", os.path.join(fd, "%05d.png"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", os.path.join(d, "clip.mp4")], check=False)
        edges = [self.first or self.t0] + self.ends
        chunk_s = [round(b - a, 4) for a, b in zip(edges, edges[1:])]
        cam = os.path.join(self.action, "script.json")
        meta = {
            "name": self.name, "config": cfg, "run_tag": tag, "prompt": self.prompt, "seed": int(os.environ.get("SEED", "42")),
            "image": self.image, "movement_script": json.load(open(cam)) if os.path.exists(cam) else self.action, "action_path": self.action,
            "attention_backend": self.attn if os.environ.get("CONFIG", "") != "bf16" else "stock",
            "precision": {"bf16": "bf16 stock", "sage2": "bf16 + SageAttention2++ (int8 QK, fp8 PV)", "kernel": "bf16 + Kestrel INT8-QK/MXFP8-PV attention only",
                          "int8": "full int8 stack (FP8 N3 GEMMs + Kestrel attention + fused glue)"}.get(cfg, cfg),
            "git_hash": os.environ.get("GIT_HASH", "unknown"), "git_dirty": os.environ.get("GIT_DIRTY", "unknown"),
            "resolution": [int(v.shape[1]), int(v.shape[2])], "frames": int(v.shape[0]), "fps": self.fps,
            "tokens_per_latent_frame": (v.shape[1] // 16) * (v.shape[2] // 16),
            "wall_s_total": round(t1 - self.t0, 4), "chunk_wall_s": chunk_s,
            "setup_s": round(self.first - self.t0, 4) if self.first else None,
            "chunk_note": "chunk_wall_s starts at the first DiT forward (setup_s = text+image encode before it); each chunk ends at its KV-cache update (sync); decode_s is the Wan VAE decode after the last chunk",
            "decode_s": round(t1 - self.ends[-1], 4) if self.ends else None, "dit_loop_s": round(self.ends[-1] - (self.first or self.t0), 4) if self.ends else None,
            "forwards": self.forwards, "gpu_start": self.gpu_start, "gpu_end": gpu_state(), "gpu_during": clocks,
            "flags": {k: os.environ[k] for k in FLAGS if k in os.environ}, "torch": torch.__version__, "cuda": torch.version.cuda,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        }
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(json.dumps({"exp": "sample", "dir": d, "frames": meta["frames"], "res": meta["resolution"], "wall_s": meta["wall_s_total"],
                          "dit_loop_s": meta["dit_loop_s"]}), flush=True)
