import argparse, glob, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "/System/Library/Fonts/Menlo.ttc"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


class Clip:
    def __init__(self, d, label):
        self.d, self.label = d, label
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.frames = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
        ch = self.meta["chunk_wall_s"]
        if self.meta.get("setup_s") is None:
            ch = ch[1:]
            first = 5
        else:
            first = 0
        self.ready, t, n = [], 0.0, 0
        lat = self.meta.get("flags", {}).get("CHUNK", "2")
        per = int(lat) * 4
        for k, c in enumerate(ch):
            t += c
            n = (1 + (int(lat) - 1) * 4) if (k == 0 and first == 0) else n + per
            self.ready.append((t, min(n, len(self.frames))))
        if first:
            self.ready = [(t, min(n + first, len(self.frames))) for t, n in self.ready]
        self.chunk_ms = [round(c * 1000) for c in ch]
        self.p, self.cache = 0, {}

    def available(self, t):
        a = 0
        for tt, n in self.ready:
            if tt <= t:
                a = n
        return a

    def step(self, t):
        a = self.available(t)
        if self.p + 1 < a:
            self.p += 1
        return self.p

    def done(self):
        return self.p >= len(self.frames) - 1

    def image(self, i):
        if i not in self.cache:
            self.cache = {i: Image.open(self.frames[i]).convert("RGB")}
        return self.cache[i].copy()

    def chunk_at(self, t):
        for k, (tt, _) in enumerate(self.ready):
            if t < tt:
                return k
        return len(self.ready) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--clip", nargs=2, action="append", metavar=("DIR", "LABEL"), required=True)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--chunk_counter", action="store_true")
    a = ap.parse_args()
    clips = [Clip(d, l) for d, l in a.clip]
    W, H = 832, 480
    f1, f2 = font(26), font(20)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W * len(clips)}x{H}",
                             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    t, dt, finish = 0.0, 1.0 / a.fps, {}
    while True:
        canvas = Image.new("RGB", (W * len(clips), H))
        for j, c in enumerate(clips):
            p = c.step(t)
            if c.done() and j not in finish:
                finish[j] = t
            im = c.image(p)
            d = ImageDraw.Draw(im)
            gen = finish.get(j, t)
            txt = f"{c.label}   {gen:5.2f} s"
            d.rectangle((0, 0, W, 40), fill=(0, 0, 0))
            d.text((14, 6), txt, font=f1, fill=(255, 255, 255) if j not in finish else (120, 255, 140))
            if a.chunk_counter:
                k = c.chunk_at(t)
                d.rectangle((0, H - 34, 330, H), fill=(0, 0, 0))
                d.text((12, H - 30), f"chunk {k:3d}  {c.chunk_ms[k]:4d} ms", font=f2, fill=(255, 220, 120))
            canvas.paste(im, (j * W, 0))
        proc.stdin.write(np.asarray(canvas).tobytes())
        if len(finish) == len(clips) and t - max(finish.values()) >= a.hold:
            break
        t += dt
    proc.stdin.close()
    proc.wait()
    print(json.dumps({"out": a.out, "finish_s": {clips[j].label: round(v, 2) for j, v in finish.items()}}))


if __name__ == "__main__":
    main()
