import argparse, glob, json, os
import numpy as np, torch, torch.nn.functional as F
from PIL import Image


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "frames", "*.png")))
    return torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fs])).permute(0, 3, 1, 2).contiguous()


def gauss(dev, k=11, sigma=1.5):
    g = torch.exp(-((torch.arange(k, device=dev) - k // 2).float() ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, k, k).contiguous()


def ssim(a, b, w):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    f = lambda x: F.conv2d(x, w, groups=3)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    return (((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).flatten(1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ref", default="bf16")
    ap.add_argument("--chunk_frames", type=int, default=8)
    a = ap.parse_args()
    import lpips
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    w = gauss(dev)
    rows = []
    for ref in sorted(glob.glob(os.path.join(a.root, "**", a.ref, "meta.json"), recursive=True)):
        clip = os.path.dirname(os.path.dirname(ref))
        R = load(os.path.dirname(ref))
        for cfg in sorted(os.listdir(clip)):
            d = os.path.join(clip, cfg)
            if cfg == a.ref or not os.path.exists(os.path.join(d, "meta.json")):
                continue
            X = load(d)
            n = min(len(R), len(X))
            psnr, ss, lpv = [], [], []
            with torch.no_grad():
                for i in range(0, n, 16):
                    r = R[i:min(i + 16, n)].to(dev).float() / 255
                    x = X[i:min(i + 16, n)].to(dev).float() / 255
                    mse = ((r - x) ** 2).flatten(1).mean(1).clamp_min(1e-12)
                    psnr += (10 * torch.log10(1 / mse)).tolist()
                    ss += ssim(r, x, w).tolist()
                    lpv += lp(r * 2 - 1, x * 2 - 1).flatten().tolist()
            k = a.chunk_frames
            per_chunk = lambda v: [round(float(np.mean(v[j:j + k])), 4) for j in range(0, len(v), k)]
            out = {"clip": os.path.relpath(clip, a.root), "config": cfg, "ref": a.ref, "frames": n,
                   "psnr_mean": round(float(np.mean(psnr)), 3), "ssim_mean": round(float(np.mean(ss)), 4), "lpips_mean": round(float(np.mean(lpv)), 4),
                   "psnr_first_chunk": round(float(np.mean(psnr[:k])), 3), "psnr_last_chunk": round(float(np.mean(psnr[-k:])), 3),
                   "lpips_first_chunk": round(float(np.mean(lpv[:k])), 4), "lpips_last_chunk": round(float(np.mean(lpv[-k:])), 4),
                   "per_chunk": {"psnr": per_chunk(psnr), "ssim": per_chunk(ss), "lpips": per_chunk(lpv)},
                   "per_frame": {"psnr": [round(v, 3) for v in psnr], "ssim": [round(v, 4) for v in ss], "lpips": [round(v, 4) for v in lpv]}}
            json.dump(out, open(os.path.join(d, f"vs_{a.ref}.json"), "w"))
            rows.append({k2: v for k2, v in out.items() if k2 not in ("per_chunk", "per_frame")})
            print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(os.path.join(a.root, f"accuracy_vs_{a.ref}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
