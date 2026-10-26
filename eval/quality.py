import argparse, json, time
import av, numpy as np, torch, torch.nn.functional as F

IMNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIPN = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def read_video(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or 16)
        frames = np.stack([f.to_ndarray(format="rgb24") for f in c.decode(s)])
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous(), fps

def batches(x, bs):
    for i in range(0, x.shape[0], bs):
        yield x[i:i + bs]

def to_dev(x, dev):
    return x.to(dev, non_blocking=True).float().div_(255)

def resize_short(x, size):
    h, w = x.shape[-2:]
    s = size / min(h, w)
    return F.interpolate(x, size=(round(h * s), round(w * s)), mode="bilinear", antialias=True, align_corners=False)

def prep(x, size, norm):
    x = resize_short(x, size)
    h, w = x.shape[-2:]
    t, l = (h - size) // 2, (w - size) // 2
    x = x[..., t:t + size, l:l + size]
    m = torch.tensor(norm[0], device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(norm[1], device=x.device).view(1, 3, 1, 1)
    return (x - m) / s

def per_second(v, fps):
    n = max(1, round(fps))
    return [round(float(v[i:i + n].mean()), 4) for i in range(0, len(v), n)]

def gauss(dev, k=11, sigma=1.5):
    g = torch.exp(-((torch.arange(k, device=dev) - k // 2).float() ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, k, k).contiguous()

def ssim(a, b, w):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    f = lambda x: F.conv2d(x, w, groups=3)
    ma, mb = f(a), f(b)
    va, vb, cab = f(a * a) - ma * ma, f(b * b) - mb * mb, f(a * b) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * cab + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))
    return m.flatten(1).mean(1)

class Models:
    def __init__(self, dev):
        from transformers import CLIPVisionModelWithProjection, Dinov2Model
        import lpips
        self.dev = dev
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-small").to(dev).eval()
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        self.win = gauss(dev)

    def embed(self, video, bs):
        d, c = [], []
        for x in batches(video, bs):
            x = to_dev(x, self.dev)
            d.append(self.dino(pixel_values=prep(x, 224, IMNET)).pooler_output)
            c.append(self.clip(pixel_values=prep(x, 224, CLIPN)).image_embeds)
        return F.normalize(torch.cat(d).float(), dim=-1), F.normalize(torch.cat(c).float(), dim=-1)

def consistency(e):
    prev = (e[1:] * e[:-1]).sum(-1).clamp_min(0)
    first = (e[1:] * e[:1]).sum(-1).clamp_min(0)
    return float(((prev + first) / 2).mean())

@torch.inference_mode()
def no_ref(m, video, fps, bs):
    d, c = m.embed(video, bs)
    diffs = []
    for i in range(0, video.shape[0] - 1, bs):
        x = video[i:i + bs + 1].to(m.dev).float()
        diffs.append((x[1:] - x[:-1]).abs().flatten(1).mean(1))
    flick = float(torch.cat(diffs).mean())
    n = max(1, round(fps))
    drift = 1 - float(F.cosine_similarity(d[:n].mean(0), d[-n:].mean(0), dim=0))
    return {"subject_consistency": consistency(d), "background_consistency": consistency(c),
            "temporal_flicker_mad": flick, "temporal_flicker": (255 - flick) / 255, "dino_drift": drift}

@torch.inference_mode()
def full_ref(m, ref, cand, fps, size, bs):
    t = min(ref.shape[0], cand.shape[0])
    ref, cand = ref[:t], cand[:t]
    psnr, ss, lp = [], [], []
    for a, b in zip(batches(ref, bs), batches(cand, bs)):
        a, b = to_dev(a, m.dev), to_dev(b, m.dev)
        if b.shape[-2:] != a.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", antialias=True, align_corners=False)
        mse = (a - b).pow(2).flatten(1).mean(1).clamp_min(1e-15)
        psnr.append(-10 * torch.log10(mse))
        a, b = resize_short(a, size).clamp(0, 1), resize_short(b, size).clamp(0, 1)
        ss.append(ssim(a, b, m.win))
        lp.append(m.lpips(a * 2 - 1, b * 2 - 1).flatten())
    psnr, ss, lp = (torch.cat(v).cpu() for v in (psnr, ss, lp))
    lps = per_second(lp, fps)
    return {"frames": t, "psnr": float(psnr.mean()), "ssim": float(ss.mean()), "lpips": float(lp.mean()),
            "lpips_first_s": lps[0], "lpips_last_s": lps[-1],
            "psnr_per_s": per_second(psnr, fps), "ssim_per_s": per_second(ss, fps), "lpips_per_s": lps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("cands", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    m = Models(dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    ref, fps = read_video(a.ref)
    t0 = time.perf_counter()
    r = {"video": a.ref, "role": "ref", "frames": ref.shape[0], "fps": fps, **no_ref(m, ref, fps, a.batch)}
    sync()
    print(json.dumps({**r, "secs": round(time.perf_counter() - t0, 3)}), flush=True)
    for p in a.cands:
        cand, cfps = read_video(p)
        t0 = time.perf_counter()
        r = {"video": p, "role": "cand", "ref": a.ref, "fps": cfps, **full_ref(m, ref, cand, fps, a.size, a.batch),
             **no_ref(m, cand, cfps, a.batch)}
        sync()
        print(json.dumps({**r, "secs": round(time.perf_counter() - t0, 3)}), flush=True)

if __name__ == "__main__":
    main()
