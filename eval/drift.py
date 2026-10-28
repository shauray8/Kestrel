import argparse, json
import av, torch, torch.nn.functional as F
from quality import IMNET, CLIPN, batches, prep, resize_short

CURVES = ("dino_cosine", "clip_text_similarity", "sharpness", "saturation_mean", "brightness_mean", "brightness_std", "near_white_fraction", "near_black_fraction", "motion_mad")

def _feat(o):
    return o if torch.is_tensor(o) else o.pooler_output

class Models:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPTokenizer, Dinov2Model
        self.device = device
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-small").to(device).eval()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    @torch.inference_mode()
    def text(self, prompt):
        tokens = self.tokenizer([prompt], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        return F.normalize(_feat(self.clip.get_text_features(**tokens)).float(), dim=-1)

    @torch.inference_mode()
    def embed(self, frames):
        dino, clip = [], []
        for batch in batches(torch.stack(frames), 8):
            x = batch.to(self.device).float().div(255)
            dino.append(self.dino(pixel_values=prep(x, 224, IMNET)).pooler_output)
            clip.append(_feat(self.clip.get_image_features(pixel_values=prep(x, 224, CLIPN))))
        d = F.normalize(torch.cat(dino).float().mean(0), dim=0)
        c = F.normalize(torch.cat(clip).float().mean(0), dim=0)
        return d.cpu(), c.cpu()

def pixel_metrics(frames):
    x = torch.stack(frames)
    gray = x[:, 0] * 0.299 + x[:, 1] * 0.587 + x[:, 2] * 0.114
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
    lap = F.conv2d(gray.unsqueeze(1), kernel).squeeze(1)
    high = x.amax(1)
    low = x.amin(1)
    saturation = ((high - low) / high.clamp_min(1)).mean((1, 2))
    return {
        "sharpness_raw": float(lap.var((1, 2), unbiased=False).mean()),
        "saturation_mean": float(saturation.mean()),
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std(unbiased=False)),
        "near_white_fraction": float((low > 245).float().mean()),
        "near_black_fraction": float((high < 10).float().mean()),
        "motion_mad": float((x[1:] - x[:-1]).abs().mean()) if len(x) > 1 else 0.0,
    }

def ratio(a, b):
    return a / b if b else None

def summary(windows):
    if not windows:
        return {"slope_per_min": {}, "last_first_ratio": {}}
    t = torch.tensor([(w["start_s"] + w["end_s"]) / 120 for w in windows], dtype=torch.float64)
    slopes, ratios = {}, {}
    for key in CURVES:
        y = torch.tensor([w[key] for w in windows], dtype=torch.float64)
        slopes[key] = float(((t - t.mean()) * (y - y.mean())).sum() / ((t - t.mean()) ** 2).sum()) if len(windows) > 1 else 0.0
        ratios[key] = ratio(float(y[-1]), float(y[0]))
    return {"slope_per_min": slopes, "last_first_ratio": ratios}

@torch.inference_mode()
def analyze(path, models, text_embedding, window_s, sample_fps):
    windows, pending = [], []
    current = 0

    def flush():
        if not pending:
            return
        dino, clip = models.embed(pending)
        row = {"window": len(windows), "start_s": current * window_s, "end_s": (current + 1) * window_s, "frames": len(pending), **pixel_metrics(pending)}
        row["dino_mean"] = dino
        row["clip_mean"] = clip
        windows.append(row)

    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate)
        next_t = 0.0
        for index, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * stream.time_base) if frame.pts is not None else index / fps
            if t + 1e-9 < next_t:
                continue
            next_t += 1.0 / sample_fps
            wi = int(t // window_s)
            if wi != current:
                flush()
                pending = []
                current = wi
            a = frame.to_ndarray(format="rgb24")
            x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
            pending.append(resize_short(x, 256).squeeze(0))
    flush()
    if not windows:
        raise ValueError(f"no decodable frames: {path}")
    first_dino = windows[0]["dino_mean"]
    first_sharpness = windows[0]["sharpness_raw"]
    for row in windows:
        dino = row.pop("dino_mean")
        clip = row.pop("clip_mean")
        row["dino_cosine"] = float((dino * first_dino).sum())
        row["clip_text_similarity"] = float((clip * text_embedding).sum())
        row["sharpness"] = ratio(row.pop("sharpness_raw"), first_sharpness)
        if row["sharpness"] is None:
            row["sharpness"] = 0.0
        for key in ("start_s", "end_s", *CURVES):
            if isinstance(row[key], float):
                row[key] = round(row[key], 6)
    return {"video": path, "window_s": window_s, "sample_fps": sample_fps, "resize_short": 256, "windows": windows, "summary": summary(windows)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    if args.window_s <= 0:
        ap.error("--window-s must be positive")
    with open(args.prompt_file) as f:
        prompt = f.read().strip()
    models = Models(args.device)
    text_embedding = models.text(prompt).cpu()
    for path in args.videos:
        print(json.dumps(analyze(path, models, text_embedding, args.window_s, 4.0), allow_nan=False), flush=True)

if __name__ == "__main__":
    main()
