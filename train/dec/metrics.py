import lpips
import torch
import torch.nn.functional as F

_LP = None


def _lpips():
    global _LP
    if _LP is None:
        _LP = lpips.LPIPS(net="alex", verbose=False).cuda().eval().requires_grad_(False)
    return _LP


def _gauss(c, s=1.5, k=11):
    g = torch.exp(-((torch.arange(k, device="cuda") - k // 2) ** 2) / (2 * s * s))
    g = (g / g.sum()).float()
    return (g[:, None] * g[None]).expand(c, 1, k, k).contiguous()


def ssim(x, y):
    x, y = (x + 1) / 2, (y + 1) / 2
    w, c1, c2 = _gauss(3), 0.01 ** 2, 0.03 ** 2
    mx, my = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
    sx = F.conv2d(x * x, w, groups=3) - mx * mx
    sy = F.conv2d(y * y, w, groups=3) - my * my
    sxy = F.conv2d(x * y, w, groups=3) - mx * my
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sx + sy + c2))
    return m.mean((1, 2, 3))


@torch.no_grad()
def frame_metrics(pred, gt, chunk=16):
    p = pred[0].float().clamp(-1, 1).transpose(0, 1)
    g = gt[0].float().transpose(0, 1)
    mse = ((p - g) ** 2).mean((1, 2, 3))
    psnr = 10 * torch.log10(4 / mse.clamp_min(1e-10))
    ss = torch.cat([ssim(p[i:i + chunk], g[i:i + chunk]) for i in range(0, len(p), chunk)])
    lp = torch.cat([_lpips()(p[i:i + chunk], g[i:i + chunk]).flatten() for i in range(0, len(p), chunk)])
    flick = ((p[1:] - p[:-1]) - (g[1:] - g[:-1])).abs().mean((1, 2, 3))
    return {"psnr": psnr.mean().item(), "ssim": ss.mean().item(), "lpips": lp.mean().item(), "flicker": flick.mean().item()}


class Meter:
    def __init__(self):
        self.rows = []

    def add(self, pred, gt):
        r = frame_metrics(pred, gt)
        self.rows.append(r)
        return r

    def mean(self):
        return {k: sum(r[k] for r in self.rows) / len(self.rows) for k in self.rows[0]} if self.rows else {}


def _hp(x):
    w = _gauss(3, s=1.5, k=7)
    return x - F.conv2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), w, groups=3)


@torch.no_grad()
def grain(pred, ref, chunk=16, patch=16, flat_q=0.3):
    p = pred[0].float().clamp(-1, 1).transpose(0, 1)
    r = ref[0].float().clamp(-1, 1).transpose(0, 1)
    acc = {k: 0.0 for k in ("hp", "hr", "tp", "tr", "thp", "thr", "fp", "fr")}
    for i in range(0, len(p), chunk):
        a, b = p[i:i + chunk + 1], r[i:i + chunk + 1]
        ha, hb = _hp(a[:chunk]), _hp(b[:chunk])
        acc["hp"] += ha.pow(2).sum().item()
        acc["hr"] += hb.pow(2).sum().item()
        if len(a) > 1:
            da, db = a[1:] - a[:-1], b[1:] - b[:-1]
            acc["tp"] += da.pow(2).sum().item()
            acc["tr"] += db.pow(2).sum().item()
            acc["thp"] += _hp(da).pow(2).sum().item()
            acc["thr"] += _hp(db).pow(2).sum().item()
        ea = F.avg_pool2d(ha.pow(2).mean(1, keepdim=True), patch).flatten(1)
        eb = F.avg_pool2d(hb.pow(2).mean(1, keepdim=True), patch).flatten(1)
        m = eb <= torch.quantile(eb, flat_q, dim=1, keepdim=True)
        acc["fp"] += (ea * m).sum().item()
        acc["fr"] += (eb * m).sum().item()
    return {"hf_ratio": acc["hp"] / acc["hr"], "temporal_ratio": acc["tp"] / acc["tr"],
            "temporal_hf_ratio": acc["thp"] / acc["thr"], "flat_hf_ratio": acc["fp"] / acc["fr"]}
