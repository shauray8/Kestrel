import torch
import torch.nn as nn
import torch.nn.functional as F

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
BLK = 128


def _q(xb, dim):
    s = xb.abs().amax(dim, keepdim=True) / FP8_MAX
    s = s.masked_fill(s == 0, 1.0)
    q = (xb / s).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return q, s


def _wblocks(w):
    co = w.shape[0]
    k = w.permute(0, 2, 3, 4, 1).reshape(co, -1).float()
    pad = -k.shape[1] % BLK
    if pad:
        k = F.pad(k, (0, pad))
    return k.view(co, -1, BLK)


@torch.no_grad()
def fq_weight(w):
    co, ci, *ks = w.shape
    q, s = _q(_wblocks(w), 2)
    d = (q.to(torch.bfloat16).float() * s).view(co, -1)[:, : ci * ks[0] * ks[1] * ks[2]]
    return d.view(co, *ks, ci).permute(0, 4, 1, 2, 3).to(w.dtype)


@torch.no_grad()
def fq_act(x):
    c = x.shape[1]
    xf = x.float()
    pad = -c % BLK
    if pad:
        xf = F.pad(xf, (0, 0, 0, 0, 0, 0, 0, pad))
    xb = xf.unflatten(1, (-1, BLK))
    q, s = _q(xb, 2)
    return (q.to(torch.bfloat16).float() * s).flatten(1, 2)[:, :c].to(x.dtype)


def ste(x, fq):
    return x + (fq - x).detach()


class QConv3d(nn.Conv3d):
    def forward(self, x):
        xq = torch.where(self.qat_on, fq_act(x), x)
        wq = torch.where(self.qat_on, fq_weight(self.weight), self.weight)
        return self._conv_forward(ste(x, xq), ste(self.weight, wq), self.bias)


def apply_qat(model, min_ch=256):
    n = 0
    for m in model.modules():
        if type(m) is nn.Conv3d and m.in_channels >= min_ch and m.out_channels >= min_ch:
            m.register_buffer("qat_on", torch.zeros((), dtype=torch.bool, device=m.weight.device), persistent=False)
            m.__class__ = QConv3d
            n += 1
    return n


def set_qat(model, on):
    for m in model.modules():
        if isinstance(m, QConv3d):
            m.qat_on.fill_(on)


@torch.no_grad()
def export_fp8(model):
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, QConv3d):
            q, s = _q(_wblocks(m.weight), 2)
            out[name] = {"w": q.flatten(1), "scale": s.squeeze(2), "shape": tuple(m.weight.shape), "layout": "co,(kt,kh,kw,ci)",
                         "block": BLK, "bias": None if m.bias is None else m.bias.detach().float().clone()}
    return out
