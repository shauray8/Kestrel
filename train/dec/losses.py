import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LAYERS = (2, 5, 8, 11)


def frames(x, idx=None):
    b, c, t, h, w = x.shape
    x = x.transpose(1, 2).reshape(b * t, c, h, w)
    return x if idx is None else x[idx]


class Dino(nn.Module):
    def __init__(self, res=224):
        super().__init__()
        import timm
        self.net = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=res).eval().requires_grad_(False)
        self.res = res
        self.register_buffer("m", MEAN)
        self.register_buffer("s", STD)

    def feats(self, x):
        x = F.interpolate((x.float() + 1) / 2, size=(self.res, self.res), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.m) / self.s).to(torch.bfloat16)
        return self.net.forward_intermediates(x, indices=list(LAYERS), output_fmt="NLC", intermediates_only=True)


class PDino(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino

    def forward(self, x, y):
        fx = self.dino.feats(x)[-1]
        with torch.no_grad():
            fy = self.dino.feats(y)[-1]
        return (1 - F.cosine_similarity(fx.float(), fy.float(), dim=-1)).mean()


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)

    def forward(self, x, y):
        return self.net(x.float(), y.float()).mean()


class Head(nn.Module):
    def __init__(self, c=384, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(c, h, 1), nn.LeakyReLU(0.2), nn.Conv1d(h, h, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv1d(h, 1, 1))

    def forward(self, f):
        return self.net(f.float().transpose(1, 2)).mean((1, 2))


class Disc(nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.dino = dino
        self.heads = nn.ModuleList([Head() for _ in LAYERS])

    def forward(self, x):
        return sum(h(f) for h, f in zip(self.heads, self.dino.feats(x)))


def d_hinge(real, fake):
    return F.relu(1 - real).mean() + F.relu(1 + fake).mean()


def g_hinge(fake):
    return -fake.mean()


def readout_loss(ro, y):
    tot = 0
    for r in ro:
        tgt = F.adaptive_avg_pool3d(y.float(), r.shape[2:])
        tot = tot + F.mse_loss(r.float(), tgt)
    return tot


def temporal_loss(p, y):
    return ((p[:, :, 1:] - p[:, :, :-1]).float() - (y[:, :, 1:] - y[:, :, :-1]).float()).abs().mean()
