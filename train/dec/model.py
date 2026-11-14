import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CFG = {
    "S": ((512, 384, 256, 96, 48, 24), (3, 3, 2, 1, 1, 0)),
    "M": ((768, 512, 384, 128, 48, 32), (4, 4, 3, 1, 1, 0)),
    "L": ((1024, 768, 512, 160, 64, 32), (6, 5, 4, 2, 1, 0)),
}
READOUT_STAGES = (2, 3)


class RMS(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(1, keepdim=True) + 1e-6)
        return (y * (1 + self.g.view(1, -1, 1, 1, 1))).to(x.dtype)


class TConv(nn.Module):
    def __init__(self, ci, co, k=3, spatial=1, zero=False):
        super().__init__()
        self.k = k
        self.conv = nn.Conv3d(ci, co, (k, 2 * spatial + 1, 2 * spatial + 1), padding=(0, spatial, spatial))
        if zero:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, state=None, key=None):
        if state is None or key not in state:
            prev = x[:, :, :1].expand(-1, -1, self.k - 1, -1, -1)
        else:
            prev = state[key]
        x = torch.cat([prev, x], 2)
        if state is not None:
            state[key] = x[:, :, -(self.k - 1):].detach()
        return self.conv(x)


def sconv(ci, co):
    return nn.Conv3d(ci, co, (1, 3, 3), padding=(0, 1, 1))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.c1, self.n2, self.c2 = RMS(c), sconv(c, c), RMS(c), sconv(c, c)
        self.t = TConv(c, c, spatial=0, zero=True)

    def forward(self, x, state=None, key=None):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + self.t(h, state, key)


class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = sconv(ci, 4 * co)

    def forward(self, x):
        y = self.conv(x)
        b, c4, t, h, w = y.shape
        c = c4 // 4
        return y.view(b, c, 2, 2, t, h, w).permute(0, 1, 4, 5, 2, 6, 3).reshape(b, c, t, 2 * h, 2 * w)


class GaussHead(nn.Module):
    def __init__(self, c, rgb=True):
        super().__init__()
        self.rgb = rgb
        self.net = nn.Sequential(RMS(c), sconv(c, c), nn.SiLU(), nn.Conv3d(c, 12 if rgb else 9, 1))
        self.dscale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[0] = math.log(math.e - 1)
            self.net[-1].bias[1] = 2.0
            self.net[-1].bias[5] = 1.0

    def forward(self, x):
        p = self.net(x).float()
        g = {"depth": F.softplus(p[:, :1]) * self.dscale.exp(), "opacity": torch.sigmoid(p[:, 1:2]),
             "log_scale": p[:, 2:5], "quat": F.normalize(p[:, 5:9], dim=1)}
        if self.rgb:
            g["rgb"] = torch.tanh(p[:, 9:12])
        return g


class Decoder(nn.Module):
    def __init__(self, size="M", mean=None, std=None, grad_ckpt_from=99, gauss=False, gauss_rgb=True):
        super().__init__()
        widths, depths = CFG[size]
        self.size, self.widths, self.depths = size, widths, depths
        self.register_buffer("mean", (mean if mean is not None else torch.zeros(32)).view(1, 32, 1, 1, 1).float())
        self.register_buffer("std", (std if std is not None else torch.ones(32)).view(1, 32, 1, 1, 1).float())
        self.proj_in = TConv(32, 4 * widths[0])
        self.stages = nn.ModuleList([nn.ModuleList([Block(w) for _ in range(d)]) for w, d in zip(widths, depths)])
        self.ups = nn.ModuleList([Up(widths[i], widths[i + 1]) for i in range(len(widths) - 1)])
        self.readouts = nn.ModuleDict({str(i): nn.Sequential(RMS(widths[i]), nn.Conv3d(widths[i], 3, 1)) for i in READOUT_STAGES})
        self.out_norm = RMS(widths[-1])
        self.out = sconv(widths[-1], 3)
        self.gauss_stage = len(widths) - 2
        if gauss:
            self.gauss_head = GaussHead(widths[self.gauss_stage], gauss_rgb)
        self.grad_ckpt_from = grad_ckpt_from

    def forward(self, z, state=None, readouts=False, gauss=False):
        x = ((z.float() - self.mean) / self.std).to(z.dtype)
        x = self.proj_in(x, state, "in")
        b, c4, l, h, w = x.shape
        c = c4 // 4
        x = x.view(b, c, 4, l, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, c, 4 * l, h, w)
        ro = []
        for i, blocks in enumerate(self.stages):
            for j, blk in enumerate(blocks):
                if self.training and i >= self.grad_ckpt_from and state is None:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x, state, f"s{i}b{j}")
            if readouts and str(i) in self.readouts:
                ro.append(self.readouts[str(i)](x))
            if gauss and i == self.gauss_stage:
                g = self.gauss_head(x)
            if i < len(self.ups):
                x = self.ups[i](x)
        y = self.out(F.silu(self.out_norm(x)))
        outs = (y,) + ((ro,) if readouts else ()) + ((g,) if gauss else ())
        return outs if len(outs) > 1 else y

    @torch.no_grad()
    def decode_stream(self, z, chunk=10):
        state, outs = {}, []
        for i in range(0, z.shape[2], chunk):
            outs.append(self(z[:, :, i:i + chunk], state=state))
        return torch.cat(outs, 2)

    def param_groups(self):
        heads = [self.proj_in, self.out, *self.readouts.values()] + ([self.gauss_head] if hasattr(self, "gauss_head") else [])
        excluded = {id(p) for m in heads for p in m.parameters()}
        muon, adam = [], []
        for p in self.parameters():
            (muon if p.ndim >= 2 and id(p) not in excluded else adam).append(p)
        return muon, adam


@torch.no_grad()
def count_flops(model, z):
    tot = [0]

    def hook(m, i, o):
        tot[0] += 2 * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * o.numel() // m.groups

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv3d)]
    model(z, readouts=True)
    for h in hs:
        h.remove()
    return tot[0]
