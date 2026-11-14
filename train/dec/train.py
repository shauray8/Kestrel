import argparse
import copy
import json
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import pynvml
import torch
import torch.nn.functional as F

from data import Batcher, GeneratedBatcher, load_latents, load_poses, read_frames
from gauss import gaussian_losses
from losses import LPIPS, Disc, Dino, PDino, d_hinge, frames, g_hinge, readout_loss, temporal_loss
from metrics import Meter
from model import Decoder, count_flops
from qat import apply_qat, export_fp8, set_qat
from muon import Muon

H100_BF16 = 989e12


def lr_at(step, a):
    if step < a.warmup:
        return (step + 1) / a.warmup
    d0 = int(a.steps * (1 - a.decay))
    if step < d0:
        return 1.0
    p = (step - d0) / max(1, a.steps - d0)
    return a.lr_min + (1 - a.lr_min) * (1 - math.sqrt(p))


class EMA:
    def __init__(self, model, half_life, start):
        self.m = copy.deepcopy(model).float().eval().requires_grad_(False)
        self.decay = 0.5 ** (1 / half_life)
        self.start, self.pause, self.hist = start, 0, deque(maxlen=200)

    def spike(self, gnorm):
        self.hist.append(gnorm)
        med = sorted(self.hist)[len(self.hist) // 2]
        if len(self.hist) > 10 and gnorm > 10 * med:
            self.pause = 250

    @torch.no_grad()
    def update(self, model, step):
        if self.pause:
            self.pause -= 1
            return
        d = 0.0 if step < self.start else self.decay
        if not hasattr(self, "pe"):
            self.pe, self.pm = list(self.m.parameters()), list(model.parameters())
        torch._foreach_lerp_(self.pe, [p.float() for p in self.pm] if self.pm[0].dtype != torch.float32 else self.pm, 1 - d)


def gpu_stats(h):
    return {"temp": pynvml.nvmlDeviceGetTemperature(h, 0), "sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, 1),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000, "util": pynvml.nvmlDeviceGetUtilizationRates(h).gpu}


@torch.no_grad()
def evaluate(model, lat, paths, n):
    model.eval()
    m = Meter()
    for i in range(min(n, len(paths))):
        z = lat[i: i + 1].cuda().to(torch.bfloat16)
        with torch.autocast("cuda", torch.bfloat16):
            y = model.decode_stream(z)
        gt = read_frames(paths[i], 0, y.shape[2]).permute(1, 0, 2, 3)[None].to(torch.bfloat16).div(127.5).sub(1)
        m.add(y, gt)
    return m.mean()


def save(path, **kw):
    tmp = str(path) + ".tmp"
    torch.save(kw, tmp)
    os.replace(tmp, path)


def decode_batch(fwd, z, cf, gauss, readouts=True):
    out = fwd(z, readouts=readouts, gauss=gauss)
    if not readouts and not gauss:
        out = (out,)
    yh = out[0][:, :, cf:]
    ro = [r[:, :, cf:] for r in out[1]] if readouts else []
    gs = {k: v[:, :, cf:] for k, v in out[-1].items()} if gauss else None
    return yh, ro, gs


def anchor_loss(pred, target, sigma):
    pred, target = frames(pred).float(), frames(target).float()
    if sigma > 0:
        radius = math.ceil(3 * sigma)
        x = torch.arange(-radius, radius + 1, device=pred.device, dtype=torch.float32)
        kernel = (-0.5 * (x / sigma).square()).exp()
        kernel = kernel / kernel.sum()
        kx, ky = kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        def blur(v):
            v = F.conv2d(F.pad(v, (radius, radius, 0, 0), mode="replicate"), kx, groups=3)
            return F.conv2d(F.pad(v, (0, 0, radius, radius), mode="replicate"), ky, groups=3)
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = blur(pred), blur(target)
    return (pred - target).abs().mean()


def main(a):
    if not 0 <= a.gen_frac <= 1:
        raise ValueError("--gen_frac must be in [0, 1]")
    if a.gen_frac and (a.gen_anchor_sigma < 0 or a.w_gen_anchor < 0 or a.temporal_smooth_w < 0):
        raise ValueError("generated loss weights and blur sigma must be nonnegative")
    ngen = min(a.batch, max(1, round(a.batch * a.gen_frac))) if a.gen_frac else 0
    nreal = a.batch - ngen
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    lat, paths, st = load_latents(a.latents, a.max_clips)
    hlat, hpaths, _ = load_latents(a.holdout_latents)
    tr = list(range(len(paths)))
    print(json.dumps({"train_clips": len(paths), "train_T": lat.shape[2], "holdout_clips": len(hpaths)}), flush=True)

    model = Decoder(a.size, st["mean"], st["std"], a.grad_ckpt_from, gauss=a.gauss, gauss_rgb=bool(a.gauss_head_rgb)).cuda().to(memory_format=torch.channels_last_3d)
    lat_std = st["std"].cuda().view(1, 32, 1, 1, 1)
    if a.init:
        sd = torch.load(a.init, map_location="cuda", weights_only=False)
        init_sd = sd["ema"] if "ema" in sd else sd["model"]
        if a.gauss:
            init_sd = {**{k: v for k, v in model.state_dict().items() if k.startswith("gauss_head.")}, **init_sd}
        model.load_state_dict(init_sd)
    if a.qat:
        print(json.dumps({"qat_layers": apply_qat(model, a.qat_min_ch)}), flush=True)
    muon_p, adam_p = model.param_groups()
    if a.opt == "muon":
        opts = [Muon(muon_p, lr=a.lr, weight_decay=a.wd, ns_steps=a.ns_steps),
                torch.optim.AdamW(adam_p, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)]
    else:
        opts = [torch.optim.AdamW([{"params": muon_p, "weight_decay": a.wd}, {"params": adam_p, "weight_decay": 0.0}],
                                  lr=a.lr, betas=(0.9, 0.999), fused=True)]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    ema = EMA(model, a.ema_half_life, a.ema_start)

    lp = LPIPS().cuda()
    dino = Dino().cuda() if (a.w_pdino > 0 or a.gan_start < a.steps) else None
    pd = PDino(dino) if dino is not None and a.w_pdino > 0 else None
    disc = Disc(dino).cuda() if a.gan_start < a.steps else None
    dopt = torch.optim.AdamW(disc.heads.parameters(), lr=a.d_lr, betas=(0.0, 0.99), fused=True) if disc else None

    step = 0
    ck = out / "last.pt"
    if ck.exists():
        sd = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(sd["model"])
        ema.m.load_state_dict(sd["ema"])
        for o, s in zip(opts, sd["opts"]):
            o.load_state_dict(s)
        if disc and sd.get("disc"):
            disc.heads.load_state_dict(sd["disc"])
            dopt.load_state_dict(sd["dopt"])
        step = sd["step"]
        print(json.dumps({"resume": step}), flush=True)

    qat_live = bool(a.qat and step >= a.qat_start)
    if a.qat:
        set_qat(model, qat_live)
        set_qat(ema.m, qat_live)
        if qat_live:
            print(json.dumps({"qat_on": step}), flush=True)
    fwd = torch.compile(model) if a.compile else model
    labels = load_poses(a.labels) if a.gauss and a.labels else None
    batcher = Batcher(lat, paths, tr, nreal, a.crop, a.ctx, seed=a.seed + step, poses=labels) if nreal else None
    gen_batcher = None
    if ngen:
        gen_labels = load_poses(a.gen_labels or a.labels or "/workspace/data/gen_labels")
        gen_batcher = GeneratedBatcher(a.gen_root, gen_labels, ngen, a.crop, a.ctx, seed=a.seed + step + 1,
                                       anchor=a.gen_anchor == "pixelumm" and a.w_gen_anchor > 0)
        print(json.dumps({"real_batch": nreal, "gen_batch": ngen, "gen_windows": len(gen_batcher.windows)}), flush=True)
    with torch.autocast("cuda", torch.bfloat16):
        flops_fwd = sum(count_flops(model, source.next()[0]) for source in (batcher, gen_batcher) if source is not None)
    pynvml.nvmlInit()
    nh = pynvml.nvmlDeviceGetHandleByIndex(0)
    log = open(out / "log.jsonl", "a")
    wb = None
    if a.wandb:
        import wandb
        wb = wandb.init(project=a.wandb, name=out.name, id=out.name, resume="allow", config=vars(a), dir=str(out))
    cf = 4 * a.ctx
    t_last, n_last = time.perf_counter(), step
    model.train()
    while step < a.steps:
        batch = batcher.next() if batcher is not None else None
        z, y = batch[:2] if batch is not None else (None, None)
        poses = batch[2] if batch is not None and labels is not None else None
        gz, anchor, gposes = gen_batcher.next() if gen_batcher is not None else (None, None, None)
        if a.qat and not qat_live and step >= a.qat_start:
            set_qat(model, True)
            set_qat(ema.m, True)
            qat_live = True
            print(json.dumps({"qat_on": step}), flush=True)
        if a.latent_noise > 0 and z is not None:
            sig = torch.rand(z.shape[0], 1, 1, 1, 1, device="cuda") * a.latent_noise
            sig = sig * (torch.rand_like(sig) < a.latent_noise_p)
            z = (z.float() + sig * lat_std * torch.randn_like(z, dtype=torch.float32)).to(z.dtype)
        s = lr_at(step, a)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * s
        gan_on = disc is not None and step >= a.gan_start and nreal > 0
        with torch.autocast("cuda", torch.bfloat16):
            gen_gauss = a.gauss and a.temporal_smooth_w == 0
            if z is not None and gz is not None and z.shape[1:] == gz.shape[1:]:
                yh, ro, gs = decode_batch(fwd, torch.cat((z, gz)), cf, a.gauss)
                gyh, yh = yh[nreal:], yh[:nreal]
                ro = [r[:nreal] for r in ro]
                ggs = {k: v[nreal:] for k, v in gs.items()} if gs is not None else None
                gs = {k: v[:nreal] for k, v in gs.items()} if gs is not None else None
            else:
                if z is not None:
                    yh, ro, gs = decode_batch(fwd, z, cf, a.gauss)
                if gz is not None:
                    gyh, _, ggs = decode_batch(fwd, gz, cf, gen_gauss, readouts=False)
            real_parts, gen_parts = {}, {}
            if z is not None:
                real_parts["l1"] = (yh.float() - y.float()).abs().mean()
                b, _, t = y.shape[:3]
                sel = torch.randperm(b * t, device="cuda")[: a.perc_frames]
                fy, fh = frames(y, sel), frames(yh, sel)
                real_parts["lpips"] = a.w_lpips * lp(fh, fy)
                if pd is not None:
                    real_parts["pdino"] = a.w_pdino * pd(fh, fy)
                real_parts["readout"] = a.w_readout * readout_loss(ro, y)
                if a.gauss and poses is not None and any(w > 0 for w in (a.w_xview, a.w_selfrender, a.w_depth, a.w_xview_lpips)):
                    real_parts.update(gaussian_losses(gs, yh, poses, a.xview_offsets, a.xview_grad, a.xview_alpha, a.xview_occ,
                                                      a.w_xview, a.w_selfrender, a.w_depth, a.w_xview_lpips, lp,
                                                      gauss_impl=a.gauss_impl, xview_src=a.xview_src))
                if a.w_temporal > 0:
                    real_parts["temporal"] = a.w_temporal * temporal_loss(yh, y)
                if gan_on:
                    ramp = min(1.0, (step - a.gan_start + 1) / a.gan_ramp)
                    real_parts["gan"] = a.w_gan * ramp * g_hinge(disc(fh))
            if gz is not None:
                if a.temporal_smooth_w > 0:
                    gen_parts["temporal_smooth"] = a.temporal_smooth_w * (gyh[:, :, 1:].float() - gyh[:, :, :-1].float()).abs().mean()
                elif gen_gauss and any(w > 0 for w in (a.w_xview, a.w_selfrender, a.w_depth, a.w_xview_lpips)):
                    gen_parts.update(gaussian_losses(ggs, gyh, gposes, a.xview_offsets, a.xview_grad, a.xview_alpha, a.xview_occ,
                                                     a.w_xview, a.w_selfrender, a.w_depth, a.w_xview_lpips, lp,
                                                     gauss_impl=a.gauss_impl, xview_src=a.xview_src))
                if anchor is not None and a.w_gen_anchor > 0:
                    gen_parts["anchor"] = a.w_gen_anchor * anchor_loss(gyh, anchor, a.gen_anchor_sigma)
                if not gen_parts:
                    raise ValueError("generated samples need an active Gaussian, anchor, or temporal smoothness loss")
                parts = {**{"real_" + k: v * (nreal / a.batch) for k, v in real_parts.items()},
                         **{"gen_" + k: v * (ngen / a.batch) for k, v in gen_parts.items()}}
            else:
                parts = real_parts
            loss = sum(parts.values())
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        for o in opts:
            o.step()
        if gan_on and step % a.d_every == 0:
            with torch.autocast("cuda", torch.bfloat16):
                dl = d_hinge(disc(fy), disc(fh.detach()))
            dopt.zero_grad(set_to_none=True)
            dl.backward()
            dopt.step()
            parts["real_d" if ngen else "d"] = dl.detach()
        step += 1
        ema.update(model, step)
        if step % a.log_every == 0:
            gnv = gn.item()
            ema.spike(gnv)
            torch.cuda.synchronize()
            now = time.perf_counter()
            sps = (step - n_last) * a.batch / (now - t_last)
            t_last, n_last = now, step
            rec = {"step": step, "loss": loss.item(), **{k: v.item() for k, v in parts.items()}, "gnorm": gnv, "lr_scale": s,
                   "samples_s": sps, "gen_mfu": 3 * flops_fwd / a.batch * sps / H100_BF16, "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                   "ema_pause": ema.pause, **gpu_stats(nh)}
            if ngen:
                rec.update(real_loss=sum(v.detach().item() for k, v in parts.items() if k.startswith("real_") and k != "real_d"),
                           gen_loss=sum(v.detach().item() for k, v in parts.items() if k.startswith("gen_")),
                           real_batch=nreal, gen_batch=ngen)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if wb:
                wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=step)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(ema.m, hlat, hpaths, a.eval_n)
            rec = {"step": step, "eval_ema": ev}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if wb:
                wb.log({f"eval/{k}": v for k, v in ev.items()}, step=step)
            model.train()
        if step % a.ckpt_every == 0 or step == a.steps:
            save(ck, model=model.state_dict(), ema=ema.m.state_dict(), opts=[o.state_dict() for o in opts], step=step,
                 disc=disc.heads.state_dict() if disc else None, dopt=dopt.state_dict() if dopt else None, args=vars(a))
            if step % a.keep_every == 0 or step == a.steps:
                save(out / f"ema_{step:07d}.pt", ema=ema.m.state_dict(), step=step, args=vars(a))
                if qat_live:
                    save(out / f"fp8_{step:07d}.pt", fp8=export_fp8(ema.m), step=step, args=vars(a))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--latents", default="/workspace/data/latents")
    p.add_argument("--size", default="S", choices=["S", "M", "L"])
    p.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=8)
    p.add_argument("--ctx", type=int, default=2)
    p.add_argument("--latent_noise", type=float, default=0.0)
    p.add_argument("--latent_noise_p", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_min", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--decay", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ns_steps", type=int, default=5)
    p.add_argument("--w_lpips", type=float, default=0.1)
    p.add_argument("--w_pdino", type=float, default=0.01)
    p.add_argument("--w_readout", type=float, default=1.0)
    p.add_argument("--w_temporal", type=float, default=0.0)
    p.add_argument("--w_gan", type=float, default=0.1)
    p.add_argument("--gan_start", type=int, default=10 ** 9)
    p.add_argument("--gan_ramp", type=int, default=2000)
    p.add_argument("--d_lr", type=float, default=1e-4)
    p.add_argument("--d_every", type=int, default=1)
    p.add_argument("--perc_frames", type=int, default=32)
    p.add_argument("--ema_half_life", type=float, default=2000)
    p.add_argument("--ema_start", type=int, default=2000)
    p.add_argument("--grad_ckpt_from", type=int, default=99)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--holdout_latents", default="/workspace/data/latents_ho")
    p.add_argument("--max_clips", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_n", type=int, default=16)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=1000)
    p.add_argument("--keep_every", type=int, default=5000)
    p.add_argument("--init", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", default="")
    p.add_argument("--gauss", action="store_true")
    p.add_argument("--gauss_head_rgb", type=int, default=0)
    p.add_argument("--gauss_impl", choices=("batched", "loop"), default="batched", help="Gaussian rasterization implementation")
    p.add_argument("--xview_src", type=int, default=0, help="Stratified random source frames per window; 0 uses all frames")
    p.add_argument("--w_xview", type=float, default=0.0)
    p.add_argument("--w_selfrender", type=float, default=0.0)
    p.add_argument("--w_depth", type=float, default=0.0)
    p.add_argument("--xview_offsets", nargs="+", type=int, default=[4, 8])
    p.add_argument("--xview_grad", action="store_true")
    p.add_argument("--xview_alpha", type=float, default=0.5)
    p.add_argument("--xview_occ", type=float, default=0.1)
    p.add_argument("--w_xview_lpips", type=float, default=0.0)
    p.add_argument("--labels", default="")
    p.add_argument("--gen_root", default="/workspace/data/gen", help="Generated rollout directory or glob; .npy latents need matching .mp4 files for anchors")
    p.add_argument("--gen_labels", default="", help="Generated label index or directory; falls back to --labels, then /workspace/data/gen_labels")
    p.add_argument("--gen_frac", type=float, default=0.0, help="Generated batch fraction rounded to samples (at least one when positive); losses weighted by actual sample fractions")
    p.add_argument("--gen_anchor", choices=["pixelumm", "none"], default="pixelumm", help="Anchor 40-frame windows with the existing PixelUMM mp4; no DCAE-V decode")
    p.add_argument("--w_gen_anchor", type=float, default=1.0, help="Weight of full-resolution Gaussian-blurred RGB L1 on generated samples only")
    p.add_argument("--gen_anchor_sigma", type=float, default=4.0, help="Spatial Gaussian anchor blur sigma in output pixels")
    p.add_argument("--temporal_smooth_w", type=float, default=0.0, help="Replace generated Gaussian losses with consecutive-frame RGB L1; keep anchor and real losses")
    p.add_argument("--qat", type=int, default=0)
    p.add_argument("--qat_min_ch", type=int, default=256)
    p.add_argument("--qat_start", type=int, default=0)
    main(p.parse_args())
