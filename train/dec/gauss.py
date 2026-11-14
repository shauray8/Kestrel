import torch
import torch.nn.functional as F


def camera_quat(r):
    m00, m01, m02 = r[..., 0, :].unbind(-1)
    m10, m11, m12 = r[..., 1, :].unbind(-1)
    m20, m21, m22 = r[..., 2, :].unbind(-1)
    s = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                     1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1).clamp_min(0).sqrt()
    q = torch.stack((torch.stack((s[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
                     torch.stack((m21 - m12, s[..., 1].square(), m10 + m01, m02 + m20), -1),
                     torch.stack((m02 - m20, m10 + m01, s[..., 2].square(), m12 + m21), -1),
                     torch.stack((m10 - m01, m20 + m02, m21 + m12, s[..., 3].square()), -1)), -2)
    q = q / (2 * s[..., None].clamp_min(1e-6))
    index = s.argmax(-1)[..., None, None].expand(*s.shape[:-1], 1, 4)
    return F.normalize(q.gather(-2, index).squeeze(-2), dim=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), -1)


def camera_rays(K, h, w):
    yy, xx = torch.meshgrid(torch.arange(h, device=K.device, dtype=torch.float32) + 0.5,
                            torch.arange(w, device=K.device, dtype=torch.float32) + 0.5, indexing="ij")
    pixels = torch.stack((xx, yy, torch.ones_like(xx)), -1).reshape(-1, 3)
    return F.normalize(pixels @ torch.linalg.inv(K).transpose(-1, -2), dim=-1)


def unproject(g, c2w, K, rgb, rays=None):
    h, w = g["depth"].shape[-2:]
    rays = camera_rays(K, h, w) if rays is None else rays
    depth = g["depth"].reshape(-1)
    points = rays * depth[:, None]
    means = points @ c2w[:3, :3].T + c2w[:3, 3]
    quats = quat_mul(camera_quat(c2w[:3, :3]), g["quat"].permute(1, 2, 0).reshape(-1, 4))
    scales = g["log_scale"].permute(1, 2, 0).reshape(-1, 3).exp()
    colors = g.get("rgb", rgb).permute(1, 2, 0).reshape(-1, 3).add(1).mul(0.5)
    return (means, quats, scales, g["opacity"].reshape(-1), colors), points[:, 2].reshape(1, h, w)


def render(gaussians, viewmats, Ks, width, height):
    from gsplat import rasterization
    means, quats, scales, opacities, colors = gaussians
    image, alpha, _ = rasterization(means=means.contiguous(), quats=quats.contiguous(), scales=scales.contiguous(),
                                     opacities=opacities.contiguous(), colors=colors.contiguous(),
                                     viewmats=viewmats.contiguous(), Ks=Ks.contiguous(), width=width, height=height,
                                     render_mode="RGB+ED")
    return image[..., :3].movedim(-1, -3).mul(2).sub(1), alpha.movedim(-1, -3), image[..., 3:].movedim(-1, -3)


def masked_l1(x, y, mask):
    return ((x - y).abs() * mask).sum() / (mask.sum() * x.shape[-3]).clamp_min(1)


def depth_loss(pred, target):
    mask = torch.isfinite(target) & (target > 0) & torch.isfinite(pred) & (pred > 0)
    n = mask.sum((-2, -1), keepdim=True).clamp_min(1)
    p, t = pred.clamp_min(1e-6).log(), torch.where(mask, target, torch.ones_like(target)).log()
    p, t = torch.where(mask, p, 0), torch.where(mask, t, 0)
    p, t = (p - p.sum((-2, -1), keepdim=True) / n) * mask, (t - t.sum((-2, -1), keepdim=True) / n) * mask
    p = p / (p.square().sum((-2, -1), keepdim=True) / n).clamp_min(1e-6).sqrt()
    t = t / (t.square().sum((-2, -1), keepdim=True) / n).clamp_min(1e-6).sqrt()
    return ((p - t).abs().sum((-2, -1), keepdim=True) / n).mean()


def source_frames(t, count):
    if count <= 0 or count >= t:
        return torch.arange(t)
    edges = torch.arange(count + 1) * t // count
    return edges[:-1] + (torch.rand(count) * (edges[1:] - edges[:-1])).long()


def batched_losses(g, rgb, target, c2w, K, rays, depths, sources, offsets, alpha, occ,
                   w_xview, w_selfrender, w_lpips, lpips):
    t, _, h, w = rgb.shape
    ds = {sign * d for d in offsets for sign in (-1, 1)} if w_xview > 0 or w_lpips > 0 else set()
    if w_selfrender > 0:
        ds.add(0)
    ds = torch.tensor(sorted(d for d in ds if abs(d) < t), dtype=torch.long)
    if not ds.numel():
        return [], [], []
    js = sources[:, None] + ds
    valid = (js >= 0) & (js < t)
    active = valid.any(-1)
    sources, js, valid = sources[active], js[active], valid[active]
    if not sources.numel():
        return [], [], []
    si, ci = valid.nonzero(as_tuple=True)
    same = sources[si] == js[si, ci]
    self_ids = same.nonzero(as_tuple=True)[0].to(rgb.device)
    cross_ids = (~same).nonzero(as_tuple=True)[0].to(rgb.device)
    sources, js, si, ci = (x.to(rgb.device) for x in (sources, js.clamp(0, t - 1), si, ci))
    n = sources.numel()
    gs = {k: v[:, sources].transpose(0, 1).float() for k, v in g.items()}
    points = rays[sources] * gs["depth"].reshape(n, -1, 1)
    rotations = c2w[sources, :3, :3]
    cloud = (points @ rotations.transpose(-1, -2) + c2w[sources, None, :3, 3],
             quat_mul(camera_quat(rotations)[:, None], gs["quat"].movedim(1, -1).reshape(n, -1, 4)),
             gs["log_scale"].movedim(1, -1).reshape(n, -1, 3).exp(),
             gs["opacity"].reshape(n, -1),
             gs.get("rgb", rgb[sources]).movedim(1, -1).reshape(n, -1, 3).add(1).mul(0.5))
    rendered, coverage, z = render(cloud, torch.linalg.inv(c2w)[js], K[js], w, h)
    pred, coverage, z = rendered[si, ci], coverage[si, ci], z[si, ci]
    targets = js[si, ci]
    tgt, ref = target[targets], depths[targets].detach()
    mask = (coverage.detach() > alpha) & torch.isfinite(z) & (z > 0)
    visible = torch.isfinite(ref) & (ref > 0) & ((z.detach() - ref).abs() <= occ * ref)
    mask = mask & (same.to(rgb.device)[:, None, None, None] | visible)
    losses = ((pred - tgt).abs() * mask).sum((-3, -2, -1)) / (mask.sum((-3, -2, -1)) * pred.shape[-3]).clamp_min(1)
    xloss = [losses[cross_ids]] if cross_ids.numel() else []
    sloss = [losses[self_ids]] if self_ids.numel() else []
    ploss = [lpips(torch.where(mask[cross_ids], pred[cross_ids], tgt[cross_ids]), tgt[cross_ids]).expand(cross_ids.numel())] if w_lpips > 0 and cross_ids.numel() else []
    return xloss, sloss, ploss


def gaussian_losses(g, rgb, poses, offsets=(4, 8), grad=False, alpha=0.5, occ=0.1,
                    w_xview=0.0, w_selfrender=0.0, w_depth=0.0, w_lpips=0.0, lpips=None,
                    gauss_impl="batched", xview_src=0):
    if gauss_impl not in ("loop", "batched"):
        raise ValueError(f"Unknown Gaussian implementation: {gauss_impl}")
    if xview_src < 0:
        raise ValueError("xview_src must be nonnegative")
    if not any(p is not None for p in poses):
        return {}
    with torch.autocast(device_type=rgb.device.type, enabled=False):
        h, w = g["depth"].shape[-2:]
        b, _, t = rgb.shape[:3]
        rgb = F.interpolate(rgb.float().transpose(1, 2).reshape(b * t, 3, *rgb.shape[-2:]),
                            size=(h, w), mode="area").reshape(b, t, 3, h, w)
        target = rgb if grad else rgb.detach()
        xloss, sloss, dloss, ploss = [], [], [], []
        for bi, pose in enumerate(poses):
            if pose is None:
                continue
            K = pose["K"].float().clone()
            K[:, :2] *= 0.5
            c2w = pose["c2w"].float()
            rays = camera_rays(K, h, w)
            depths = g["depth"][bi].float().transpose(0, 1) * rays[..., 2].reshape(t, 1, h, w)
            if w_depth > 0 and "depth" in pose:
                dloss.append(depth_loss(depths, pose["depth"].float()))
            if w_xview <= 0 and w_selfrender <= 0 and w_lpips <= 0:
                continue
            sources = source_frames(t, xview_src)
            if gauss_impl == "batched":
                xl, sl, pl = batched_losses({k: v[bi] for k, v in g.items()}, rgb[bi], target[bi],
                                            c2w, K, rays, depths, sources, offsets, alpha, occ,
                                            w_xview, w_selfrender, w_lpips, lpips)
                xloss.extend(xl)
                sloss.extend(sl)
                ploss.extend(pl)
                continue
            viewmats = torch.linalg.inv(c2w)
            for i in sources.tolist():
                js = sorted({i + sign * d for d in offsets for sign in (-1, 1)
                             if 0 <= i + sign * d < t}) if w_xview > 0 or w_lpips > 0 else []
                if w_selfrender > 0:
                    js = sorted(set(js + [i]))
                if not js:
                    continue
                cloud, _ = unproject({k: v[bi, :, i].float() for k, v in g.items()}, c2w[i], K[i], rgb[bi, i], rays[i])
                rendered, coverage, z = render(cloud, viewmats[js], K[js], w, h)
                for k, j in enumerate(js):
                    mask = (coverage[k:k + 1].detach() > alpha) & torch.isfinite(z[k:k + 1]) & (z[k:k + 1] > 0)
                    if i != j:
                        ref = depths[j:j + 1].detach()
                        mask = mask & torch.isfinite(ref) & (ref > 0) & ((z[k:k + 1].detach() - ref).abs() <= occ * ref)
                    pred, tgt = rendered[k:k + 1], target[bi, j:j + 1]
                    if i == j:
                        sloss.append(masked_l1(pred, tgt, mask))
                    else:
                        xloss.append(masked_l1(pred, tgt, mask))
                        if w_lpips > 0:
                            ploss.append(lpips(torch.where(mask, pred, tgt), tgt))
        parts = {}
        for name, values, weight in (("xview", xloss, w_xview), ("selfrender", sloss, w_selfrender),
                                      ("depth", dloss, w_depth), ("xview_lpips", ploss, w_lpips)):
            if values and weight > 0:
                values = torch.stack(values) if gauss_impl == "loop" or name == "depth" else torch.cat(values)
                parts[name] = weight * values.mean()
        return parts
