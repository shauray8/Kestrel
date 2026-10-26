import torch
from sageattention import _fused
import sageattention.core as _core

_sm89 = _core.sm89_compile
E4 = torch.float8_e4m3fn
VMAX = 2.25
_GRAN_PER_WARP = 2

class KVQAttention:
    def __init__(self, v_headroom: float = 2.0):
        self.v_headroom = v_headroom
        self.epoch = 0
        self.state = {}
        self.stats = {"hist_quant": 0, "calls": 0}

    def new_epoch(self):
        self.epoch += 1

    def _hist(self, layer, k, v, Lh):
        st = self.state.get(layer)
        if st is not None and st["epoch"] == self.epoch and st["Lh"] == Lh and st["L"] == k.shape[1]:
            return st
        L = k.shape[1]
        Lpad = (L + 63) // 64 * 64
        La = (Lh // 64) * 64
        b, _, h, d = k.shape
        km = k[:, :Lh].mean(dim=1, keepdim=True, dtype=torch.float32).to(k.dtype)
        if st is not None and st["L"] == L:
            k_i8, k_sc, v_f8, v_scale = st["k_i8"], st["k_sc"], st["v_f8"], st["v_scale"]
        else:
            k_i8 = torch.empty((b, L, h, d), dtype=torch.int8, device=k.device)
            k_sc = torch.empty((b, h, Lpad // 64), dtype=torch.float32, device=k.device)
            v_f8 = torch.zeros((b, d, h, Lpad), dtype=E4, device=k.device)
            v_scale = torch.empty((b, h, d), dtype=torch.float32, device=k.device)
        if La > 0:
            sc = torch.empty((b, h, La // 64), dtype=torch.float32, device=k.device)
            _fused.quant_per_block_int8_fuse_sub_mean_cuda(k[:, :La], km.squeeze(1).contiguous(), k_i8[:, :La], sc, 64, 0)
            k_sc[:, :, : La // 64] = sc
            vt = torch.empty((b, d, h, La), dtype=v.dtype, device=v.device)
            _fused.transpose_pad_permute_cuda(v[:, :La], vt, 0)
            vf = torch.empty((b, d, h, La), dtype=E4, device=v.device)
            _fused.scale_fuse_quant_cuda(vt, vf, v_scale, La, VMAX / self.v_headroom, 0)
            v_f8[..., :La] = vf
        else:
            v_scale.copy_((v.float().abs().amax(dim=1) * self.v_headroom / VMAX).clamp(min=1e-8))
        st = {"epoch": self.epoch, "Lh": Lh, "L": L, "La": La, "km": km, "k_i8": k_i8, "k_sc": k_sc, "v_f8": v_f8, "v_scale": v_scale}
        self.state[layer] = st
        self.stats["hist_quant"] += 1
        return st

    @staticmethod
    def _qv(vt, v_scale):
        s = v_scale.permute(0, 2, 1)[..., None]
        return (vt.float() / s).clamp(-VMAX, VMAX).to(E4)

    def __call__(self, layer, q, k, v, sm_scale=None):
        self.stats["calls"] += 1
        q, k, v = q.to(torch.bfloat16).contiguous(), k.to(torch.bfloat16).contiguous(), v.to(torch.bfloat16).contiguous()
        L, Lq = k.shape[1], q.shape[1]
        Lh = L - Lq
        st = self._hist(layer, k, v, Lh)
        La = st["La"]
        q_i8, q_sc, k_tail_i8, k_tail_sc = _per_warp(q, k[:, La:], st["km"])
        st["k_i8"][:, La:] = k_tail_i8
        st["k_sc"][:, :, La // 64:] = k_tail_sc
        n = L - La
        npad = (n + 63) // 64 * 64
        vt = torch.empty((1, v.shape[3], v.shape[2], npad), dtype=v.dtype, device=v.device)
        _fused.transpose_pad_permute_cuda(v[:, La:], vt, 0)
        st["v_f8"][..., La:] = self._qv(vt, st["v_scale"])
        o = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        _sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
            q_i8, st["k_i8"], st["v_f8"], o, q_sc, st["k_sc"], st["v_scale"], 0, 0, _GRAN_PER_WARP,
            sm_scale if sm_scale is not None else q.shape[-1] ** -0.5, 0)
        return o

def _per_warp(q, k, km):
    b, lq, h, d = q.shape
    lk = k.shape[1]
    q_i8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_i8 = torch.empty((b, lk, h, d), dtype=torch.int8, device=q.device)
    q_sc = torch.empty((b, h, ((lq + 127) // 128) * (128 // 32)), device=q.device, dtype=torch.float32)
    k_sc = torch.empty((b, h, (lk + 63) // 64), device=q.device, dtype=torch.float32)
    _fused.quant_per_warp_int8_cuda(q, q_i8, q_sc, 128, 32, 0)
    _fused.quant_per_block_int8_fuse_sub_mean_cuda(k.contiguous(), km.squeeze(1), k_i8, k_sc, 64, 0)
    return q_i8, q_sc, k_i8, k_sc
