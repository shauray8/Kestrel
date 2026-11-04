import os
import torch
from kestrel import ops

def _src(x):
    return x if x.dtype in (torch.float16, torch.bfloat16) and x.is_contiguous() else x.half().contiguous()


class FAttention:
    def __init__(self, mode=None, load_mode=None, exp_mode=None, rescale_threshold=None, q_tile=None, mx=None, bal=None, qk=None):
        self.mode = (mode if mode is not None else os.environ.get("FA_MODE", "fp8")).lower()
        self.load_mode = int(load_mode if load_mode is not None else os.environ.get("FA_LOAD_MODE", "0"))
        self.exp_mode = int(exp_mode if exp_mode is not None else os.environ.get("FA_EXP_MODE", "0"))
        self.rescale_threshold = float(rescale_threshold if rescale_threshold is not None else os.environ.get("FA_RESCALE_THRESHOLD", "0"))
        self.q_tile = int(q_tile if q_tile is not None else os.environ.get("FA_Q_TILE", "128"))
        self.mx = bool(int(mx if mx is not None else os.environ.get("FA_MX", "0")))
        self.bal = bool(int(bal if bal is not None else os.environ.get("FA_BAL", "0")))
        self.i8 = (qk if qk is not None else os.environ.get("FA_QK", "fp8")).lower() == "int8"
        if self.mode not in ("fp8", "fp16"):
            raise ValueError("FA_MODE must be fp8 or fp16")
        if self.load_mode not in (0, 1, 3):
            raise ValueError("FA_LOAD_MODE must be 0 (cp.async), 1 (TMA), or 3 (producer-warp TMA); mode 2 requires TMA multicast unsupported on RTX 50")
        if self.exp_mode not in (0, 1, 2):
            raise ValueError("FA_EXP_MODE must be 0 (exp2), 1 (polynomial quarter), or 2 (packed f16x2)")
        if not 0 <= self.rescale_threshold <= 0.5:
            raise ValueError("FA_RESCALE_THRESHOLD must be between 0 and 0.5 in log2 units")
        if self.q_tile not in (64, 128):
            raise ValueError("FA_Q_TILE must be 64 or 128")
        if self.mx and self.load_mode != 3:
            raise ValueError("FA_MX=1 requires FA_LOAD_MODE=3")
        if self.bal and (self.load_mode != 3 or self.q_tile != 128):
            raise ValueError("FA_BAL=1 requires FA_LOAD_MODE=3 and FA_Q_TILE=128")
        if self.i8 and not self.mx:
            raise ValueError("FA_QK=int8 requires FA_MX=1")
        self.epoch = 0
        self.state = {}
        self.stats = {"hist_quant": 0, "calls": 0}

    def new_epoch(self):
        self.epoch += 1

    def _hist(self, layer, q, k, v):
        Lq, L, h = q.shape[1], k.shape[1], k.shape[2]
        Lh = L - Lq
        signature = tuple((x.device, x.dtype, tuple(x.shape), tuple(x.stride())) for x in (q, k, v))
        st = self.state.get(layer)
        fresh = st is None or st["signature"] != signature or st["Lh"] != Lh
        if fresh:
            nq, nk = (Lq + 63) // 64, (L + 63) // 64
            st = {"signature": signature, "Lh": Lh, "La": Lh // 64 * 64, "epoch": -1}
            for name, shape, dtype in (("q8", (h, nq, 64, 128), torch.uint8),
                                       ("k8", (h, nk, 64, 128), torch.uint8),
                                       ("v8", (h, nk, 128, 64), torch.uint8),
                                       ("qs", (h, nq), torch.float32),
                                       ("ks", (h, nk), torch.float32),
                                       ("vs", (h, nk), torch.float32)):
                st[name] = torch.empty(shape, dtype=dtype, device=q.device)
            self.state[layer] = st
        rebuild = st["epoch"] != self.epoch
        ext = ops.fa()
        if fresh and self.load_mode in (1, 3):
            st["plan"] = ext.fa_plan(st["k8"], st["v8"])
        if rebuild and self.i8:
            st["kmean"] = torch.mean(k[0], 0, dtype=torch.float32)
        qk = (127, False, True, st.get("kmean")) if self.i8 else (8, False, False, None)
        for x, packed, scales, transpose, (target, pow2, int8, mean) in ((k, "k8", "ks", False, qk),
                                                                         (v, "v8", "vs", True, (448 if self.mx else 1, self.mx, False, None))):
            buf = _src(x[0])
            if rebuild and st["La"]:
                ext.fa_quant(buf, st[packed], st[scales], 0, st["La"] // 64, transpose, target, pow2, int8, mean)
            ext.fa_quant(buf, st[packed], st[scales], st["La"] // 64, (L + 63) // 64, transpose, target, pow2, int8, mean)
        if rebuild:
            st["epoch"] = self.epoch
            self.stats["hist_quant"] += 1
        return st

    def __call__(self, layer, q, k, v, sm_scale=None):
        self.stats["calls"] += 1
        if self.mode == "fp16":
            return ops.n2().n2_attention(q[0].half().contiguous(), k[0].half().contiguous(), v[0].half().contiguous())[None].to(q.dtype)
        st = self._hist(layer, q, k, v)
        ext = ops.fa()
        ext.fa_quant(_src(q[0]), st["q8"], st["qs"], 0, (q.shape[1] + 63) // 64, False, 127 if self.i8 else 8, False, self.i8)
        out = ext.fa_attention(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"], q.shape[1], k.shape[1],
                               q.shape[-1] ** -0.5 if sm_scale is None else sm_scale, self.load_mode, self.exp_mode, self.rescale_threshold,
                               st.get("plan"), self.q_tile, self.mx, self.bal, self.i8, q.dtype == torch.bfloat16)
        return out[None] if out.dtype == q.dtype else out[None].to(q.dtype)
