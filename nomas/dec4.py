import math
import os
import weakref
from functools import wraps

import torch
import triton as tr
import triton.language as tl
from torch import nn
from torch.nn import functional as F

from kestrel import ops
from nomas.gemm import N3Linear, _replace_linears

_SCALES = {}
_INSTALLED = False


@tr.jit
def _pack_kernel(X, U, Q, S, M: tl.constexpr, K: tl.constexpr,
                 INV_G, SWIGLU: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0) * B + tl.arange(0, B)
    lane = tl.arange(0, 16)
    offset = block[:, None] * 16 + lane[None, :]
    x = tl.load(X + offset, offset < M * K, 0).to(tl.float32)
    if SWIGLU:
        up = tl.load(U + offset, offset < M * K, 0).to(tl.float32)
        x = ((x * tl.sigmoid(x)).to(tl.bfloat16).to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
    scale = tl.minimum(tl.div_rn(tl.max(tl.abs(x), 1) * INV_G, 6.0), 448.0).to(tl.float8e4nv)
    sf = scale.to(tl.float32)
    y = x * tl.where(sf[:, None] > 0, tl.div_rn(INV_G, sf[:, None]), 0.0)
    a = tl.abs(y)
    code = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32)
            + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
            + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32)
            + (a > 5.0).to(tl.int32))
    code = code | tl.where(y < 0, 8, 0)
    pairs = tl.reshape(code << ((lane[None, :] % 2) * 4), (B, 8, 2))
    packed = tl.sum(pairs, 2).to(tl.uint8)
    qoffset = block[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Q + qoffset, packed, qoffset < M * K // 2)
    row, col = block // (K // 16), block % (K // 16)
    soffset = (((row // 128) * (K // 64) + col // 4) * 512
               + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4)
    tl.store(S + soffset, scale.to(tl.uint8, bitcast=True), row < M)


def _pack(x, scale, up=None):
    k, m = x.shape[-1], x.numel() // x.shape[-1]
    q = torch.empty((m, k // 2), device=x.device, dtype=torch.uint8)
    s = torch.zeros((tr.cdiv(m, 128) * 128, k // 16), device=x.device, dtype=torch.uint8)
    _pack_kernel[(tr.cdiv(m * (k // 16), 32),)](x, x if up is None else up, q, s,
                                             m, k, 1.0 / scale, up is not None, 32)
    return q, s


def _scale(x, buffer):
    key = (buffer.device, buffer.data_ptr())
    cached = _SCALES.get(key)
    if cached is not None and cached[0]() is not None:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up Nomas NVFP4 once before CUDA graph capture")
    value = float(buffer)
    if value == 0:
        if x is None:
            return None
        maximum = float(x.detach().abs().amax())
        if not math.isfinite(maximum):
            raise ValueError("Nonfinite Nomas NVFP4 calibration input")
        value = max(maximum / (6 * 448), 1e-12)
        buffer.fill_(value)
    _SCALES[key] = (weakref.ref(buffer), value)
    return value


def _mm(q, s, w, ws, alpha, bias=None):
    y = torch._scaled_mm(q.view(torch.float4_e2m1fn_x2),
                         w.view(torch.float4_e2m1fn_x2).t(),
                         scale_a=s.view(torch.float8_e4m3fn),
                         scale_b=ws.view(torch.float8_e4m3fn), out_dtype=torch.bfloat16)
    return y.mul(alpha) if bias is None else torch.add(bias, y, alpha=alpha)


@torch.library.custom_op("nomas_dec4::linear", mutates_args=("activation_scale",))
def _linear(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, weight_scale: float,
            bias: torch.Tensor | None, activation_scale: torch.Tensor) -> torch.Tensor:
    scale = _scale(x, activation_scale)
    q, s = _pack(x, scale)
    return _mm(q, s, w, ws, scale * weight_scale, bias).reshape(*x.shape[:-1], w.shape[0])


@_linear.register_fake
def _linear_fake(x, w, ws, weight_scale, bias, activation_scale):
    return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("nomas_dec4::mlp", mutates_args=("activation_scale", "hidden_scale"))
def _mlp(x: torch.Tensor, gw: torch.Tensor, gs: torch.Tensor, gg: float,
         uw: torch.Tensor, us: torch.Tensor, ug: float,
         dw: torch.Tensor, ds: torch.Tensor, dg: float,
         activation_scale: torch.Tensor, hidden_scale: torch.Tensor) -> torch.Tensor:
    scale = _scale(x, activation_scale)
    q, s = _pack(x, scale)
    gate = _mm(q, s, gw, gs, scale * gg)
    up = _mm(q, s, uw, us, scale * ug)
    mid_scale = _scale(None, hidden_scale)
    if mid_scale is None:
        hidden = F.silu(gate) * up
        mid_scale = _scale(hidden, hidden_scale)
        hq, hs = _pack(hidden, mid_scale)
    else:
        hq, hs = _pack(gate, mid_scale, up)
    return _mm(hq, hs, dw, ds, mid_scale * dg).reshape(*x.shape[:-1], dw.shape[0])


@_mlp.register_fake
def _mlp_fake(x, gw, gs, gg, uw, us, ug, dw, ds, dg, activation_scale, hidden_scale):
    return x.new_empty((*x.shape[:-1], dw.shape[0]), dtype=torch.bfloat16)


def _weight(linear):
    if hasattr(linear, "fp8_weight"):
        q, s = linear.fp8_weight, linear.fp8_scale
        n, k = q.shape
        return (q.float().reshape(n // 128, 128, k // 128, 128)
                * s[:, None, :, None]).reshape(n, k)
    if isinstance(linear, N3Linear):
        return linear.wq.float() * linear.sw[:, None]
    return linear.weight.detach().float()


class NVFP4Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        weight = _weight(linear)
        if not weight.is_cuda or weight.ndim != 2 or any(d % 128 for d in weight.shape):
            raise ValueError("Nomas NVFP4 requires CUDA weights with dimensions divisible by 128")
        if torch.cuda.get_device_capability(weight.device)[0] < 10:
            raise ValueError("Nomas NVFP4 requires Blackwell or newer")
        self.out_features, self.in_features = weight.shape
        maximum = float(weight.abs().amax())
        if not math.isfinite(maximum):
            raise ValueError("Nonfinite Nomas NVFP4 weight")
        self.weight_scale = max(maximum / (6 * 448), 1e-12)
        q, s = _pack(weight.contiguous(), self.weight_scale)
        self.register_buffer("wq", q)
        self.register_buffer("sw", s)
        bias = getattr(linear, "bias", None)
        self.register_buffer("bias", None if bias is None else bias.detach().bfloat16().contiguous())
        self.register_buffer("activation_scale", torch.zeros((), device=weight.device, dtype=torch.float32))
        self.train(False)

    @property
    def weight(self):
        return torch.empty(0, device=self.wq.device, dtype=torch.bfloat16)

    def _apply(self, fn, recurse=True):
        for name, tensor in self._buffers.items():
            if tensor is not None:
                target = fn(torch.empty(0, device=tensor.device, dtype=tensor.dtype))
                self._buffers[name] = tensor.to(device=target.device)
        return self

    def forward(self, x):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Nomas NVFP4 requires eval + no_grad inference")
        return _linear(x.bfloat16().contiguous(), self.wq, self.sw, self.weight_scale,
                       self.bias, self.activation_scale)


def mlp_forward(module, x):
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("Nomas NVFP4 requires eval + no_grad inference")
    gate, up, down = module.gate_proj, module.up_proj, module.down_proj
    return _mlp(x.bfloat16().contiguous(), gate.wq, gate.sw, gate.weight_scale,
                up.wq, up.sw, up.weight_scale, down.wq, down.sw, down.weight_scale,
                gate.activation_scale, down.activation_scale)


class PaddedN3Linear(N3Linear):
    def __init__(self, linear):
        weight = _weight(linear)
        n, k = weight.shape
        self.input_size, self.output_size = k, n
        with torch.device("meta"):
            source = nn.Linear(tr.cdiv(k, 64) * 64, tr.cdiv(n, 128) * 128, bias=False)
        source.weight = nn.Parameter(F.pad(weight.bfloat16(), (0, (-k) % 64, 0, (-n) % 128)), requires_grad=False)
        super().__init__(source)

    def forward(self, x):
        if self.in_features != self.input_size:
            x = F.pad(x, (0, self.in_features - self.input_size))
        return super().forward(x)[..., :self.output_size]


def _shape(linear):
    if hasattr(linear, "fp8_weight"):
        return linear.fp8_weight.shape
    if isinstance(linear, N3Linear):
        return linear.wq.shape
    return linear.weight.shape


def convert_decoder(module, attention=False):
    _patch_mlp()
    count, fallback = 0, 0
    for layer in module.layers:
        targets = [(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")]
        if attention:
            targets += [(layer.attn, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        for parent, name in targets:
            child = getattr(parent, name)
            if not isinstance(child, (NVFP4Linear, PaddedN3Linear)):
                if getattr(child, "bias", None) is not None:
                    raise ValueError("PixelUMM NVFP4 projections must be bias-free")
                if any(d % 128 for d in _shape(child)):
                    setattr(parent, name, PaddedN3Linear(child))
                    fallback += 1
                else:
                    setattr(parent, name, NVFP4Linear(child))
                    count += 1
    return count, fallback + _replace_linears(module)


def _patch_mlp():
    from hymm.models.vae.pixelumm_decoder.model import GenMLP

    if getattr(GenMLP.forward, "_nomas_dec4", False):
        return
    original = GenMLP.forward
    @wraps(original)
    def forward(self, x):
        if all(isinstance(getattr(self, name), NVFP4Linear)
               for name in ("gate_proj", "up_proj", "down_proj")):
            return mlp_forward(self, x)
        return original(self, x)
    forward._nomas_dec4 = True
    GenMLP.forward = forward


def install():
    global _INSTALLED
    enabled = os.environ.get("NOMAS_DEC_GEMM") == "nvfp4" or os.environ.get("NOMAS_DEC_FP4") == "1"
    if not enabled or _INSTALLED:
        return
    selector = os.environ.get("NOMAS_DEC_GEMM", "")
    if selector not in ("", "0", "nvfp4"):
        raise ValueError("NOMAS_DEC_FP4=1 conflicts with NOMAS_DEC_GEMM=" + selector)
    os.environ.setdefault("N3_CONT", "1")
    ops.n3()
    from hymm.models.vae.pixelumm_decoder import loader

    original = loader._install_inference_optimizations
    attention = os.environ.get("NOMAS_DEC_FP4_ATTN") == "1"
    class Config:
        def __init__(self, source):
            self.source = source
        def __getattr__(self, name):
            return False if name in ("cutlass_fp8", "deepgemm_fp8") else getattr(self.source, name)
    @wraps(original)
    def install_optimizations(module, config, logger):
        if next(module.parameters()).is_cuda:
            count, fp8 = convert_decoder(module, attention)
            logger.info(f"PixelUMM Kestrel NVFP4: {count} projections, {fp8} N3 FP8 projections; first-use calibration")
        else:
            def convert_on_cuda(decoder, inputs):
                if not inputs[0].is_cuda:
                    raise RuntimeError("Nomas NVFP4 decoder requires CUDA inputs")
                count, fp8 = convert_decoder(decoder, attention)
                handle.remove()
                logger.info(f"PixelUMM Kestrel NVFP4: {count} projections, {fp8} N3 FP8 projections; first-use calibration")
            handle = module.register_forward_pre_hook(convert_on_cuda)
            logger.info("PixelUMM Kestrel NVFP4: deferring weight conversion until first CUDA forward")
        return original(module, Config(config), logger)
    loader._install_inference_optimizations = install_optimizations
    _INSTALLED = True
