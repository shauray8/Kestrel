import ast
import inspect
import math
import os
import sys
import textwrap
import weakref
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from kestrel import ops

_TARGET = 16.0
_SCALES = {}
_INSTALLED = set()


def _scale(x, activation_scale):
    key = (activation_scale.device, activation_scale.data_ptr())
    cached = _SCALES.get(key)
    if cached is not None and cached[0]() is not None:
        return cached[1]
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up Nomas N3 once before CUDA graph capture")
    value = float(activation_scale)
    if value == 0:
        if x is None:
            return None
        maximum = float(x.detach().abs().amax())
        if not math.isfinite(maximum):
            raise ValueError("Nonfinite Nomas N3 calibration input")
        value = max(maximum, 1.0) / _TARGET
        activation_scale.fill_(value)
    _SCALES[key] = (weakref.ref(activation_scale), value)
    return value


@torch.library.custom_op("nomas_n3::linear", mutates_args=("activation_scale",))
def _linear(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
            bias: torch.Tensor | None, activation_scale: torch.Tensor) -> torch.Tensor:
    return ops.n3().n3_linear(x, w, sw, bias, _scale(x, activation_scale))


@_linear.register_fake
def _linear_fake(x, w, sw, bias, activation_scale):
    return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("nomas_n3::mlp", mutates_args=("activation_scale1", "activation_scale2"))
def _mlp(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor, b1: torch.Tensor | None,
         w2: torch.Tensor, s2: torch.Tensor, b2: torch.Tensor | None,
         activation_scale1: torch.Tensor, activation_scale2: torch.Tensor) -> torch.Tensor:
    ext = ops.n3()
    sa = _scale(x, activation_scale1)
    mid = _scale(None, activation_scale2)
    if mid is None:
        hidden = F.gelu(ext.n3_linear(x, w1, s1, b1, sa), approximate="tanh")
        return ext.n3_linear(hidden, w2, s2, b2, _scale(hidden, activation_scale2))
    hidden = ext.n3_linear_gelu_f8(x, w1, s1, b1, sa, mid)
    return ext.n3_linear_f8in(hidden, w2, s2, b2, mid)


@_mlp.register_fake
def _mlp_fake(x, w1, s1, b1, w2, s2, b2, activation_scale1, activation_scale2):
    return x.new_empty((*x.shape[:-1], w2.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("nomas_n3::residual", mutates_args=("activation_scale",))
def _residual(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor,
              bias: torch.Tensor | None, activation_scale: torch.Tensor,
              residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    result = residual.clone(memory_format=torch.contiguous_format)
    return ops.n3().n3_linear_residual(x, w, sw, bias, _scale(x, activation_scale), result, gate)


@_residual.register_fake
def _residual_fake(x, w, sw, bias, activation_scale, residual, gate):
    return torch.empty_like(residual, memory_format=torch.contiguous_format)


@torch.library.custom_op("nomas_n3::mlp_residual", mutates_args=("activation_scale1", "activation_scale2"))
def _mlp_residual(x: torch.Tensor, w1: torch.Tensor, s1: torch.Tensor,
                  b1: torch.Tensor | None, w2: torch.Tensor, s2: torch.Tensor,
                  b2: torch.Tensor | None, activation_scale1: torch.Tensor,
                  activation_scale2: torch.Tensor, residual: torch.Tensor,
                  gate: torch.Tensor) -> torch.Tensor:
    ext = ops.n3()
    sa = _scale(x, activation_scale1)
    mid = _scale(None, activation_scale2)
    result = residual.clone(memory_format=torch.contiguous_format)
    if mid is None:
        hidden = F.gelu(ext.n3_linear(x, w1, s1, b1, sa), approximate="tanh")
        return ext.n3_linear_residual(hidden, w2, s2, b2, _scale(hidden, activation_scale2), result, gate)
    hidden = ext.n3_linear_gelu_f8(x, w1, s1, b1, sa, mid)
    return ext.n3_linear_f8in_residual(hidden, w2, s2, b2, mid, result, gate)


@_mlp_residual.register_fake
def _mlp_residual_fake(x, w1, s1, b1, w2, s2, b2, activation_scale1, activation_scale2, residual, gate):
    return torch.empty_like(residual, memory_format=torch.contiguous_format)


def _quantized_weight(linear):
    if hasattr(linear, "weight_q"):
        source = linear.weight_q
        n, k = source.shape
        def rows(first, last):
            scale = linear.weight_scale[first:last].repeat_interleave(128, dim=1)[:, :k]
            return (source[first:last].float() * scale).bfloat16().float()
    elif hasattr(linear, "fp8_weight"):
        source = linear.fp8_weight
        n, k = source.shape
        def rows(first, last):
            scale = linear.fp8_scale[first // 128:(last + 127) // 128]
            return (source[first:last].float().reshape(-1, 128, k // 128, 128)
                    * scale[:, None, :, None]).reshape(last - first, k)
    else:
        source = linear.weight.detach()
        n, k = source.shape
        def rows(first, last):
            return source[first:last].float()
    weight = torch.empty((n, k), device=source.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(n, device=source.device, dtype=torch.float32)
    for first in range(0, n, 128):
        last = min(first + 128, n)
        values = rows(first, last)
        scale = (values.abs().amax(1) / _TARGET).clamp_min(1e-12)
        weight[first:last].copy_((values / scale[:, None]).to(torch.float8_e4m3fn))
        scales[first:last].copy_(scale)
    return weight, scales


class N3Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        weight, scale = _quantized_weight(linear)
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.register_buffer("wq", weight)
        self.register_buffer("sw", scale)
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
            raise RuntimeError("Nomas N3 requires eval + no_grad inference")
        return _linear(x.bfloat16().contiguous(), self.wq, self.sw, self.bias, self.activation_scale)

    def forward_residual(self, x, residual, gate):
        return _residual(x.bfloat16().contiguous(), self.wq, self.sw,
                         self.bias, self.activation_scale, residual, gate.contiguous())


def _mlp_eligible(module):
    return (isinstance(module.fc1, N3Linear) and isinstance(module.fc2, N3Linear)
            and isinstance(module.act, nn.GELU) and module.act.approximate == "tanh"
            and isinstance(module.norm, nn.Identity) and module.drop1.p == 0
            and module.drop2.p == 0 and not module.training)


def _mlp_args(module, x):
    a, b = module.fc1, module.fc2
    return (x.bfloat16().contiguous(), a.wq, a.sw, a.bias,
            b.wq, b.sw, b.bias, a.activation_scale, b.activation_scale)


def _project_residual(module, x, residual, gate):
    shaped_gate = gate.unsqueeze(1) if gate.ndim == 2 else gate
    if residual.dtype == torch.float32 and gate.dtype == torch.float32:
        if isinstance(module, N3Linear):
            return module.forward_residual(x, residual, shaped_gate)
        if hasattr(module, "fc1") and _mlp_eligible(module):
            return _mlp_residual(*_mlp_args(module, x), residual, shaped_gate.contiguous())
    return residual + module(x) * shaped_gate


def _replace_linears(model):
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, N3Linear):
                continue
            weight = getattr(child, "weight_q", None)
            if weight is None:
                weight = getattr(child, "fp8_weight", None)
            if weight is None and isinstance(child, nn.Linear):
                weight = child.weight
            if weight is not None and weight.ndim == 2 and weight.shape[0] % 128 == 0 and weight.shape[1] % 64 == 0:
                setattr(parent, name, N3Linear(child))
                count += 1
    return count


class _CacheRewrite(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "compile_cache_prepost"):
            return ast.copy_location(ast.Constant(False), node)
        return node

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.op, ast.Add):
            return node
        call = node.right
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "apply_gate" and len(call.args) == 1):
            return node
        projected = call.args[0]
        gate = next((kw.value for kw in call.keywords if kw.arg == "gate"), None)
        if (gate is None or not isinstance(projected, ast.Call)
                or not isinstance(projected.func, ast.Attribute)
                or not isinstance(projected.func.value, ast.Name)
                or projected.func.value.id != "self" or len(projected.args) != 1):
            return node
        return ast.copy_location(ast.Call(ast.Name("_nomas_project_residual", ast.Load()),
                                         [projected.func, projected.args[0], node.left, gate], []), node)


def _patch_dit():
    from hymm.models.dit import nomas_cpt
    from hymm.models.dit.modules import int8_stored_bf16_linear as storage
    from hymm.models.dit.modules.mlp_layers import MLP

    original_mlp = MLP.forward
    @wraps(original_mlp)
    def forward_mlp(self, x):
        if _mlp_eligible(self):
            return _mlp(*_mlp_args(self, x))
        return original_mlp(self, x)
    MLP.forward = forward_mlp

    original_cache = nomas_cpt.MMDoubleStreamBlock.forward_with_cache
    tree = _CacheRewrite().visit(ast.parse(textwrap.dedent(inspect.getsource(original_cache))))
    ast.fix_missing_locations(tree)
    namespace = dict(original_cache.__globals__)
    namespace["_nomas_project_residual"] = _project_residual
    exec(compile(tree, inspect.getsourcefile(original_cache), "exec"), namespace)
    nomas_cpt.MMDoubleStreamBlock.forward_with_cache = namespace[original_cache.__name__]

    original_load = storage.load_w8_bf16_model
    @wraps(original_load)
    def load(*args, **kwargs):
        model = original_load(*args, **kwargs)
        _replace_linears(model)
        return model
    for name, module in tuple(sys.modules.items()):
        if name.startswith("hymm.") and module is not None:
            for key, value in tuple(vars(module).items()):
                if value is original_load:
                    setattr(module, key, load)


def _cached(module):
    if not isinstance(module, N3Linear):
        return None
    entry = _SCALES.get((module.activation_scale.device, module.activation_scale.data_ptr()))
    return entry[1] if entry is not None and entry[0]() is not None else None


def dec_qkv(attention, x):
    mods = attention.q_proj, attention.k_proj, attention.v_proj
    scales = [_cached(m) for m in mods]
    if os.environ.get("NOMAS_DEC_FUSE") != "1" or scales[0] is None or len(set(scales)) != 1 or not x.is_contiguous():
        return [m(x) for m in mods]
    ext = ops.n3()
    x8 = ext.n3_quantize(x.bfloat16(), scales[0])
    return [ext.n3_linear_f8in(x8, m.wq, m.sw, m.bias, scales[0]) for m in mods]


def _fuse_gate_up(mlp):
    g, u = mlp.gate_proj, mlp.up_proj
    w, s = torch.cat((g.wq, u.wq)), torch.cat((g.sw, u.sw))
    n = g.wq.shape[0]
    g._buffers["wq"], u._buffers["wq"], g._buffers["sw"], u._buffers["sw"] = w[:n], w[n:], s[:n], s[n:]
    mlp._nomas_gu = (w, s, n)


def dec_mlp(mlp, x):
    g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    sg, su, sd = _cached(g), _cached(u), _cached(d)
    if (os.environ.get("NOMAS_DEC_FUSE") != "1" or sg is None or sg != su or sd is None or not x.is_contiguous()
            or g.bias is not None or u.bias is not None or g.wq.shape[0] % 128):
        y = d(torch.nn.functional.silu(g(x)) * u(x))
        if (os.environ.get("NOMAS_DEC_FUSE") == "1" and not hasattr(mlp, "_nomas_gu") and not torch.cuda.is_current_stream_capturing()
                and _cached(g) is not None and _cached(g) == _cached(u) and g.bias is None and u.bias is None):
            _fuse_gate_up(mlp)
            mlp._nomas_cublas = (torch.tensor(_cached(g), device=x.device), torch.tensor(_cached(d), device=x.device),
                                 torch.ones((), device=x.device), d.sw.view(1, -1).bfloat16())
        return y
    if not hasattr(mlp, "_nomas_gu"):
        if torch.cuda.is_current_stream_capturing():
            return d(torch.nn.functional.silu(g(x)) * u(x))
        _fuse_gate_up(mlp)
    w, s, n = mlp._nomas_gu
    ext = ops.n3()
    if os.environ.get("NOMAS_DEC_CUBLAS") == "1" and d.bias is None:
        x8 = ext.n3_quantize(x.bfloat16().reshape(-1, x.shape[-1]).contiguous(), sg)
        st = mlp.__dict__.get("_nomas_cublas")
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("NOMAS_DEC_CUBLAS scales must be prepared before graph capture")
            st = mlp.__dict__["_nomas_cublas"] = (torch.tensor(sg, device=x.device), torch.tensor(sd, device=x.device),
                                                  torch.ones((), device=x.device), d.sw.view(1, -1).bfloat16())
        gu = torch._scaled_mm(x8, w.t(), scale_a=st[0], scale_b=st[2], out_dtype=torch.bfloat16)
        h = ext.n3_silu_mul_quant_cols(gu, s, sd)
        y = torch._scaled_mm(h, d.wq.t(), scale_a=st[1], scale_b=st[2], out_dtype=torch.bfloat16)
        return y.mul_(st[3]).view(*x.shape[:-1], d.out_features)
    gu = ext.n3_linear_split(x.bfloat16().reshape(-1, x.shape[-1]), w, s, None, sg, n)
    h = ext.n3_silu_mul_quant(gu, sd)
    return ext.n3_linear_f8in(h, d.wq, d.sw, d.bias, sd).view(*x.shape[:-1], d.out_features)


def _patch_decoder():
    from hymm.models.vae.pixelumm_decoder import loader, model
    model.GenMLP.forward = lambda self, x: dec_mlp(self, x)

    original = loader._install_inference_optimizations
    class Config:
        def __init__(self, source):
            self.source = source
        def __getattr__(self, name):
            return False if name in ("cutlass_fp8", "deepgemm_fp8") else getattr(self.source, name)
    @wraps(original)
    def install_optimizations(module, config, logger):
        count = _replace_linears(module)
        logger.info(f"PixelUMM Kestrel N3: {count} FP8 row-scaled linears; first-use activation calibration")
        return original(module, Config(config), logger)
    loader._install_inference_optimizations = install_optimizations


def _patch_linear_dtype():
    import hymm.models.dit.modules.int8_stored_bf16_linear as w8
    import hymm.models.dit.modules.embed_layers as emb
    orig = w8.linear_dtype
    fn = lambda module: torch.bfloat16 if isinstance(module, N3Linear) else orig(module)
    w8.linear_dtype = emb.linear_dtype = fn


def install():
    enabled = []
    if os.environ.get("NOMAS_GEMM") == "n3":
        _patch_linear_dtype()
    for flag, patch in (("NOMAS_GEMM", _patch_dit), ("NOMAS_DEC_GEMM", _patch_decoder)):
        value = os.environ.get(flag, "")
        if value not in ("", "0", "n3"):
            raise ValueError(f"{flag} must be n3 or unset")
        if value == "n3":
            if flag not in _INSTALLED:
                os.environ.setdefault("N3_CONT", "1")
                ops.n3()
                patch()
                _INSTALLED.add(flag)
            enabled.append(flag)
    return enabled
