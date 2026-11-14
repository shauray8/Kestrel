import contextvars
import functools
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl
from kestrel import ops

_INSTALLED = False
_ORIGINAL = {}
_PACKED = OrderedDict()
_GRAPH_PACKED = contextvars.ContextVar("nomas_glue_graph_packed", default=None)
_CUDA = os.environ.get("NOMAS_GLUE_CUDA", "0") == "1"


@triton.jit(do_not_specialize=["L"])
def _mod_kernel(X, S, A, Y, C: tl.constexpr, L,
                SS, AS, EPS: tl.constexpr,
                INV, FP8: tl.constexpr, NORM_FP32: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * C + c, c < C, 0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    var = tl.sum(tl.where(c < C, (x - mean) * (x - mean), 0), 0) / C
    n = (x - mean) * tl.rsqrt(var + EPS)
    if not NORM_FP32:
        n = n.to(X.dtype.element_ty).to(tl.float32)
    s = tl.load(S + r // L * SS + c, c < C, 0)
    a = tl.load(A + r // L * AS + c, c < C, 0)
    a = (1 + a.to(tl.float32)).to(A.dtype.element_ty).to(tl.float32)
    z = n * a
    if not NORM_FP32 and X.dtype.element_ty == tl.bfloat16 and A.dtype.element_ty == tl.bfloat16:
        z = z.to(tl.bfloat16).to(tl.float32)
    z = z + s.to(tl.float32)
    if not NORM_FP32 and X.dtype.element_ty == tl.bfloat16 and A.dtype.element_ty == tl.bfloat16 and S.dtype.element_ty == tl.bfloat16:
        z = z.to(tl.bfloat16)
    if FP8:
        z = z.to(tl.bfloat16).to(tl.float32) * INV
    tl.store(Y + r * C + c, z, c < C)


@torch.library.custom_op("nomas_glue::mod", mutates_args=())
def _mod_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
              eps: float, activation_scale: float, norm_fp32: bool) -> torch.Tensor:
    if _CUDA:
        return ops.nomas_glue().mod(x, shift, scale, eps, activation_scale, norm_fp32)
    norm_dtype = torch.float32 if norm_fp32 else x.dtype
    dtype = torch.float8_e4m3fn if activation_scale else torch.promote_types(torch.promote_types(norm_dtype, shift.dtype), scale.dtype)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _mod_kernel[(x.numel() // x.shape[-1],)](
        x, shift, scale, out, x.shape[-1], x.shape[1], shift.stride(0), scale.stride(0),
        eps, 1 / activation_scale if activation_scale else 1., bool(activation_scale), norm_fp32,
        triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False)
    return out


@_mod_norm.register_fake
def _mod_fake(x, shift, scale, eps, activation_scale, norm_fp32):
    norm_dtype = torch.float32 if norm_fp32 else x.dtype
    dtype = torch.float8_e4m3fn if activation_scale else torch.promote_types(torch.promote_types(norm_dtype, shift.dtype), scale.dtype)
    return torch.empty_like(x, dtype=dtype)


def mod_norm(x, shift, scale, eps, activation_scale):
    norm_fp32 = torch.is_autocast_enabled("cuda") and x.dtype in (torch.float16, torch.bfloat16)
    return _mod_norm(x, shift, scale, eps, activation_scale, norm_fp32)


@triton.jit
def _rms_kernel(X, W, Y, R, Z, C: tl.constexpr, EPS: tl.constexpr,
                RES: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + r * C + c, c < C, 0)
    if RES:
        x = (x.to(tl.float32) + tl.load(R + r * C + c, c < C, 0).to(tl.float32)).to(X.dtype.element_ty)
        tl.store(Z + r * C + c, x, c < C)
    f = x.to(tl.float32)
    n = (f * tl.rsqrt(tl.sum(f * f, 0) / C + EPS)).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + c, c < C, 0).to(tl.float32)
    tl.store(Y + r * C + c, n * w, c < C)


@torch.library.custom_op("nomas_glue::rms", mutates_args=())
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _CUDA:
        return ops.nomas_glue().rms(x, weight, eps)
    out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))
    _rms_kernel[(x.numel() // x.shape[-1],)](x, weight, out, x, x, x.shape[-1], eps,
        False, triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False)
    return out


@rms_norm.register_fake
def _rms_fake(x, weight, eps):
    return torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))


@torch.library.custom_op("nomas_glue::add_rms", mutates_args=())
def add_rms(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if _CUDA:
        return ops.nomas_glue().add_rms(x, residual, weight, eps)
    added = torch.empty_like(x)
    out = torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))
    _rms_kernel[(x.numel() // x.shape[-1],)](x, weight, out, residual, added,
        x.shape[-1], eps, True, triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False)
    return added, out


@add_rms.register_fake
def _add_rms_fake(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.promote_types(x.dtype, weight.dtype))


@triton.jit(do_not_specialize=["L"])
def _rope_kernel(X, W, U, COS, SIN, Y, H: tl.constexpr, D: tl.constexpr,
                 L, EPS: tl.constexpr, DEC: tl.constexpr,
                 ROPE: tl.constexpr, WF32: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(X + r * D + d).to(tl.float32)
    if DEC:
        half = D // 2
        t = tl.sum(tl.where(d < half, x * x, 0), 0) / half
        hw = tl.sum(tl.where(d >= half, x * x, 0), 0) / half
        n = x * tl.rsqrt(tl.where(d < half, t, hw) + EPS)
        w = tl.where(d < half, tl.load(W + d % half), tl.load(U + d % half)).to(tl.float32)
        n = n * w
        section = tl.where(d < half, half, D // 4)
        start = tl.where(d < half, 0, half + (d - half) // (D // 4) * (D // 4))
        partner = start + (d - start + section // 2) % section
        sign = tl.where(d - start < section // 2, -1., 1.)
    else:
        n = (x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)).to(X.dtype.element_ty).to(tl.float32)
        n = n * tl.load(W + d).to(tl.float32)
        if not WF32:
            n = n.to(X.dtype.element_ty).to(tl.float32)
        n = n.to(Y.dtype.element_ty).to(tl.float32)
        partner = d ^ 1
        sign = tl.where(d % 2 == 0, -1., 1.)
    if ROPE:
        rotated = tl.gather(n, partner, 0) * sign
        token = r // H % L
        c = tl.load(COS + token * D + d).to(tl.float32)
        s = tl.load(SIN + token * D + d).to(tl.float32)
        n = n * c + rotated * s
    tl.store(Y + r * D + d, n)


@torch.library.custom_op("nomas_glue::norm_rope", mutates_args=())
def norm_rope(x: torch.Tensor, weight: torch.Tensor, other: torch.Tensor,
              cos: torch.Tensor, sin: torch.Tensor, eps: float, decoder: bool,
              rotary: bool, dtype: torch.dtype) -> torch.Tensor:
    if _CUDA:
        return ops.nomas_glue().norm_rope(x, weight, other, cos, sin, eps, decoder, rotary, dtype)
    out = torch.empty_like(x, dtype=dtype)
    _rope_kernel[(x.numel() // x.shape[-1],)](x, weight, other, cos, sin, out,
        x.shape[2], x.shape[3], x.shape[1], eps, decoder, rotary,
        weight.dtype == torch.float32, enable_fp_fusion=False)
    return out


@norm_rope.register_fake
def _rope_fake(x, weight, other, cos, sin, eps, decoder, rotary, dtype):
    return torch.empty_like(x, dtype=dtype)


def _hidden(x, norm, shift, scale, linears):
    from .gemm import N3Linear, _scale
    if (isinstance(norm, torch.nn.LayerNorm) and not norm.elementwise_affine
            and x.is_cuda and x.is_contiguous() and shift.ndim == 2):
        scales = [_scale(None, m.activation_scale) for m in linears] if all(isinstance(m, N3Linear) for m in linears) else [None]
        sa = scales[0] if all(s == scales[0] for s in scales) else None
        return mod_norm(x, shift, scale, norm.eps, sa or 0.), sa
    from hymm.models.dit.modules.modulate_layers import modulate
    return modulate(norm(x), shift=shift, scale=scale), None


def _linear(module, x, scale):
    if scale:
        return ops.n3().n3_linear_f8in(x, module.wq, module.sw, module.bias, scale)
    return module(x)


_DEV = {}
_DEV_CONTENT = {}


def _on(t, device):
    if t.device == device:
        return t
    try:
        version = t._version
    except RuntimeError:
        version = None
    key = (t.data_ptr(), tuple(t.shape), t.dtype, device, version)
    hit = _DEV.get(key)
    if hit is None or hit[0] is not t:
        sig = (tuple(t.shape), t.dtype, device, t.contiguous().view(torch.uint8).numpy().tobytes()) if t.device.type == "cpu" else None
        dev = _DEV_CONTENT.get(sig) if sig is not None else None
        if dev is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare Nomas device tables before CUDA graph capture")
            dev = t.to(device)
            if sig is not None:
                if len(_DEV_CONTENT) > 64:
                    _DEV_CONTENT.clear()
                _DEV_CONTENT[sig] = dev
        if len(_DEV) > 256:
            _DEV.clear()
        hit = _DEV[key] = (t, dev)
    return hit[1]


device_tensor = _on


def _qkv(block, stream, x, shift, scale, freqs):
    if freqs is not None:
        freqs = tuple(_on(f, x.device) for f in freqs)
    from hymm.models.dit.modules.norm_layers import RMSNorm
    layers = [getattr(block, stream + "_attn_" + n) for n in ("q", "k", "v")]
    hidden, sa = _hidden(x, getattr(block, stream + "_norm1"), shift, scale, layers)
    q, k, v = [_linear(m, hidden, sa).reshape(x.shape[0], x.shape[1], block.heads_num, -1) for m in layers]
    fused = (q.is_cuda and q.shape[-1] == 128 and all(
        isinstance(getattr(block, stream + "_attn_" + n + "_norm"), RMSNorm)
        and hasattr(getattr(block, stream + "_attn_" + n + "_norm"), "weight") for n in ("q", "k")))
    for name, z in (("q", q), ("k", k)):
        norm = getattr(block, stream + "_attn_" + name + "_norm")
        if fused:
            cos, sin = freqs if freqs is not None else (z, z)
            result = norm_rope(z, norm.weight, norm.weight, cos, sin, norm.eps,
                               False, freqs is not None, v.dtype)
        else:
            result = norm(z).to(v)
        if name == "q":
            q = result
        else:
            k = result
    if freqs is not None and not fused:
        from hymm.models.dit.modules.posemb_layers import apply_rotary_emb
        q, k = apply_rotary_emb(q, k, freqs, head_first=False)
    return q, k, v


def _post(block, stream, x, attention, parts):
    from .gemm import _project_residual, _mlp_eligible, _scale
    proj, mlp = getattr(block, stream + "_attn_proj"), getattr(block, stream + "_mlp")
    x = _project_residual(proj, attention, x, parts[2])
    hidden, sa = _hidden(x, getattr(block, stream + "_norm2"), parts[3], parts[4], [mlp.fc1])
    if sa and _mlp_eligible(mlp):
        mid = _scale(None, mlp.fc2.activation_scale)
        if mid:
            ext = ops.n3()
            h = ext.n3_linear_f8in_gelu_f8(hidden, mlp.fc1.wq, mlp.fc1.sw, mlp.fc1.bias, sa, mid)
            if x.dtype == torch.float32 and parts[5].dtype == torch.float32:
                return ext.n3_linear_f8in_residual(h, mlp.fc2.wq, mlp.fc2.sw,
                    mlp.fc2.bias, mid, x, parts[5].unsqueeze(1))
            y = ext.n3_linear_f8in(h, mlp.fc2.wq, mlp.fc2.sw, mlp.fc2.bias, mid)
            return x + y * parts[5].unsqueeze(1)
    if sa:
        hidden = _linear(mlp.fc1, hidden, sa)
        hidden = mlp.norm(mlp.drop1(mlp.act(hidden)))
        y = mlp.drop2(mlp.fc2(hidden))
        return x + y * parts[5].unsqueeze(1)
    return _project_residual(mlp, hidden, x, parts[5])


def _block(self, img, txt, vec, vec_txt=None, freqs_cis=None, text_mask=None,
           kv_cache=None, cache_valid_len=0, update_cache=False, attn_mode=None,
           cache_txt=True, cache_mask=None, intra_seg_txt_offsets=None,
           intra_seg_vid_windows=None, intra_seg_tokens_per_frame=0, sparse_context=None):
    from hymm.models.dit import nomas_cpt
    if self.training or torch.is_grad_enabled() or not img.is_cuda or self.moe_config is not None:
        return _ORIGINAL["block"](self, img, txt, vec, vec_txt, freqs_cis, text_mask,
            kv_cache, cache_valid_len, update_cache, attn_mode, cache_txt, cache_mask,
            intra_seg_txt_offsets, intra_seg_vid_windows, intra_seg_tokens_per_frame, sparse_context)
    if vec.ndim != 2 or kv_cache is None:
        raise ValueError("Nomas glue cache block requires [B,C] timestep and KV cache")
    if freqs_cis is not None and not isinstance(freqs_cis, tuple):
        return _ORIGINAL["block"](self, img, txt, vec, vec_txt, freqs_cis, text_mask,
            kv_cache, cache_valid_len, update_cache, attn_mode, cache_txt, cache_mask,
            intra_seg_txt_offsets, intra_seg_vid_windows, intra_seg_tokens_per_frame, sparse_context)
    ip = self.img_mod(vec).chunk(6, -1)
    tp = self.txt_mod(vec if vec_txt is None else vec_txt).chunk(6, -1)
    iq, ik, iv = _qkv(self, "img", img, ip[0], ip[1], freqs_cis)
    tq, tk, tv = _qkv(self, "txt", txt, tp[0], tp[1], None)
    ia, ta = nomas_cpt.causal_attention_with_cache_mp((iq, tq), (ik, tk), (iv, tv),
        kv_cache, cache_valid_len, attn_mode or self.attn_mode, text_mask,
        update_cache, cache_txt, cache_mask, intra_seg_txt_offsets,
        intra_seg_vid_windows, intra_seg_tokens_per_frame, self.seg_ar_attention_backend, sparse_context)
    return _post(self, "img", img, ia, ip), _post(self, "txt", txt, ta, tp)


@triton.jit
def _source(CACHE, SCALE, IMG, TXT, IDX, r, d, head, H: tl.constexpr,
            HIST, NI, CS,
            SS, INT8: tl.constexpr, NK):
    token = tl.load(IDX + r, r < NK, 0)
    a = tl.load(CACHE + token[:, None] * CS + head * 128 + d[None, :],
                (r[:, None] < NK) & (token[:, None] < HIST), 0).to(tl.float32)
    if INT8:
        s = tl.load(SCALE + token * SS + head, (r < NK) & (token < HIST), 0)
        a = (a * s[:, None]).to(tl.bfloat16).to(tl.float32)
    b = tl.load(IMG + (token[:, None] - HIST) * H * 128 + head * 128 + d[None, :],
        (r[:, None] < NK) & (token[:, None] >= HIST) & (token[:, None] < HIST + NI), 0).to(tl.float32)
    c = tl.load(TXT + (token[:, None] - HIST - NI) * H * 128 + head * 128 + d[None, :],
        (r[:, None] < NK) & (token[:, None] >= HIST + NI), 0).to(tl.float32)
    return tl.where(token[:, None] < HIST, a, tl.where(token[:, None] < HIST + NI, b, c))


@triton.jit(do_not_specialize=["HIST", "NI", "NK", "NB", "FIRST"])
def _indexed_quant(CACHE, SCALE, IMG, TXT, IDX, OUT, S, MEAN,
                   H: tl.constexpr, HIST, NI,
                   CS, SS, INT8: tl.constexpr,
                   NK, NB, FIRST,
                   VALUE: tl.constexpr, REPEAT: tl.constexpr = 1):
    block, head = tl.program_id(0) + FIRST, tl.program_id(1)
    row = tl.arange(0, 64)
    dim = tl.arange(0, 128)
    r = block * 64 + row
    source_head = head // REPEAT
    x = _source(CACHE, SCALE, IMG, TXT, IDX, r, dim, source_head, H, HIST, NI, CS, SS, INT8, NK)
    if not VALUE:
        x = tl.where(r[:, None] < NK, x - tl.load(MEAN + source_head * 128 + dim)[None, :], 0.)
    maximum = tl.max(tl.max(tl.abs(x), 1), 0)
    scale = tl.maximum(maximum / (448. if VALUE else 127.), 1.e-12)
    if VALUE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
        quant = (x / scale).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        shuffled = (row & 48) + ((row & 6) << 1) + (row & 1) + ((row & 8) >> 2)
        offset = dim[None, :] * 64 + shuffled[:, None]
    else:
        quant = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(x / scale), -127.), 127.).to(tl.int8).to(tl.uint8, bitcast=True)
        offset = row[:, None] * 128 + dim[None, :]
    tl.store(OUT + (head * NB + block) * 8192 + offset, quant)
    tl.store(S + head * NB + block, scale)


@triton.jit(do_not_specialize=["HIST", "N"])
def _history_mean(CACHE, SCALE, IDX, OUT, H: tl.constexpr, HIST,
                  CS, SS, INT8: tl.constexpr,
                  N):
    head, block = tl.program_id(0), tl.program_id(1)
    r = block * 128 + tl.arange(0, 128)
    d = tl.arange(0, 128)
    t = tl.load(IDX + r, r < N, 0)
    x = tl.load(CACHE + t[:, None] * CS + head * 128 + d[None, :], r[:, None] < N, 0).to(tl.float32)
    if INT8:
        s = tl.load(SCALE + t * SS + head, r < N, 0)
        x = (x * s[:, None]).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + (block * H + head) * 128 + d, tl.sum(x, 0) / N)


def _quant(grid, *args, **kwargs):
    mode = os.environ.get("NOMAS_IQUANT")
    if mode in ("1", "fast") or (mode == "v" and args[17]):
        a = args
        return ops.iquant().indexed_quant(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], grid[1], a[8], a[9], a[10],
            a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18] if len(a) > 18 else 1, os.environ["NOMAS_IQUANT"] == "fast")
    if _CUDA:
        return ops.nomas_glue().indexed_quant(*args)
    return _indexed_quant[grid](*args, **kwargs)


def _quant_warmup(*args, **kwargs):
    if _CUDA:
        return ops.nomas_glue()
    return _indexed_quant.warmup(*args, **kwargs)


def _mean(grid, *args, **kwargs):
    if _CUDA:
        return ops.nomas_glue().history_mean(*args)
    return _history_mean[grid](*args, **kwargs)


def _buffers(q, nk, count):
    h, nq, nb = q.shape[2], triton.cdiv(count, 64), triton.cdiv(nk, 64)
    key = q.device, h, nq, nb
    active = _GRAPH_PACKED.get()
    packed = _PACKED if active is None else active
    st = packed.pop(key, None)
    if st is None:
        st = {n: torch.empty(shape, dtype=dtype, device=q.device) for n, shape, dtype in (
            ("q8", (h, nq, 64, 128), torch.uint8), ("k8", (h, nb, 64, 128), torch.uint8),
            ("v8", (h, nb, 128, 64), torch.uint8), ("qs", (h, nq), torch.float32),
            ("ks", (h, nb), torch.float32), ("vs", (h, nb), torch.float32))}
        st["plan"] = ops.fa().fa_plan(st["k8"], st["v8"])
        st["zero"] = torch.zeros((h, 128), device=q.device)
    packed[key] = st
    if active is None:
        while len(packed) > 4:
            packed.popitem(last=False)
    return st


def _direct_group(q, k, v, cache, length, qi, ki, history):
    from .attn import _engine
    h, nk, count = q[0].shape[2], ki.numel(), qi.numel()
    st = _buffers(q[0], nk, count)
    nb, aligned = st["k8"].shape[1], history["n"] // 64
    mean = history["mean"]
    integer = cache["k"].dtype == torch.int8
    for values, name, scales, transpose in ((k, "k8", "ks", False), (v, "v8", "vs", True)):
        source = cache["v" if transpose else "k"][0]
        scale = cache.get("v_scale" if transpose else "k_scale", source)[0]
        if aligned and name in history:
            st[name][:, :aligned].copy_(history[name])
            st[scales][:, :aligned].copy_(history[scales])
            first = aligned
        else:
            first = 0
        _quant((nb - first, h), source, scale, values[0][0], values[1][0], ki,
            st[name], st[scales], mean, h, length, values[0].shape[1], source.stride(0),
            scale.stride(0), integer, nk, nb, first, transpose, enable_fp_fusion=False)
        if aligned and name not in history:
            history[name] = st[name][:, :aligned].clone()
            history[scales] = st[scales][:, :aligned].clone()
    ext = ops.fa()
    _quant((st["q8"].shape[1], h), q[0][0], q[0][0], q[0][0], q[1][0], qi,
        st["q8"], st["qs"], st["zero"], h, 0, q[0].shape[1], q[0].stride(1),
        0, False, count, st["q8"].shape[1], 0, False, enable_fp_fusion=False)
    engine = _engine()
    engine.stats["calls"] += 1
    splits = _splits(triton.cdiv(count, 128) * h, nb) if os.environ.get("NOMAS_FA_SPLIT") == "1" and engine.q_tile == 128 else 1
    if splits > 1:
        tab = torch.arange(nb, device=q[0].device, dtype=torch.int32)
        val = torch.full((nb,), 64, device=q[0].device, dtype=torch.int32)
        val[-1:].fill_(nk - (nb - 1) * 64)
        ws = torch.empty(splits * h * count * 130, device=q[0].device)
        return ext.fa_attention_split(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"], tab, val,
            count, splits, 128 ** -.5, engine.exp_mode, engine.rescale_threshold, st["plan"], ws)
    return ext.fa_attention(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"],
        count, nk, 128 ** -.5, engine.load_mode, engine.exp_mode, engine.rescale_threshold,
        st["plan"], engine.q_tile, engine.mx, engine.bal, True, True)


def _splits(ctas, blocks, sms=170):
    best, cost = 1, -(-ctas // sms)
    for s in range(2, 9):
        if blocks < 4 * s:
            break
        c = -(-ctas * s // sms) / s * (1 + 0.03 * s)
        if c < 0.9 * cost:
            best, cost = s, c
    return best


def _groups(img_len, txt_mask, history_mask, length, offsets, windows):
    from .attn import _indices
    device = txt_mask.device
    hi, ti = history_mask.nonzero().flatten(), txt_mask.nonzero().flatten()
    image = torch.arange(img_len, device=device)
    if offsets is None:
        qi = torch.cat((image, img_len + ti))
        return [(qi, torch.cat((hi, length + qi)))], hi
    offsets = tuple(max(0, min(int(x), ti.numel())) for x in offsets)
    groups = {}
    windows = [(max(0, min(int(a), img_len)), max(0, min(int(b), img_len)), int(c)) for a, b, c in windows]
    boundaries = sorted({p for a, b, _ in windows for p in (a, b)})
    for left, right in zip(boundaries, boundaries[1:]):
        covering = [a for l, r, a in windows if l <= left and r >= right]
        if covering:
            groups.setdefault(tuple(sorted({a for a in covering if 0 <= a < len(offsets)-1})), []).append((left, right))
    groups.setdefault((), []).append((img_len, img_len + offsets[0]))
    for a, (left, right) in enumerate(zip(offsets, offsets[1:])):
        groups.setdefault((a,), []).append((img_len + left, img_len + right))
    result = []
    for actions, intervals in groups.items():
        compact = _indices(intervals, device)
        qi = torch.where(compact < img_len, compact, img_len + ti[(compact-img_len).clamp(min=0)]) if ti.numel() else compact
        text = ti.index_select(0, _indices([(0, offsets[0])] + [(offsets[a], offsets[a+1]) for a in actions], device))
        ki = torch.cat((hi, length + image, length + img_len + text))
        if qi.numel() and ki.numel():
            result.append((qi, ki))
    return result, hi


def prepare_attention(kv_cache, length, ni, nt, heads, text_mask, cache_mask,
                      offsets, windows, device, refresh=False):
    from .attn import _cache_token
    windows = windows or []
    offsets = offsets if windows else None
    token = (_cache_token(kv_cache, cache_mask, length), _cache_token({"k": text_mask}, None, nt),
             ni, None if offsets is None else tuple(offsets), tuple(tuple(w) for w in windows))
    state = kv_cache.get("_nomas_glue")
    if state is not None and state["token"] == token and not refresh:
        return state
    tm = torch.ones(nt, dtype=torch.bool, device=device) if text_mask is None else text_mask[0].bool()
    hm = torch.ones(length, dtype=torch.bool, device=device) if cache_mask is None else cache_mask[0, :length].bool()
    groups, hi = _groups(ni, tm, hm, length, offsets, windows)
    same = (state is not None and len(groups) == len(state["groups"])
        and state["history"]["n"] == hi.numel()
        and state["history"]["mean"].shape == (heads, 128)
        and state["history"]["mean"].device == kv_cache["k"].device
        and all(a.shape == c.shape and b.shape == d.shape
                for (a, b), (c, d) in zip(groups, state["groups"])))
    if same:
        for (qi, ki), (old_qi, old_ki) in zip(groups, state["groups"]):
            old_qi.copy_(qi)
            old_ki.copy_(ki)
        history = state["history"]
    else:
        history = {"n": hi.numel(), "mean": torch.empty((heads, 128), device=device)}
        state = {"groups": groups, "history": history}
    state["token"] = token
    if hi.numel():
        source = kv_cache["k"][0]
        scale = kv_cache.get("k_scale", kv_cache["k"])[0]
        parts = torch.empty((triton.cdiv(hi.numel(), 128), heads, 128), device=device)
        _mean((heads, parts.shape[0]), source, scale, hi, parts,
            heads, length, source.stride(0), scale.stride(0),
            source.dtype == torch.int8, hi.numel(), enable_fp_fusion=False)
        history["mean"].copy_(parts.sum(0))
    else:
        history["mean"].zero_()
    aligned = hi.numel() // 64
    if aligned:
        for key, name, scales, transpose in (("k", "k8", "ks", False), ("v", "v8", "vs", True)):
            source = kv_cache[key][0]
            scale = kv_cache.get(key + "_scale", kv_cache[key])[0]
            if name not in history:
                history[name] = torch.empty((heads, aligned, 128, 64) if transpose else
                    (heads, aligned, 64, 128), dtype=torch.uint8, device=device)
                history[scales] = torch.empty((heads, aligned), device=device)
            _quant((aligned, heads), source, scale, source, source, hi,
                history[name], history[scales], history["mean"], heads, length, 0,
                source.stride(0), scale.stride(0), source.dtype == torch.int8,
                hi.numel(), aligned, 0, transpose, enable_fp_fusion=False)
    meta = torch.empty((1, 0, heads, 128), dtype=torch.bfloat16, device=device)
    for qi, ki in state["groups"]:
        st = _buffers(meta, ki.numel(), qi.numel())
        for key, name, scales, transpose in (("k", "k8", "ks", False), ("v", "v8", "vs", True)):
            source = kv_cache[key][0]
            scale = kv_cache.get(key + "_scale", kv_cache[key])[0]
            first = aligned
            _quant_warmup(source, scale, meta, meta, ki, st[name], st[scales],
                history["mean"], heads, length, ni, source.stride(0), scale.stride(0),
                source.dtype == torch.int8, ki.numel(), st[name].shape[1], first, transpose,
                grid=(st[name].shape[1] - first, heads), enable_fp_fusion=False)
        _quant_warmup(meta, meta, meta, meta, qi, st["q8"], st["qs"], st["zero"],
            heads, 0, ni, heads * 128, 0, False, qi.numel(), st["q8"].shape[1], 0, False,
            grid=(st["q8"].shape[1], heads), enable_fp_fusion=False)
    kv_cache["_nomas_glue"] = state
    return state


def _attention(q, k, v, kv_cache, cache_valid_len=0, attn_mode=None,
               text_mask=None, update_cache=False, cache_txt=True, cache_mask=None,
               intra_seg_txt_offsets=None, intra_seg_vid_windows=None,
               tokens_per_frame=0, seg_ar_attention_backend="magi", sparse_context=None):
    from hymm.models.dit import nomas_cpt
    eligible = (not update_cache and not torch.is_grad_enabled() and not nomas_cpt.get_sequence_parallel_state()
        and seg_ar_attention_backend == "flex" and attn_mode == "flash3" and sparse_context is None and cache_txt
        and q[0].shape[0] == 1 and kv_cache.get("k") is not None
        and kv_cache["k"].shape[0] == 1 and q[0].shape[-1] == 128
        and all(x.dtype == torch.bfloat16 and x.is_contiguous() for x in (*q, *k, *v))
        and kv_cache["k"].dtype in (torch.int8, torch.bfloat16))
    if not eligible:
        if update_cache:
            kv_cache.pop("_nomas_glue", None)
        return _ORIGINAL["attention"](q, k, v, kv_cache, cache_valid_len, attn_mode,
            text_mask, update_cache, cache_txt, cache_mask, intra_seg_txt_offsets,
            intra_seg_vid_windows, tokens_per_frame, seg_ar_attention_backend, sparse_context)
    length, ni, nt = cache_valid_len, q[0].shape[1], q[1].shape[1]
    offsets = None if intra_seg_txt_offsets is None or not intra_seg_vid_windows else intra_seg_txt_offsets[0]
    tpf = int(tokens_per_frame) if tokens_per_frame and tokens_per_frame > 0 else 1
    windows = [[int(a) * tpf, int(b) * tpf, int(c)] for a, b, c in (intra_seg_vid_windows or [])]
    state = prepare_attention(kv_cache, length, ni, nt, q[0].shape[2],
        text_mask, cache_mask, offsets, windows, q[0].device)
    out = torch.zeros((1, ni + nt, q[0].shape[2], 128), dtype=q[0].dtype, device=q[0].device)
    for qi, ki in state["groups"]:
        part = _direct_group(q, k, v, kv_cache, length, qi, ki, state["history"])
        out[0].index_copy_(0, qi, part)
    return out[:, :ni].reshape(1, ni, -1), out[:, ni:].reshape(1, nt, -1)


@triton.jit(do_not_specialize=["T", "SPATIAL", "TOTAL"])
def _camera_add_kernel(X, F, M, Y, C: tl.constexpr, T,
                       SPATIAL, TOTAL,
                       MASK: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    frame = i // C // SPATIAL % T
    batch = i // (C * SPATIAL * T)
    f = tl.load(F + (batch * T + frame) * C + i % C, i < TOTAL, 0).to(tl.float32)
    if MASK:
        f = f * tl.load(M + batch * T + frame, i < TOTAL, 0).to(tl.float32)
    x = tl.load(X + i, i < TOTAL, 0).to(tl.float32)
    tl.store(Y + i, x + f, i < TOTAL)


@torch.library.custom_op("nomas_glue::camera_add", mutates_args=())
def camera_add(x: torch.Tensor, frames: torch.Tensor, mask: torch.Tensor | None,
               spatial: int) -> torch.Tensor:
    if _CUDA:
        return ops.nomas_glue().camera_add(x, frames, mask, spatial)
    dtype = torch.float32 if mask is not None else torch.promote_types(x.dtype, frames.dtype)
    out = torch.empty_like(x, dtype=dtype)
    _camera_add_kernel[(triton.cdiv(x.numel(), 1024),)](x, frames,
        frames if mask is None else mask, out, x.shape[-1], frames.shape[1],
        spatial, x.numel(), mask is not None, 1024, enable_fp_fusion=False)
    return out


@camera_add.register_fake
def _camera_add_fake(x, frames, mask, spatial):
    dtype = torch.float32 if mask is not None else torch.promote_types(x.dtype, frames.dtype)
    return torch.empty_like(x, dtype=dtype)


def _camera_blockwise(self, q, k, v, tt, th, tw, attn_boundaries_latent):
    if (not self.training and not torch.is_grad_enabled() and q.is_cuda
            and len(attn_boundaries_latent) == tt + 1
            and all(isinstance(b, int) and b == i for i, b in enumerate(attn_boundaries_latent))
            and k.shape[1] >= tt):
        start, end = self._sp_local_window(tt * th * tw, q.shape[1])
        frames = torch.arange(start, end, device=v.device) // (th * tw)
        return v.index_select(1, frames)
    return _ORIGINAL["camera_blockwise"](self, q, k, v, tt, th, tw, attn_boundaries_latent)


def _camera(self, x, tt, th, tw, keyboard, has_camera_mask=None,
            segment_boundaries_latent=None, packed_img_token_counts=None,
            packed_kb_token_counts=None, packed_sample_shapes=None,
            packed_has_camera=None, packed_segment_boundaries_latent=None,
            packed_has_camera_per_segment=None, rope_time_offset=0):
    from hymm.models.camera import action_module
    eligible = (not self.training and not torch.is_grad_enabled() and x.is_cuda
        and self.per_chunk_camera_attn and self.camera_chunk_size == 1
        and packed_img_token_counts is None and not action_module.get_sequence_parallel_state()
        and (tt > 1 or keyboard.shape[1] == 1) and keyboard.shape[1] >= tt
        and x.shape[1] == tt * th * tw)
    if not eligible:
        return _ORIGINAL["camera"](self, x, tt, th, tw, keyboard, has_camera_mask,
            segment_boundaries_latent, packed_img_token_counts, packed_kb_token_counts,
            packed_sample_shapes, packed_has_camera, packed_segment_boundaries_latent,
            packed_has_camera_per_segment, rope_time_offset)
    if has_camera_mask is not None and has_camera_mask.ndim == 2 and not (
            has_camera_mask.shape[1] == 1 and segment_boundaries_latent is None):
        if segment_boundaries_latent is None or len(segment_boundaries_latent) != has_camera_mask.shape[1] + 1:
            return _ORIGINAL["camera"](self, x, tt, th, tw, keyboard, has_camera_mask,
                segment_boundaries_latent, rope_time_offset=rope_time_offset)
    state = prepare_camera(self, keyboard, has_camera_mask, tt, th, tw,
                           segment_boundaries_latent, rope_time_offset)
    return camera_add(x.contiguous(), state["frames"], state["mask"], th * tw)


def prepare_camera(self, keyboard, has_camera_mask, tt, th, tw,
                   segment_boundaries_latent=None, rope_time_offset=0, refresh=False):
    from .attn import _cache_token
    token = (_cache_token({"k": keyboard, "v": has_camera_mask}, None, tt),
             th, tw, int(rope_time_offset), None if segment_boundaries_latent is None else tuple(segment_boundaries_latent))
    state = getattr(self, "_nomas_camera", None)
    if state is None or state["token"] != token or refresh:
        from .gemm import N3Linear, _linear as n3_linear
        projection = self.keyboard_attn_kv
        embedded = self.keyboard_embed(keyboard)
        if isinstance(projection, N3Linear):
            half = projection.out_features // 2
            bias = None if projection.bias is None else projection.bias[half:]
            value = n3_linear(embedded.bfloat16().contiguous(), projection.wq[half:],
                projection.sw[half:], bias, projection.activation_scale)
        elif isinstance(projection, torch.nn.Linear):
            half = projection.out_features // 2
            bias = None if projection.bias is None else projection.bias[half:]
            value = torch.nn.functional.linear(embedded, projection.weight[half:], bias)
        else:
            kv = projection(embedded)
            value = kv[..., kv.shape[-1] // 2:].contiguous()
        frames = self.proj_keyboard(value)[:, :tt].contiguous()
        mask = None
        if has_camera_mask is not None:
            if has_camera_mask.ndim == 1 or (
                    has_camera_mask.ndim == 2 and has_camera_mask.shape[1] == 1 and segment_boundaries_latent is None):
                mask = has_camera_mask.float().reshape(keyboard.shape[0], 1).expand(-1, tt).contiguous()
            elif has_camera_mask.ndim == 2:
                ids = torch.arange(tt, device=keyboard.device) + int(rope_time_offset)
                ids = ids.clamp(max=max(int(self.freqs_cos.shape[0]) - 1, 0))
                interior = torch.tensor(segment_boundaries_latent[1:-1], device=ids.device, dtype=ids.dtype)
                segments = torch.bucketize(ids, interior, right=True).clamp(max=has_camera_mask.shape[1] - 1)
                mask = has_camera_mask.float().index_select(1, segments).contiguous()
            else:
                raise ValueError("Camera mask must have shape [B] or [B,segments]")
        if (state is not None and state["frames"].shape == frames.shape
                and state["frames"].dtype == frames.dtype and state["frames"].device == frames.device
                and (state["mask"] is None) == (mask is None)):
            state["frames"].copy_(frames)
            if mask is not None:
                state["mask"].copy_(mask)
            state.update(token=token, input=keyboard, gate=has_camera_mask)
        else:
            state = {"token": token, "frames": frames, "mask": mask, "input": keyboard, "gate": has_camera_mask}
            self._nomas_camera = state
    return state


def _preprocess_camera(self, norm_extrinsics, norm_intrinsics, action_onehot,
                       oh, ow, img_dtype, device):
    from .attn import _cache_token
    if self.training or torch.is_grad_enabled() or self.camera_injection_mode not in ("action_attn", "extrinsics_attn"):
        return _ORIGINAL["preprocess_camera"](self, norm_extrinsics, norm_intrinsics,
            action_onehot, oh, ow, img_dtype, device)
    inputs = {"k": norm_extrinsics, "v": norm_intrinsics, "k_scale": action_onehot}
    token = (_cache_token(inputs, None, 0), oh, ow, img_dtype, str(device), self.camera_injection_mode)
    state = getattr(self, "_nomas_camera_embedding", None)
    if state is None or state["token"] != token:
        value = _ORIGINAL["preprocess_camera"](self, norm_extrinsics, norm_intrinsics,
            action_onehot, oh, ow, img_dtype, device)
        state = {"token": token, "inputs": inputs, "value": value}
        self._nomas_camera_embedding = state
    return state["value"]


_PAGED = {}
_DUMPED = [0, 0]


def _paged_buffers(device, heads, nb):
    key = device, heads, nb
    st = _PAGED.get(key)
    if st is None:
        st = {"k8": torch.empty((heads, nb, 64, 128), dtype=torch.uint8, device=device),
              "v8": torch.empty((heads, nb, 128, 64), dtype=torch.uint8, device=device),
              "ks": torch.empty((heads, nb), device=device), "vs": torch.empty((heads, nb), device=device)}
        st["plan"] = ops.fa().fa_plan(st["k8"], st["v8"])
        _PAGED[key] = st
    return st


def _decoder_paged(q, k, v, paged):
    from . import attn
    output = torch.empty_like(q)
    heads, kv_heads = q.shape[2], k.shape[2]
    engine, ext = attn._engine(), ops.fa()
    nb, count, index = paged["blocks"], paged["count"], paged["index"]
    st = _paged_buffers(q.device, heads, nb)
    for batch in range(q.shape[0]):
        partial = torch.empty((triton.cdiv(count, 128), kv_heads, 128), device=k.device)
        _mean((kv_heads, partial.shape[0]), k[batch], k[batch], paged["all"], partial,
            kv_heads, 0, k.stride(1), 0, False, count, enable_fp_fusion=False)
        mean = partial.sum(0)
        for source, name, scales, value in ((k, "k8", "ks", False), (v, "v8", "vs", True)):
            _quant((nb, heads), source[batch], source[batch], source[batch], source[batch], index,
                st[name], st[scales], mean, kv_heads, 0, source.shape[1], source.stride(1), 0, False,
                index.numel(), nb, 0, value, heads // kv_heads, enable_fp_fusion=False)
        for qi, tab, val in paged["groups"]:
            n = qi.numel()
            qb = _buffers(q, 64, n)
            _quant((qb["q8"].shape[1], heads), q[batch], q[batch], q[batch], q[batch], qi,
                qb["q8"], qb["qs"], qb["zero"], heads, 0, q.shape[1], q.stride(1), 0, False,
                n, qb["q8"].shape[1], 0, False, 1, enable_fp_fusion=False)
            part = ext.fa_attention_paged(qb["q8"], st["k8"], st["v8"], qb["qs"], st["ks"], st["vs"],
                tab, val, n, 128 ** -.5, engine.exp_mode, engine.rescale_threshold, st["plan"], engine.q_tile)
            engine.stats["calls"] += 1
            output[batch].index_copy_(0, qi, part)
    return output


def _decoder_attention(q, k, v, block_mask=None):
    from . import attn
    attn._check(q, k, v, gqa=True)
    if not all(x.is_contiguous() for x in (q, k, v)) or block_mask is None:
        return _ORIGINAL["decoder_attention"](q, k, v, block_mask)
    entry = attn._LAYOUTS.get(id(block_mask))
    if entry is None or entry[0] is not block_mask:
        raise ValueError("Decoder mask was not built by PixelUMM local_attention_layout")
    dump = os.environ.get("NOMAS_DEC_DUMP")
    if dump and _DUMPED[0] < 6 and _DUMPED.__setitem__(1, _DUMPED[1] + 1) is None and _DUMPED[1] % 7 == 1:
        torch.save({"q": q, "k": k, "v": v, "plan": entry[1], "paged": entry[2]}, f"{dump}/dec_{_DUMPED[0]}.pt")
        _DUMPED[0] += 1
    if os.environ.get("NOMAS_DEC_PAGED") == "1" and len(entry) > 2:
        return _decoder_paged(q, k, v, entry[2])
    output = torch.empty_like(q)
    heads, kv_heads = q.shape[2], k.shape[2]
    engine, ext = attn._engine(), ops.fa()
    for batch in range(q.shape[0]):
        for qi, ki in entry[1]:
            count, nk = qi.numel(), ki.numel()
            st = _buffers(q, nk, count)
            partial = torch.empty((triton.cdiv(nk, 128), kv_heads, 128), device=k.device)
            _mean((kv_heads, partial.shape[0]), k[batch], k[batch], ki, partial,
                kv_heads, 0, k.stride(1), 0, False, nk, enable_fp_fusion=False)
            mean = partial.sum(0)
            for source, indices, name, scales, value, center, source_heads, repeat in (
                (q, qi, "q8", "qs", False, st["zero"], heads, 1),
                (k, ki, "k8", "ks", False, mean, kv_heads, heads // kv_heads),
                (v, ki, "v8", "vs", True, mean, kv_heads, heads // kv_heads)):
                blocks = st[name].shape[1]
                _quant((blocks, heads), source[batch], source[batch], source[batch],
                    source[batch], indices, st[name], st[scales], center, source_heads,
                    0, source.shape[1], source.stride(1), 0, False, indices.numel(),
                    blocks, 0, value, repeat, enable_fp_fusion=False)
            part = ext.fa_attention(st["q8"], st["k8"], st["v8"], st["qs"], st["ks"], st["vs"],
                count, nk, 128 ** -.5, engine.load_mode, engine.exp_mode, engine.rescale_threshold,
                st["plan"], engine.q_tile, engine.mx, engine.bal, True, True)
            output[batch].index_copy_(0, qi, part)
    return output


def install():
    global _INSTALLED
    value = os.environ.get("NOMAS_GLUE", "")
    if value not in ("", "0", "1"):
        raise ValueError("NOMAS_GLUE must be 1 or unset")
    if value != "1" or _INSTALLED:
        return []
    from hymm.models.dit import nomas_cpt
    from hymm.models.vae.pixelumm_decoder import model
    from hymm.models.camera.action_module import ActionModule
    _ORIGINAL.update(block=nomas_cpt.MMDoubleStreamBlock.forward_with_cache,
                     attention=nomas_cpt.causal_attention_with_cache_mp,
                     rms=model.RMSNorm.forward, prepare=model.GenAttention._prepare_qk,
                     layer=model.GenLayer.forward, camera=ActionModule.forward,
                     camera_blockwise=ActionModule._blockwise_cross_attn,
                     preprocess_camera=nomas_cpt.CPTHYNomasDiffusionTransformer._preprocess_camera)
    nomas_cpt.CPTHYNomasDiffusionTransformer._preprocess_camera = _preprocess_camera
    ActionModule.forward = _camera
    ActionModule._blockwise_cross_attn = _camera_blockwise
    nomas_cpt.MMDoubleStreamBlock.forward_with_cache = _block
    if os.environ.get("NOMAS_ATTN") == "fa":
        nomas_cpt.causal_attention_with_cache_mp = _attention
    if os.environ.get("NOMAS_DEC_ATTN") == "fa":
        from . import attn
        _ORIGINAL["decoder_attention"] = attn.decoder_attention
        attn.decoder_attention = _decoder_attention
    @functools.wraps(_ORIGINAL["rms"])
    def rms(self, x):
        if x.is_cuda and x.is_contiguous() and not torch.is_grad_enabled():
            return rms_norm(x, self.weight, self.eps)
        return _ORIGINAL["rms"](self, x)
    @functools.wraps(_ORIGINAL["prepare"])
    def prepare(self, x, cos, sin, norm_t, norm_hw, dtype):
        if x.is_cuda and x.is_contiguous() and not torch.is_grad_enabled() and x.shape[-1] == 128:
            return norm_rope(x, norm_t.weight, norm_hw.weight, cos, sin,
                             norm_t.eps, True, True, dtype)
        return _ORIGINAL["prepare"](self, x, cos, sin, norm_t, norm_hw, dtype)
    @functools.wraps(_ORIGINAL["layer"])
    def layer(self, x, cos, sin, block_mask=None):
        if not x.is_cuda or not x.is_contiguous() or torch.is_grad_enabled():
            return _ORIGINAL["layer"](self, x, cos, sin, block_mask)
        attn = self.attn(self.input_norm(x), cos, sin, block_mask)
        if attn.dtype != x.dtype or attn.shape != x.shape:
            x = x + attn
            hidden = self.post_attn_norm(x)
        else:
            x, hidden = add_rms(x, attn, self.post_attn_norm.weight, self.post_attn_norm.eps)
        return x + self.mlp(hidden)
    model.RMSNorm.forward = rms
    model.GenAttention._prepare_qk = prepare
    model.GenLayer.forward = layer
    _INSTALLED = True
    return ["NOMAS_GLUE"]
