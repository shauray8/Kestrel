import contextvars
import functools
import importlib
import os
from collections import OrderedDict

import torch

from kestrel import ops
from kestrel.fa import FAttention


_ACTIVE = contextvars.ContextVar("nomas_fa", default=None)
_ENGINES = {}
_INSTALLED = set()
_LAYOUTS = {}
_ORIGINAL = {}


class _PackedFA(FAttention):
    def __init__(self, balanced=True, state_limit=4):
        super().__init__(mode="fp8", load_mode=3, q_tile=128, mx=1, bal=int(balanced), qk="int8")
        self.state = OrderedDict()
        self.state_limit = state_limit
        self.history = None
        self.history_len = 0

    def _hist(self, layer, q, k, v):
        signature = (tuple(q.shape), tuple(k.shape), q.dtype, q.device)
        st = self.state.pop(signature, None)
        if st is None:
            h = q.shape[2]
            nq, nk = (q.shape[1] + 63) // 64, (k.shape[1] + 63) // 64
            st = {}
            for name, shape, dtype in (
                ("q8", (h, nq, 64, 128), torch.uint8),
                ("k8", (h, nk, 64, 128), torch.uint8),
                ("v8", (h, nk, 128, 64), torch.uint8),
                ("qs", (h, nq), torch.float32),
                ("ks", (h, nk), torch.float32),
                ("vs", (h, nk), torch.float32),
            ):
                st[name] = torch.empty(shape, dtype=dtype, device=q.device)
            st["plan"] = ops.fa().fa_plan(st["k8"], st["v8"])
        self.state[signature] = st
        while self.state_limit is not None and len(self.state) > self.state_limit:
            self.state.popitem(last=False)
        history = self.history
        aligned = self.history_len // 64
        ext = ops.fa()
        if "mean" not in history:
            source = k[0, :self.history_len] if self.history_len else k[0]
            history["mean"] = source.mean(0, dtype=torch.float32).contiguous()
        mean = history["mean"]
        for x, name, scale, transpose, target, power, integer in (
            (k, "k8", "ks", False, 127, False, True),
            (v, "v8", "vs", True, 448, True, False),
        ):
            if aligned and name not in history:
                ext.fa_quant(x[0], st[name], st[scale], 0, aligned, transpose,
                             target, power, integer, mean if integer else None)
                history[name] = st[name][:, :aligned].clone(memory_format=torch.contiguous_format)
                history[scale] = st[scale][:, :aligned].clone(memory_format=torch.contiguous_format)
                self.stats["hist_quant"] += int(name == "k8")
            elif aligned:
                st[name][:, :aligned].copy_(history[name])
                st[scale][:, :aligned].copy_(history[scale])
            ext.fa_quant(x[0], st[name], st[scale], aligned, st[name].shape[1],
                         transpose, target, power, integer, mean if integer else None)
        return st

    def run(self, q, k, v, history, history_len, scale=None):
        self.history, self.history_len = history, history_len
        try:
            return self(None, q.contiguous(), k.contiguous(), v.contiguous(), scale)
        finally:
            self.history = None


def _engine():
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device).cuda_stream
    key = (device, stream)
    engine = _ENGINES.get(key)
    if engine is None:
        balanced = stream == torch.cuda.default_stream(device).cuda_stream
        engine = _PackedFA(balanced)
        _ENGINES[key] = engine
    return engine


def _check(q, k, v, gqa=False):
    if torch.is_grad_enabled():
        raise RuntimeError("Nomas FA requires no-grad inference")
    if (q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1] or q.shape[-1] != 128
            or (q.shape[2] % k.shape[2] if gqa else q.shape[2] != k.shape[2])
            or not q.is_cuda or q.dtype != torch.bfloat16
            or k.dtype != q.dtype or v.dtype != q.dtype
            or k.device != q.device or v.device != q.device):
        raise ValueError("Nomas FA requires CUDA BF16 [B,L,H,128] tensors")


def _indices(intervals, device):
    parts = [torch.arange(a, b, device=device) for a, b in intervals if b > a]
    return torch.cat(parts) if parts else torch.empty(0, device=device, dtype=torch.long)


def _session(cache=None, history_len=None):
    active = _ACTIVE.get()
    if active is not None:
        return active
    return {"state": cache if cache is not None else {}, "length": history_len}


def _group(q, k, v, batch, qi, ki, session, history_len):
    if not qi.numel() or not ki.numel():
        return None
    state = session["state"].setdefault("history", {})
    key = (batch, q.shape[2], q.device, history_len)
    history = state.setdefault(key, {})
    return _engine().run(
        q[batch:batch+1].index_select(1, qi),
        k[batch:batch+1].index_select(1, ki),
        v[batch:batch+1].index_select(1, ki), history, history_len)[0]


def dense_attention(q, k, v, query_mask, key_mask, *, cache=None, history_len=None):
    _check(q, k, v)
    session = _session(cache, history_len)
    prefix = session["length"]
    prefix = max(k.shape[1] - q.shape[1], 0) if prefix is None else prefix
    output = torch.zeros_like(q)
    for batch in range(q.shape[0]):
        qi = query_mask[batch].nonzero().flatten()
        hi = key_mask[batch, :prefix].nonzero().flatten()
        ci = key_mask[batch, prefix:].nonzero().flatten() + prefix
        ki = torch.cat((hi, ci))
        part = _group(q, k, v, batch, qi, ki, session, hi.numel())
        if part is not None:
            output[batch].index_copy_(0, qi, part)
    return output


def intra_attention(q_full, full_k, full_v, kv_mask, cur_txt_mask,
                    cache_valid_len, img_q_len, txt_offsets_per_sample,
                    vid_windows, seg_ar_attention_backend="flex", *, cache=None):
    _check(q_full, full_k, full_v)
    session = _session(cache, cache_valid_len)
    output = torch.zeros_like(q_full)
    device = q_full.device
    for batch in range(q_full.shape[0]):
        hi = kv_mask[batch, :cache_valid_len].nonzero().flatten()
        ti = cur_txt_mask[batch].nonzero().flatten()
        valid = ti.numel()
        offsets = txt_offsets_per_sample[batch] if batch < len(txt_offsets_per_sample) else None
        if offsets is None:
            qi = torch.cat((torch.arange(img_q_len, device=device), img_q_len + ti))
            ki = torch.cat((hi, cache_valid_len + qi))
            part = _group(q_full, full_k, full_v, batch, qi, ki, session, hi.numel())
            if part is not None:
                output[batch].index_copy_(0, qi, part)
            continue
        offsets = tuple(max(0, min(int(x), valid)) for x in offsets)
        groups = {}
        windows = [(max(0, min(int(a), img_q_len)), max(0, min(int(b), img_q_len)), int(c))
                   for a, b, c in vid_windows]
        boundaries = sorted({x for a, b, _ in windows for x in (a, b)})
        for left, right in zip(boundaries, boundaries[1:]):
            covering = [action for a, b, action in windows if a <= left and b >= right]
            if covering:
                actions = tuple(sorted({a for a in covering if 0 <= a < len(offsets)-1}))
                groups.setdefault(actions, []).append((left, right))
        groups.setdefault((), []).append((img_q_len, img_q_len + offsets[0]))
        for action, (left, right) in enumerate(zip(offsets, offsets[1:])):
            groups.setdefault((action,), []).append((img_q_len + left, img_q_len + right))
        for actions, intervals in groups.items():
            compact_qi = _indices(intervals, device)
            qi = torch.where(compact_qi < img_q_len, compact_qi,
                             img_q_len + ti[(compact_qi-img_q_len).clamp(min=0)]) if valid else compact_qi
            text_intervals = [(0, offsets[0])] + [(offsets[a], offsets[a+1]) for a in actions]
            text = ti.index_select(0, _indices(text_intervals, device))
            ki = torch.cat((hi, torch.arange(cache_valid_len, cache_valid_len+img_q_len,
                                            device=device), cache_valid_len+img_q_len+text))
            part = _group(q_full, full_k, full_v, batch, qi, ki, session, hi.numel())
            if part is not None:
                output[batch].index_copy_(0, qi, part)
    return output


def _cache_token(cache, cache_mask, length):
    tensors = [cache.get(name) for name in ("k", "v", "k_scale", "v_scale")]
    if cache_mask is not None:
        tensors.append(cache_mask)
    def token(x):
        if x is None:
            return None
        try:
            version = x._version
        except RuntimeError:
            version = None
        return (x.data_ptr(), tuple(x.shape), str(x.device), version)
    return length, tuple(token(x) for x in tensors)


def _install_dit():
    module = importlib.import_module("hymm.models.dit.nomas_cpt")
    ranges = importlib.import_module("hymm.models.dit.modules.attention.range_attention")
    original = module.causal_attention_with_cache_mp
    _ORIGINAL["dense"] = module.flex_masked_attention
    _ORIGINAL["intra"] = module._intra_seg_cache_forward
    @functools.wraps(original)
    @torch.compiler.disable
    def attention(q, k, v, kv_cache, cache_valid_len=0, attn_mode=None,
                  text_mask=None, update_cache=False, cache_txt=True, cache_mask=None,
                  intra_seg_txt_offsets=None, intra_seg_vid_windows=None,
                  tokens_per_frame=0, seg_ar_attention_backend="magi", sparse_context=None):
        state = kv_cache.setdefault("_nomas_fa", {})
        token = _cache_token(kv_cache, cache_mask, cache_valid_len)
        if state.get("token") != token:
            state.clear()
            state["token"] = token
        active = _ACTIVE.set({"state": state, "length": cache_valid_len})
        try:
            return original(q, k, v, kv_cache, cache_valid_len, attn_mode, text_mask,
                            update_cache, cache_txt, cache_mask, intra_seg_txt_offsets,
                            intra_seg_vid_windows, tokens_per_frame,
                            seg_ar_attention_backend, sparse_context)
        finally:
            _ACTIVE.reset(active)
            if update_cache:
                state.clear()
    module.causal_attention_with_cache_mp = attention
    module.flex_masked_attention = dense_attention
    module._intra_seg_cache_forward = intra_attention
    ranges.flex_masked_attention = dense_attention


def _decoder_plan(grid, tile, device):
    frames, height, width = grid
    nh, nw = (height+tile-1)//tile, (width+tile-1)//tile
    spans = {}
    cursor = 0
    groups = {}
    for y in range(nh):
        for x in range(nw):
            length = frames * min(tile, height-y*tile) * min(tile, width-x*tile)
            spans[y, x] = (cursor, cursor+length)
            cursor += length
            center = (min(max(y, 1), max(nh-2, 1)), min(max(x, 1), max(nw-2, 1)))
            groups.setdefault(center, []).append((y, x))
    result = []
    for (cy, cx), tiles in groups.items():
        qi = _indices([spans[t] for t in tiles], device)
        ki = _indices([spans[y, x] for y in range(max(0, cy-1), min(nh, cy+2))
                       for x in range(max(0, cx-1), min(nw, cx+2))], device)
        result.append((qi, ki))
    return result


def decoder_attention(q, k, v, block_mask=None):
    if q.shape[2] % k.shape[2]:
        raise ValueError("Decoder Q heads must be divisible by KV heads")
    repeat = q.shape[2] // k.shape[2]
    _check(q, k, v, gqa=True)
    if block_mask is None:
        plan = [(torch.arange(q.shape[1], device=q.device), torch.arange(k.shape[1], device=k.device))]
    else:
        entry = _LAYOUTS.get(id(block_mask))
        if entry is None or entry[0] is not block_mask:
            raise ValueError("Decoder mask was not built by PixelUMM local_attention_layout")
        plan = entry[1]
    output = torch.empty_like(q)
    for batch in range(q.shape[0]):
        for qi, ki in plan:
            kg = k[batch:batch+1].index_select(1, ki)
            vg = v[batch:batch+1].index_select(1, ki)
            if repeat != 1:
                kg = kg.repeat_interleave(repeat, 2)
                vg = vg.repeat_interleave(repeat, 2)
            part = _engine().run(q[batch:batch+1].index_select(1, qi), kg, vg, {}, 0)
            output[batch].index_copy_(0, qi, part[0])
    return output


def _install_decoder():
    module = importlib.import_module("hymm.models.vae.pixelumm_decoder.model")
    layout = module.local_attention_layout
    _ORIGINAL["decoder"] = module.GenAttention.forward
    _ORIGINAL["decoder_flex"] = module._flex_attention
    @functools.wraps(layout)
    def local_attention_layout(grid, tile, device):
        result = layout(grid, tile, device)
        if result is not None and id(result[2]) not in _LAYOUTS:
            _LAYOUTS[id(result[2])] = (result[2], _decoder_plan(grid, tile, device))
        return result
    @torch.compiler.disable
    def forward(self, x, cos, sin, block_mask=None):
        arch = self.arch
        batch, count, _ = x.shape
        q = self.q_proj(x).view(batch, count, arch.num_heads, arch.head_dim)
        k = self.k_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        v = self.v_proj(x).view(batch, count, arch.num_kv_heads, arch.head_dim)
        q = self._prepare_qk(q, cos, sin, self.q_norm_t, self.q_norm_hw, v.dtype)
        k = self._prepare_qk(k, cos, sin, self.k_norm_t, self.k_norm_hw, v.dtype)
        out = decoder_attention(q, k, v, block_mask)
        return self.o_proj(out.reshape(batch, count, arch.num_heads * arch.head_dim))
    module.local_attention_layout = local_attention_layout
    module.GenAttention.forward = forward


def install():
    if os.environ.get("NOMAS_ATTN", "").lower() == "fa" and "dit" not in _INSTALLED:
        _install_dit()
        _INSTALLED.add("dit")
    if os.environ.get("NOMAS_DEC_ATTN", "").lower() == "fa" and "decoder" not in _INSTALLED:
        _install_decoder()
        _INSTALLED.add("decoder")
    return bool(_INSTALLED)
