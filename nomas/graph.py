import ast
from collections import OrderedDict
from contextlib import contextmanager
import contextvars
from functools import wraps
import inspect
import os
import sys
import textwrap

import torch


_CURRENT = contextvars.ContextVar("nomas_graph", default=None)
_installed = False
_POOLS = {}
_STREAMS = {}


def _tensor_key(x):
    shape = (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
    try:
        version = x._version
    except RuntimeError:
        version = None
    return shape, x.data_ptr(), version




def _constant(factory):
    runner = _CURRENT.get()
    if runner is None:
        return factory()
    slot = runner.cursor
    runner.cursor += 1
    if slot == len(runner.constants):
        if runner.sealed:
            raise RuntimeError("NOMAS_GRAPH control flow changed after warmup")
        runner.constants.append(factory())
    return runner.constants[slot]


def _memoize(fn, resident=False):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        def build():
            result = fn(*args, **kwargs)
            runner = _CURRENT.get()
            return tuple(x.to(runner.device) for x in result) if resident and runner is not None else result
        return _constant(build)
    return wrapped




def _stage(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.contiguous_format)
    if isinstance(value, list):
        return [_stage(x) for x in value]
    if isinstance(value, tuple):
        return tuple(_stage(x) for x in value)
    if isinstance(value, dict):
        return {k: _stage(v) for k, v in value.items()}
    return value


def _copy_stage(target, source):
    if isinstance(source, torch.Tensor):
        if target.data_ptr() != source.data_ptr():
            target.copy_(source)
    elif isinstance(source, (list, tuple)):
        for a, b in zip(target, source):
            _copy_stage(a, b)
    elif isinstance(source, dict):
        for k, v in source.items():
            _copy_stage(target[k], v)


class GraphRunner:
    def __init__(self, fn, inputs, dynamic, label, prepare=None, eager_first=True):
        self.fn = fn
        self.dynamic = tuple(dynamic)
        self.label = label
        self.device = next(inputs[k].device for k in self.dynamic)
        pool_key = (self.device, "decoder" if label.startswith("PixelUMM") else "dit")
        if pool_key not in _STREAMS:
            _STREAMS[pool_key] = torch.cuda.Stream(device=self.device)
        self.stream = _STREAMS[pool_key]
        storage = {"k", "v", "k_scale", "v_scale", "storage_format", "swa_capacity", "swa_sampling"}
        self.inputs = {k: ([{name: value for name, value in cache.items() if name in storage}
                            for cache in v] if k == "kv_cache_list" and v is not None else _stage(v))
                       for k, v in inputs.items()}
        self.constants = []
        self.cursor = 0
        from nomas.attn import _PackedFA
        self.engine = _PackedFA(balanced=False, state_limit=None)
        self.packed = {}
        self.histories = {}
        self.camera_states = {}
        self.graph = None
        self.output = None
        self.sealed = False
        self.replays = 0
        self.calls = 0
        self.warmup_steps = int(eager_first)
        self.prepare = prepare
        self.pool = _POOLS.setdefault(pool_key, torch.cuda.graph_pool_handle())

    @contextmanager
    def _scope(self):
        token = _CURRENT.set(self)
        glue = sys.modules.get("nomas.glue")
        packed = getattr(glue, "_GRAPH_PACKED", None)
        packed_token = packed.set(self.packed) if packed is not None else None
        self.cursor = 0
        try:
            yield
        finally:
            if packed is not None:
                packed.reset(packed_token)
            _CURRENT.reset(token)

    def bind(self, inputs):
        for name, value in inputs.items():
            if name == "kv_cache_list" and value is not None:
                length = inputs["cache_valid_len"]
                for staged, external in zip(self.inputs[name], value):
                    for key in ("k", "v", "k_scale", "v_scale"):
                        if key in external and staged[key].data_ptr() != external[key].data_ptr():
                            staged[key][:, :length].copy_(external[key][:, :length])
            elif name == "cache_mask_list" and value is not None:
                length = inputs["cache_valid_len"]
                for staged, external in zip(self.inputs[name], value):
                    staged[:, :length].copy_(external[:, :length])
            elif name not in self.dynamic:
                _copy_stage(self.inputs[name], value)
        if self.prepare is not None:
            with self._scope():
                self.prepare(self, True)

    def __call__(self, **inputs):
        if torch.is_grad_enabled():
            raise RuntimeError("NOMAS_GRAPH requires no-grad inference")
        current = torch.cuda.current_stream(self.device)
        for name in self.dynamic:
            self.inputs[name].copy_(inputs[name])
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            if self.calls < self.warmup_steps:
                with self._scope():
                    output = self.fn(**self.inputs)
            elif self.graph is None:
                if self.prepare is not None:
                    with self._scope():
                        self.prepare(self, False)
                self.graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(self.graph, pool=self.pool, stream=self.stream,
                                          capture_error_mode="thread_local"), self._scope():
                        self.output = self.fn(**self.inputs)
                except Exception as exc:
                    raise RuntimeError(f"NOMAS_GRAPH {self.label} capture failed: {exc}") from exc
                self.sealed = True
                self.graph.replay()
                self.replays += 1
                output = self.output
            else:
                self.graph.replay()
                self.replays += 1
                output = self.output
        current.wait_stream(self.stream)
        self.calls += 1
        return output


def _plans_output(q, k, v, plan, session, zero):
    from nomas import attn
    output = torch.zeros_like(q) if zero else torch.empty_like(q)
    for batch, qi, ki, history in plan:
        part = attn._group(q, k, v, batch, qi, ki, session, history)
        if part is not None:
            output[batch].index_copy_(0, qi, part)
    return output


def _dense(q, k, v, query_mask, key_mask, *, cache=None, history_len=None):
    from nomas import attn
    attn._check(q, k, v)
    session = attn._session(cache, history_len)
    prefix = session["length"]
    prefix = max(k.shape[1] - q.shape[1], 0) if prefix is None else prefix
    def build():
        plan = []
        for batch in range(q.shape[0]):
            qi = query_mask[batch].nonzero().flatten()
            hi = key_mask[batch, :prefix].nonzero().flatten()
            ci = key_mask[batch, prefix:].nonzero().flatten() + prefix
            plan.append((batch, qi, torch.cat((hi, ci)), hi.numel()))
        return plan
    return _plans_output(q, k, v, _constant(build), session, True)


def _intra(q_full, full_k, full_v, kv_mask, cur_txt_mask, cache_valid_len,
           img_q_len, txt_offsets_per_sample, vid_windows,
           seg_ar_attention_backend="flex", *, cache=None):
    from nomas import attn
    attn._check(q_full, full_k, full_v)
    session = attn._session(cache, cache_valid_len)
    device = q_full.device
    def build():
        plan = []
        for batch in range(q_full.shape[0]):
            hi = kv_mask[batch, :cache_valid_len].nonzero().flatten()
            ti = cur_txt_mask[batch].nonzero().flatten()
            valid = ti.numel()
            offsets = txt_offsets_per_sample[batch] if batch < len(txt_offsets_per_sample) else None
            if offsets is None:
                qi = torch.cat((torch.arange(img_q_len, device=device), img_q_len + ti))
                plan.append((batch, qi, torch.cat((hi, cache_valid_len + qi)), hi.numel()))
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
                compact = attn._indices(intervals, device)
                qi = torch.where(compact < img_q_len, compact,
                                 img_q_len + ti[(compact-img_q_len).clamp(min=0)]) if valid else compact
                ranges = [(0, offsets[0])] + [(offsets[a], offsets[a+1]) for a in actions]
                text = ti.index_select(0, attn._indices(ranges, device))
                ki = torch.cat((hi, torch.arange(cache_valid_len, cache_valid_len+img_q_len,
                                                device=device), cache_valid_len+img_q_len+text))
                plan.append((batch, qi, ki, hi.numel()))
        return plan
    return _plans_output(q_full, full_k, full_v, _constant(build), session, True)


class _StaticMasks(ast.NodeTransformer):
    names = {"cur_img_mask", "cur_txt_mask", "cache_write_mask", "history_mask",
             "cur_full_mask", "q_mask", "kv_mask", "vid_windows_tok"}

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in self.names:
            node.value = ast.Call(ast.Name("_nomas_graph_constant", ast.Load()),
                                  [ast.Lambda(ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                                                          kw_defaults=[], defaults=[]), node.value)], [])
        return node


def _static_masks(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    tree.body[0].decorator_list = []
    tree = _StaticMasks().visit(tree)
    namespace = dict(fn.__globals__)
    namespace["_nomas_graph_constant"] = _constant
    namespace["flex_masked_attention"] = _dense
    namespace["_intra_seg_cache_forward"] = _intra
    exec(compile(ast.fix_missing_locations(tree), inspect.getsourcefile(fn), "exec"), namespace)
    return namespace[fn.__name__]


def _install_attention(cpt, decoder, dit=True):
    from nomas import attn
    original_engine = attn._engine
    def engine():
        active = _CURRENT.get()
        return active.engine if active is not None else original_engine()
    attn._engine = engine
    if dit and "dit" in attn._INSTALLED:
        original = cpt.causal_attention_with_cache_mp
        baseline = attn._ORIGINAL.get("causal")
        if baseline is None:
            baseline = original
            while hasattr(baseline, "__wrapped__"):
                baseline = baseline.__wrapped__
        if baseline.__module__ == cpt.__name__ and os.environ.get("NOMAS_GLUE") != "1":
            patched = _static_masks(baseline)
            @wraps(original)
            def attention(*args, **kwargs):
                if _CURRENT.get() is None:
                    return original(*args, **kwargs)
                bound = inspect.signature(baseline).bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                state = _CURRENT.get().histories.setdefault(id(values["kv_cache"]), {})
                token = attn._ACTIVE.set({"state": state, "length": values["cache_valid_len"]})
                try:
                    return patched(*args, **kwargs)
                finally:
                    attn._ACTIVE.reset(token)
            cpt.causal_attention_with_cache_mp = attention
        cpt.flex_masked_attention = _dense
        cpt._intra_seg_cache_forward = _intra
        ranges = sys.modules.get("hymm.models.dit.modules.attention.range_attention")
        if ranges is not None:
            ranges.flex_masked_attention = _dense
    if dit:
        cpt.CPTHYNomasDiffusionTransformer.get_rotary_pos_embed = _memoize(
            cpt.CPTHYNomasDiffusionTransformer.get_rotary_pos_embed, resident=True)
        cpt.get_rotary_pos_embed_with_offset = _memoize(cpt.get_rotary_pos_embed_with_offset, resident=True)
    decoder.mrope_tables = _memoize(decoder.mrope_tables)
    decoder.local_attention_layout = _memoize(decoder.local_attention_layout)


def _shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape), value.dtype, value.device
    if isinstance(value, (tuple, list)):
        return tuple(_shape(x) for x in value)
    if isinstance(value, dict):
        return tuple((k, _shape(v)) for k, v in value.items() if not k.startswith("_nomas"))
    return value


def _mask_key(mask, length=None):
    if mask is None:
        return None
    mask = mask if length is None else mask[:, :length]
    return bytes(mask.flatten().to(device="cpu", dtype=torch.uint8).tolist())


def _dit_key(values):
    parts = []
    for name, value in values.items():
        if name == "kv_cache_list" and value is not None:
            part = tuple((tuple(cache["k"].shape[2:]), cache["k"].dtype,
                          cache["k"].device, "swa_capacity" in cache) for cache in value)
        elif name == "cache_mask_list" and value is not None:
            part = tuple(_mask_key(mask, values["cache_valid_len"]) for mask in value)
        elif name == "text_mask":
            part = _shape(value), _mask_key(value)
        else:
            part = _shape(value)
        parts.append((name, part))
    return tuple(parts) + (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))


def _prepare_dit(model, runner, refresh):
    from nomas import glue
    values = runner.inputs
    tt, th, tw = (size // patch for size, patch in zip(values["hidden_states"].shape[2:], model.patch_size))
    offset = values["global_frame_counter"]
    if offset == 0:
        freqs = model.get_rotary_pos_embed((tt, th, tw), device=runner.device)
    else:
        from hymm.models.dit import nomas_cpt as cpt
        freqs = cpt.get_rotary_pos_embed_with_offset((tt, th, tw), offset, model.hidden_size,
            model.heads_num, model.rope_dim_list, model.rope_theta)
    for table in freqs:
        glue.device_tensor(table, runner.device)
    for block in model.double_blocks:
        for stream in ("img", "txt"):
            for kind in ("q", "k"):
                norm = getattr(block, stream + "_attn_" + kind + "_norm")
                if hasattr(norm, "weight"):
                    glue.device_tensor(norm.weight, runner.device)
    windows = [[int(a) * th * tw, int(b) * th * tw, int(c)]
               for a, b, c in values["intra_seg_vid_windows"] or []]
    offsets = values["intra_seg_txt_offsets"]
    for index, cache in enumerate(values["kv_cache_list"]):
        glue.prepare_attention(cache, values["cache_valid_len"], tt * th * tw,
            values["text_states"].shape[1], model.heads_num, values["text_mask"],
            None if values["cache_mask_list"] is None else values["cache_mask_list"][index],
            None if offsets is None else offsets[0], windows, runner.device, refresh=refresh)
    keyboard = values["camera_embedding"]
    if keyboard is not None and model.camera_injection_mode in ("action_attn", "extrinsics_attn"):
        for index, module in enumerate(model.action_modules[:len(model.double_blocks)]):
            if not module.per_chunk_camera_attn or module.camera_chunk_size != 1:
                continue
            module._nomas_camera = runner.camera_states.get(index)
            runner.camera_states[index] = glue.prepare_camera(module, keyboard,
                values["camera_has_mask"], tt, th, tw,
                rope_time_offset=values["camera_rope_time_offset"], refresh=refresh)


def _remember(cache, key, runner, limit):
    cache[key] = runner
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _install_dit(cls):
    original = cls.forward_with_cache
    signature = inspect.signature(original)
    @wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        if values["update_cache"]:
            return original(self, *args, **kwargs)
        if (getattr(self, "nomas_sparse_config", {}).get("enabled", False)
                or values.get("camera_inputs") is not None):
            raise ValueError("NOMAS_GRAPH requires dense attention and precomputed camera_embedding")
        if any(cache.get("k") is None or cache["k"].dtype not in (torch.bfloat16, torch.int8)
               for cache in values["kv_cache_list"] or []):
            raise ValueError("NOMAS_GRAPH supports resident BF16/INT8 history only")
        if values["guidance"] is None:
            guidance = getattr(self, "_nomas_graph_guidance", None)
            if guidance is None or guidance.device != values["hidden_states"].device:
                guidance = torch.full((1,), 6016.0, dtype=torch.bfloat16, device=values["hidden_states"].device)
                self._nomas_graph_guidance = guidance
            values["guidance"] = guidance
        active = getattr(self, "_nomas_dit_graph", None)
        source = (_tensor_key(values["text_states"]), values["cache_valid_len"], values["global_frame_counter"],
                  _shape(values["hidden_states"]), _shape(values["timestep"]), _shape(values["camera_embedding"]),
                  torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        boundary = (active is None or getattr(self, "_sparse_step_index", -1) == 0
                    or getattr(self, "_nomas_graph_source", None) != source)
        if boundary:
            if values.get("hide_cached_action_tails") and values.get("cache_mask_list") is not None:
                for mask in values["cache_mask_list"]:
                    for start, end in getattr(mask, "_action_tail_ranges", []):
                        mask[:, int(start):int(end)] = False
            values["hide_cached_action_tails"] = False
            key = _dit_key(values)
            cache = getattr(self, "_nomas_dit_graphs", OrderedDict())
            prepared = os.environ.get("NOMAS_GLUE") == "1" and values["cache_txt"]
            runner = cache.get(key) if prepared else None
            if runner is None:
                calibrated = getattr(self, "_nomas_graph_calibrated", False)
                runner = GraphRunner(lambda **kw: original(self, **kw), values,
                    ("hidden_states", "timestep"), "DiT",
                    prepare=(lambda r, refresh: _prepare_dit(self, r, refresh)) if prepared else None,
                    eager_first=not (prepared and calibrated))
                runner.key = key
            runner.bind(values)
            _remember(cache, key, runner, int(os.environ.get("NOMAS_GRAPH_DIT_KEYS", "4")))
            self._nomas_dit_graphs = cache
            self._nomas_dit_graph = runner
            self._nomas_graph_source = source
        output = self._nomas_dit_graph(**values)
        self._nomas_graph_calibrated = True
        return output
    cls.forward_with_cache = forward


def _install_decoder(cls):
    original = cls.forward
    signature = inspect.signature(original)
    @wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self")
        key = _shape(values), torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
        cache = getattr(self, "_nomas_decoder_graphs", OrderedDict())
        runner = cache.get(key)
        if runner is None:
            runner = GraphRunner(lambda **kw: original(self, **kw), values, ("latent", "xt"), "PixelUMM window")
            runner.key = key
        _remember(cache, key, runner, int(os.environ.get("NOMAS_GRAPH_DECODER_KEYS", "2")))
        self._nomas_decoder_graphs = cache
        self._nomas_decoder_graph = runner
        output = runner(**values)
        return output.clone() if runner.graph is not None else output
    cls.forward = forward


class _HotPath(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if not isinstance(call, ast.Call):
            return self.generic_visit(node)
        name = ast.unparse(call.func)
        if name in {"torch.cuda.synchronize", "self._nomas_sync"}:
            return ast.copy_location(ast.Pass(), node)
        if name.startswith("logger.") or name == "_seg_profiler.new_step":
            if name == "_seg_profiler.new_step" or any(
                    isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute)
                    and part.func.attr == "item" for part in ast.walk(call)):
                return ast.copy_location(ast.Pass(), node)
            if name.startswith("logger.") and "num_inference_steps" in ast.unparse(call):
                return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)


def _install_pipeline(cls):
    source = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    method = next(node for node in source.body[0].body
                  if isinstance(node, ast.FunctionDef) and node.name == "_process_single_prompt")
    method.decorator_list = []
    tree = ast.Module(body=[method], type_ignores=[])
    if os.environ.get("NOMAS_PIPE_ASYNC_DECODE") == "1":
        from nomas.pipe import _BoundaryPatch
        patch = _BoundaryPatch()
        tree = patch.visit(tree)
        if patch.boundaries != 1:
            raise RuntimeError("NOMAS_GRAPH cannot locate the async decoder boundary")
    tree = ast.fix_missing_locations(_HotPath().visit(tree))
    namespace = dict(sys.modules[cls.__module__].__dict__)
    exec(compile(tree, inspect.getsourcefile(cls), "exec"), namespace)
    patched = torch.no_grad()(namespace[method.name])
    current = cls._process_single_prompt
    cells = dict(zip(current.__code__.co_freevars, current.__closure__ or ()))
    if "original_process" in cells:
        cells["original_process"].cell_contents = patched
        current.__wrapped__ = patched
    else:
        cls._process_single_prompt = patched


def _install_cache(cls):
    original = cls._initialize_kv_cache
    signature = inspect.signature(original)
    @wraps(original)
    def initialize(self, *args, **kwargs):
        if (getattr(self, "qvg_kv_storage", False)
                or getattr(self.args.inference, "interactive_kv_cache_layerwise_cpu_offload", False)):
            raise ValueError("NOMAS_GRAPH requires resident BF16/INT8 KV, not QVG or CPU cache offload")
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        key = tuple((name, value) for name, value in bound.arguments.items() if name != "self")
        key += (self.int8_kv_storage,)
        saved = getattr(self, "_nomas_graph_kv", None)
        if saved is None or saved[0] != key:
            original(self, *args, **kwargs)
            self._nomas_graph_kv = key, self.kv_cache, self.cache_mask_list
            return
        _, self.kv_cache, self.cache_mask_list = saved
        for cache in self.kv_cache:
            for name in tuple(cache):
                if name not in {"k", "v", "k_scale", "v_scale", "storage_format"}:
                    del cache[name]
        for mask in self.cache_mask_list:
            mask.zero_()
            if hasattr(mask, "_action_tail_ranges"):
                mask._action_tail_ranges = []
        self.cache_valid_len = 0
        self.global_frame_counter = 0
        self.segment_token_lengths = []
    cls._initialize_kv_cache = initialize


def _patch_timestep_embedding():
    import math
    from hymm.models.dit.modules import embed_layers as emb
    table = {}

    def timestep_embedding(t, dim, max_period=10000):
        key = (dim, max_period, t.device)
        freqs = table.get(key)
        if freqs is None:
            half = dim // 2
            freqs = table[key] = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        args = t.flatten()[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.view(*t.shape, dim)
    emb.timestep_embedding = timestep_embedding


def _patch_no_pad_attention():
    from hymm.models.dit.modules import attention as att

    def no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False):
        q, k, v = (x.transpose(1, 2) for x in qkv.unbind(2))
        mask = key_padding_mask.bool()[:, None, None, :]
        if causal:
            positions = torch.arange(q.shape[2], device=q.device)
            mask = mask & (positions[:, None] >= positions[None, :])
        dtype = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        out = torch.nn.functional.scaled_dot_product_attention(q.to(dtype), k.to(dtype), v.to(dtype),
            attn_mask=mask, dropout_p=dropout_p, is_causal=False, scale=softmax_scale)
        return (out.transpose(1, 2) * key_padding_mask[:, :, None, None]).to(qkv.dtype)
    att.flash_attn_no_pad = att.flash_attn_no_pad_v3 = no_pad


def install():
    global _installed
    if os.environ.get("NOMAS_GRAPH") != "1" or _installed:
        return False
    dit = os.environ.get("NOMAS_GRAPH_DIT", "1")
    if dit not in ("0", "1"):
        raise ValueError("NOMAS_GRAPH_DIT must be 0 or 1")
    if dit == "1":
        _patch_timestep_embedding()
        _patch_no_pad_attention()
    from hymm.models.dit import nomas_cpt as cpt
    from hymm.models.vae.pixelumm_decoder import model as decoder
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    from nomas import attn
    required = {"dit", "decoder"} if dit == "1" else {"decoder"}
    if not required.issubset(attn._INSTALLED):
        raise ValueError("NOMAS_GRAPH requires NOMAS_DEC_ATTN=fa and, for DiT graphs, NOMAS_ATTN=fa")
    if os.environ.get("NOMAS_GLUE") == "1":
        glue = sys.modules.get("nomas.glue")
        if glue is None or not glue._INSTALLED:
            raise ValueError("Install NOMAS_GLUE before NOMAS_GRAPH")
    if any(int(os.environ.get(name, default)) < 1 for name, default in
           (("NOMAS_GRAPH_DIT_KEYS", "4"), ("NOMAS_GRAPH_DECODER_KEYS", "2"))):
        raise ValueError("NOMAS_GRAPH key limits must be positive")
    _install_attention(cpt, decoder, dit=dit == "1")
    if dit == "1":
        _install_dit(cpt.CPTHYNomasDiffusionTransformer)
    _install_decoder(decoder.DCAEVPixelUMMDecoder)
    if dit == "1":
        _install_pipeline(CPTCausalInferencePipeline)
        _install_cache(CPTCausalInferencePipeline)
    _installed = True
    return True
