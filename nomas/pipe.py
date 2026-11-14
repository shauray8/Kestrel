import ast
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import gc
import inspect
import json
import os
from types import MethodType
import textwrap
import time
import weakref

import torch
from loguru import logger


_installed = False
_pipeline = None
_ALL_PROMPTS = []


def _report(name, started, device, **values):
    torch.cuda.current_stream(device).synchronize()
    values.update(stage=name, wall_seconds=time.perf_counter() - started,
                  allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                  reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
    logger.info("NOMAS_PIPE " + json.dumps(values, sort_keys=True))


def _remove_offload(module):
    if hasattr(module, "_model_cpu_offload"):
        delattr(module, "_model_cpu_offload")


def _key(prompt, data_type="video", is_uncond=False):
    return prompt, data_type, bool(is_uncond)


def _prepare_text(self, prompts, negative_prompt):
    if self._nomas_text_ready:
        return
    started = time.perf_counter()
    encoder = self.text_encoder
    _remove_offload(encoder)
    encoder.to(device=self.device)
    encoder.device = self.device
    _report("text_encoder_H2D_once", started, self.device)
    started = time.perf_counter()
    requests = list(dict.fromkeys([_key(p) for p in prompts] + [_key("")]))
    if (float(self.args.inference.cfg_scale_video) > 1
            or float(getattr(self.args.inference, "camera_cfg_scale", 1)) > 1):
        requests.append(_key(negative_prompt or "", is_uncond=True))
    with torch.no_grad():
        for prompt, data_type, is_uncond in requests:
            states, mask = self._nomas_original_encode(prompt, data_type, is_uncond)
            self._nomas_text_cache[_key(prompt, data_type, is_uncond)] = (
                states.detach().cpu(), None if mask is None else mask.detach().cpu())
            del states, mask
    _report("encode_all_text_once", started, self.device,
            cache_entries=len(self._nomas_text_cache))
    started = time.perf_counter()
    del encoder.model
    gc.collect()
    torch.cuda.empty_cache()
    self._nomas_text_ready = True
    _report("free_text_encoder", started, self.device)
    started = time.perf_counter()
    self.transformer.to(device=self.device)
    self.vae.to(device=self.device)
    _report("resident_models_H2D_once", started, self.device)


def _cached_encode(self, prompt, data_type="video", is_uncond=False):
    key = _key(prompt, data_type, is_uncond)
    if key not in self._nomas_text_cache:
        raise RuntimeError("NOMAS_PIPE text cache miss after encoder release: " + repr((prompt[:300], data_type, is_uncond))
                           + " cached=" + repr([(k[0][:120], k[1], k[2]) for k in self._nomas_text_cache]))
    states, mask = self._nomas_text_cache[key]
    return (states.to(device=self.device, dtype=self.target_dtype),
            None if mask is None else mask.to(device=self.device, dtype=self.target_dtype))


def _decode_windows_cpu(session, starts):
    model = session.model
    try:
        return model._predict_x0(
            session.buffer, model._noise_tile(session.buffer), all_ranks=False,
            t_starts=starts, pad_short=session._tail_fix(), output_cpu=True)
    finally:
        model._noise_cache = None


class _DecodeJob:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.stream = torch.cuda.Stream(device=pipeline.device)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.session = None
        self.parts = []
        self.future = None
        self.started = time.perf_counter()
        self.inference_mode = torch.is_inference_mode_enabled()
        self.autocast = torch.is_autocast_enabled("cuda")
        self.autocast_dtype = torch.get_autocast_dtype("cuda")

    def push(self, latents, pixel_frames, final):
        owned = latents.detach().clone()
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self.pipeline.device))
        previous = self.future
        self.future = self.pool.submit(self._push, owned, ready, pixel_frames, final, previous)

    def _push(self, latents, ready, pixel_frames, final, previous):
        if previous is not None:
            previous.result()
        p = self.pipeline
        started = time.perf_counter()
        with torch.cuda.device(p.device), torch.cuda.stream(self.stream), \
                torch.inference_mode(self.inference_mode), torch.no_grad(), \
                torch.autocast("cuda", enabled=self.autocast, dtype=self.autocast_dtype):
            self.stream.wait_event(ready)
            latents.record_stream(self.stream)
            if self.session is None:
                self.session = p.vae.create_streaming_decode_session(to_cpu=True, retain_history=False)
                self.session._decode_windows = MethodType(_decode_windows_cpu, self.session)
            stable = self.session.push(p._prepare_vae_decode_latents(latents), pixel_frames=pixel_frames)
            if stable.numel():
                self.parts.append(p._postprocess_vae_video(stable).cpu())
            if final:
                tail = self.session.finish()
                if tail.numel():
                    self.parts.append(p._postprocess_vae_video(tail).cpu())
            self.stream.synchronize()
            _report("async_decode_segment", started, p.device, frames=pixel_frames, final=final)

    def result(self, num_frames):
        if self.future is None:
            raise RuntimeError("NOMAS_PIPE async decoder received no segment")
        self.future.result()
        self.pool.shutdown(wait=True)
        video = torch.cat(self.parts, dim=2)
        if video.shape[2] != num_frames:
            raise RuntimeError(f"NOMAS_PIPE async output has {video.shape[2]} frames, expected {num_frames}")
        _report("async_decode_finish", self.started, self.pipeline.device, frames=num_frames)
        return video

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.stream.synchronize()


def _boundary(self, latents, latent_end, num_frames, final):
    job = getattr(self, "_nomas_decode_job", None)
    if job is not None:
        end = num_frames if final else min(num_frames, latent_end * self.t_ratio)
        job.push(latents, end - self._nomas_decode_pixel_end, final)
        self._nomas_decode_pixel_end = end


def _sync(self, *args, **kwargs):
    if getattr(self, "_nomas_decode_job", None) is not None:
        torch.cuda.current_stream(self.device).synchronize()
    else:
        torch.cuda.synchronize(*args, **kwargs)


class _BoundaryPatch(ast.NodeTransformer):
    def __init__(self):
        self.boundaries = 0

    def visit_With(self, node):
        node = self.generic_visit(node)
        for item in node.items:
            context = item.context_expr
            if (isinstance(context, ast.Call) and isinstance(context.func, ast.Name)
                    and context.func.id == "stage" and context.args
                    and isinstance(context.args[0], ast.Constant)
                    and context.args[0].value == "cache_write"):
                self.boundaries += 1
                extra = ast.parse(
                    "if is_last_chunk_of_seg:\n"
                    "    noise_pred = output = latent_input = cache_input = None\n"
                    "    cache_latents = cache_snapshot_latents = cache_cond_latents = None\n"
                    "    text_states_cfg = text_mask_cfg = chunk_cam_emb = chunk_cam_mask = None\n"
                    "    cache_write_text_states = cache_write_text_mask = None\n"
                    "    cache_write_intra_offsets = cache_write_cam_emb = cache_write_cam_mask = None\n"
                    "    self._nomas_boundary(latents, chunk_latent_end, num_frames, is_last_chunk)"
                ).body
                return [node, *extra]
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "synchronize"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda"
                and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "torch"):
            node.func = ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr="_nomas_sync", ctx=ast.Load())
        return node


def _patch_boundaries(method):
    original = inspect.unwrap(method)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree.body[0].decorator_list = []
    patch = _BoundaryPatch()
    tree = ast.fix_missing_locations(patch.visit(tree))
    if patch.boundaries != 1:
        raise RuntimeError(f"NOMAS_PIPE expected one cache_write stage, found {patch.boundaries}")
    namespace = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), namespace)
    return torch.no_grad()(namespace[original.__name__])


def install():
    global _installed
    if os.environ.get("NOMAS_PIPE") != "1" or _installed:
        return
    import hymm.inference as inference
    from hymm.pipeline.base import BasePipeline
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    from omegaconf import open_dict

    async_decode = os.environ.get("NOMAS_PIPE_ASYNC_DECODE") == "1"
    compile_mode = os.environ.get("NOMAS_PIPE_COMPILE_MODE")
    if compile_mode:
        if compile_mode not in {"default", "max-autotune-no-cudagraphs"}:
            raise ValueError("NOMAS_PIPE_COMPILE_MODE must be default or max-autotune-no-cudagraphs; cache mutation and external kernels are not graph-capture safe")
        from hymm.models.dit.modules import nomas_compile_prepost as compiled
        for name, fn in (("compiled_pre_attention", compiled.pre_attention),
                         ("compiled_post_attention", compiled.post_attention)):
            def factory(fn=fn):
                if compile_mode == "default":
                    return torch.compile(fn, fullgraph=True, options={"triton.cudagraphs": False})
                return torch.compile(fn, fullgraph=True, mode=compile_mode)
            wrapped = lru_cache(maxsize=1)(factory)
            setattr(compiled, name, wrapped)
            import hymm.models.dit.nomas_cpt as cpt
            setattr(cpt, name, wrapped)

    original_load = BasePipeline._load_models
    original_encode = BasePipeline._encode_single_prompt
    original_loader = inference._load_prompt_entries
    original_decode = BasePipeline._decode_latents
    original_process = CPTCausalInferencePipeline._process_single_prompt
    if async_decode:
        original_process = _patch_boundaries(original_process)

    @wraps(original_load)
    def load(self, *args, **kwargs):
        global _pipeline
        if self.device.type != "cuda" or int(self.args.parallelism.sp_size) != 1:
            raise ValueError("NOMAS_PIPE requires one CUDA GPU and sp_size=1")
        if self.args.inference.rollout_type != "seg_ar" or self.args.vae.name != "pixelumm":
            raise ValueError("NOMAS_PIPE requires seg_ar and PixelUMM")
        if not (getattr(self.args.vae, "cutlass_fp8", False)
                or getattr(self.args.vae, "deepgemm_fp8", False)):
            raise ValueError("NOMAS_PIPE resident mode requires FP8 PixelUMM")
        from hymm.models.vae.pixelumm_decoder.prequantized import is_prequantized_pixelumm
        if not is_prequantized_pixelumm(self.args.vae.checkpoint):
            raise ValueError("NOMAS_PIPE requires serialized FP8 PixelUMM weights to avoid GPU conversion before text encoding")
        with open_dict(self.args):
            self.args.parallelism.dit_cpu_offload = True
            self.args.parallelism.dit_cpu_offload_for_decode = False
            self.args.parallelism.text_encoder_cpu_offload = True
            self.args.parallelism.shard_text_encoder = False
            self.args.vae.cpu_offload = True
        original_load(self, *args, **kwargs)
        for model in (self.transformer, self.vae, self.text_encoder):
            _remove_offload(model)
        with open_dict(self.args):
            self.args.parallelism.dit_cpu_offload = False
            self.args.parallelism.text_encoder_cpu_offload = False
            self.args.vae.cpu_offload = False
        if os.environ.get("NOMAS_DEC_GPU_OUT") == "1":
            for obj in [self.vae, *getattr(self.vae, "modules", lambda: [])()]:
                if getattr(obj, "decode_output_cpu", None) is True:
                    obj.decode_output_cpu = False
        self._nomas_text_cache = {}
        self._nomas_text_ready = False
        self._nomas_original_encode = original_encode.__get__(self, type(self))
        _pipeline = weakref.ref(self)

    @wraps(original_loader)
    def loader(*args, **kwargs):
        global _ALL_PROMPTS
        entries = original_loader(*args, **kwargs)
        _ALL_PROMPTS = [part.strip() for entry in entries for part in entry[0].split(inference.PROMPT_SEPARATOR)]
        p = _pipeline() if _pipeline is not None else None
        if p is not None:
            _prepare_text(p, _ALL_PROMPTS, p.args.inference.neg_prompt)
        return entries

    @wraps(original_decode)
    def decode(self, latents, num_frames):
        job = getattr(self, "_nomas_decode_job", None)
        return job.result(num_frames) if job is not None else original_decode(self, latents, num_frames)

    @wraps(original_process)
    def process(self, *args, **kwargs):
        bound = inspect.signature(original_process).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        _prepare_text(self, list(dict.fromkeys(_ALL_PROMPTS + list(values["prompts"]))), values["negative_prompt"])
        if async_decode:
            if (values["preserve_kv_cache"] or values["segment_callback"] is not None
                    or bool(getattr(self.args.inference, "seg_ar_segment_vae_decode", False))
                    or bool(getattr(self.args.inference, "profile_memory", False))
                    or getattr(self.vae, "_distributed_decode_mode", "none") != "none"):
                raise ValueError("NOMAS_PIPE_ASYNC_DECODE requires offline seg_ar, no segment callbacks, segment VAE disabled, profile_memory=false, and PixelUMM parallel_mode=none")
            self._nomas_decode_job = _DecodeJob(self)
            self._nomas_decode_pixel_end = 0
        started = time.perf_counter()
        try:
            return original_process(self, *args, **kwargs)
        finally:
            job = getattr(self, "_nomas_decode_job", None)
            if job is not None:
                job.close()
                self._nomas_decode_job = None
            _report("prompt_total", started, self.device)

    BasePipeline._load_models = load
    BasePipeline._encode_single_prompt = _cached_encode
    BasePipeline._decode_latents = decode
    CPTCausalInferencePipeline._process_single_prompt = process
    inference._load_prompt_entries = loader
    CPTCausalInferencePipeline._nomas_boundary = _boundary
    CPTCausalInferencePipeline._nomas_sync = _sync
    _installed = True
