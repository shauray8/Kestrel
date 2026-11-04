import argparse
import ast
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas"))

import torch
from loguru import logger
from omegaconf import OmegaConf
from hymm.models.vae.pixelumm_decoder import maybe_load_pixelumm_decoder
from hymm.pipeline.base import BasePipeline
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
from nomas.pipe import _BoundaryPatch, _DecodeJob, _cached_encode, _key, _patch_boundaries


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--frames", type=int, default=240)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--width", type=int, default=832)
args = parser.parse_args()
tree = ast.parse("def f():\n    with stage('cache_write'):\n        pass\n    segment_decode_elapsed = 0.0\n    segment_decode_elapsed = time.time() - segment_decode_start\n")
patch = _BoundaryPatch()
patched_tree = patch.visit(tree)
assert patch.boundaries == 1
assert sum(isinstance(n, ast.Attribute) and n.attr == "_nomas_boundary" for n in ast.walk(patched_tree)) == 1
_patch_boundaries(CPTCausalInferencePipeline._process_single_prompt)
config = OmegaConf.create({"vae": {"name": "pixelumm", "checkpoint": args.checkpoint,
    "cutlass_fp8": True, "cpu_offload": True, "parallel_mode": "none",
    "num_steps": 1, "temporal_window_latents": 10, "temporal_overlap_latents": 5}})
vae = maybe_load_pixelumm_decoder(config, device=torch.device("cuda"), logger=logger)
vae.eval().requires_grad_(False)
p = SimpleNamespace(vae=vae, device=torch.device("cuda"), target_dtype=torch.bfloat16)
p._prepare_vae_decode_latents = lambda x: BasePipeline._prepare_vae_decode_latents(p, x)
p._postprocess_vae_video = BasePipeline._postprocess_vae_video
latent_frames = (args.frames - 1) // 4 + 1

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    torch.manual_seed(42)
    latent = torch.randn(1, 32, latent_frames, args.height // 32, args.width // 32,
                         device="cuda", dtype=torch.bfloat16)
    states = torch.randn(1, 256, 3584, dtype=torch.bfloat16)
    mask = torch.ones(1, 256, dtype=torch.bfloat16)
    p._nomas_text_cache = {_key("test"): (states, mask)}
    cached, cached_mask = _cached_encode(p, "test")
    torch.testing.assert_close(cached.cpu(), states, rtol=0, atol=0)
    torch.testing.assert_close(cached_mask.cpu(), mask, rtol=0, atol=0)
    for iteration in range(2):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        reference = vae.decode(p._prepare_vae_decode_latents(latent), return_dict=False,
                               pixel_frames=args.frames)[0]
        reference = p._postprocess_vae_video(reference).cpu()
        torch.cuda.synchronize()
        serial_seconds = time.perf_counter() - started
        serial_peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        job = _DecodeJob(p)
        try:
            split = latent_frames // 2
            job.push(latent[:, :, :split], split * 4, False)
            job.push(latent[:, :, split:], args.frames - split * 4, True)
            actual = job.result(args.frames)
        finally:
            job.close()
        torch.cuda.synchronize()
        async_seconds = time.perf_counter() - started
        error = (actual.float() - reference.float()).abs()
        print(dict(iteration=iteration, shape=list(actual.shape),
                   serial_seconds=serial_seconds, streaming_seconds=async_seconds,
                   max_abs=float(error.max()), mean_abs=float(error.mean()),
                   serial_peak_gib=serial_peak,
                   streaming_peak_gib=torch.cuda.max_memory_allocated() / 2**30))
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.004)
