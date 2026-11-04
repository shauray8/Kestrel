import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch


sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
               os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas")]


def compare(label, actual, expected, args):
    delta = actual.float() - expected.float()
    if not torch.isfinite(actual).all():
        raise AssertionError(label)
    print(json.dumps({"case": label, "max_abs": float(delta.abs().max()),
                      "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12))}), flush=True)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)


def timing(fn, repeats):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return {"cuda_ms_per_step": start.elapsed_time(end) / repeats,
            "wall_ms_per_step": 1000 * (time.perf_counter() - wall) / repeats}


def cache_blocks(model, capacity, history, quantized):
    from hymm.models.dit.modules.int8_kv_cache import make_int8_kv_block, quantize_kv_pair_into
    blocks, masks = [], []
    for _ in model.double_blocks:
        shape = (1, capacity, model.heads_num, 128)
        if quantized:
            block = make_int8_kv_block(1, capacity, model.heads_num, 128, device="cuda")
            if history:
                k = torch.randn((1, history, model.heads_num, 128), device="cuda", dtype=torch.bfloat16)
                quantize_kv_pair_into(block, k, torch.randn_like(k), 0)
        else:
            block = {name: torch.randn(shape, device="cuda", dtype=torch.bfloat16) for name in ("k", "v")}
        mask = torch.zeros((1, capacity), device="cuda", dtype=torch.bool)
        mask[:, :history] = True
        if history:
            mask[:, 13:29] = False
        blocks.append(block)
        masks.append(mask)
    return blocks, masks


def dit_check(args):
    from hymm.models.dit.nomas_cpt import CPTHYNomasDiffusionTransformer
    from nomas.gemm import _replace_linears
    model = CPTHYNomasDiffusionTransformer(
        in_channels=32, out_channels=32, hidden_size=args.hidden_size,
        heads_num=args.hidden_size // 128, mm_double_blocks_depth=args.blocks,
        mm_single_blocks_depth=0, text_projection=args.text_projection, text_states_dim=256,
        rope_dim_list=[16, 56, 56], seg_ar_attention_backend="flex",
        camera_injection_mode="extrinsics_attn" if args.camera else None,
        action_config=dict(keyboard_dim_in=64, hidden_size=128, img_hidden_size=args.hidden_size,
            keyboard_hidden_dim=args.hidden_size, heads_num=args.hidden_size // 128,
            rope_dim_list=[16, 56, 56], per_chunk_camera_attn=True, camera_chunk_size=1) if args.camera else None,
        device="cuda", dtype=torch.bfloat16).to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    model.final_layer.linear.weight.normal_(0, args.hidden_size ** -0.5)
    for block in model.double_blocks:
        for modulation in (block.img_mod, block.txt_mod):
            modulation.linear.weight.normal_(0, 0.02)
            modulation.linear.bias.normal_(0, 0.02)
    _replace_linears(model)
    eager = type(model).forward_with_cache.__wrapped__
    frames, height, width = args.latent_size
    tokens = frames * (height // 2) * (width // 2)
    text = torch.randn((1, args.text_tokens, 256), device="cuda", dtype=torch.bfloat16)
    text_mask = torch.ones((1, args.text_tokens), device="cuda", dtype=torch.bool)
    text_mask[:, -3:] = False
    text_mask[:, 5] = False
    seen = {}
    calibrated = None
    for clip in range(2):
        text = torch.randn_like(text)
        keyboard = torch.randn((1, 2 * frames, 64), device="cuda", dtype=torch.bfloat16)
        for segment, history in enumerate((0, args.history)):
            caches, masks = cache_blocks(model, history + tokens + args.text_tokens, history, args.int8_history)
            inputs = dict(hidden_states=torch.randn((1, 32, frames, height, width), device="cuda", dtype=torch.bfloat16),
                          timestep=torch.full((1,), 1000.0, device="cuda"), text_states=text,
                          text_mask=text_mask, kv_cache_list=caches, cache_mask_list=masks,
                          cache_valid_len=history, global_frame_counter=segment * frames,
                          update_cache=False, cache_txt=True)
            if args.camera:
                inputs.update(camera_embedding=keyboard, camera_rope_time_offset=segment * frames,
                              camera_has_mask=torch.tensor([bool((clip + segment) % 2)], device="cuda"))
            if args.intra:
                valid = args.text_tokens - 4
                inputs.update(intra_seg_txt_offsets=[[4, valid // 2, valid]],
                              intra_seg_vid_windows=[[0, frames // 2, 0], [frames // 2, frames, 1]])
            before = seen[segment].replays if args.glue and segment in seen else 0
            for step in range(args.steps):
                model._sparse_step_index = step
                inputs["hidden_states"].normal_()
                inputs["timestep"].fill_(1000.0 * (1 - step / args.steps))
                if calibrated is None:
                    eager(model, **inputs)
                expected = eager(model, **inputs)
                actual = model.forward_with_cache(**inputs)
                if calibrated is None:
                    calibrated = {name: value.clone() for name, value in model.named_buffers()
                                  if name.endswith("activation_scale")}
                else:
                    for name, value in model.named_buffers():
                        if name in calibrated:
                            torch.testing.assert_close(value, calibrated[name], rtol=0, atol=0)
                if actual[1] != expected[1]:
                    raise AssertionError("DiT replay changed cache-token metadata")
                compare(f"dit_clip{clip}_segment{segment}_step{step}", actual[0], expected[0], args)
            runner = model._nomas_dit_graph
            if runner.graph is None:
                raise AssertionError("DiT forward was not captured")
            if args.glue and clip and runner is not seen[segment]:
                raise AssertionError("Same-shape denoising graph was not reused across prompts")
            expected_replays = args.steps - int(not args.glue or (clip == 0 and segment == 0))
            if runner.replays - before != expected_replays:
                raise AssertionError("Unexpected eager warmup after process calibration")
            seen[segment] = runner
            print(json.dumps({"case": f"dit_clip{clip}_segment{segment}", "video_tokens": tokens,
                              "history": history, "blocks": args.blocks, "captured_steps": expected_replays,
                              "eager": timing(lambda: eager(model, **inputs), args.repeats),
                              "graph": timing(lambda: model.forward_with_cache(**inputs), args.repeats)}), flush=True)
            graph = runner.graph
            model.forward_with_cache(**dict(inputs, update_cache=True))
            if model._nomas_dit_graph is not runner or runner.graph is not graph:
                raise AssertionError("Eager cache commit must preserve reusable denoising graphs")
    del model, seen, runner, graph, caches, masks, inputs, expected, actual, calibrated
    gc.collect()
    torch.cuda.empty_cache()


def decoder_check(args):
    from hymm.models.vae.pixelumm_decoder.model import DCAEVPixelUMMDecoder, GenExpertArch
    from nomas.gemm import _replace_linears
    model = DCAEVPixelUMMDecoder(GenExpertArch(hidden_size=256, intermediate_size=768,
        num_layers=args.decoder_blocks, num_heads=2, num_kv_heads=1),
        local_attention_tile=args.decoder_tile).cuda().bfloat16().eval().requires_grad_(False)
    _replace_linears(model)
    eager = type(model).forward.__wrapped__
    frames, height, width = args.decoder_latent_size
    inputs = dict(latent=torch.randn((1, 32, frames, height, width), device="cuda", dtype=torch.bfloat16),
                  xt=torch.randn((1, 3, frames * 4, height * 32, width * 32), device="cuda", dtype=torch.bfloat16))
    saved = None
    for window in range(8):
        inputs["latent"].normal_()
        inputs["xt"].normal_()
        expected = eager(model, **inputs)
        actual = model(**inputs)
        compare(f"decoder_window{window}", actual, expected, args)
        if window == 3:
            first, saved = actual, actual.clone()
    torch.testing.assert_close(first, saved, rtol=0, atol=0)
    runner = model._nomas_decoder_graph
    if runner.graph is None or runner.replays != 8 - runner.warmup_steps:
        raise AssertionError("Decoder window forward was not captured")
    print(json.dumps({"case": "decoder_window", "latent_shape": list(inputs["latent"].shape),
                      "local_tile": args.decoder_tile, "captured_windows": runner.replays,
                      "eager": timing(lambda: eager(model, **inputs), args.repeats),
                      "graph": timing(lambda: model(**inputs), args.repeats)}), flush=True)
    changed = dict(latent=inputs["latent"][:, :, :1].contiguous(),
                   xt=inputs["xt"][:, :, :4].contiguous())
    expected = eager(model, **changed)
    actual = model(**changed)
    compare("decoder_tail_shape", actual, expected, args)
    if model._nomas_decoder_graph is runner:
        raise AssertionError("Decoder tail shape reused a mismatched graph")
    inputs["latent"].normal_()
    inputs["xt"].normal_()
    compare("decoder_return_to_regular_window", model(**inputs), eager(model, **inputs), args)
    if model._nomas_decoder_graph is not runner:
        raise AssertionError("Returning to the regular shape must reuse its graph")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--latent-size", type=int, nargs=3, default=[4, 16, 16])
    parser.add_argument("--text-tokens", type=int, default=32)
    parser.add_argument("--history", type=int, default=192)
    parser.add_argument("--int8-history", action="store_true")
    parser.add_argument("--intra", action="store_true")
    parser.add_argument("--glue", action="store_true", default=True)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--text-projection", choices=("linear", "single_refiner"), default="single_refiner")
    parser.add_argument("--no-glue", dest="glue", action="store_false")
    parser.add_argument("--decoder-blocks", type=int, default=2)
    parser.add_argument("--decoder-latent-size", type=int, nargs=3, default=[2, 5, 7])
    parser.add_argument("--decoder-tile", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("graph_check requires the RTX CUDA runtime; no CPU substitute")
    if (args.hidden_size % 128 or min(args.latent_size) < 2 or args.text_tokens < 16
            or args.steps < 2 or args.decoder_latent_size[0] < 2):
        raise ValueError("Use hidden_size divisible by 128, latent dimensions >=2, text_tokens>=16, steps>=2")
    os.environ.update(NOMAS_ATTN="fa", NOMAS_DEC_ATTN="fa", NOMAS_GEMM="n3",
                      NOMAS_DEC_GEMM="n3", NOMAS_GRAPH="1")
    os.environ["NOMAS_GLUE"] = "1" if args.glue else "0"
    from nomas import attn, gemm, graph
    from hymm.models.dit.nomas_cpt import CPTHYNomasDiffusionTransformer
    original_dit = CPTHYNomasDiffusionTransformer.forward_with_cache
    attn.install()
    gemm.install()
    if args.glue:
        from nomas import glue
        glue.install()
    graph.install()
    if graph.install():
        raise AssertionError("Graph install must be idempotent")
    attn._engine().bal = False
    torch.manual_seed(29)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if os.environ.get("NOMAS_GRAPH_DIT", "1") == "1":
            dit_check(args)
        elif CPTHYNomasDiffusionTransformer.forward_with_cache is not original_dit:
            raise AssertionError("Decoder-only graphs must not wrap the DiT forward")
        decoder_check(args)
    print(json.dumps({"case": "pass", "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


if __name__ == "__main__":
    main()
