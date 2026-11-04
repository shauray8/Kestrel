import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas"))

from kestrel import ops
from nomas import attn, gemm, glue
from hymm.models.dit import nomas_cpt
from hymm.models.dit.modules.norm_layers import RMSNorm
from hymm.models.dit.modules.posemb_layers import apply_rotary_emb
from hymm.models.dit.modules.int8_kv_cache import make_int8_kv_block, quantize_kv_pair_into, dequantize_kv
from hymm.models.vae.pixelumm_decoder import model as decoder
from hymm.models.camera.action_module import ActionModule


def compare(name, expected, actual, limit=.005, atol=.04):
    if isinstance(expected, tuple):
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare(name + "/" + str(i), a, b, limit, atol)
        return
    a, b = expected.float(), actual.float()
    difference = a - b
    rel = float(difference.norm() / a.norm().clamp_min(1.e-12))
    maximum = float(difference.abs().max())
    record = dict(case=name, shape=list(a.shape), relative_l2=rel,
                  max_absolute=maximum, exact_equal=torch.equal(expected, actual),
                  finite=bool(torch.isfinite(b).all()))
    invalid = not record["finite"] or (rel > limit and maximum > atol)
    if expected.dtype == torch.float8_e4m3fn and actual.dtype == expected.dtype:
        magnitude = torch.maximum(a.abs(), b.abs()).clamp_min(2. ** -6)
        spacing = torch.exp2(torch.floor(torch.log2(magnitude)) - 3)
        record["max_fp8_ulp"] = float((difference.abs() / spacing).max())
        invalid = not record["finite"] or record["max_fp8_ulp"] > 1 or rel > .02
    print(json.dumps(record), flush=True)
    if invalid:
        raise AssertionError(record)


def bench(name, baseline, candidate, repeats):
    for _ in range(3):
        baseline()
        candidate()
    result = dict(case=name)
    for label, fn in (("original", baseline), ("glue", candidate)):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        a.record()
        for _ in range(repeats):
            fn()
        b.record()
        b.synchronize()
        result[label + "_gpu_ms"] = a.elapsed_time(b) / repeats
        result[label + "_wall_ms"] = (time.perf_counter() - start) * 1000 / repeats
    result["speedup"] = result["original_wall_ms"] / result["glue_wall_ms"]
    print(json.dumps(result), flush=True)


def tables(tokens, dim=128):
    angle = torch.randn(tokens, dim // 2, device="cuda").repeat_interleave(2, -1)
    return angle.cos(), angle.sin()


def modulation_check(args):
    for amp in (True, False):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            for tokens in (args.text_tokens, 11700):
                for dtype, modulation_dtype in ((torch.float32, torch.float32),
                        (torch.float32, torch.bfloat16), (torch.bfloat16, torch.bfloat16)):
                    x = torch.randn(1, tokens, 2048, device="cuda", dtype=dtype)
                    packed = torch.randn(1, 6 * 2048, device="cuda", dtype=modulation_dtype) * .1
                    shift, scale = packed.chunk(6, -1)[:2]
                    reference = lambda: F.layer_norm(x, (2048,), eps=1.e-6) * (1 + scale[:, None]) + shift[:, None]
                    candidate = lambda: glue.mod_norm(x, shift, scale, 1.e-6, 0.)
                    name = f"layernorm_mod/{tokens}/{dtype}/{modulation_dtype}/autocast{amp}"
                    expected, actual = reference(), candidate()
                    if expected.dtype != actual.dtype:
                        raise AssertionError(f"{name}: dtype {actual.dtype} != {expected.dtype}")
                    compare(name, expected, actual)
                    bench(name, reference, candidate, args.repeats)
                    sa = .25
                    quantized = lambda: (reference().bfloat16().float() / sa).to(torch.float8_e4m3fn)
                    fused_quantized = lambda: glue.mod_norm(x, shift, scale, 1.e-6, sa)
                    compare(name + "/fp8", quantized(), fused_quantized(), .01)
                    bench(name + "/fp8", quantized, fused_quantized, args.repeats)


def rotary_check(args):
    for weight_dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(1, 11700, 16, 128, device="cuda", dtype=torch.bfloat16)
        norm = RMSNorm(128, dtype=weight_dtype, device="cuda").eval()
        norm.weight.uniform_(.8, 1.2)
        cos, sin = tables(x.shape[1])
        def reference():
            n = norm(x).to(x)
            return apply_rotary_emb(n, n, (cos, sin), head_first=False)[0]
        candidate = lambda: glue.norm_rope(x, norm.weight, norm.weight, cos, sin, norm.eps, False, True, x.dtype)
        compare("dit_rms_rope/" + str(weight_dtype), reference(), candidate())
        bench("dit_rms_rope/" + str(weight_dtype), reference, candidate, args.repeats)
    arch = decoder.GenExpertArch()
    attention = decoder.GenAttention(arch).cuda().bfloat16().eval()
    for tokens, heads in ((15600, 32), (46800, 8)):
        x = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        angle = torch.randn(1, tokens, 1, 128, device="cuda")
        cos, sin = angle.cos(), angle.sin()
        nt, nh = attention.q_norm_t, attention.q_norm_hw
        nt.weight.uniform_(.8, 1.2)
        nh.weight.uniform_(.8, 1.2)
        reference = lambda: glue._ORIGINAL["prepare"](attention, x, cos, sin, nt, nh, x.dtype)
        candidate = lambda: attention._prepare_qk(x, cos, sin, nt, nh, x.dtype)
        compare(f"decoder_split_rms_mrope/{tokens}/{heads}", reference(), candidate())
        bench(f"decoder_split_rms_mrope/{tokens}/{heads}", reference, candidate, args.repeats)
        del x, cos, sin, angle
    for tokens in (15600, 46800):
        x = torch.randn(1, tokens, 4096, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        norm = decoder.RMSNorm(4096, 1.e-6).cuda().bfloat16()
        reference = lambda: (x + residual, glue._ORIGINAL["rms"](norm, x + residual))
        candidate = lambda: glue.add_rms(x, residual, norm.weight, norm.eps)
        compare(f"decoder_add_rms/{tokens}", reference(), candidate())
        bench(f"decoder_add_rms/{tokens}", reference, candidate, args.repeats)
        del x, residual


def attention_check(args):
    ni, nt, length, h = 11700, args.text_tokens, 7020, 16
    q = tuple(torch.randn(1, n, h, 128, device="cuda", dtype=torch.bfloat16) for n in (ni, nt))
    k = tuple(torch.randn_like(z) for z in q)
    v = tuple(torch.randn_like(z) for z in q)
    history = torch.randn(1, length, h, 128, device="cuda", dtype=torch.bfloat16)
    cache = make_int8_kv_block(1, length + ni + nt, h, 128, device="cuda")
    quantize_kv_pair_into(cache, history, torch.randn_like(history), 0)
    tm = torch.ones(1, nt, device="cuda", dtype=torch.bool)
    tm[:, -7:] = False
    hm = torch.ones(1, cache["k"].shape[1], device="cuda", dtype=torch.bool)
    from hymm.models.dit.modules.uniform_history_cache import temporal_context_frame_indices
    selected_frames = temporal_context_frame_indices(18, 12)
    frame_mask = torch.zeros(18, dtype=torch.bool, device="cuda")
    frame_mask[list(selected_frames)] = True
    hm[:, :length].view(1, 18, 390).copy_(frame_mask[None, :, None].expand(1, 18, 390))
    hm[:, 17:32] = False
    offsets = [[16, nt // 3, 2 * nt // 3, nt - 8]]
    windows = [[0, 3900, 0], [3900, 7800, 1], [7800, 11700, 2]]
    for masked in (False, True):
        cache.pop("_nomas_glue", None)
        original_cache = {}
        off, win = (offsets, windows) if masked else (None, None)
        frame_windows = [[a // 390, b // 390, c] for a, b, c in win] if win else None
        def reference():
            fullq = torch.cat(q, 1)
            fullk = torch.cat((dequantize_kv(cache, "k", length), *k), 1)
            fullv = torch.cat((dequantize_kv(cache, "v", length), *v), 1)
            qm = torch.cat((torch.ones(1, ni, dtype=torch.bool, device="cuda"), tm), 1)
            km = torch.cat((hm[:, :length], qm), 1)
            if masked:
                out = attn.intra_attention(fullq, fullk, fullv, km, tm, length, ni, off, win, "flex", cache=original_cache)
            else:
                out = attn.dense_attention(fullq, fullk, fullv, qm, km, cache=original_cache, history_len=length)
            return out[:, :ni].flatten(2), out[:, ni:].flatten(2)
        candidate = lambda: glue._attention(q, k, v, cache, length, "flash3", tm,
            False, True, hm, off, frame_windows, 390, "flex")
        compare("direct_int8_history/" + str(masked), reference(), candidate(), .04)
        bench("direct_int8_history/" + str(masked), reference, candidate, args.repeats)
        for _ in range(7):
            for z in (*q, *k, *v):
                z.normal_()
            candidate()
        compare("direct_int8_history_changed_current/" + str(masked), reference(), candidate(), .04)
        state = cache["_nomas_glue"]
        if state["history"].get("k8") is None:
            raise AssertionError("History was not packed once for reuse")
        qi, ki = state["groups"][0]
        glue._direct_group(q, k, v, cache, length, qi, ki, state["history"])
        packed = glue._buffers(q[0], ki.numel(), qi.numel())
        expected = torch.empty_like(packed["v8"])
        scales = torch.empty_like(packed["vs"])
        selected = torch.cat((dequantize_kv(cache, "v", length), *v), 1).index_select(1, ki)
        ops.fa().fa_quant(selected[0], expected, scales, 0, expected.shape[1], True, 448, True, False)
        if not torch.equal(packed["v8"], expected) or not torch.equal(packed["vs"], scales):
            raise AssertionError("Indexed INT8-decode/FA-V packing differs from native FA packing")
        addresses = {name: state["history"][name].data_ptr() for name in ("mean", "k8", "ks", "v8", "vs")}
        quantize_kv_pair_into(cache, torch.randn_like(history), torch.randn_like(history), 0)
        refreshed = glue.prepare_attention(cache, length, ni, nt, h, tm, hm, off[0] if off else None,
            win, "cuda", refresh=True)
        if refreshed is not state or any(state["history"][name].data_ptr() != address for name, address in addresses.items()):
            raise AssertionError("History refresh changed graph-captured buffer addresses")
        original_cache.clear()
        compare("direct_int8_history_refreshed/" + str(masked), reference(), candidate(), .04)


def decoder_attention_check(args):
    grid = (10, 30, 52)
    mask = decoder.local_attention_layout(grid, 8, "cuda")[2]
    q = torch.randn(1, 15600, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 15600, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    baseline = lambda: glue._ORIGINAL["decoder_attention"](q, k, v, mask)
    candidate = lambda: attn.decoder_attention(q, k, v, mask)
    compare("decoder_indexed_gqa_tile_packing", baseline(), candidate(), .04)
    bench("decoder_indexed_gqa_tile_packing", baseline, candidate, args.repeats)


def block_check(args):
    block = nomas_cpt.MMDoubleStreamBlock(2048, 16, 4., attn_mode="flash3",
        dtype=torch.bfloat16, device="cuda").eval()
    block.seg_ar_attention_backend = "flex"
    block.compile_cache_prepost = False
    gemm._replace_linears(block)
    x = torch.randn(1, 11700, 2048, device="cuda", dtype=torch.float32)
    text = torch.randn(1, args.text_tokens, 2048, device="cuda", dtype=torch.float32)
    vec = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16)
    cache = make_int8_kv_block(1, 11700 + args.text_tokens, 16, 128, device="cuda")
    mask = torch.ones(1, args.text_tokens, dtype=torch.bool, device="cuda")
    freqs = tables(11700)
    kwargs = dict(freqs_cis=freqs, text_mask=mask, kv_cache=cache, cache_valid_len=0,
                  attn_mode="flash3")
    reference = lambda: glue._ORIGINAL["block"](block, x, text, vec, **kwargs)
    candidate = lambda: block.forward_with_cache(x, text, vec, **kwargs)
    reference()
    candidate()
    compare("real_dit_n3_block", reference(), candidate(), .015)
    bench("real_dit_n3_block", reference, candidate, args.repeats)
    other = make_int8_kv_block(1, 11700 + args.text_tokens, 16, 128, device="cuda")
    frozen_x = x.clone()
    written = glue._ORIGINAL["block"](block, x, text, vec, **dict(kwargs, update_cache=True))
    actual = block.forward_with_cache(x, text, vec, **dict(kwargs, kv_cache=other, update_cache=True))
    compare("real_dit_n3_cache_write", written, actual, .02)
    for key in ("k", "v"):
        compare("real_dit_n3_written_" + key, dequantize_kv(cache, key, cache["k"].shape[1]),
            dequantize_kv(other, key, other["k"].shape[1]), .02)
    if not torch.equal(x, frozen_x):
        raise AssertionError("Fused post-attention residual mutated the block input")


def camera_check(args):
    camera = ActionModule(keyboard_dim_in=64, hidden_size=128, img_hidden_size=2048,
        keyboard_hidden_dim=2048, heads_num=16, rope_dim_list=[16, 56, 56], per_chunk_camera_attn=True,
        camera_chunk_size=1, seg_ar_attention_backend="flex").cuda().bfloat16().eval()
    x = torch.randn(1, 11700, 2048, device="cuda", dtype=torch.float32)
    keyboard = torch.randn(1, 30, 64, device="cuda", dtype=torch.bfloat16)
    for mask, boundaries in ((None, None),
            (torch.tensor([0.], device="cuda"), None),
            (torch.tensor([[1.]], device="cuda"), None),
            (torch.tensor([[1., 0., 1.]], device="cuda"), [0, 10, 20, 30])):
        camera._nomas_camera = None
        reference_mask = mask.flatten() if mask is not None and mask.ndim == 2 and mask.shape[1] == 1 and boundaries is None else mask
        reference = lambda: glue._ORIGINAL["camera"](camera, x, 30, 15, 26, keyboard,
            has_camera_mask=reference_mask, segment_boundaries_latent=boundaries)
        candidate = lambda: camera(x, 30, 15, 26, keyboard,
            has_camera_mask=mask, segment_boundaries_latent=boundaries)
        baseline = reference()
        actual = candidate()
        compare("camera_singleton/" + str(boundaries) + "/" + str(mask is None), baseline, actual, .001, .01)
        saved = camera._nomas_camera["frames"]
        x.normal_()
        compare("camera_cached_changed_video", reference(), candidate(), .001, .01)
        if camera._nomas_camera["frames"] is not saved:
            raise AssertionError("Camera frame outputs recomputed with fixed conditioning")
        bench("camera_singleton/" + str(boundaries) + "/" + str(mask is None), reference, candidate, args.repeats)
    keyboard_full = torch.randn(1, 60, 64, device="cuda", dtype=torch.bfloat16)
    for segment, offset, keys in ((0, 0, keyboard_full), (1, 30, keyboard_full),
            (1, 30, keyboard_full[:, 30:].contiguous())):
        for gate, boundaries in ((None, None), (torch.tensor([False], device="cuda"), None),
                (torch.tensor([True], device="cuda"), None),
                (torch.tensor([[True, False]], device="cuda"), [0, 30, 60])):
            kwargs = dict(has_camera_mask=gate, segment_boundaries_latent=boundaries,
                          rope_time_offset=offset)
            for phase in ("denoise", "cache_write"):
                x.normal_()
                expected = glue._ORIGINAL["camera"](camera, x, 30, 15, 26, keys, **kwargs)
                actual = camera(x, 30, 15, 26, keys, **kwargs)
                compare(f"camera_clip60/segment{segment}/offset{offset}/{keys.shape[1]}/{phase}/{boundaries}/{gate is None}",
                        expected, actual, .001, .01)
                if camera._nomas_camera["frames"].shape[1] != 30:
                    raise AssertionError("Clip-level keyboard was not reduced to singleton frame values")
    from types import SimpleNamespace
    from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
    pipeline = SimpleNamespace(device=keyboard_full.device)
    for segment in range(2):
        keys, gate = CPTCausalInferencePipeline._slice_chunk_camera(pipeline, keyboard_full,
            "extrinsics_attn", segment * 30, (segment + 1) * 30, segment, [True, False], 1)
        torch.testing.assert_close(keys, keyboard_full[:, segment * 30:(segment + 1) * 30], rtol=0, atol=0)
        kwargs = dict(has_camera_mask=gate, segment_boundaries_latent=None, rope_time_offset=segment * 30)
        compare(f"camera_real_pipeline_kwargs/segment{segment}",
            glue._ORIGINAL["camera"](camera, x, 30, 15, 26, keys, **kwargs),
            camera(x, 30, 15, 26, keys, **kwargs), .001, .01)
    strided = torch.randn(1, 11700, 4096, device="cuda")[..., ::2]
    compare("camera_noncontiguous_video",
        glue._ORIGINAL["camera"](camera, strided, 30, 15, 26, keyboard_full),
        camera(strided, 30, 15, 26, keyboard_full), .001, .01)
    q = torch.randn(1, 60, 16, 128, device="cuda", dtype=torch.bfloat16)
    k, values = torch.randn_like(q), torch.randn_like(q)
    arguments = dict(q=q, k=k, v=values, tt=30, th=1, tw=2,
                     attn_boundaries_latent=list(range(31)))
    compare("camera_actual_blockwise_hook",
        glue._ORIGINAL["camera_blockwise"](camera, **arguments),
        camera._blockwise_cross_attn(**arguments), 0., 0.)
    v = torch.randn(1, 16, 390, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, 16, 1, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    compare("camera_single_key_identity", value.expand_as(v), F.scaled_dot_product_attention(v, key, value), 0., 0.)
    refresh_mask = torch.tensor([True], device="cuda")
    camera(x, 30, 15, 26, keyboard, has_camera_mask=refresh_mask)
    changed = keyboard.clone().add_(1)
    old = camera._nomas_camera["frames"].clone()
    address = camera._nomas_camera["frames"].data_ptr()
    camera(x, 30, 15, 26, changed, has_camera_mask=refresh_mask)
    if torch.equal(camera._nomas_camera["frames"], old) or camera._nomas_camera["frames"].data_ptr() != address:
        raise AssertionError("New camera conditioning did not refresh fixed-address frame outputs")
    gemm._replace_linears(camera)
    camera._nomas_camera = None
    baseline = lambda: glue._ORIGINAL["camera"](camera, x, 30, 15, 26, keyboard)
    candidate = lambda: camera(x, 30, 15, 26, keyboard)
    baseline()
    compare("camera_singleton_n3", baseline(), candidate(), .002, .02)
    bench("camera_singleton_n3", baseline, candidate, args.repeats)
    for offset in (0, 30):
        kwargs = dict(rope_time_offset=offset)
        expected = glue._ORIGINAL["camera"](camera, x, 30, 15, 26, keyboard_full, **kwargs)
        actual = camera(x, 30, 15, 26, keyboard_full, **kwargs)
        compare(f"camera_clip60_n3/offset{offset}", expected, actual, .002, .02)


def device_tables_check(args):
    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.tensor([1., 2., 3., 4.])
    b = torch.tensor([1., 3., 2., 4.])
    compare("device_table_same_checksum", b.to(device), glue.device_tensor(b, device), 0., 0.)
    first = glue.device_tensor(a, device)
    if first is glue.device_tensor(b, device):
        raise AssertionError("Distinct CPU table contents shared a device table")
    a.add_(1)
    compare("device_table_mutation", a.to(device), glue.device_tensor(a, device), 0., 0.)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--text-tokens", type=int, default=512)
    parser.add_argument("--skip-block", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("glue_check requires a CUDA GPU; no CPU substitute")
    torch.manual_seed(2026)
    os.environ.update(NOMAS_GLUE="1", NOMAS_ATTN="fa", NOMAS_DEC_ATTN="fa")
    os.environ.setdefault("N3_CONT", "1")
    attn.install()
    glue.install()
    ops.fa()
    ops.n3()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for check in (device_tables_check, modulation_check, rotary_check, attention_check, decoder_attention_check, camera_check,
                      *(() if args.skip_block else (block_check,))):
            check(args)
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps(dict(status="passed", peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)), flush=True)


if __name__ == "__main__":
    main()
