import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--hy-nomas", default=os.environ.get("HY_NOMAS_ROOT", "/workspace/Hy-Nomas"))
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--decoder", action="store_true")
args = parser.parse_args()
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), args.hy_nomas]

from hymm.models.dit import nomas_cpt
from hymm.models.dit.modules.attention.range_attention import flex_masked_attention
from nomas import attn

original_intra = nomas_cpt._intra_seg_cache_forward
os.environ["NOMAS_ATTN"] = "fa"
if args.decoder:
    os.environ["NOMAS_DEC_ATTN"] = "fa"
attn.install()
torch.manual_seed(2026)


def compare(name, reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    error = candidate-reference
    rms = (error.square().mean()/reference.square().mean().clamp_min(1e-20)).sqrt().item()
    cosine = torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()
    print(json.dumps(dict(case=name, relative_rmse=rms, cosine=cosine,
                          max_abs=error.abs().max().item())), flush=True)
    if not torch.isfinite(candidate).all() or rms > 0.12 or cosine < 0.99:
        raise AssertionError(name)


def bench(name, baseline, candidate):
    for _ in range(args.warmup):
        baseline()
        candidate()
    result = {"case": name}
    for label, fn in (("original_flex", baseline), ("kestrel_fa", candidate)):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for _ in range(args.iters):
            fn()
        end.record()
        end.synchronize()
        result[label+"_ms"] = start.elapsed_time(end)/args.iters
        result[label+"_wall_ms"] = (time.perf_counter()-wall)*1000/args.iters
    result["speedup"] = result["original_flex_wall_ms"]/result["kestrel_fa_wall_ms"]
    print(json.dumps(result), flush=True)


with torch.inference_mode():
    device = "cuda"
    q = torch.randn((1, 11700, 16, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 7020+11700, 16, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    qm = torch.ones(q.shape[:2], device=device, dtype=torch.bool)
    km = torch.ones(k.shape[:2], device=device, dtype=torch.bool)
    cache = {}
    baseline = lambda: flex_masked_attention(q, k, v, qm, km)
    candidate = lambda: attn.dense_attention(q, k, v, qm, km, cache=cache, history_len=7020)
    before = attn._engine().stats["hist_quant"]
    compare("dense_11700x18720x16x128", baseline(), candidate())
    for _ in range(7):
        q.normal_()
        k[:, 7020:].normal_()
        v[:, 7020:].normal_()
        candidate()
    compare("dense_step8_changed_current", baseline(), candidate())
    if attn._engine().stats["hist_quant"]-before != 1:
        raise AssertionError("Immutable history must be quantized exactly once over 8 steps")
    bench("dense_11700x18720x16x128", baseline, candidate)
    qm[:, 13::701] = False
    km[:, 19::503] = False
    cache.clear()
    compare("dense_mask_holes", baseline(), candidate())
    del q, k, v, qm, km, baseline, candidate, cache

    text = 96
    q = torch.randn((1, 11700+text, 16, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 7020+11700+text, 16, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tm = torch.ones((1, text), device=device, dtype=torch.bool)
    tm[:, 88:] = False
    tm[:, 12] = False
    km = torch.ones(k.shape[:2], device=device, dtype=torch.bool)
    km[:, 7020+11700:] = tm
    km[:, 17:32] = False
    offsets = [[16, 40, 64, 87]]
    windows = [[0, 3900, 0], [3900, 7800, 1], [7800, 11700, 2]]
    cache = {}
    parameters = (q, k, v, km, tm, 7020, 11700, offsets, windows, "flex")
    baseline = lambda: original_intra(*parameters)
    candidate = lambda: attn.intra_attention(*parameters, cache=cache)
    compare("action_text_and_history_holes", baseline(), candidate())
    bench("action_text_and_history_holes", baseline, candidate)
    del q, k, v, km, tm, parameters, baseline, candidate, cache

    if args.decoder:
        from hymm.models.vae.pixelumm_decoder import model
        grid = (10, 30, 52)
        layout = model.local_attention_layout(grid, 8, device)
        mask = layout[2] if layout is not None else None
        q = torch.randn((1, 15600, 32, 128), device=device, dtype=torch.bfloat16)
        k = torch.randn((1, 15600, 8, 128), device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        baseline = lambda: attn._ORIGINAL["decoder_flex"](
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), mask).transpose(1, 2)
        candidate = lambda: attn.decoder_attention(q, k, v, mask)
        compare("decoder_gqa_local_10x30x52", baseline(), candidate())
        bench("decoder_gqa_local_10x30x52", baseline, candidate)
        del q, k, v, baseline, candidate
        q = torch.randn((1, 3120, 16, 128), device=device, dtype=torch.bfloat16)
        k = torch.randn((1, 4096, 16, 128), device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        qm = torch.ones(q.shape[:2], device=device, dtype=torch.bool)
        km = torch.ones(k.shape[:2], device=device, dtype=torch.bool)
        reference = flex_masked_attention(q, k, v, qm, km)
        default = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(default)
        for _ in range(3):
            first = attn._engine().run(q, k, v, {}, 0)
            with torch.cuda.stream(side):
                if attn._engine().bal:
                    raise AssertionError("Non-default streams must not share balanced FA scratch")
                second = attn._engine().run(q, k, v, {}, 0)
        default.wait_stream(side)
        compare("concurrent_default_stream", reference, first)
        compare("concurrent_decoder_stream", reference, second)
    print(json.dumps(dict(fa_stats=attn._engine().stats,
                          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
