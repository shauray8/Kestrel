import json
import math
import os
from pathlib import Path

import torch


CDF_FRACTIONS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
BUDGETS = (0.25, 0.5, 0.75)


@torch.no_grad()
def mass_metrics(q, k, v, sink_tokens, current_tokens, query_block=128, block_sizes=(4, 16, 64),
                 cdf_fractions=CDF_FRACTIONS, budgets=BUDGETS):
    if q.ndim == 4:
        if q.shape[0] != 1:
            raise ValueError("mass capture requires batch size one")
        q, k, v = q[0], k[0], v[0]
    nq, heads, dim = q.shape
    nk = k.shape[0]
    if not 0 <= sink_tokens <= nk - current_tokens <= nk:
        raise ValueError("sink, recent and current regions must be disjoint")
    shape = (math.ceil(nq / query_block), heads)
    result = {"region_mass": torch.empty((*shape, 3))}
    for size in block_sizes:
        for name, width in (("cdf", len(cdf_fractions)), ("mass", len(budgets)), ("recall", len(budgets)),
                            ("key_fraction", len(budgets)), ("error", len(budgets)), ("error_renorm", len(budgets))):
            result[f"b{size}_{name}"] = torch.empty((*shape, width))
    with torch.autocast(device_type=q.device.type, enabled=False):
        for head in range(heads):
            kh, vh = k[:, head].float(), v[:, head].float()
            for qi, start in enumerate(range(0, nq, query_block)):
                p = (q[start:start + query_block, head].float() @ kh.T / math.sqrt(dim)).softmax(-1)
                dense = p @ vh
                norm = dense.norm().clamp_min(torch.finfo(torch.float32).tiny)
                result["region_mass"][qi, head] = torch.stack((p[:, :sink_tokens].sum(-1).mean(),
                    p[:, sink_tokens:nk - current_tokens].sum(-1).mean(), p[:, nk - current_tokens:].sum(-1).mean())).cpu()
                key_order = p.argsort(dim=-1, descending=True)
                for size in block_sizes:
                    blocks = math.ceil(nk / size)
                    padded = torch.nn.functional.pad(p.mean(0), (0, blocks * size - nk))
                    mass = padded.reshape(blocks, size).sum(-1)
                    order = mass.argsort(descending=True)
                    cdf = mass[order].cumsum(0)
                    counts = torch.full((blocks,), size, device=q.device, dtype=torch.long)
                    counts[-1] = nk - (blocks - 1) * size
                    key_counts = counts[order].cumsum(0)
                    result[f"b{size}_cdf"][qi, head] = torch.stack([
                        cdf[max(0, math.ceil(fraction * blocks) - 1)] for fraction in cdf_fractions]).cpu()
                    for bi, budget in enumerate(budgets):
                        limit = math.floor(nk * budget)
                        selected = torch.zeros(blocks, device=q.device, dtype=torch.bool)
                        selected[order] = key_counts <= limit
                        mask = selected.repeat_interleave(size)[:nk]
                        restricted = p * mask
                        retained = restricted.sum(-1, keepdim=True)
                        approx = restricted @ vh
                        renorm = approx / retained.clamp_min(torch.finfo(torch.float32).tiny)
                        recall = mask[key_order[:, :limit]].float().mean() if limit else p.new_zeros(())
                        values = {"mass": retained.mean(), "recall": recall,
                                  "key_fraction": mask.float().mean(), "error": (approx - dense).norm() / norm,
                                  "error_renorm": (renorm - dense).norm() / norm}
                        for name, value in values.items():
                            result[f"b{size}_{name}"][qi, head, bi] = value.cpu()
    return result


class Capture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.chunks = {int(x) for x in os.environ.get("ATTN_MASS_CHUNKS", "1,5,11").split(",")}
        self.steps = {int(x) for x in os.environ.get("ATTN_MASS_STEPS", "0,1,3").split(",")}
        self.layers = {int(x) for x in os.environ.get("ATTN_MASS_LAYERS", "0,7,15,23,29").split(",")}
        self.records = []
        self.active = None
        self.begin("run")

    def begin(self, tag):
        self.tag, self.start, self.step, self.forward = tag, None, -1, None

    def save(self, q, k, v, module, cache, frame_seqlen, layer):
        chunk, step = self.forward
        lo = (k.storage_offset() - cache["k"].storage_offset()) // cache["k"].stride(1)
        current = q.shape[1]
        current_start = lo + k.shape[1] - current
        sink = max(0, min(module.sink_size * frame_seqlen, current_start) - lo)
        metrics = mass_metrics(q, k, v, sink, current)
        name = f"{self.tag}_c{chunk}_s{step}_l{layer}"
        info = {"tag": self.tag, "chunk": chunk, "step": step, "layer": layer,
                "q_shape": list(q.shape), "kv_shape": list(k.shape), "window_cache_offset": lo,
                "region_tokens": [sink, k.shape[1] - sink - current, current],
                "query_starts": list(range(0, q.shape[1], 128)),
                "query_counts": [min(128, q.shape[1] - i) for i in range(0, q.shape[1], 128)]}
        detail = {**info, "metrics": {key: value.tolist() for key, value in metrics.items()}}
        (self.directory / f"{name}.json").write_text(json.dumps(detail, separators=(",", ":"), allow_nan=False))
        if os.environ.get("ATTN_MASS_NPZ") == "1":
            import numpy as np
            arrays = {key: value.numpy() for key, value in metrics.items()}
            if os.environ.get("ATTN_MASS_QKV") == "1":
                arrays.update({key: value.detach().cpu().float().numpy() for key, value in (("q", q), ("k", k), ("v", v))})
            np.savez_compressed(self.directory / f"{name}.npz", **arrays)
        self.records.append({**info, "file": f"{name}.json", "mean": {
            key: value.mean((0, 1)).tolist() for key, value in metrics.items()}})
        self.flush()

    def flush(self):
        summary = {"query_block": 128, "block_sizes": [4, 16, 64], "cdf_fractions": CDF_FRACTIONS,
                   "budgets": BUDGETS, "region_order": ["sink", "recent", "current"],
                   "selection": "rank blocks by mean softmax mass over the query block; take ranked prefix fitting floor(budget * keys)",
                   "cdf": "mass of ceil(fraction * KV blocks) highest-mass blocks",
                   "recall": "mean per-query intersection with dense top floor(budget * keys) individual keys, divided by that count",
                   "error": "Frobenius relative L2 over queries and value channels, separately per head",
                   "axes": ["query_block", "head", "cdf_fraction or budget or region"],
                   "chunks": sorted(self.chunks), "steps": sorted(self.steps) + ["kv"], "layers": sorted(self.layers),
                   "captures": self.records}
        (self.directory / "summary.json").write_text(json.dumps(summary, separators=(",", ":"), allow_nan=False))


def install(pipe, mf, enabled=lambda: True):
    if os.environ.get("GRAPHS") or os.environ.get("VM_ATTN") == "1" or os.environ.get("SPEC") == "1":
        raise ValueError("ATTN_MASS requires eager dense self-attention (no GRAPHS, VM_ATTN or SPEC)")
    capture = Capture(os.environ["ATTN_MASS_OUT"])
    original_model, original_attention = pipe.model.forward, mf.attention

    def forward(*args, **kwargs):
        capture.forward = None
        if enabled():
            start = kwargs["current_start"]
            if start != capture.start:
                capture.start, capture.step = start, -1
            t = kwargs["t"]
            if bool((t == 0).all()):
                step = "kv"
            else:
                capture.step += 1
                step = capture.step
            chunk = start // (int(os.environ.get("CHUNK", "3")) * kwargs["frame_seqlen"])
            if chunk in capture.chunks and (step == "kv" or step in capture.steps):
                capture.forward = (chunk, step)
        try:
            return original_model(*args, **kwargs)
        finally:
            capture.forward = None

    def wrap(module, layer):
        original = module.forward

        def self_attention(x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0,
                           max_attention_size=1_000_000, frame_seqlen=None, seq_lens_int=None):
            if capture.forward is not None and layer in capture.layers:
                fs = frame_seqlen if frame_seqlen is not None else int(grid_sizes[0, 1:].prod())
                capture.active = (module, kv_cache, fs, layer)
            try:
                return original(x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                                frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int)
            finally:
                capture.active = None

        module.forward = self_attention

    def attention(q, k, v, *args, **kwargs):
        if capture.active is not None:
            capture.save(q, k, v, *capture.active)
        return original_attention(q, k, v, *args, **kwargs)

    for layer, block in enumerate(pipe.model.blocks):
        wrap(block.self_attn, layer)
    pipe.model.forward, mf.attention = forward, attention
    capture.flush()
    return capture
