import math
import operator
from typing import NamedTuple

import torch
import torch.nn.functional as F


BQ = 128


class ABPlan(NamedTuple):
    indices: torch.Tensor
    log_bias: torch.Tensor
    kept: int
    total: int


def _cluster(queries, valid, clusters):
    counts = valid.sum(-1, keepdim=True)
    mean = queries.sum(1) / counts
    distance = (queries - mean[:, None]).square().sum(-1)
    center = distance.masked_fill(~valid, -1).argmax(-1)
    nearest = torch.full_like(distance, float("inf"))
    assignment = torch.zeros_like(distance, dtype=torch.int64)
    rows = torch.arange(queries.shape[0], device=queries.device)
    for cluster in range(clusters):
        distance = (queries - queries[rows, center, None]).square().sum(-1)
        assignment = torch.where(valid & (distance < nearest), cluster, assignment)
        nearest = torch.minimum(nearest, distance).masked_fill(~valid, -1)
        if cluster + 1 < clusters:
            center = nearest.argmax(-1)
    membership = F.one_hot(assignment, clusters).transpose(1, 2).float()
    membership.mul_(valid[:, None])
    populations = membership.sum(-1)
    centroids = torch.bmm(membership, queries) / populations.clamp_min(1)[..., None]
    return centroids, populations / counts


def _sample(scores, samples, offset):
    mass = scores.double().clamp_min_(1e-30)
    ordered = mass.sort(-1, descending=True).values
    tail = ordered.flip(-1).cumsum(-1).flip(-1)
    ranks = torch.arange(samples, dtype=torch.float64, device=scores.device)
    scale = ((samples - ranks) / tail[:, :samples]).amax(-1, keepdim=True)
    probability = (mass * scale).clamp_max_(1)
    certain = probability == 1
    remaining = samples - certain.sum(-1, keepdim=True)
    fractional = probability.masked_fill(certain, 0)
    cumulative = fractional.cumsum(-1)
    positions = torch.arange(scores.shape[-1], device=scores.device)
    last = torch.where(fractional > 0, positions, -1).amax(-1, keepdim=True)
    cumulative = torch.where(
        positions >= last, remaining, torch.minimum(cumulative, remaining)
    )
    previous = F.pad(cumulative[:, :-1], (1, 0))
    selected = certain | (
        (torch.ceil(cumulative - offset) > torch.ceil(previous - offset))
        & (fractional > 0)
    )
    indices = torch.where(selected, positions, scores.shape[-1]).sort(-1).values
    indices = indices[:, :samples]
    return indices, -probability.gather(1, indices).log2().float()


@torch.no_grad()
def make_plan(
    q,
    k,
    sparsity=0.9,
    sink_tokens=9360,
    current_tokens=None,
    clusters=8,
    selection_block=4,
    generator=None,
):
    if q.ndim != 3 or k.ndim != 3 or q.shape[1:] != k.shape[1:] or q.shape[-1] != 128:
        raise ValueError("Q/K must have shape [tokens, heads, 128] with matching heads")
    if min(q.shape[:2]) <= 0 or k.shape[0] <= 0:
        raise ValueError("Q/K token and head counts must be positive")
    if not q.is_cuda or q.device != k.device:
        raise ValueError("Q/K must share a CUDA device")
    if q.dtype != torch.float16 or k.dtype != torch.float16:
        raise ValueError("Q/K must be float16")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("Q/K must be contiguous")
    sparsity = float(sparsity)
    if not math.isfinite(sparsity) or not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between zero and one")
    selection_block = operator.index(selection_block)
    clusters = operator.index(clusters)
    sink_tokens = operator.index(sink_tokens)
    current_tokens = q.shape[0] if current_tokens is None else operator.index(current_tokens)
    if selection_block not in (4, 64):
        raise ValueError("selection_block must be 4 or 64")
    if not 1 <= clusters <= BQ:
        raise ValueError("clusters must be between one and 128")
    if sink_tokens < 0 or current_tokens < 0:
        raise ValueError("sink_tokens and current_tokens must be nonnegative")
    query_tokens, heads, _ = q.shape
    key_tokens = k.shape[0]
    query_blocks = (query_tokens + BQ - 1) // BQ
    blocks = (key_tokens + selection_block - 1) // selection_block
    prefix = (min(sink_tokens, key_tokens) + selection_block - 1) // selection_block
    current_tokens = min(current_tokens, key_tokens)
    suffix_start = (key_tokens - current_tokens) // selection_block if current_tokens else blocks
    suffix_start = max(prefix, suffix_start)
    suffix = blocks - suffix_start
    mandatory = prefix + suffix
    tile_blocks = 64 // selection_block
    requested = max(1, math.ceil((1 - sparsity) * blocks))
    budget = min(blocks, max(mandatory, (requested + tile_blocks - 1) // tile_blocks * tile_blocks))
    shape = (heads, query_blocks, budget)
    kept = heads * query_blocks * budget
    total = heads * query_blocks * blocks
    if budget == blocks:
        indices = torch.arange(blocks, dtype=torch.int32, device=q.device)
        return ABPlan(indices.expand(shape).contiguous(), torch.zeros(shape, dtype=torch.float32, device=q.device), kept, total)
    prefix_ids = torch.arange(prefix, dtype=torch.int32, device=q.device)
    suffix_ids = torch.arange(suffix_start, blocks, dtype=torch.int32, device=q.device)
    if budget == mandatory:
        indices = torch.cat((prefix_ids, suffix_ids)).expand(shape).contiguous()
        return ABPlan(indices, torch.zeros(shape, dtype=torch.float32, device=q.device), kept, total)
    indices = torch.empty(shape, dtype=torch.int32, device=q.device)
    log_bias = torch.zeros(shape, dtype=torch.float32, device=q.device)
    indices[..., :prefix] = prefix_ids
    indices[..., budget - suffix:] = suffix_ids
    samples = budget - mandatory
    representatives = blocks * (selection_block // 2)
    pair_counts = (key_tokens - 2 * torch.arange(representatives, device=q.device)).clamp(0, 2)
    log_mass = pair_counts.float().log()
    divisor = pair_counts.clamp_min(1)[:, None]
    offsets = torch.rand((heads, query_blocks, 1), dtype=torch.float64, device=q.device, generator=generator)
    chunk = max(1, min(64, 8 * 1024 * 1024 // (clusters * representatives)))
    token_offsets = torch.arange(BQ, device=q.device)
    for head in range(heads):
        keys = F.pad(k[:, head].float(), (0, 0, 0, blocks * selection_block - key_tokens))
        pooled = keys.reshape(representatives, 2, 128).sum(1) / divisor
        for start in range(0, query_blocks, chunk):
            stop = min(query_blocks, start + chunk)
            queries = q[start * BQ:min(query_tokens, stop * BQ), head].float()
            queries = F.pad(queries, (0, 0, 0, (stop - start) * BQ - queries.shape[0]))
            queries = queries.reshape(stop - start, BQ, 128)
            lengths = (query_tokens - torch.arange(start, stop, device=q.device) * BQ).clamp_max(BQ)
            centroids, weights = _cluster(queries, token_offsets < lengths[:, None], clusters)
            logits = torch.matmul(centroids, pooled.T) * (128 ** -0.5)
            logits.add_(log_mass)
            probabilities = logits.softmax(-1)
            probabilities.mul_(weights[..., None])
            scores = probabilities.sum(1).reshape(stop - start, blocks, selection_block // 2).sum(-1)
            chosen, bias = _sample(scores[:, prefix:suffix_start], samples, offsets[head, start:stop])
            indices[head, start:stop, prefix:prefix + samples] = chosen + prefix
            log_bias[head, start:stop, prefix:prefix + samples] = bias
    return ABPlan(indices, log_bias, kept, total)
