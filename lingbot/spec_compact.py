import torch


def bucket_capacity(count, total, bucket=128, pad=16):
    count = total if count > total - pad else count
    return ((count + bucket - 1) // bucket) * bucket


def compact(active, capacity, pad=16):
    active = active.flatten()
    total = active.numel()
    order = torch.arange(total, device=active.device)
    count = active.sum()
    dense = count > total - pad
    write_count = torch.where(dense, total, count)
    permutation = torch.argsort(torch.where(active, order, order + total))
    slots = torch.arange(capacity, device=active.device)
    source = torch.minimum(slots, (write_count - 1).clamp_min(0))
    idx = permutation[source]
    rank = torch.empty_like(permutation).scatter_(0, permutation, order)
    return idx, count, write_count, rank, active | dense


def expand_rows(rows, rank, active):
    return torch.where(active[:, None], rows[rank.clamp_max(rows.shape[0] - 1)], 0)


def distance_bank(bank):
    return [torch.cat((b, torch.ones_like(b[:, :1]), b.square().sum(-1, keepdim=True)), -1).T
            for b in bank.split(8192)]


def redundancy(query, bank):
    q = torch.cat((-2 * query, query.square().sum(-1, keepdim=True), torch.ones_like(query[:, :1])), -1)
    result = torch.full((query.shape[0],), float("inf"), device=query.device)
    for b in bank:
        distance = (q @ b).clamp_min_(0).sqrt_().amin(1)
        result = torch.minimum(result, distance)
    return result
