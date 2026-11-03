from typing import NamedTuple


class CachePlan(NamedTuple):
    global_end: int
    local_end: int
    write_offset: int
    lo: int
    advance: tuple | None


def cache_plan(current_start, tokens, old_global, old_local, cache_size, sink_tokens, max_attention_size, local=True):
    end = current_start + tokens
    advance = None
    if not local:
        hi = end
    elif end > old_global and tokens + old_local > cache_size:
        drop = tokens + old_local - cache_size
        if old_local - drop - sink_tokens > 0:
            advance = (sink_tokens, drop, old_local)
        hi = old_local + end - old_global - drop
    else:
        hi = old_local + end - old_global
    return CachePlan(end, hi, hi - tokens, max(0, hi - max_attention_size), advance)
