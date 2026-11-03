import inspect, os
from functools import wraps


def install(pipe, paths, mode=None):
    mode = mode or os.environ.get("KV_REUSE_MODE", "last")
    if mode not in ("last", "extrap"):
        raise ValueError("KV_REUSE_MODE must be last or extrap")
    inner, gen = pipe.model.forward, pipe.generate
    signature = inspect.signature(gen)
    state = {"active": False}

    def appended(kw):
        p = pipe.model.patch_size
        f, h, w = kw["x"][0].shape[1:]
        n = (f // p[0]) * (h // p[1]) * (w // p[2])
        for cache in kw["kv_cache"]:
            end = min(kw["current_start"] + n, cache["k"].shape[1])
            yield cache["k"][:, end - n:end], cache["v"][:, end - n:end]

    def forward(*a, **kw):
        if not state["active"]:
            return inner(*a, **kw)
        if state["start"] != kw["current_start"]:
            state.update(start=kw["current_start"], step=0, previous=None)
        step, count = state["step"], state["count"]
        if step == count:
            if mode == "extrap":
                previous_t, last_t = (float(pipe.scheduler.timesteps[i]) for i in state["indices"][-2:])
                alpha = last_t / (previous_t - last_t)
                for current, previous in zip(appended(kw), state["previous"]):
                    for dst, src in zip(current, previous):
                        dst.copy_(dst.float().mul_(1 + alpha).add_(src.float(), alpha=-alpha))
                state["previous"] = None
            commit = getattr(pipe, "_kv_reuse_commit", None)
            if commit is not None:
                commit(kw["x"][0])
            state["step"] += 1
            paths["kv_forward_skipped"] += 1
            return None
        out = inner(*a, **kw)
        state["step"] += 1
        if mode == "extrap" and step == count - 2:
            state["previous"] = [(k.clone(), v.clone()) for k, v in appended(kw)]
        return out

    @wraps(gen)
    def generate(*a, **kw):
        bound = signature.bind(*a, **kw)
        bound.apply_defaults()
        indices = bound.arguments["timesteps_index"]
        if len(indices) < (2 if mode == "extrap" else 1):
            raise ValueError("KV reuse requires denoise steps; extrap requires at least two")
        state.update(active=True, start=None, step=0, count=len(indices), indices=indices, previous=None)
        try:
            return gen(*a, **kw)
        finally:
            state.update(active=False, previous=None)

    pipe.model.forward = forward
    pipe.generate = generate
