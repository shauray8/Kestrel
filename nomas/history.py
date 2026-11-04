import inspect, textwrap

_EDITS = {
    "commit_uniform_history": (
        ("segment[key].to(device='cpu', copy=True, non_blocking=False).contiguous()", "segment[key].contiguous()"),
        ("device='cpu', dtype=torch.bool, copy=True, non_blocking=False,", "device=current_mask.device, dtype=torch.bool, copy=True,"),
    ),
    "_rebuild_history": (
        ("getattr(old, key).index_select(1, indices)", "getattr(old, key).index_select(1, indices.to(getattr(old, key).device))"),
        ("old.mask.index_select(1, indices)", "old.mask.index_select(1, indices.to(old.mask.device))"),
    ),
}


def install():
    import hymm.models.dit.modules.uniform_history_cache as m
    for name, edits in _EDITS.items():
        src = textwrap.dedent(inspect.getsource(getattr(m, name)))
        for old, new in edits:
            if src.count(old) != 1:
                raise RuntimeError(f"NOMAS_GPU_ARCHIVE cannot patch {name}: {old!r}")
            src = src.replace(old, new)
        exec(compile(src, m.__file__, "exec"), m.__dict__)
