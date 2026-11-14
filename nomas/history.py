import dataclasses, inspect, os, textwrap

_EDITS = {
    "commit_uniform_history": (
        ("segment[key].to(device='cpu', copy=True, non_blocking=False).contiguous()", "segment[key].contiguous()"),
        ("device='cpu', dtype=torch.bool, copy=True, non_blocking=False,", "device=current_mask.device, dtype=torch.bool, copy=True,"),
    ),
    "_rebuild_history": (
        ("getattr(old, key).index_select(1, indices)", "getattr(old, key).index_select(1, indices.to(getattr(old, key).device)).to(cache[key].device)"),
        ("old.mask.index_select(1, indices)", "old.mask.index_select(1, indices.to(old.mask.device)).to(mask.device)"),
    ),
}


def _offload(cache, keep):
    records = cache.get("swa_archive", [])
    keep = int(os.environ.get("NOMAS_ARCHIVE_GPU_SEGMENTS", keep))
    if keep <= 0:
        return
    if os.environ.get("NOMAS_ARCHIVE_DROP") == "1":
        sink = int(os.environ.get("NOMAS_ARCHIVE_SINK", "1"))
        if len(records) > sink + keep:
            cache["swa_archive"] = records[:sink] + records[-keep:]
        return
    moved = []
    for i, rec in enumerate(records):
        if i >= len(records) - keep or not getattr(rec.k, "is_cuda", False):
            moved.append(rec)
            continue
        fields = {f: getattr(rec, f).to("cpu") for f in ("k", "v", "mask", "k_scale", "v_scale") if getattr(rec, f) is not None}
        moved.append(dataclasses.replace(rec, **fields))
    cache["swa_archive"] = moved


def install():
    import hymm.models.dit.modules.uniform_history_cache as m
    for name, edits in _EDITS.items():
        src = textwrap.dedent(inspect.getsource(getattr(m, name)))
        for old, new in edits:
            if src.count(old) != 1:
                raise RuntimeError(f"NOMAS_GPU_ARCHIVE cannot patch {name}: {old!r}")
            src = src.replace(old, new)
        exec(compile(src, m.__file__, "exec"), m.__dict__)
    original = m.history_frame_indices

    def sinked(total_frames, past_frames, sampling="uniform", *, gamma=2.0):
        k = int(os.environ.get("NOMAS_SINK_FRAMES", "0"))
        if k <= 0 or total_frames <= past_frames or past_frames <= k:
            return original(total_frames, past_frames, sampling, gamma=gamma)
        rest = original(total_frames - k, past_frames - k, sampling, gamma=gamma)
        return tuple(range(k)) + tuple(f + k for f in rest)
    m.history_frame_indices = sinked
    keep = int(os.environ.get("NOMAS_ARCHIVE_GPU_SEGMENTS", "0"))
    if keep > 0 or os.environ.get("NOMAS_ARCHIVE_POLICY_DYNAMIC") == "1":
        commit = m.commit_uniform_history

        def bounded(cache, *args, **kwargs):
            out = commit(cache, *args, **kwargs)
            _offload(cache, keep)
            return out
        m.commit_uniform_history = bounded
        for mod in list(__import__("sys").modules.values()):
            if mod is not m and getattr(mod, "commit_uniform_history", None) is commit:
                mod.commit_uniform_history = bounded
