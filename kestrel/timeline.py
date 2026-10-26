from collections import defaultdict

from torch.autograd import DeviceType

def _bucket(name):
    n = name.lower()
    for key, words in (("gemm", ("gemm", "cutlass", "sm90", "sm100", "sm120", "n1_", "nvjet", "cublas")), ("attn", ("attn", "sage", "flash", "fmha")),
                       ("norm_glue", ("norm", "mod_", "gated", "residual")), ("quant", ("quant", "fp4", "fp8", "scale")),
                       ("conv", ("conv", "implicit")), ("copy", ("copy", "memcpy", "memset", "cat", "fill"))):
        if any(w in n for w in words):
            return key
    return "elementwise_other"

def summarize(prof, wall_s):
    ev = sorted(((e.time_range.start, e.time_range.end, e.name) for e in prof.events() if e.device_type == DeviceType.CUDA and e.time_range.end > e.time_range.start),
                key=lambda t: t[0])
    if not ev:
        return {}
    busy, cur_s, cur_e, gaps = 0.0, ev[0][0], ev[0][1], []
    for s, e, _ in ev[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = ev[-1][1] - ev[0][0]
    by = defaultdict(lambda: [0, 0.0])
    sizes = {"<5us": [0, 0.0], "5-20us": [0, 0.0], "20-100us": [0, 0.0], ">100us": [0, 0.0]}
    names = defaultdict(lambda: [0, 0.0])
    for s, e, n in ev:
        d = e - s
        b = by[_bucket(n)]; b[0] += 1; b[1] += d
        k = "<5us" if d < 5 else "5-20us" if d < 20 else "20-100us" if d < 100 else ">100us"
        sizes[k][0] += 1; sizes[k][1] += d
        m = names[n[:80]]; m[0] += 1; m[1] += d
    tot = sum(e - s for s, e, _ in ev)
    gaps.sort()
    return {"wall_s": round(wall_s, 3), "gpu_span_s": round(span / 1e6, 3), "gpu_busy_s": round(busy / 1e6, 3),
            "busy_frac_of_span": round(busy / span, 4), "kernels": len(ev),
            "idle_gaps": {"n": len(gaps), "total_ms": round(sum(gaps) / 1e3, 1), "p50_us": round(gaps[len(gaps) // 2], 1) if gaps else 0,
                          "gt100us_total_ms": round(sum(g for g in gaps if g > 100) / 1e3, 1)},
            "by_class_pct": {k: {"n": v[0], "pct": round(100 * v[1] / tot, 1)} for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])},
            "by_size": {k: {"n": v[0], "pct_time": round(100 * v[1] / tot, 1)} for k, v in sizes.items()},
            "top": [{"name": k, "n": v[0], "pct": round(100 * v[1] / tot, 1)} for k, v in sorted(names.items(), key=lambda kv: -kv[1][1])[:15]]}
