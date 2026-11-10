import json, random, statistics, sys, time
import torch

sys.path[:0] = ["/root/src", "/root/src/bench"]
from kestrel import ops
import attn_head2head as h

try:
    import pynvml
    pynvml.nvmlInit()
    NV = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NV = None


def telemetry():
    if NV is None:
        return {}
    return {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(NV, pynvml.NVML_CLOCK_SM), "power_w": pynvml.nvmlDeviceGetPowerUsage(NV) / 1000,
            "temp_c": pynvml.nvmlDeviceGetTemperature(NV, pynvml.NVML_TEMPERATURE_GPU)}


def sample(fn, n):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / n


def run(seed, rounds=15, calls=20, lks=(9360, 18720, 28080)):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    ext = ops.fa()
    dev = {"gpu": torch.cuda.get_device_name(), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "torch": torch.__version__, "cuda": torch.version.cuda, "seed": seed}
    out = []
    with torch.inference_mode():
        for lk in lks:
            q, k, v = (torch.randn(n, 12, 128, device="cuda", dtype=torch.bfloat16) for n in (3120, lk, lk))
            ref = h.reference(q, k, v, 64)
            cases = {"sage2pp": h.sage2_kernel(q[None], k[None], v[None]),
                     "ours_int8": h.ours(ext, q, k, v, True, True, False, True),
                     "ours_fp8": h.ours(ext, q, k, v, True, True, False, False)}
            fns, err = {}, {}
            for name, (fn, norm, _, _) in cases.items():
                o = norm(fn())[:512].float()
                err[name] = float((o - ref).norm() / ref.norm())
                fns[name] = fn
            for fn in fns.values():
                for _ in range(10):
                    fn()
            torch.cuda.synchronize()
            samples, tel = {n: [] for n in fns}, []
            for r in range(rounds):
                order = list(fns)
                rng.shuffle(order)
                for name in order:
                    samples[name].append(sample(fns[name], calls))
                tel.append({"round": r, "order": order, **telemetry()})
            out.append({**dev, "Lk": lk, "rounds": rounds, "calls": calls, "rel_l2": err,
                        "median_ms": {n: statistics.median(s) for n, s in samples.items()}, "samples_ms": samples, "telemetry": tel})
            print(json.dumps({"Lk": lk, "median_ms": out[-1]["median_ms"], "rel_l2": err}), flush=True)
    return out


if __name__ == "__main__":
    print(json.dumps(run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)))
