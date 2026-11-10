import json, pathlib, statistics, time
import modal

SAGE_REV = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
OUT = pathlib.Path.cwd() / "modal_results"
GPU = "RTX-PRO-6000"

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ninja-build")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("numpy", "nvidia-ml-py", "packaging", "setuptools", "wheel", "ninja")
    .run_commands("python -c \"import torch; print('TORCH', torch.__version__, torch.version.cuda)\"")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0", "MAX_JOBS": "16", "TORCH_EXTENSIONS_DIR": "/root/torch_ext",
          "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared"})
    .run_commands(
        f"git clone https://github.com/thu-ml/SageAttention /root/SageAttention && cd /root/SageAttention && git checkout {SAGE_REV}",
        "cd /root/SageAttention && pip install . --no-build-isolation",
        gpu=GPU,
    )
    .add_local_dir(ROOT / "kestrel", "/root/src/kestrel", copy=True, ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "bench", "/root/src/bench", copy=True, ignore=["**/__pycache__", "modal"])
    .add_local_file(HERE / "robust_h2h.py", "/root/robust_h2h.py", copy=True)
    .run_commands("cd /root && python -c \"import sys; sys.path[:0]=['/root/src']; from kestrel import ops; ops.fa()\"", gpu=GPU)
)

app = modal.App("kestrel-fa-vs-sage2pp", image=image)


@app.function(gpu=GPU, timeout=1800)
def trial(seed: int):
    import subprocess, sys
    sys.path.insert(0, "/root")
    import robust_h2h
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    rows = robust_h2h.run(seed)
    for r in rows:
        r["nvidia_smi"] = smi
    return json.dumps(rows, default=str)


@app.function(gpu=GPU, timeout=1800)
def trial_mps(seed: int, percent: float):
    import os, subprocess
    env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mpslog"}
    os.makedirs("/tmp/mps", exist_ok=True); os.makedirs("/tmp/mpslog", exist_ok=True)
    start = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, capture_output=True, text=True)
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percent)
    code = ("import sys,json,torch; sys.path[:0]=['/root']; import robust_h2h; "
            "rows=robust_h2h.run(%d); print('ROWS'+json.dumps(rows, default=str))" % seed)
    p = subprocess.run(["python", "-c", code], env=env, capture_output=True, text=True, cwd="/root")
    rows = [json.loads(l[4:]) for l in p.stdout.splitlines() if l.startswith("ROWS")]
    if not rows:
        return json.dumps({"error": p.stderr[-3000:], "mps_start": start.returncode, "mps_err": start.stderr[-500:]})
    for r in rows[0]:
        r["mps_percent"] = percent
    return json.dumps(rows[0])


@app.local_entrypoint()
def mps(procs: int = 6, percent: float = 90.43):
    outs = [json.loads(x) for x in trial_mps.starmap([(s, percent) for s in range(procs)])]
    errs = [o for o in outs if isinstance(o, dict)]
    if errs:
        print("ERROR", errs[0]); return
    results = [row for rows in outs for row in rows]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"h2h_mps{percent}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"saved {path}; reported SMs {sorted({r['sms'] for r in results})}")
    summarize(results)


@app.local_entrypoint()
def main(procs: int = 6):
    results = [row for rows in trial.map(range(procs)) for row in json.loads(rows)]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"h2h_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"saved {path}")
    summarize(results)


def summarize(results):
    for lk in sorted({r["Lk"] for r in results}):
        rs = [r for r in results if r["Lk"] == lk]
        for ours in ("ours_int8", "ours_fp8"):
            m = [r["median_ms"]["sage2pp"] / r["median_ms"][ours] - 1 for r in rs]
            paired = [s / o - 1 for r in rs for s, o in zip(r["samples_ms"]["sage2pp"], r["samples_ms"][ours])]
            wins = sum(x > 0 for x in paired)
            print(f"Lk={lk:5d} {ours:9s} per-process sage/ours-1: {', '.join(f'{x*100:+.1f}%' for x in m)} | "
                  f"median {statistics.median(m)*100:+.1f}% | paired rounds ours faster {wins}/{len(paired)} | "
                  f"err sage {statistics.mean(r['rel_l2']['sage2pp'] for r in rs):.4f} ours {statistics.mean(r['rel_l2'][ours] for r in rs):.4f}")
