import json, pathlib, subprocess, time
import modal

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
OUT = pathlib.Path.cwd() / "modal_results"
INPUTS = pathlib.Path(__import__("os").environ.get("SHOWCASE_INPUTS", pathlib.Path.cwd() / "inputs"))
GPU = "RTX-PRO-6000"
SAGE_REV = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
W = "/workspace"
models = modal.Volume.from_name("kestrel-models", create_if_missing=True)
samples = modal.Volume.from_name("kestrel-samples", create_if_missing=True)

dl_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ninja-build", "ffmpeg", "util-linux", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.13.0", "torchvision", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("numpy", "nvidia-ml-py", "packaging", "setuptools", "wheel", "ninja", "einops", "huggingface_hub")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0a", "MAX_JOBS": "16", "CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -shared",
          "TORCH_EXTENSIONS_DIR": "/root/torch_ext", "OMP_NUM_THREADS": "8"})
    .run_commands(
        f"git clone -q https://github.com/robbyant/lingbot-world-v2 {W}/lingbot-world-v2",
        f"git clone -q https://github.com/madebyollin/taehv.git {W}/taehv",
        f"git clone -q https://github.com/thu-ml/SageAttention {W}/SageAttention && cd {W}/SageAttention && git checkout -q {SAGE_REV}",
    )
    .add_local_dir(ROOT, f"{W}/kestrel", copy=True, ignore=["**/__pycache__", ".git", "**/.DS_Store", "bench/modal"])
    .run_commands(
        f"cd {W}/lingbot-world-v2 && git apply {W}/kestrel/lingbot/patches/lingbot_sdpa_timing.diff && git apply {W}/kestrel/lingbot/patches/model_fast_ebcast.diff",
        f"cd {W}/lingbot-world-v2 && grep -v -E '^(torch|torchvision|torchaudio|flash_attn|flash-attn)' requirements.txt > /tmp/req.txt && pip install -q -r /tmp/req.txt",
        f"pip install -q -e {W}/kestrel --no-deps && pip install -q -r {W}/kestrel/eval/requirements.txt",
        f"cd {W}/SageAttention && git apply {W}/kestrel/lingbot/patches/sage_stream.diff && EXT_PARALLEL=4 NVCC_APPEND_FLAGS=--threads=8 pip install -q --no-build-isolation .",
        gpu=GPU,
    )
    .run_commands(f"cd {W} && python -c \"from kestrel import ops; [getattr(ops, n)() for n in ('n1', 'n3', 'fa', 'rope_kv', 'fused_mod')]\"", gpu=GPU)
)

app = modal.App("kestrel-lingbot14b")


@app.function(image=dl_image, volumes={W + "/models": models}, timeout=7200, cpu=8, memory=16384)
def download():
    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot_download("robbyant/lingbot-world-v2-14b-causal-fast", local_dir=f"{W}/models/lingbot14b")
    for f in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", f, local_dir=f"{W}/models/lingbot_assets")
    snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", allow_patterns=["google/*"], local_dir=f"{W}/models/lingbot_assets")
    models.commit()
    return subprocess.run(["du", "-sh", f"{W}/models/lingbot14b", f"{W}/models/lingbot_assets"], capture_output=True, text=True).stdout


@app.function(image=image, gpu=GPU, volumes={W + "/models": models, W + "/samples": samples}, timeout=6 * 3600, memory=131072, cpu=16)
def matrix(tag: str, variants: list[str], sets: list[str]):
    pathlib.Path(f"{W}/gpu.lock").touch()
    pathlib.Path(f"{W}/calib").mkdir(exist_ok=True)
    rundir = f"{W}/samples/{tag}"
    cmd = ["python", f"{W}/kestrel/bench/matrix.py", rundir, *variants, "--set", "MODEL=14b", "--set", f"KESTREL_MODELS={W}/models"]
    for s in sets:
        cmd += ["--set", s]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=W)
    samples.commit()
    logs = {}
    for f in sorted(pathlib.Path(rundir).glob("*.log")):
        txt = f.read_text(errors="replace").splitlines()
        logs[f.name] = [l for l in txt if any(k in l for k in ("dit_loop_total_s", "tae_live", "\"exp\"", "Error", "Traceback", "error"))][-40:]
    q = pathlib.Path(rundir) / "quality.jsonl"
    return json.dumps({"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:],
                       "logs": logs, "quality": q.read_text().splitlines() if q.exists() else [],
                       "smi": subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"], capture_output=True, text=True).stdout})


@app.local_entrypoint()
def fetch():
    print(download.remote())


@app.local_entrypoint()
def run(tag: str = "l14_97", variants: str = "stock,best_fa,best", sets: str = "FRAMES=97"):
    out = json.loads(matrix.remote(tag, variants.split(","), [s for s in sets.split(";") if s]))
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path} rc={out['rc']} wall={out['wall_s']}s {out['smi']}")
    for name, lines in out["logs"].items():
        print("==", name)
        for l in lines[-8:]:
            print("  ", l[:300])
    for l in out["quality"]:
        print("Q", l[:400])
    if out["rc"]:
        print(out["stderr"][-2000:])


clip_image = image.add_local_dir(INPUTS, f"{W}/inputs") if INPUTS.is_dir() else image
STACK = {"FUSED_GLUE": "1", "KESTREL_E_BCAST": "1", "CAM_HOIST": "1", "ATTN": "fa", "FUSED_FFN": "1", "ROPE32": "1", "ROPE_KV": "1",
         "XATTN": "1", "N3": "1", "FA_LOAD_MODE": "3", "FA_EXP_MODE": "0", "FA_Q_TILE": "128", "FA_RESCALE_THRESHOLD": "0.5",
         "RMS_ROPE": "1", "GLUE2": "1", "FA_MX": "1", "FA_BAL": "1", "N3_CONT": "1", "ONLY": "n2_calib_rope32,n1n2_rope32"}


@app.function(image=clip_image, gpu=GPU, volumes={W + "/models": models, W + "/samples": samples}, timeout=4 * 3600, memory=131072, cpu=16)
def clip(job: dict, frames: int, tag: str):
    import os
    pathlib.Path(f"{W}/gpu.lock").touch()
    env = {**os.environ, **STACK, "MODEL": "14b", "RESIDENT": "1", "KESTREL_MODELS": f"{W}/models", "CHUNK": "2", "TAE_LIVE": "1",
           "LB_TIMING": "1", "MAX_AREA": "399424", "CONFIG": "int8", "CALIB_CACHE": f"{W}/samples/calib14", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "SAMPLE_DIR": f"{W}/samples/{tag}", "SAMPLE_NAME": job["name"], "IMAGE": f"{W}/{job['image']}", "PROMPT_FILE": f"{W}/{job['prompt']}",
           "ACTION": f"{W}/{job['action']}", "EXAMPLE": "examples/05", "FRAMES": str(frames), "SEED": "42", "KESTREL_OUT": "/tmp/out",
           "KESTREL_GPU_LOCK": f"{W}/gpu.lock"}
    t0 = time.time()
    p = subprocess.run(["python", f"{W}/kestrel/lingbot/run.py"], cwd=f"{W}/lingbot-world-v2", env=env, capture_output=True, text=True)
    pathlib.Path(f"{W}/samples/{tag}/logs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{W}/samples/{tag}/logs/{job['name']}.log").write_text(p.stdout + p.stderr)
    samples.commit()
    keep = [l for l in (p.stdout + p.stderr).splitlines() if any(k in l for k in ("dit_loop_total_s", "tae_live", "n1_e2e", "resident", "calib_cache", "Error", "Traceback"))]
    return json.dumps({"name": job["name"], "rc": p.returncode, "wall_s": round(time.time() - t0, 1), "lines": keep[-12:]})


@app.local_entrypoint()
def showcase(frames: int = 481, tag: str = "showcase14", only: str = ""):
    jobs = json.loads((INPUTS / "jobs.json").read_text())
    if only:
        jobs = [j for j in jobs if any(o in j["name"] for o in only.split(","))]
    first = json.loads(clip.remote(jobs[0], frames, tag))
    print(json.dumps(first))
    outs = [first] + [json.loads(o) for o in clip.starmap([(j, frames, tag) for j in jobs[1:]])]
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{tag}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(outs, indent=1))
    for o in outs:
        print(o["name"], "rc", o["rc"], "wall", o["wall_s"], *[l[:200] for l in o["lines"][-3:]], sep="\n  ")


@app.function(image=image, gpu=GPU, timeout=1800)
def smoke():
    import torch, sageattention
    from kestrel import ops
    ops.fa(); ops.n3()
    return f"{torch.__version__} {torch.cuda.get_device_name()} sage {sageattention.__file__}"


@app.local_entrypoint()
def build():
    print(smoke.remote())
