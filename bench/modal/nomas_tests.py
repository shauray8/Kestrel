"""Run with: modal run --profile shauray8 bench/modal/nomas_tests.py --stage kernels.

The JSON report is also the test/input inventory; diagnostic-only scripts are not
reported as numerical passes. No full-clip quality, residency, or overlap claim is
made by these isolated checks. CONFIG.md rejects decoder NVFP4 quality and native
CUDA glue parity; both remain experiments here, not enabled deployment defaults.

Upstream: https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/0bd2405
Weights: https://huggingface.co/tencent/Hy-Nomas/tree/main/VAE
There is no FP8 serialization script in nomas/ or in the pinned upstream scripts/.
The published PixelUMM_FT_FP8 artifact is used only after checking its format.
Upload real inference.save_latents=true output to kestrel-samples/nomas/latents.npy
for decode_cmp, and a genuinely different toolchain's output to
kestrel-samples/nomas/prof/ab_reference.pt for ab_compare. Neither is synthesized.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time

import modal

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE
OUT = HERE / "out"
W = "/workspace"
REPO = f"{W}/kestrel"
HY = f"{W}/Hy-Nomas"
GPU = "RTX-PRO-6000"
HY_REV = "0bd240561567c0245d8ab8695c962337a8ea77fa"
MODEL_DIR = f"{W}/models/nomas"
SAMPLE_DIR = f"{W}/samples/nomas"
FP8 = f"{MODEL_DIR}/VAE/PixelUMM_FT_FP8/diffusion_pytorch_model_fp8.safetensors"
BF16 = f"{MODEL_DIR}/VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
LATENTS = f"{SAMPLE_DIR}/latents.npy"
DUMPS = f"{W}/prof/dec"
models = modal.Volume.from_name("kestrel-models", create_if_missing=True)
samples = modal.Volume.from_name("kestrel-samples", create_if_missing=True)

# Use the pinned project's dependency declarations, without replacing cu132 torch
# or installing Hopper-only FA3/Magi backends unused by these flex/FA checks.
_EXTRA_DEPS = (
    "import pathlib,subprocess,sys,tomllib; "
    f"p=tomllib.loads(pathlib.Path('{HY}/pyproject.toml').read_text()); "
    "skip=('torch==','flash-attn','magi-attention'); "
    "deps=[d for d in p['project']['optional-dependencies']['all'] if not d.startswith(skip)]; "
    "subprocess.run([sys.executable,'-m','pip','install',*deps],check=True)"
)
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "ninja-build", "ffmpeg", "util-linux", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.13.0", "torchvision", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("numpy", "nvidia-ml-py", "packaging", "setuptools", "wheel", "ninja", "psutil", "pytest")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0a", "MAX_JOBS": "16", "CC": "gcc", "CXX": "g++",
          "LDSHARED": "g++ -shared", "TORCH_EXTENSIONS_DIR": "/root/torch_ext", "OMP_NUM_THREADS": "8",
          "HY_NOMAS_ROOT": HY, "NOMAS_ROOT": HY, "PYTHONPATH": f"{W}/flash_attn_shim:{REPO}:{HY}",
          "CUTLASS_PATH": f"{W}/cutlass", "N3_CONT": "1", "NOMAS_GLUE_CUDA": "0"})
    .run_commands(
        f"git clone -q https://github.com/Tencent-Hunyuan/Hy-Nomas.git {HY} && git -C {HY} checkout -q {HY_REV}",
        f"git clone -q --depth 1 https://github.com/NVIDIA/cutlass.git {W}/cutlass",
        f"pip install -q -e {HY}",
        "python -c " + __import__("shlex").quote(_EXTRA_DEPS),
    )
    .add_local_dir(ROOT, REPO, copy=True, ignore=["**/__pycache__", ".git", "**/.DS_Store", "bench/modal"])
    .add_local_dir(ROOT.parent / "runs" / "nomas" / "flash_attn_shim", f"{W}/flash_attn_shim", copy=True, ignore=["**/__pycache__"])
    .run_commands(
        f"pip install -q -e {REPO} --no-deps",
        # n1/fused_mod/n2 are exercised by the dependent Kestrel checks too.
        f"cd {W} && python -c \"from kestrel import ops; [getattr(ops,n)() for n in ('n1','n3','n2','fa','nvfp4','fused_mod','nomas_glue','iquant')]\"",
        gpu=GPU,
    )
)
dl_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub==0.36.0", "hf_transfer").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
app = modal.App("kestrel-nomas-tests")


def _test(name, stage, script, args=(), inputs=(), criteria="Script assertions and exit 0", diagnostic=False, env=None):
    return dict(name=name, function=stage, script=script, args=list(args), required_inputs=list(inputs),
                pass_criteria=criteria, diagnostic_only=diagnostic, env=env or {})


def inventory():
    """Every executable in nomas/tests, plus its direct Kestrel kernel checks."""
    random = ["CUDA sm_120a; random tensors, no model weights"]
    source = random + [f"Hy-Nomas source at {HY_REV}"]
    dump = [f"{DUMPS}/dec_*.pt: q/k/v/plan/paged, emitted by glue_check via NOMAS_DEC_DUMP"]
    tests = [
        _test("kestrel_n3", "kernels", "tests/test_n3_linear.py", inputs=random,
              criteria="Bitwise N1/N3 parity, fused residual/split and CUDA graph assertions"),
        _test("kestrel_fa", "kernels", "tests/test_fa_attn.py", inputs=random,
              args=(), criteria="pytest assertions; optional Sage target-shape cases explicitly skipped"),
        _test("kestrel_nvfp4", "kernels", "tests/test_nvfp4.py", inputs=random,
              criteria="Timing/error measurements only; no numerical gate in script", diagnostic=True),
        _test("attn_check", "kernels", "nomas/tests/attn_check.py", ["--hy-nomas", HY, "--decoder"], source,
              "Finite; relative RMSE <=0.12, cosine >=0.99; one history quantization; stream isolation"),
        _test("gemm_check", "kernels", "nomas/tests/gemm_check.py", inputs=source,
              criteria="Relative L2 <=0.15, fused residual and BF16 fallback assertions; original CUTLASS baseline"),
        _test("gemm_check_bf16", "kernels", "nomas/tests/gemm_check.py", ["--decoder-backend", "bf16"], source,
              "Same N3 gates against BF16; additional run, not a replacement for CUTLASS failure"),
        _test("glue_check", "kernels", "nomas/tests/glue_check.py", ["--repeats", "5"], source,
              "All per-case tolerance/dtype/cache/camera assertions; default relative L2 .005 or maxabs .04",
              env={"NOMAS_DEC_DUMP": DUMPS}),
        _test("glue_cuda_check", "kernels", "nomas/tests/glue_cuda_check.py", ["--graphs", "--compile"], random,
              "Exact integer/indexed quant; <=1 ULP floats; FP8 mismatch fraction <1e-4; statistics relative L2 <1e-6"),
        _test("graph_check", "kernels", "nomas/tests/graph_check.py", ["--camera", "--intra", "--int8-history"], source,
              "Random small models, NOT checkpoint-based: rtol/atol .02 plus capture/replay/reuse/cache assertions"),
        _test("dec4_synthetic", "kernels", "nomas/tests/dec4_check.py", ["--attention", "--graph", "--compile"], source,
              "Packing/lifecycle; linear relative L2 <=.25, MLP <=.35, fused <=.02, graph/compile <=.001; rejected quality experiment"),
        _test("fa_split_check", "kernels", "nomas/tests/fa_split_check.py", inputs=random,
              criteria="CONTRACT PASS: split error / base error <=1.1 (script does not set failure rc)"),
        _test("dec_fuse_check", "kernels", "nomas/tests/dec_fuse_check.py", inputs=random,
              criteria="FUSE PASS: MLP and QKV bitexact (script does not set failure rc)"),
        _test("dec_cublas_check", "kernels", "nomas/tests/dec_cublas_check.py", inputs=random,
              criteria="CONTRACT PASS: cuBLAS relative L2 <=1.1*fused N3 error"),
        _test("gemm_rowwise", "kernels", "nomas/tests/gemm_rowwise.py", inputs=random,
              criteria="Timing/relative L2 measurements; cuBLAS errors are printed, not asserted", diagnostic=True),
        _test("gemm_shapes", "kernels", "nomas/tests/gemm_shapes.py", inputs=random,
              criteria="Timing/TFLOPS/clock/power measurements only", diagnostic=True),
        _test("iquant_check", "kernels", "nomas/tests/iquant_check.py", [DUMPS], random + dump,
              "Printed byte/scale equality for IEEE and fast modes; random INT8 DiT cases run even without dumps", diagnostic=True),
        _test("dec_paged_check", "kernels", "nomas/tests/dec_paged_check.py", [DUMPS], dump,
              "Nonempty dump corpus; CONTRACT PASS: new/old relative L2 <=1.1"),
        _test("dec_qtile_check", "kernels", "nomas/tests/dec_qtile_check.py", [DUMPS], dump,
              "Prints q_tile128/64 bitexact and timings; no assertion", diagnostic=True),
        _test("ab_toolchain", "kernels", "nomas/tests/ab_toolchain.py", ["current"], random + ["Optional decoder dumps"],
              "Produces /workspace/prof/ab_current.pt and timings; no cross-toolchain comparison", diagnostic=True),
        _test("ab_compare", "kernels", "nomas/tests/ab_compare.py",
              [f"{W}/prof/ab_reference.pt", f"{W}/prof/ab_current.pt"],
              [f"{W}/prof/ab_reference.pt from a different toolchain", f"{W}/prof/ab_current.pt"],
              "Every tensor bitexact; DIFF is treated as failure even though script exits 0"),
        _test("pipe_check_240", "pipeline", "nomas/tests/pipe_check.py", ["--checkpoint", FP8, "--frames", "240"],
              [HY, FP8], "One AST hook, bitexact cached text, stream/full decode rtol .01 atol .004; 480x832", env={"NOMAS_DEC_GEMM": "n3", "N3_CONT": "1"}),
        _test("pipe_check_241", "pipeline", "nomas/tests/pipe_check.py", ["--checkpoint", FP8, "--frames", "241"],
              [HY, FP8], "Same gate including non-multiple-of-four tail; 480x832", env={"NOMAS_DEC_GEMM": "n3", "N3_CONT": "1"}),
        _test("dec4_checkpoint", "pipeline", "nomas/tests/dec4_check.py", ["--checkpoint", FP8, "--attention", "--graph", "--compile"],
              [HY, FP8], "Real checkpoint layer 0, checkpoint-inferred FFN; same NVFP4 gates; not visual acceptance"),
        _test("decode_cmp", "pipeline", "nomas/tests/decode_cmp.py",
              ["--checkpoint", FP8, "--latents", LATENTS, "--pixel-frames", "240", "--overlap", "5"],
              [HY, FP8, f"{LATENTS}: actual 240-frame inference latents [32,60,15,26] or [1,32,60,15,26]"],
              "Same-seed FP8/NVFP4: PSNR >=30 dB and SSIM >=.95 (LPIPS intentionally not installed); not a visual override"),
    ]
    return tests


def _smi():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,power.draw", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return dict(rc=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(rc=None, error=str(exc))


def _summary(stage, results):
    return dict(stage=stage, upstream_revision=HY_REV, gpu=_smi(), results=results,
                inventory=inventory(), limitations=[
                    "No nomas/ FP8 serialization script exists; download uses the published FP8 artifact and checks metadata.",
                    "Pinned upstream CUTLASS FP8 compiles only sm_89 and conversion requires SM89; its original baseline may fail on sm_120a. Failures are retained, not patched away.",
                    "No test loads INT8 DiT, DCAE-V or Qwen weights: graph_check/dec4 synthetic instantiate random models. DiT/DCAE assets are staged for future full inference; Qwen is not downloaded.",
                    "decode_cmp needs real saved 240-frame latents; ab_compare needs an independent toolchain result. Missing inputs are blocked, never empty-loop passes.",
                    "pipe_check.md sections 2-6 require a real multi-item prompt/camera suite, matched full inference outputs, NVML/Nsight and human visual review; these isolated scripts do not implement that acceptance plan.",
                    "NVFP4 decoder and native CUDA glue are unaccepted/rejected experiments per CONFIG.md; diagnostic results are not deployment acceptance.",
                ], counts={state: sum(r["status"] == state for r in results)
                           for state in ("passed", "failed", "blocked", "diagnostic", "skipped")})


def _fp8_reason():
    if not pathlib.Path(FP8).is_file():
        return f"Missing serialized FP8 checkpoint {FP8}; run download. No local serialization script exists."
    import struct
    with open(FP8, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        metadata = json.loads(stream.read(size)).get("__metadata__", {})
    if metadata.get("format") != "pixelumm_fp8_bf16_single_v1":
        return f"Published checkpoint format {metadata.get('format')!r} is incompatible with pinned prequantized loader; missing conversion script."
    return None


def _run(test, logdir, reason=None):
    command = [sys.executable, "-u", f"{REPO}/{test['script']}", *test["args"]]
    if test["name"] == "kestrel_fa":
        command = [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", f"{REPO}/{test['script']}"]
    result = {**test, "command": command, "rc": None, "wall_s": 0.0, "stdout_tail": "", "stderr_tail": ""}
    if reason:
        return {**result, "status": "blocked", "reason": reason}
    start = time.perf_counter()
    stdout, stderr = logdir / (test["name"] + ".stdout.log"), logdir / (test["name"] + ".stderr.log")
    try:
        with stdout.open("w") as out, stderr.open("w") as err:
            p = subprocess.run(command, cwd=REPO, env={**os.environ, **test["env"]}, stdout=out, stderr=err, timeout=3600)
        result["rc"] = p.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    result["wall_s"] = round(time.perf_counter() - start, 3)
    text = stdout.read_text(errors="replace") if stdout.exists() else ""
    result.update(stdout_tail=text[-12000:], stderr_tail=stderr.read_text(errors="replace")[-12000:] if stderr.exists() else "",
                  stdout_log=str(stdout), stderr_log=str(stderr))
    contract_fail = bool(re.search(r"^(?:CONTRACT|FUSE) FAIL\b", text, re.MULTILINE))
    if test["name"] == "ab_compare" and "DIFF" in text:
        contract_fail = True
    result["status"] = ("failed" if result["rc"] != 0 or contract_fail else
                        "diagnostic" if test["diagnostic_only"] else "passed")
    if contract_fail:
        result["reason"] = "Script printed a failed numerical contract despite its exit status"
    if test["name"] == "iquant_check" and not list(pathlib.Path(DUMPS).glob("dec_*.pt")):
        result["coverage_note"] = "No decoder dumps: only the script's random INT8 DiT cases ran"
    return result


def _execute(stage):
    models.reload()
    samples.reload()
    prof = pathlib.Path(f"{SAMPLE_DIR}/prof")
    (prof / "dec").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{W}/prof").symlink_to(prof, target_is_directory=True) if not pathlib.Path(f"{W}/prof").exists() else None
    logdir = pathlib.Path(SAMPLE_DIR) / "logs" / (stage + "_" + time.strftime("%Y%m%dT%H%M%S"))
    logdir.mkdir(parents=True, exist_ok=True)
    results = []
    for test in inventory():
        if test["function"] != stage:
            continue
        reason = None
        if test["name"] in ("dec_paged_check", "dec_qtile_check") and not list(pathlib.Path(DUMPS).glob("dec_*.pt")):
            reason = "Missing decoder q/k/v capture; glue_check did not reach NOMAS_DEC_DUMP. Refuse an empty-loop pass."
        if test["name"] == "ab_compare":
            missing = [p for p in test["args"] if not pathlib.Path(p).is_file()]
            if missing:
                reason = "Missing independently produced toolchain comparison input(s): " + ", ".join(missing)
        if stage == "pipeline":
            try:
                reason = _fp8_reason()
            except (OSError, ValueError) as exc:
                reason = "Cannot inspect FP8 checkpoint: " + str(exc)
            if test["name"] == "decode_cmp" and not pathlib.Path(LATENTS).is_file():
                reason = (reason + "; " if reason else "") + f"Missing real inference.save_latents=true output at {LATENTS}; random latents are not a quality substitute."
        results.append(_run(test, logdir, reason))
    summary = _summary(stage, results)
    (logdir / "summary.json").write_text(json.dumps(summary, indent=2))
    samples.commit()
    return json.dumps(summary)


@app.function(image=dl_image, volumes={W + "/models": models}, timeout=7200, cpu=8, memory=16384)
def download():
    from huggingface_hub import HfApi, hf_hub_download
    results = []
    prefixes = ("VAE/DCAE-V/", "VAE/PixelUMM_FT/", "VAE/PixelUMM_FT_FP8/", "DiT/distilled_int8/")
    info = HfApi().model_info("tencent/Hy-Nomas", files_metadata=True)
    for item in info.siblings:
        if not item.rfilename.startswith(prefixes):
            continue
        start = time.perf_counter()
        path = pathlib.Path(MODEL_DIR) / item.rfilename
        result = dict(name=item.rfilename, function="download", required_inputs=["HF tencent/Hy-Nomas"],
                      revision=info.sha, args=["local_dir=" + MODEL_DIR], rc=None, stdout_tail="", stderr_tail="")
        try:
            if path.is_file() and path.stat().st_size == item.size:
                result.update(status="skipped", reason="Existing complete file (matches HF size)", rc=0)
            else:
                hf_hub_download("tencent/Hy-Nomas", item.rfilename, revision=info.sha, local_dir=MODEL_DIR)
                result.update(status="passed", rc=0)
        except Exception as exc:
            result.update(status="failed", stderr_tail=str(exc))
        result["wall_s"] = round(time.perf_counter() - start, 3)
        results.append(result)
    try:
        reason = _fp8_reason()
    except (OSError, ValueError) as exc:
        reason = str(exc)
    results.append(dict(name="serialized_pixelumm", function="download", args=[], required_inputs=[FP8],
                        status="blocked" if reason else "passed", rc=None if reason else 0, wall_s=0,
                        stdout_tail="", stderr_tail="", reason=reason or "Published FP8 checkpoint has the pinned loader's format; no derived checkpoint required"))
    models.commit()
    return json.dumps(_summary("download", results))


@app.function(image=image, gpu=GPU, volumes={W + "/models": models, W + "/samples": samples}, timeout=24 * 3600, memory=131072, cpu=16)
def kernels():
    return _execute("kernels")


@app.function(image=image, gpu=GPU, volumes={W + "/models": models, W + "/samples": samples}, timeout=8 * 3600, memory=131072, cpu=16)
def pipeline():
    return _execute("pipeline")


@app.local_entrypoint()
def main(stage: str = "kernels"):
    if stage not in ("kernels", "pipeline", "download", "all"):
        raise ValueError("stage must be kernels, pipeline, download, or all")
    summaries = {}
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (("download", "kernels", "pipeline") if stage == "all" else (stage,)):
        try:
            summary = json.loads({"download": download, "kernels": kernels, "pipeline": pipeline}[name].remote())
        except Exception as exc:
            # An image/build/infrastructure failure must not prevent the other stages.
            summary = dict(stage=name, status="failed", error=str(exc), inventory=inventory())
        summaries[name] = summary
        (OUT / f"nomas_tests_{name}.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    if stage == "all":
        (OUT / "nomas_tests_all.json").write_text(json.dumps(summaries, indent=2))
