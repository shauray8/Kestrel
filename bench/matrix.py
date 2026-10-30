import argparse, datetime, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = "/workspace/gpu.lock"

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def overrides(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set requires KEY=VALUE: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--set requires a nonempty key: {item}")
        out[key] = value
    return out

def variant_config(presets, variant, extra, out):
    config = {**presets["common"], **presets["variants"][variant], **extra}
    config["KESTREL_OUT"] = str(out)
    config.setdefault("KESTREL_GPU_LOCK", LOCK)
    return config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--lingbot", default="/workspace/lingbot-world-v2")
    ap.add_argument("--ref", default="stock")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    preset_path = ROOT / "bench" / "presets.json"
    presets = json.loads(preset_path.read_text())
    try:
        extra = overrides(args.set)
    except ValueError as e:
        ap.error(str(e))
    unknown = [v for v in args.variants if v not in presets["variants"]]
    if unknown:
        ap.error(f"unknown variants: {', '.join(unknown)}")
    rundir = Path(args.rundir).resolve()
    rundir.mkdir(parents=True, exist_ok=True)
    sha = commit()
    failed = False
    for variant in args.variants:
        out = rundir / "videos" / variant
        out.mkdir(parents=True, exist_ok=True)
        config = variant_config(presets, variant, extra, out)
        env = os.environ.copy()
        env.update(config)
        command = [sys.executable, str(ROOT / "lingbot" / "run.py")]
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        started = now()
        with (rundir / f"{variant}.log").open("w") as log:
            rc = subprocess.call(command, cwd=args.lingbot, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = {"variant": variant, "git_commit": sha, "env": config, "command": command,
                  "cwd": str(Path(args.lingbot).resolve()), "start_time": started, "end_time": now(), "rc": rc}
        (rundir / f"{variant}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(json.dumps({k: record[k] for k in ("variant", "rc", "start_time", "end_time")}, separators=(",", ":")), flush=True)
        failed |= rc != 0
    ref_videos = sorted(p for p in (rundir / "videos" / args.ref).glob("*.mp4")
                        if not p.stem.endswith("_taelive") and "_bf16_c" in p.stem and "calib" not in p.stem)
    candidates = sorted(p for p in (rundir / "videos").glob("*/*.mp4")
                        if not p.stem.endswith("_taelive") and "calib" not in p.stem)
    quality = rundir / "quality.jsonl"
    if len(ref_videos) != 1:
        quality.write_text("")
        print(json.dumps({"quality_rc": None, "error": f"expected one {args.ref} bf16 video, found {len(ref_videos)}"}), flush=True)
        failed = True
    elif candidates:
        command = ["flock", extra.get("KESTREL_GPU_LOCK", LOCK), sys.executable, str(ROOT / "eval" / "quality.py"), "--ref", str(ref_videos[0]), *map(str, candidates)]
        with quality.open("w") as out, (rundir / "quality.log").open("w") as err:
            rc = subprocess.call(command, stdout=out, stderr=err)
        print(json.dumps({"quality_rc": rc, "videos": len(candidates)}), flush=True)
        failed |= rc != 0
    else:
        quality.write_text("")
        failed = True
    with (rundir / "report.md").open("w") as out:
        rc = subprocess.call([sys.executable, str(ROOT / "bench" / "matrix_report.py"), str(rundir),
                              "--quality", str(quality), "--md"], stdout=out)
    failed |= rc != 0
    print(json.dumps({"report": str(rundir / "report.md"), "report_rc": rc}), flush=True)
    return int(failed)

if __name__ == "__main__":
    raise SystemExit(main())
