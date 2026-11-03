import argparse
import fnmatch
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def sources(patterns):
    found = sorted(ROOT.glob("*.cu"))
    if not patterns:
        return found
    selected = [p for p in found if any(fnmatch.fnmatchcase(p.stem, q) for q in patterns)]
    unmatched = [q for q in patterns if not any(fnmatch.fnmatchcase(p.stem, q) for p in found)]
    if unmatched:
        raise ValueError("unknown feature patterns: " + ", ".join(unmatched))
    return selected


def summarize(text, prefer_error):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if prefer_error and ("error" in line.lower() or "fatal" in line.lower())]
    return (errors[0] if errors else lines[-1] if lines else "")[:400] or None


def invoke(command, directory, timeout, log, prefer_error=False):
    try:
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.write_text(f"timeout after {timeout:g}s\n")
        return False, "", f"timeout after {timeout:g}s"
    except OSError as exc:
        log.write_text(str(exc) + "\n")
        return False, "", str(exc)
    log.write_text("$ " + " ".join(map(str, command)) + "\n" + result.stdout + result.stderr)
    if result.returncode == 0:
        return True, result.stdout, None
    return False, result.stdout, summarize(result.stderr + result.stdout, prefer_error) or f"exit status {result.returncode}"


def parse_metric(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("probe must emit exactly one JSON metric")
    record = json.loads(lines[0])
    if set(record) != {"metric", "value", "unit"}:
        raise ValueError("invalid metric fields")
    if not isinstance(record["metric"], str) or not isinstance(record["unit"], str):
        raise ValueError("metric and unit must be strings")
    value = record["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("metric must be a finite number")
    return record


def commands(source, directory, args):
    host = source.with_suffix(".py")
    if host.is_file():
        binary = directory / "probe.cubin"
        build = [args.nvcc, "-O3", "-std=c++17", "-arch=sm_120a", "-cubin", str(source), "-o", str(binary)]
        run = [sys.executable, str(host), str(binary)]
    else:
        binary = directory / "probe"
        build = [args.nvcc, "-O3", "-std=c++17", "-arch=sm_120a", "-lineinfo", str(source), "-lcuda", "-o", str(binary)]
        run = [str(binary)]
    return binary, build, run + ([] if args.iterations is None else [str(args.iterations)])


def probe(source, args):
    directory = args.output_dir / source.stem
    directory.mkdir(parents=True, exist_ok=True)
    binary, build, run = commands(source, directory, args)
    record = dict(feature=source.stem, compiled=False, ran=False, metric="compile_failed", value=None, unit=None, error=None)
    if args.run_only:
        compiled = binary.is_file()
        record["error"] = None if compiled else f"missing {binary}"
    else:
        compiled, _, record["error"] = invoke(build, directory, args.compile_timeout, directory / "compile.log", prefer_error=True)
    record["compiled"] = compiled
    if not compiled:
        return record
    record["metric"] = "build_only" if args.build_only else "run_failed"
    if args.build_only:
        return record
    ran, text, record["error"] = invoke(run, directory, args.timeout, directory / "run.log")
    if ran:
        try:
            record.update(parse_metric(text))
            record["ran"] = True
        except (ValueError, TypeError) as exc:
            with (directory / "run.log").open("a") as log:
                log.write("\n" + str(exc) + "\n")
            record["metric"] = "invalid_output"
            record["error"] = str(exc)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", default=[])
    parser.add_argument("--list", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--compile-timeout", type=float, default=180)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = sources(args.features)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print("\n".join(p.stem for p in selected))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if (args.iterations is not None and args.iterations < 1) or args.timeout <= 0 or args.compile_timeout <= 0:
        parser.error("iterations and timeouts must be positive")
    args.output_dir = args.output_dir.resolve()
    failed = False
    for source in selected:
        record = probe(source, args)
        print(json.dumps(record, allow_nan=False), flush=True)
        failed |= not record["compiled"] or (not args.build_only and not record["ran"])
    return int(args.strict and failed)


if __name__ == "__main__":
    sys.exit(main())
