import argparse, json, re, statistics
from pathlib import Path

NUM = r"[-+0-9.eE]+"
KV = re.compile(r"(\w+)=(" + NUM + r")")
QKEYS = ("psnr", "lpips", "lpips_first_s", "lpips_last_s", "subject_consistency", "background_consistency", "temporal_flicker", "dino_drift")

def number(s):
    return int(s) if re.fullmatch(r"[-+]?\d+", s) else float(s)

def quality_tag(path):
    m = re.match(r"\d+_(.*)_c\d+\.mp4$", Path(path).name)
    return m.group(1) if m else None

def load_quality(files):
    by_video, by_tag = {}, {}
    for file in files:
        for line in Path(file).read_text().splitlines():
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "video" not in q:
                continue
            video = q["video"]
            by_video.setdefault(video, {}).update(q)
            tag = quality_tag(video)
            if tag:
                by_tag.setdefault(tag, []).append(q)
    return by_video, by_tag

def parse_log(path):
    records, cur, tagged = [], None, {}
    for line in path.read_text(errors="replace").splitlines():
        if "[LBT] chunk " in line:
            if cur is None:
                cur = {"chunks_data": []}
            cur["chunks_data"].append({k: number(v) for k, v in KV.findall(line)})
            continue
        if "[LBT] dit_loop_total_s=" in line:
            if cur is not None:
                cur.update({k: number(v) for k, v in KV.findall(line)})
            continue
        start = line.find('{"exp":')
        if start < 0:
            continue
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        exp = obj.get("exp")
        if exp in ("paths", "graphs") and "tag" in obj:
            tagged[exp, obj["tag"]] = {k: v for k, v in obj.items() if k not in ("exp", "tag")}
            continue
        if exp == "n1_e2e" and "tag" in obj:
            if cur is not None:
                steady = cur["chunks_data"][2:]
                r = {"variant": path.stem, "tag": obj["tag"], "out": obj.get("out")}
                for key in ("denoise_s", "kvupdate_s", "chunk_total_s"):
                    r[key] = statistics.median(c[key] for c in steady) if steady else None
                for key in ("dit_loop_total_s", "chunks", "latent_frames", "peak_mem_GB"):
                    r[key] = cur.get(key)
                r["generate_s"] = obj.get("generate_s")
                for key in ("fps_e2e", "tae_ms_per_latent", "chunk_start_to_first_pixel_ms_median"):
                    if key in cur:
                        r[key] = cur[key]
                i2p = cur.get("input_to_pixels_ms", {})
                r["i2p_p50_ms"], r["i2p_p95_ms"] = i2p.get("p50"), i2p.get("p95")
                r["paths"], r["graphs"] = cur.get("paths"), cur.get("graphs")
                r["spec_mean_steps"], r["timeline_busy_frac"] = cur.get("spec_mean_steps"), cur.get("timeline_busy_frac")
                records.append(r)
            cur = None
        elif exp in ("tae_live", "paths", "graphs", "spec", "timeline"):
            if cur is None:
                cur = {"chunks_data": []}
            if exp == "tae_live":
                cur.update(obj)
            elif exp in ("paths", "graphs"):
                cur[exp] = {k: v for k, v in obj.items() if k != "exp"}
            elif exp == "spec":
                cur["spec_mean_steps"] = obj.get("mean_steps")
            elif exp == "timeline":
                cur["timeline_busy_frac"] = obj.get("busy_frac_of_span")
    for r in records:
        for exp in ("paths", "graphs"):
            if (exp, r["tag"]) in tagged:
                r[exp] = tagged[exp, r["tag"]]
    return records

def fallback(r):
    p = r.get("paths") or {}
    flat = {}
    for k, v in p.items():
        if k == "tag":
            continue
        if isinstance(v, dict):
            flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            flat[k] = v
    vals = [f"{k}={v}" for k, v in flat.items() if "fallback" in k.lower() and (not isinstance(v, (int, float)) or v)]
    if not vals and p:
        vals = [f"{k}={v}" for k, v in flat.items() if not isinstance(v, (int, float)) or v]
    return ",".join(vals) or "-"

def fmt(v, digits=3):
    return "-" if v is None else f"{v:.{digits}f}" if isinstance(v, float) else str(v)

def markdown(records, base):
    print("| variant | tag | s/chunk | speedup | fps | i2p p50/p95 ms | LPIPS | drift | fallback |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in records:
        speed = base["chunk_total_s"] / r["chunk_total_s"] if base and not r["variant"].startswith("prof_") and base.get("chunk_total_s") and r.get("chunk_total_s") else None
        speed_text = f"{fmt(speed, 2)}x" if speed is not None else "-"
        latency = f"{fmt(r.get('i2p_p50_ms'), 1)}/{fmt(r.get('i2p_p95_ms'), 1)}"
        print(f"| {r['variant']} | {r['tag']} | {fmt(r.get('chunk_total_s'), 4)} | {speed_text} | {fmt(r.get('fps_e2e'), 2)} | {latency} | {fmt(r.get('lpips'), 4)} | {fmt(r.get('dino_drift'), 4)} | {fallback(r)} |")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--quality", action="append", default=[])
    ap.add_argument("--baseline", metavar="VARIANT:TAG")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    records = [r for p in sorted(Path(args.logdir).glob("*.log")) for r in parse_log(p)]
    if args.quality:
        by_video, by_tag = load_quality(args.quality)
        for r in records:
            q = by_video.get(r.get("out"))
            if q is None:
                matches = by_tag.get(r["tag"], [])
                q = matches[0] if len(matches) == 1 else None
            if q:
                r.update(q)
    if args.baseline:
        if ":" not in args.baseline:
            ap.error("--baseline must be VARIANT:TAG")
        variant, tag = args.baseline.split(":", 1)
        base = next((r for r in records if r["variant"] == variant and r["tag"] == tag), None)
        if base is None:
            ap.error(f"baseline record not found: {args.baseline}")
    else:
        base = next((r for r in records if r["tag"] == "bf16" and not r["variant"].startswith("prof_")), None)
    if args.md:
        markdown(records, base)
    else:
        for r in records:
            print(json.dumps(r, separators=(",", ":")))

if __name__ == "__main__":
    main()
