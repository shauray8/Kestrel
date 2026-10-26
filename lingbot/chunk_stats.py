import json, re, statistics, sys

text = open(sys.argv[1]).read()
min_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 4
pat = re.compile(r"chunk (\d+) denoise_s=([\d.]+) steps=4 kvupdate_s=([\d.]+) chunk_total_s=([\d.]+)")
chunks = []
for line in text.splitlines():
    m = pat.search(line)
    if m:
        chunks.append(tuple(float(x) for x in m.groups()))
        continue
    if line.startswith('{"exp": "n1_e2e", "tag"'):
        tag = json.loads(line)["tag"]
        steady = [c for c in chunks if c[0] >= min_chunk]
        if steady:
            print(json.dumps({"tag": tag, "n": len(steady), "median_chunk_s": round(statistics.median(c[3] for c in steady), 4),
                              "median_denoise_s": round(statistics.median(c[1] for c in steady), 4),
                              "median_kv_s": round(statistics.median(c[2] for c in steady), 4)}))
        chunks = []
print(json.dumps({"tracebacks": text.count("Traceback")}))
