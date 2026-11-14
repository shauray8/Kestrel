import subprocess, threading, time, torch
from kestrel import ops
ext = ops.n3()
torch.manual_seed(0)
def clk(stop, out):
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        out.append(r); time.sleep(0.1)
def bench(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True); t = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); t.append(a.elapsed_time(b))
    return sorted(t)[n // 2]
shapes = [("dec q", 15600, 4096, 4096), ("dec k/v", 15600, 4096, 1024), ("dec o", 15600, 4096, 4096), ("dec gate+up", 15600, 4096, 24576),
          ("dec down", 15600, 12288, 4096), ("dit qkv", 11700, 2048, 6144), ("dit fc1", 11700, 2048, 8192), ("dit fc2", 11700, 8192, 2048)]
for name, m, k, n in shapes:
    x = (torch.randn(m, k, device="cuda") * 2).bfloat16()
    w = (torch.randn(n, k, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    sw = torch.rand(n, device="cuda") * 0.01 + 0.001
    x8 = ext.n3_quantize(x, 0.25)
    ms = bench(lambda: ext.n3_linear_f8in(x8, w, sw, None, 0.25))
    stop, samples = threading.Event(), []
    th = threading.Thread(target=clk, args=(stop, samples)); th.start()
    for _ in range(200): ext.n3_linear_f8in(x8, w, sw, None, 0.25)
    torch.cuda.synchronize(); stop.set(); th.join()
    one = torch.ones((), device="cuda")
    try:
        cb = bench(lambda: torch._scaled_mm(x8, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
    except Exception as e:
        cb = float("nan")
    tf = 2 * m * n * k / ms / 1e9
    print(f"{name:12s} M{m} K{k} N{n}: n3 {ms:.3f} ms {tf:6.0f} TF | cuBLAS fp8 {cb:.3f} ms {2*m*n*k/cb/1e9:6.0f} TF | clocks/W/throttle {samples[len(samples)//2] if samples else '?'}", flush=True)
