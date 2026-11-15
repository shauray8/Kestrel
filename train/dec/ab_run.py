import subprocess, sys
from pathlib import Path
import torch
init = "/workspace/runs/distill_dcae2/ema_0002000.pt"
out, steps, *overrides = sys.argv[1:]
if Path(out).exists():
    raise SystemExit(f"exists: {out}")
args = dict(torch.load(init, map_location="cpu", weights_only=False)["args"])
args.update(init=init, out=out, steps=int(steps), seed=0, trackio="kestrel-decoder", gen_frac=0.75, lookahead=0, extra_tail=0, lpips_jitter=0, eval_every=500, ckpt_every=500, keep_every=500)
cmd = [sys.executable, "train.py"]
for k, v in args.items():
    if isinstance(v, bool):
        if v: cmd.append(f"--{k}")
    elif v is not None:
        cmd.append(f"--{k}"); cmd.extend(map(str, v if isinstance(v, (list, tuple)) else [v]))
subprocess.run(cmd + overrides, check=True)
