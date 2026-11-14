import glob, os, sys, torch
os.environ.setdefault("NOMAS_GLUE", "1")
from nomas import glue, attn

def reference(q, k, v, plan):
    out = torch.empty_like(q, dtype=torch.float32)
    rep = q.shape[2] // k.shape[2]
    for qi, ki in plan:
        qq = q[0, qi].float().transpose(0, 1)
        kk = k[0, ki].float().repeat_interleave(rep, 1).transpose(0, 1)
        vv = v[0, ki].float().repeat_interleave(rep, 1).transpose(0, 1)
        p = torch.softmax(qq @ kk.transpose(1, 2) * 128 ** -.5, -1)
        out[0, qi] = (p @ vv).transpose(0, 1)
    return out

def rel(a, b):
    return ((a.float() - b).norm() / b.norm()).item()

def bench(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(n):
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    return sorted(ts)[n // 2]

glue._ORIGINAL["decoder_attention"] = attn.decoder_attention
torch.set_grad_enabled(False)
worst = 0.
for f in sorted(glob.glob(sys.argv[1] + "/dec_*.pt")):
    d = torch.load(f, map_location="cuda")
    q, k, v = d["q"], d["k"], d["v"]
    ref = reference(q, k, v, d["plan"])
    old = glue._decoder_attention.__wrapped__ if hasattr(glue._decoder_attention, "__wrapped__") else None
    entry = (None, d["plan"])
    class M: pass
    os.environ["NOMAS_DEC_PAGED"] = "0"
    attn._LAYOUTS.clear(); mask = M(); attn._LAYOUTS[id(mask)] = (mask, d["plan"], d["paged"])
    run_old = lambda: glue._decoder_attention(q, k, v, mask)
    run_new = lambda: glue._decoder_paged(q, k, v, d["paged"])
    o_old, o_new = run_old(), run_new()
    e_old, e_new = rel(o_old, ref), rel(o_new, ref)
    t_old, t_new, t_old2 = bench(run_old), bench(run_new), bench(run_old)
    worst = max(worst, e_new / e_old)
    print(f"{os.path.basename(f)} q{tuple(q.shape)} k{tuple(k.shape)} relL2 old {e_old:.5f} new {e_new:.5f} ratio {e_new/e_old:.3f} "
          f"maxabs old {(o_old.float()-ref).abs().max().item():.4f} new {(o_new.float()-ref).abs().max().item():.4f} "
          f"ms old {t_old:.3f}/{t_old2:.3f} new {t_new:.3f} speedup {min(t_old,t_old2)/t_new:.2f}x", flush=True)
print("CONTRACT", "PASS" if worst <= 1.1 else "FAIL", f"worst_ratio={worst:.3f}")
