import inspect, json, os
from functools import lru_cache, wraps
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import wan.modules.model_fast as mf

SYNCFREE = os.environ.get("SPEC_SYNCFREE") == "1"
if SYNCFREE:
    from lingbot.spec_compact import bucket_capacity, compact, distance_bank, expand_rows, redundancy
    from kestrel import ops
EPS = float(os.environ.get("SPEC_EPS", "0.24"))
PAD = 16
ST = {"sparse": None}
_orig_sa = mf.CausalWanSelfAttention.forward

def _sa_forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000,
                frame_seqlen=None, seq_lens_int=None):
    sp = ST["sparse"]
    if sp is None:
        return _orig_sa(self, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                        frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int)
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    v = self.v(x).view(b, s, n, d)
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    if "count" in sp:
        q = sp["ext"].rope_qk_scatter(q.contiguous(), k.contiguous(), v.contiguous(), sp["cos_full"], sp["sin_full"],
                                      kv_cache["k"], kv_cache["v"], sp["lend"] - sp["L"], sp["idx"], sp["count"])
    else:
        q, k = _rope(q, sp).type_as(v), _rope(k, sp).type_as(v)
        kv_cache["k"][:, sp["pos"]] = k.to(kv_cache["k"].dtype)
        kv_cache["v"][:, sp["pos"]] = v.to(kv_cache["v"].dtype)
    lo, hi = max(0, sp["lend"] - max_attention_size), sp["lend"]
    return self.o(mf.attention(q, kv_cache["k"][:, lo:hi], kv_cache["v"][:, lo:hi]).flatten(2))

def _rope(x, sp):
    xr = x.float().unflatten(3, (-1, 2))
    a, b = xr[..., 0], xr[..., 1]
    return torch.stack([a * sp["cos"] - b * sp["sin"], a * sp["sin"] + b * sp["cos"]], -1).flatten(3)

def _rope_table(freqs, f, h, w, start):
    c = freqs.shape[1]
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([fs[0][start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], -1).reshape(f * h * w, 1, -1)
    return fi.real.float(), fi.imag.float()

class _Rows(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return _rows(self.m, x)

def _rows(m, x):
    y = m(x)
    sp = ST["sparse"]
    if SYNCFREE and not hasattr(m, "val"):
        return y
    return y[:, sp["idx"]] if sp is not None and y.shape[1] == sp["L"] else y

def _tok(x):
    c, f, h, w = x.shape
    return x.view(c, f, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(f * (h // 2) * (w // 2), c * 4)

def _lat(m):
    return m.repeat_interleave(2, 1).repeat_interleave(2, 2)

@lru_cache(maxsize=8)
def _smooth_weights(frames, device):
    r = torch.arange(9, dtype=torch.float32, device=device) - 4
    g = torch.exp(-0.5 * r * r)
    g = g / g.sum()
    i = torch.arange(frames, device=device)
    wt = torch.exp(-0.5 * (i[None] - i[:, None]).float() ** 2) * (i[None] >= i[:, None])
    return g, wt / wt.sum(1, keepdim=True)

def _smooth(s):
    if SYNCFREE:
        g, wt = _smooth_weights(s.shape[0], s.device)
    else:
        r = torch.arange(9, dtype=torch.float32, device=s.device) - 4
        g = torch.exp(-0.5 * r * r)
        g = g / g.sum()
        i = torch.arange(s.shape[0], device=s.device)
        wt = torch.exp(-0.5 * (i[None] - i[:, None]).float() ** 2) * (i[None] >= i[:, None])
        wt = wt / wt.sum(1, keepdim=True)
    x = F.pad(s[:, None], (4, 4, 4, 4), mode="reflect")
    x = F.conv2d(F.conv2d(x, g.view(1, 1, 1, 9)), g.view(1, 1, 9, 1))[:, 0]
    return torch.einsum("fj,jhw->fhw", wt, x)

def _redundancy(d, bank):
    q, b = _tok(d), _tok(bank)
    m = torch.full((q.shape[0],), float("inf"), device=q.device)
    for s in range(0, b.shape[0], 8192):
        m = torch.minimum(m, torch.cdist(q, b[s:s + 8192]).amin(1))
    return m

def _sparse_forward(model, kw, idx, grid, sp):
    f, h, w = grid
    L = f * h * w
    t = kw["t"]
    x = model.patch_embedding(torch.cat([kw["x"][0], kw["y"][0]])[None]).flatten(2).transpose(1, 2)[:, idx]
    n = x.shape[1]
    with torch.amp.autocast("cuda", dtype=torch.float32):
        e = model.time_embedding(mf.sinusoidal_embedding_1d(model.freq_dim, t.flatten()).unflatten(0, (1, 1)).float())
        e0 = model.time_projection(e).unflatten(2, (6, model.dim))
    if os.environ.get("KESTREL_E_BCAST") != "1":
        e, e0 = e.expand(1, n, -1), e0.expand(1, n, 6, -1)
    context = model.text_embedding(torch.stack([torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                                                for u in kw["context"]]))
    sp.update(idx=idx, L=L)
    if "count" not in sp:
        sp.update(cos=sp["cos_full"][idx], sin=sp["sin_full"][idx], pos=sp["lend"] - L + idx)
    dcd = kw.get("dit_cond_dict")
    grid_sizes = torch.tensor([[f, h, w]], dtype=torch.long)
    try:
        ST["sparse"] = sp
        if dcd is not None and "c2ws_plucker_emb" in dcd:
            pz = model.patch_size
            p = torch.cat([rearrange(i, "1 c (f c1) (h c2) (w c3) -> 1 (f h w) (c c1 c2 c3)", c1=pz[0], c2=pz[1], c3=pz[2])
                           for i in dcd["c2ws_plucker_emb"]], dim=1)[:, idx]
            p = _rows(model.patch_embedding_wancamctrl, p)
            dcd = dict(dcd, c2ws_plucker_emb=p + _rows(model.c2ws_hidden_states_layer2,
                                                        F.silu(_rows(model.c2ws_hidden_states_layer1, p))))
        args = dict(e=e0, seq_lens=torch.tensor([n], dtype=torch.long), grid_sizes=grid_sizes, freqs=model.freqs,
                    context=context, context_lens=None, dit_cond_dict=dcd, max_attention_size=kw["max_attention_size"],
                    frame_seqlen=kw["frame_seqlen"], cross_attn_first_call=kw["cross_attn_first_call"])
        for i, blk in enumerate(model.blocks):
            x = blk(x, kv_cache=kw["kv_cache"][i], crossattn_cache=kw["crossattn_cache"][i],
                    current_start=kw["current_start"], **args)
    finally:
        ST["sparse"] = None
    out = model.head(x, e)[0]
    if "count" in sp:
        full = expand_rows(out, sp["rank"], sp["model_active"])
    else:
        full = out.new_zeros(L, out.shape[-1])
        full[idx] = out
    return model.unpatchify(full[None], grid_sizes)[0].float()

def _fit(cal, K):
    s = torch.cat([c[0] for c in cal], 1)
    ok = torch.cat([c[1] for c in cal], 1) <= EPS
    q = lambda v, y: float(v.float().flatten().kthvalue(max(1, round(float(y.float().mean()) * v.numel()))).values) if y.any() else -1.0
    tau_red = q(s[0], ok[0])
    pend = ~ok[0]
    ps, py = [], []
    for k in range(1, K - 1):
        ps.append(s[k][pend]); py.append(ok[k][pend])
        pend = pend & ~ok[k]
    tau_stab = q(torch.cat(ps), torch.cat(py)) if ps else -1.0
    first = lambda acc: torch.where(acc.any(0), acc.float().argmax(0) + 1, torch.full_like(acc[0], K, dtype=torch.long))
    tf = torch.cat([s[:1] <= tau_red, s[1:] <= tau_stab])
    return tau_red, tau_stab, {"oracle_mean_steps": round(float(first(ok).float().mean()), 3),
                               "oracle_accept_rate": [round(float(v), 3) for v in ok.float().mean(1)],
                               "tf_mean_steps_est": round(float(first(tf).float().mean()), 3)}

def install(pipe):
    model, inner, gen = pipe.model, pipe.model.forward, pipe.generate
    K0 = inspect.signature(gen).parameters["timesteps_index"].default
    mf.CausalWanSelfAttention.forward = _sa_forward
    if SYNCFREE:
        mf.CausalWanSelfAttention._spec_sparse = staticmethod(lambda: ST["sparse"] is not None)
        from lingbot.ropekv_hook import install as install_rope
        install_rope(mf, ops)
    S = {"tau": [float(os.environ[k]) if k in os.environ else None for k in ("SPEC_TAU_RED", "SPEC_TAU_STAB")],
         "wrapped": False}
    ext = ops.rope_kv() if SYNCFREE else None

    def chunk_start(xt, kw):
        f, h, w = xt.shape[1], xt.shape[2] // 2, xt.shape[3] // 2
        if S["ref"] is None:
            S["ref"] = kw["y"][0][4:, :1].float()
        frames = [S["ref"]]
        if S["hist"]:
            H = torch.cat(S["hist"], 1)
            nh, sink, win = H.shape[1], min(pipe.sink_size, H.shape[1]), pipe.local_attn_size
            recent = nh if win < 0 else max(win - pipe.sink_size - f, 0)
            frames.append(H[:, sorted(set(range(sink)) | set(range(max(sink, nh - recent), nh)))])
        S.update(cs=kw["current_start"], k=0, grid=(f, h, w), bank=torch.cat(frames, 1), prev=None, sp=None, draws=[],
                 committed=torch.zeros(f, h, w, dtype=torch.bool, device=xt.device),
                 step=torch.full((f, h, w), S["K"], dtype=torch.long, device=xt.device),
                 score=torch.zeros(f, h, w, device=xt.device))
        if SYNCFREE:
            if kw["current_start"] == 0:
                for cache in kw["kv_cache"]:
                    cache["_spec_end"] = (0, 0)
            S.update(bank=distance_bank(_tok(S["bank"])), ts=pipe.scheduler.timesteps.to(xt.device).double(),
                     sigmas=pipe.scheduler.sigmas.to(xt.device).double())
            if not isinstance(S["steps"], torch.Tensor):
                S.update(steps=torch.zeros((), dtype=torch.long, device=xt.device),
                         act=torch.zeros(S["K"], dtype=torch.long, device=xt.device))

    def fwd(*a, **kw):
        xt, t = kw["x"][0], kw["t"]
        if kw["current_start"] != S["cs"]:
            chunk_start(xt, kw)
        if S["k"] == S["K"]:
            S["hist"].append(xt.float())
            return inner(*a, **kw)
        S["k"] = k = S["k"] + 1
        K, (f, h, w), committed = S["K"], S["grid"], S["committed"]
        act = ~committed
        if k == 1 or (SYNCFREE and S["calib"]):
            na = f * h * w
            pred = inner(*a, **kw)[0]
        elif SYNCFREE:
            if S["capacity"] == 0:
                na = 0
                pred = torch.zeros_like(xt)
            else:
                idx, na, count, rank, model_active = compact(act, S["capacity"])
                S["sp"].update(count=count, rank=rank, model_active=model_active)
                pred = _sparse_forward(model, kw, idx, (f, h, w), S["sp"])
        else:
            idx = act.flatten().nonzero()[:, 0]
            na = idx.numel()
            if na > f * h * w - PAD:
                pred = inner(*a, **kw)[0]
            elif na == 0:
                pred = torch.zeros_like(xt)
            else:
                if S["sp"] is None:
                    cos, sin = _rope_table(model.freqs, f, h, w, kw["current_start"] // kw["frame_seqlen"])
                    S["sp"] = {"cos_full": cos, "sin_full": sin, "lend": int(kw["kv_cache"][0]["local_end_index"])}
                pred = _sparse_forward(model, kw, torch.cat([idx, idx[-1:].expand((-na) % PAD)]), (f, h, w), S["sp"])
        S["act"][k - 1] += na
        ts = S["ts"] if SYNCFREE else pipe.scheduler.timesteps.to(xt.device).double()
        sigmas = S["sigmas"] if SYNCFREE else pipe.scheduler.sigmas.to(xt.device).double()
        sigma_index = torch.argmin((ts - t.double()).abs())
        sigma = sigmas.gather(0, sigma_index.reshape(1)).squeeze(0) if SYNCFREE else sigmas[sigma_index]
        draft = (xt.double() - sigma * pred.double()).float()
        if S["prev"] is not None and (SYNCFREE or na < f * h * w):
            cm = _lat(committed)[None]
            draft = torch.where(cm, S["prev"], draft)
            pred = torch.where(cm, ((xt.double() - S["prev"].double()) / sigma).float(), pred)
        if k < K:
            med = _tok(draft).norm(dim=1).median()
            if k == 1:
                s = redundancy(_tok(draft), S["bank"]) if SYNCFREE else _redundancy(draft, S["bank"])
            else:
                s = _tok(draft - S["prev"]).norm(dim=1)
            s = s / med
            S["score"] = torch.where(act, s.view(f, h, w), S["score"])
            sm = _smooth(S["score"])
            if S["calib"]:
                S["draws"].append((draft, sm.flatten()))
            else:
                acc = act & (sm <= S["tau"][0 if k == 1 else 1])
                committed |= acc
                if SYNCFREE:
                    S["step"] = torch.where(acc, k, S["step"])
                else:
                    S["step"][acc] = k
        elif S["calib"] and S["draws"]:
            ref = _tok(draft)
            med = ref.norm(dim=1).median()
            scores = torch.stack([v for _, v in S["draws"]])
            errors = torch.stack([(_tok(d) - ref).norm(dim=1) / med for d, _ in S["draws"]])
            S["cal"].append((scores, errors) if SYNCFREE else (scores.cpu(), errors.cpu()))
        if SYNCFREE and k == 1 and k < K and not S["calib"]:
            count, lend = torch.stack(((~committed).sum(), kw["kv_cache"][0]["local_end_index"].reshape(()))).cpu().tolist()
            S["capacity"] = bucket_capacity(count, f * h * w)
            cos, sin = _rope_table(model.freqs, f, h, w, kw["current_start"] // kw["frame_seqlen"])
            S["sp"] = {"cos_full": cos.contiguous(), "sin_full": sin.contiguous(), "lend": lend, "ext": ext}
        if k == K:
            S["steps"] += S["step"].sum() if SYNCFREE else int(S["step"].sum())
            S["toks"] += f * h * w
            S["chunks"] += 1
        S["prev"] = draft
        return [pred]

    @wraps(gen)
    def generate(*a, **kw):
        if not S["wrapped"]:
            for blk in model.blocks:
                for nm in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"):
                    setattr(blk, nm, _Rows(getattr(blk, nm)))
            S["wrapped"] = True
        K = len(kw.get("timesteps_index", K0))
        S.update(K=K, calib=None in S["tau"], cs=None, k=0, hist=[], ref=None, steps=0, toks=0, chunks=0,
                 act=[0] * K, cal=[])
        out = gen(*a, **kw)
        if S["calib"] and S["cal"]:
            if SYNCFREE:
                S["cal"] = [(scores.cpu(), errors.cpu()) for scores, errors in S["cal"]]
            tau_red, tau_stab, info = _fit(S["cal"], K)
            S["tau"] = [tau_red, tau_stab]
            print(json.dumps({"exp": "spec_calib", "eps": EPS, "tau_red": round(tau_red, 5), "tau_stab": round(tau_stab, 5),
                              "chunks": len(S["cal"]), **info}), flush=True)
        elif not S["calib"]:
            if SYNCFREE:
                stats = torch.cat((S["steps"][None], S["act"])).cpu().tolist()
                S["steps"], S["act"] = stats[0], stats[1:]
            mean = S["steps"] / max(S["toks"], 1)
            print(json.dumps({"exp": "spec", "eps": EPS, "tau_red": S["tau"][0], "tau_stab": S["tau"][1],
                              "chunks": S["chunks"], "mean_steps": round(mean, 3),
                              "active_frac_per_step": [round(v / max(S["toks"], 1), 3) for v in S["act"]],
                              "ideal_speedup": round((K + 1) / (mean + 1), 3)}), flush=True)
        return out

    pipe.model.forward = fwd
    pipe.generate = generate
    pipe._kv_reuse_commit = lambda xt: S["hist"].append(xt.float())
