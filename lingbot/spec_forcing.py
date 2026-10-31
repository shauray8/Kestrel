import inspect, json, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import wan.modules.model_fast as mf

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
    q = _rope(self.norm_q(self.q(x)).view(b, s, n, d), sp).type_as(v)
    k = _rope(self.norm_k(self.k(x)).view(b, s, n, d), sp).type_as(v)
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
    return y[:, sp["idx"]] if sp is not None and y.shape[1] == sp["L"] else y

def _tok(x):
    c, f, h, w = x.shape
    return x.view(c, f, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(f * (h // 2) * (w // 2), c * 4)

def _lat(m):
    return m.repeat_interleave(2, 1).repeat_interleave(2, 2)

def _smooth(s):
    r = torch.arange(9, dtype=torch.float32, device=s.device) - 4
    g = torch.exp(-0.5 * r * r)
    g = g / g.sum()
    x = F.pad(s[:, None], (4, 4, 4, 4), mode="reflect")
    x = F.conv2d(F.conv2d(x, g.view(1, 1, 1, 9)), g.view(1, 1, 9, 1))[:, 0]
    i = torch.arange(s.shape[0], device=s.device)
    wt = torch.exp(-0.5 * (i[None] - i[:, None]).float() ** 2) * (i[None] >= i[:, None])
    return torch.einsum("fj,jhw->fhw", wt / wt.sum(1, keepdim=True), x)

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
    sp.update(idx=idx, L=L, cos=sp["cos_full"][idx], sin=sp["sin_full"][idx], pos=sp["lend"] - L + idx)
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
    S = {"tau": [float(os.environ[k]) if k in os.environ else None for k in ("SPEC_TAU_RED", "SPEC_TAU_STAB")],
         "wrapped": False}

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
        if k == 1:
            na = f * h * w
            pred = inner(*a, **kw)[0]
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
        ts = pipe.scheduler.timesteps.to(xt.device).double()
        sigma = pipe.scheduler.sigmas.to(xt.device).double()[torch.argmin((ts - t.double()).abs())]
        draft = (xt.double() - sigma * pred.double()).float()
        if S["prev"] is not None and na < f * h * w:
            cm = _lat(committed)[None]
            draft = torch.where(cm, S["prev"], draft)
            pred = torch.where(cm, ((xt.double() - S["prev"].double()) / sigma).float(), pred)
        if k < K:
            med = _tok(draft).norm(dim=1).median()
            s = (_redundancy(draft, S["bank"]) if k == 1 else _tok(draft - S["prev"]).norm(dim=1)) / med
            S["score"] = torch.where(act, s.view(f, h, w), S["score"])
            sm = _smooth(S["score"])
            if S["calib"]:
                S["draws"].append((draft, sm.flatten()))
            else:
                acc = act & (sm <= S["tau"][0 if k == 1 else 1])
                committed |= acc
                S["step"][acc] = k
        elif S["calib"] and S["draws"]:
            ref = _tok(draft)
            med = ref.norm(dim=1).median()
            S["cal"].append((torch.stack([v for _, v in S["draws"]]).cpu(),
                             torch.stack([(_tok(d) - ref).norm(dim=1) / med for d, _ in S["draws"]]).cpu()))
        if k == K:
            S["steps"] += int(S["step"].sum()); S["toks"] += f * h * w; S["chunks"] += 1
        S["prev"] = draft
        return [pred]

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
            tau_red, tau_stab, info = _fit(S["cal"], K)
            S["tau"] = [tau_red, tau_stab]
            print(json.dumps({"exp": "spec_calib", "eps": EPS, "tau_red": round(tau_red, 5), "tau_stab": round(tau_stab, 5),
                              "chunks": len(S["cal"]), **info}), flush=True)
        elif not S["calib"]:
            mean = S["steps"] / max(S["toks"], 1)
            print(json.dumps({"exp": "spec", "eps": EPS, "tau_red": S["tau"][0], "tau_stab": S["tau"][1],
                              "chunks": S["chunks"], "mean_steps": round(mean, 3),
                              "active_frac_per_step": [round(v / max(S["toks"], 1), 3) for v in S["act"]],
                              "ideal_speedup": round((K + 1) / (mean + 1), 3)}), flush=True)
        return out

    pipe.model.forward = fwd
    pipe.generate = generate
