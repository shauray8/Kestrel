import json, os, time
import torch
import wan.modules.model_fast as mf
from kestrel import ops

def _sinusoid(dim, position):
    position = position.type(torch.float64)
    half = dim // 2
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, device=position.device).to(position).div(half)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

class Graphs:
    def __init__(self, model, ok):
        fwd = mf.CausalWanSelfAttention.forward
        assert fwd.__module__ in (mf.__name__, "ropekv_hook"), "GRAPHS replaces CausalWanSelfAttention.forward"
        self.rkv = ops.rope_kv() if fwd.__module__ == "ropekv_hook" else None
        self.model, self.ok, self.inner, self.rope = model, ok, model.forward, mf.causal_rope_apply
        self.capture, self.static, self.probed = os.environ.get("GRAPHS") == "1", False, False
        self.rope32 = os.environ.get("ROPE32") == "1" or self.rkv is not None
        mf.causal_rope_apply, mf.sinusoidal_embedding_1d = self._rope, _sinusoid
        mf.CausalWanSelfAttention.forward = lambda blk, *a, **kw: self.self_attn(blk, *a, **kw)
        model.forward = self.forward
        self.reset()

    def self_attn(self, blk, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000,
                  frame_seqlen=None, seq_lens_int=None):
        b, s, n, d = *x.shape[:2], blk.num_heads, blk.head_dim
        q = blk.norm_q(blk.q(x)).view(b, s, n, d)
        k = blk.norm_k(blk.k(x)).view(b, s, n, d)
        v = blk.v(x).view(b, s, n, d)
        f, h, w = (int(u) for u in grid_sizes[0].tolist())
        fs = frame_seqlen or h * w
        kk, vv, sf = kv_cache["k"], kv_cache["v"], current_start // fs
        end, sink, size = current_start + s, blk.sink_size * fs, kk.shape[1]
        g, l = kv_cache.get("g", 0), kv_cache.get("l", 0)
        le = l + end - g
        if end > g and s + l > size:
            ev = s + l - size
            ro = l - ev - sink
            kk[:, sink:sink + ro] = kk[:, sink + ev:sink + ev + ro].clone()
            vv[:, sink:sink + ro] = vv[:, sink + ev:sink + ev + ro].clone()
            le -= ev
        if self.rkv is not None and all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (q, k, v, kk, vv)):
            self._set_tab(sf, f, h, w)
            rq = self.rkv.rope_qk_append(q, k, v, *self.tab, kk, vv, le - s)
        else:
            rq = mf.causal_rope_apply(q, grid_sizes, freqs, start_frame=sf).type_as(v)
            kk[:, le - s:le] = mf.causal_rope_apply(k, grid_sizes, freqs, start_frame=sf).type_as(v)
            vv[:, le - s:le] = v
        lo = max(0, le - max_attention_size)
        x = mf.attention(rq, kk[:, lo:le], vv[:, lo:le])
        kv_cache["g"], kv_cache["l"] = end, le
        return blk.o(x.flatten(2))

    def _probe(self):
        a = torch.randn(2, 1, 256, 12, 128, device="cuda", dtype=torch.bfloat16)
        st, gr = a[0].clone(), torch.cuda.CUDAGraph()
        mf.attention(st, st, st)
        with torch.cuda.graph(gr):
            o = mf.attention(st, st, st)
        st.copy_(a[1]); gr.replay()
        if not torch.equal(o, mf.attention(st, st, st)):
            raise RuntimeError("self-attention kernel escapes CUDA graph capture (launches off the current stream); "
                               "for sageattention apply lingbot/patches/sage_stream.diff and rebuild")
        self.probed = True

    def reset(self):
        self.graphs, self.pool, self.tab, self.tab_key, self.roll_cs = {}, None, None, None, None
        self.stats = {"replays": 0, "eager": 0, "captures": 0, "capture_s": 0.0}

    def report(self, tag):
        print(json.dumps({"exp": "graphs", "tag": tag, "mode": os.environ.get("GRAPHS"), **self.stats,
                          "capture_s": round(self.stats["capture_s"], 3)}), flush=True)

    def _rope(self, x, grid_sizes, freqs, start_frame=0):
        if not self.static:
            return self.rope(x, grid_sizes, freqs, start_frame)
        if self.rope32:
            cos, sin = self.tab
            xr = x.float().reshape(*x.shape[:3], -1, 2)
            a, b = xr[..., 0], xr[..., 1]
            return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(3).type_as(x)
        return torch.view_as_real(torch.view_as_complex(x[0].to(torch.float64).reshape(*x.shape[1:3], -1, 2)) * self.tab).flatten(2)[None].type_as(x)

    def _set_tab(self, sf, f, h, w):
        if self.tab_key == (sf, f, h, w):
            return
        c = self.model.freqs.shape[1]
        fq = self.model.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        fi = torch.cat([fq[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1), fq[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                        fq[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
        if self.tab is None:
            self.tab = (fi.real.float(), fi.imag.float()) if self.rope32 else fi
        elif self.rope32:
            self.tab[0].copy_(fi.real); self.tab[1].copy_(fi.imag)
        else:
            self.tab.copy_(fi)
        self.tab_key = (sf, f, h, w)

    def forward(self, **kw):
        x, kvs, cs, fs = kw["x"][0], kw["kv_cache"], kw["current_start"], kw["frame_seqlen"]
        s, g, l, size = fs * x.shape[1], kvs[0].get("g", 0), kvs[0].get("l", 0), kvs[0]["k"].shape[1]
        roll = cs + s > g
        if not (self.capture and self.ok() and not kw["cross_attn_first_call"] and l == size
                and (cs == g if roll else cs + s == g and self.roll_cs == cs)):
            self.stats["eager"] += 1
            return self.inner(**kw)
        p = self.model.patch_size
        self._set_tab(cs // fs, x.shape[1] // p[0], x.shape[2] // p[1], x.shape[3] // p[2])
        ins = (x, kw["t"], kw["context"][0], kw["y"][0], kw["dit_cond_dict"]["c2ws_plucker_emb"][0])
        key = (roll, tuple(x.shape))
        if key not in self.graphs:
            if not self.probed:
                self._probe()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            st = [torch.empty_strided(u.size(), u.stride(), dtype=u.dtype, device=u.device).copy_(u) for u in ins]
            gr = torch.cuda.CUDAGraph()
            self.static = True
            try:
                with torch.cuda.graph(gr, pool=self.pool):
                    out = self.inner(**{**kw, "x": [st[0]], "t": st[1], "context": [st[2]], "y": [st[3]],
                                        "dit_cond_dict": {**kw["dit_cond_dict"], "c2ws_plucker_emb": (st[4],)}})[0]
            finally:
                self.static = False
            self.pool = self.pool or gr.pool()
            self.graphs[key] = (gr, st, out)
            gr.replay(); torch.cuda.synchronize()
            self.stats["captures"] += 1; self.stats["capture_s"] += time.perf_counter() - t0
        else:
            gr, st, out = self.graphs[key]
            for a, b in zip(st, ins):
                a.copy_(b)
            gr.replay()
            self.stats["replays"] += 1
            for d in kvs:
                d["g"], d["l"] = cs + s, size
        if roll:
            self.roll_cs = cs
        return [out.clone()]
