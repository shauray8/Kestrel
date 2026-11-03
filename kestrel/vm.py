import math
import struct
from functools import cache

import torch

from kestrel import ops

HALT, GEMM_F8, GEMM_F8_GELU, MOD_NORM, RESIDUAL_MOD_NORM, GEMM_F8_GATED_RES, JOIN, ROPE_KV_APPEND, KV_ADVANCE, ATTN_WINDOW, QUANTIZE, RMS_NORM, GATED_RESIDUAL = range(13)
OPS = ("halt", "gemm_f8", "gemm_f8_gelu", "mod_norm", "residual_mod_norm", "gemm_f8_gated_res", "join", "rope_kv_append", "kv_advance", "attn_window", "quantize", "rms_norm", "gated_residual")
ERRORS = ("ok", "timeout", "opcode", "overflow")
BM, BN, RB, AQ, MAXT = 128, 128, 8, 64, 16
_DT = {torch.float8_e4m3fn: 0, torch.bfloat16: 1, torch.float32: 2}

def _cdiv(a, b):
    return (a + b - 1) // b

@cache
def grid():
    return ops.vm().grid()

class Program:
    def __init__(self):
        self.tt, self.shapes, self.keep, self.instrs, self.stages, self.slots = [], [], [], [], [], {}

    @staticmethod
    def _row(t, scale, vscale):
        assert t.is_cuda and t.dim() == 2 and t.stride(1) == 1 and t.dtype in _DT and scale > 0
        assert vscale is None or (vscale.dtype == torch.float32 and vscale.is_contiguous() and vscale.numel() == t.size(0))
        bits = struct.unpack("<I", struct.pack("<f", scale))[0]
        return [t.data_ptr(), 0 if vscale is None else vscale.data_ptr(), t.stride(0) if t.size(0) > 1 else 0,
                t.size(0) | t.size(1) << 32, _DT[t.dtype] | bits << 32]

    def tensor(self, t, scale=1.0, vscale=None, name=None):
        assert len(self.tt) < MAXT
        self.keep += [t, vscale]
        self.tt.append(self._row(t, scale, vscale))
        self.shapes.append((t.size(0), t.size(1), t.dtype))
        if name:
            self.slots[name] = len(self.tt) - 1
        return len(self.tt) - 1

    def bind(self, name, t, scale=1.0, vscale=None):
        i = self.slots[name]
        assert (t.size(0), t.size(1), t.dtype) == self.shapes[i] and (vscale is None) == (self.tt[i][1] == 0)
        self.tt[i] = self._row(t, scale, vscale)

    def table(self):
        return [v for row in self.tt for v in row]

    def add(self, op, m=0, n=0, t=(), f=0.0, after=()):
        self.instrs.append([op, m, n, tuple(t) + (-1,) * (6 - len(t)), f, list(after)])
        return len(self.instrs) - 1

    def edge(self, src, dst):
        self.instrs[dst][5].append(src)

    def join(self, after):
        assert after
        return self.add(JOIN, after=after)

    def _stage(self, fn):
        b = len(self.instrs)
        out = fn()
        self.stages.append((b, len(self.instrs) - b))
        return out

    def gemm(self, a, w, out, bias=-1, gelu=False, gate=-1, after=None):
        (M, K, da), (N, Kw, dw), (Mo, No, do) = self.shapes[a], self.shapes[w], self.shapes[out]
        assert da == dw == torch.float8_e4m3fn and K == Kw and K % 64 == 0 and N % BN == 0 and (Mo, No) == (M, N)
        if gate >= 0:
            R, C, dg = self.shapes[gate]
            assert not gelu and do == dg == torch.float32 and C == N and M % R == 0
            op = GEMM_F8_GATED_RES
        else:
            assert do == torch.float8_e4m3fn or (do == torch.bfloat16 and not gelu)
            op = GEMM_F8_GELU if gelu else GEMM_F8
        return self._stage(lambda: [[self.add(op, m, n, (a, w, bias, out, gate), after=after(m, n) if after else ())
                                     for n in range(N // BN)] for m in range(_cdiv(M, BM))])

    def rows(self, op, x, y=-1, gate=-1, shift=-1, scale=-1, out=-1, eps=1e-6, after=None):
        M, C, dx = self.shapes[x]
        assert dx == torch.float32 and C in (1024, 1536, 2048)
        for i in (gate, shift, scale):
            assert i < 0 or (self.shapes[i][1] == C and M % self.shapes[i][0] == 0 and self.shapes[i][2] == torch.float32)
        return self._stage(lambda: [self.add(op, m, 0, (x, y, gate, shift, scale, out), eps, after(m) if after else ())
                                    for m in range(_cdiv(M, RB))])

    def rope_kv_append(self, q, k, v, cs, kc, vc, write_offset, after=None):
        L, C, dq = self.shapes[q]
        assert dq == self.shapes[k][2] == self.shapes[v][2] == self.shapes[kc][2] == self.shapes[vc][2] == torch.bfloat16
        assert self.shapes[k] == self.shapes[v] == (L, C, dq)
        assert self.shapes[cs][0] == L and self.shapes[cs][2] == torch.float32 and C % self.shapes[cs][1] == 0 and C % 8 == 0
        assert self.shapes[kc] == self.shapes[vc] and self.shapes[kc][1] == C and 0 <= write_offset <= self.shapes[kc][0] - L
        return self._stage(lambda: [self.add(ROPE_KV_APPEND, m, write_offset, (q, k, v, cs, kc, vc),
                                                   after=after(m) if after else ()) for m in range(_cdiv(L, BM))])

    def kv_advance(self, k, v, sink, drop, used):
        assert self.shapes[k] == self.shapes[v] and self.shapes[k][2] == torch.bfloat16 and self.shapes[k][1] % 8 == 0
        assert 0 <= sink < used <= self.shapes[k][0] and drop > 0 and sink % 16 == drop % 16 == used % 16 == 0
        count = used - sink - drop
        tiles = [self.add(KV_ADVANCE, sink + i, sink + drop + i, (k, v), min(16, count - i))
                 for i in range(0, count, 16)]
        stride = drop // 16
        for i in range(len(tiles) - stride):
            self.edge(tiles[i], tiles[i + stride])
        return tiles
    def quantize(self, x, out, after=None):
        M, C, dx = self.shapes[x]
        assert dx in (torch.bfloat16, torch.float32) and self.shapes[out] == (M, C, torch.float8_e4m3fn)
        return self._stage(lambda: [self.add(QUANTIZE, m, 0, (x, out), after=after(m) if after else ())
                                    for m in range(_cdiv(M, RB))])

    def rms_norm(self, x, weight, out, eps=1e-6, after=None):
        M, C, dx = self.shapes[x]
        assert dx == torch.bfloat16 and self.shapes[weight] == (1, C, torch.bfloat16) and self.shapes[out] == (M, C, dx)
        return self._stage(lambda: [self.add(RMS_NORM, m, 0, (x, weight, out), eps, after=after(m) if after else ())
                                    for m in range(_cdiv(M, RB))])

    def gated_residual(self, x, y, gate=-1, after=None):
        return self.rows(GATED_RESIDUAL, x, y=y, gate=gate, after=after)
    def attn_window(self, q, k, v, out, after=None):
        Lq, C, dq = self.shapes[q]
        Lk, Ck, dk = self.shapes[k]
        assert dq == dk == self.shapes[v][2] == self.shapes[out][2] == torch.bfloat16
        assert C > 0 and C % 128 == 0 and self.shapes[v] == (Lk, C, dq) and self.shapes[out] == (Lq, C, dq) and Ck == C
        scale_log2 = math.log2(math.e) / math.sqrt(128)
        return self._stage(lambda: [[self.add(ATTN_WINDOW, m, h, (q, k, v, out), scale_log2,
                                                  after=after(m, h) if after else ())
                                     for h in range(C // 128)] for m in range(_cdiv(Lq, AQ))])

    def window(self, tiles, w):
        for m in range(w or len(tiles), len(tiles)):
            for n, i in enumerate(tiles[m]):
                self.edge(tiles[m - w][n], i)
    def self_attention(self, q, k, v, cs, kc, vc, lo, hi, write_offset, advance=None, norm_q=None, norm_k=None, eps=1e-6):
        out = torch.empty_like(q)
        qi, ki, vi, ci, kci, vci = (self.tensor(t, name=name) for t, name in
                                    zip((q, k, v, cs, kc, vc), ("q", "k", "v", "cs", "kc", "vc")))
        aki, avi = self.tensor(kc[lo:hi], name="ak"), self.tensor(vc[lo:hi], name="av")
        oi = self.tensor(out, name="out")
        done = []
        if norm_q is not None:
            w = self.tensor(norm_q, name="norm_q")
            done += self.rms_norm(qi, w, qi, eps)
        if norm_k is not None:
            w = self.tensor(norm_k, name="norm_k")
            done += self.rms_norm(ki, w, ki, eps)
        normalized = self.join(done) if done else None
        shifted = None
        if advance is not None:
            shifted = self.join(self.kv_advance(kci, vci, *advance))
        deps = [x for x in (normalized, shifted) if x is not None]
        app = self.rope_kv_append(qi, ki, vi, ci, kci, vci, write_offset, after=lambda m: deps)
        appended = self.join(app)
        self.attn_window(qi, aki, avi, oi, after=lambda m, h: [appended])
        return out

    def build(self, device="cuda"):
        n = len(self.instrs)
        dependents = [[] for _ in range(n)]
        for i, ins in enumerate(self.instrs):
            for p in ins[5]:
                dependents[p].append(i)
        assert all(dependents[i] for i, ins in enumerate(self.instrs) if ins[0] == JOIN)
        rows, deps = [], []
        for i, (op, m, nn, t, f, after) in enumerate(self.instrs):
            rows.append([op, m, nn, *t, struct.unpack("<i", struct.pack("<f", f))[0], len(after), len(deps), len(dependents[i]), 0, 0, 0])
            deps += dependents[i]
        roots = [i for i, ins in enumerate(self.instrs) if not ins[5]]
        assert all(self.instrs[i][0] != JOIN for i in roots)
        self.n, self.nq = n, sum(ins[0] != JOIN for ins in self.instrs)
        self.ins = torch.tensor(rows, dtype=torch.int32, device=device)
        self.deps = torch.tensor(deps, dtype=torch.int32, device=device)
        self.roots = torch.tensor(roots, dtype=torch.int32, device=device)
        self.state = torch.zeros(5 + 2 * n, dtype=torch.int32, device=device)
        self.noprof = torch.empty(0, dtype=torch.int64, device=device)
        return self

    def run(self, ctas=None, timeout_ms=200.0, prof=None, table=None):
        ops.vm().run(self.ins, table or self.table(), self.deps, self.roots, self.state, self.noprof if prof is None else prof, self.nq,
                     ctas or grid(), int(timeout_ms * 1e6))

    def run_tiles(self):
        for b, c in self.stages:
            ops.vm().tiles(self.ins, self.table(), b, c)

    def status(self):
        head, tail, done, err, info = self.state[:5].tolist()
        return {"err": ERRORS[err], "info": info, "done": done, "n": self.n, "enqueued": tail}
def self_attention_program(q, k, v, cs, kc, vc, lo, hi, write_offset, advance=None, norm_q=None, norm_k=None, eps=1e-6):
    p = Program()
    out = p.self_attention(q, k, v, cs, kc, vc, lo, hi, write_offset, advance, norm_q, norm_k, eps)
    return p.build(), out

def _quantized(p, x, sa, prefix="", after=None):
    x8 = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    xi, ti = p.tensor(x, name=prefix + "x"), p.tensor(x8, sa, name=prefix + "x8")
    qt = p.quantize(xi, ti, after=after)
    joins = [p.join(qt[m * BM // RB:(m + 1) * BM // RB]) for m in range(_cdiv(x.size(0), BM))]
    return ti, joins

def _linear(p, tx, joins, w, sw, b, y, prefix=""):
    return p.gemm(tx, p.tensor(w, vscale=sw, name=prefix + "w"), p.tensor(y, name=prefix + "y"),
                  p.tensor(b.view(1, -1), name=prefix + "b"), after=lambda m, n: [joins[m]])

def linear_program(x, sa, w, sw, b, y, norm=None):
    p = Program()
    tx, joins = _normed(p, x, sa, "", norm)
    _linear(p, tx, joins, w, sw, b, y)
    return p.build()

def self_qkv_program(x, shift, scale, eps, sa, wq, swq, bq, wk, swk, bk, wv, swv, bv, q, k, v):
    p = Program()
    norm = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    mod = p.rows(MOD_NORM, p.tensor(x, name="residual"), shift=p.tensor(shift, name="shift"),
                 scale=p.tensor(scale, name="scale"), out=p.tensor(norm, name="norm"), eps=eps)
    tx, joins = _quantized(p, norm, sa, after=lambda m: [mod[m]])
    _linear(p, tx, joins, wq, swq, bq, q, "q")
    _linear(p, tx, joins, wk, swk, bk, k, "k")
    _linear(p, tx, joins, wv, swv, bv, v, "v")
    return p.build()

def _normed(p, x, sa, prefix="", norm=None):
    if norm is None:
        return _quantized(p, x, sa, prefix)
    shift, scale, eps = norm
    y = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    rows = p.rows(MOD_NORM, p.tensor(x, name=prefix + "residual"), shift=p.tensor(shift, name=prefix + "shift"),
                  scale=p.tensor(scale, name=prefix + "scale"), out=p.tensor(y, name=prefix + "norm"), eps=eps)
    return _quantized(p, y, sa, prefix, after=lambda m: [rows[m]])

def cross_qkv_program(x, context, sa, ska, wq, swq, bq, wk, swk, bk, wv, swv, bv, q, k, v, norm=None):
    p = Program()
    tx, qjoins = _normed(p, x, sa, "q", norm)
    tk, kjoins = _quantized(p, context, ska, "k")
    _linear(p, tx, qjoins, wq, swq, bq, q, "q")
    _linear(p, tk, kjoins, wk, swk, bk, k, "k")
    _linear(p, tk, kjoins, wv, swv, bv, v, "v")
    return p.build()

def cross_attention_program(q, k, v, norm_q, norm_k=None, eps=1e-6):
    p = Program()
    qi, ki, vi = p.tensor(q, name="q"), p.tensor(k, name="k"), p.tensor(v, name="v")
    out = torch.empty_like(q)
    done = []
    if norm_q is not None:
        done += p.rms_norm(qi, p.tensor(norm_q, name="norm_q"), qi, eps)
    if norm_k is not None:
        done += p.rms_norm(ki, p.tensor(norm_k, name="norm_k"), ki, eps)
    normalized = p.join(done) if done else None
    p.attn_window(qi, ki, vi, p.tensor(out, name="out"), after=(lambda m, h: [normalized]) if normalized is not None else None)
    return p.build(), out

def o_residual_program(attn, sa, w, sw, b, y, x, gate):
    p = Program()
    tx, joins = _quantized(p, attn, sa)
    tiles = _linear(p, tx, joins, w, sw, b, y)
    done = [p.join(row) for row in tiles]
    p.gated_residual(p.tensor(x, name="residual"), p.tensor(y, name="out"), p.tensor(gate, name="gate"),
                     after=lambda m: [done[m * RB // BM]])
    return p.build()


def profile(p, it=10, ctas=None):
    prof = torch.zeros(4, len(OPS), dtype=torch.int64, device="cuda")
    for _ in range(it):
        p.run(ctas, prof=prof)
    wait, ex, sig, cnt = prof.tolist()
    return {OPS[o]: {"n": cnt[o] // it, "wait_us": wait[o] / cnt[o] / 1e3, "exec_us": ex[o] / cnt[o] / 1e3, "signal_us": sig[o] / cnt[o] / 1e3}
            for o in range(len(OPS)) if cnt[o]}

def ffn(p, tx, w0, sw0, b0, h8, s_mid, w2, sw2, b2, out, gate=-1, window=None, after=None):
    th = p.tensor(h8, s_mid, name="h8")
    up = p.gemm(tx, p.tensor(w0, vscale=sw0, name="w0"), th, p.tensor(b0.view(1, -1), name="b0"), gelu=True, after=after)
    mid = [p.join(row) for row in up]
    dn = p.gemm(th, p.tensor(w2, vscale=sw2, name="w2"), out, p.tensor(b2.view(1, -1), name="b2"), gate=gate, after=lambda m, n: [mid[m]])
    p.window(up, window)
    return up, dn

def ffn_program(x8, sa, w0, sw0, b0, h8, s_mid, w2, sw2, b2, y, window=None):
    p = Program()
    p.up, p.dn = ffn(p, p.tensor(x8, sa), w0, sw0, b0, h8, s_mid, w2, sw2, b2, p.tensor(y), window=window)
    return p.build()

def ffn_glue_program(x, ca, shift, scale, gate, h0, sa, w0, sw0, b0, h8, s_mid, w2, sw2, b2, eps=1e-6, window=None):
    p = Program()
    tx, th0 = p.tensor(x, name="x"), p.tensor(h0, sa, name="h0")
    p.pre = p.rows(RESIDUAL_MOD_NORM, tx, y=p.tensor(ca, name="ca"), shift=p.tensor(shift, name="shift"), scale=p.tensor(scale, name="scale"),
                   out=th0, eps=eps)
    pj = [p.join(p.pre[m * BM // RB:(m + 1) * BM // RB]) for m in range(_cdiv(x.size(0), BM))]
    p.up, p.dn = ffn(p, th0, w0, sw0, b0, h8, s_mid, w2, sw2, b2, tx, gate=p.tensor(gate, name="gate"), window=window,
                     after=lambda m, n: [pj[m]])
    return p.build()
