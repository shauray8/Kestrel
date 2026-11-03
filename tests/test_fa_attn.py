import itertools, json, os
import pytest
import torch
from kestrel import ops
from kestrel.fa import FAttention

SHAPES = list(itertools.product((3120, 6240), (9360, 18720, 28080)))
LOADS = tuple(map(int, os.getenv("FA_TEST_LOAD_MODES", "0,1,3").split(",")))
EXPS = tuple(map(int, os.getenv("FA_TEST_EXP_MODES", "0,1,2").split(",")))
THRESHOLDS = tuple(map(float, os.getenv("FA_TEST_THRESHOLDS", "0,0.5").split(",")))
MODES = list(itertools.product(LOADS, EXPS, THRESHOLDS))
QUANT_REL = 0.15
KERNEL_REL = 0.05
VKEY = torch.tensor([(p & 48) + ((p >> 2) & 3) * 2 + (p & 1) + ((p >> 1) & 1) * 8 for p in range(64)]).argsort()
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@pytest.fixture(autouse=True)
def fp32_matmul():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old

def tensors(lq, lk, h=2, kind="random"):
    gen = torch.Generator(device="cuda").manual_seed(2026)
    q, k, v = (torch.randn(l, h, 128, device="cuda", dtype=torch.float16, generator=gen) for l in (lq, lk, lk))
    if kind == "constant":
        q.fill_(0.5); k.fill_(-0.25); v.fill_(0.375)
    elif kind == "zero":
        q.zero_(); k.zero_(); v.zero_()
    return q, k, v

def reference(q, k, v, scale=None, chunk=64):
    k, v = k.float().transpose(0, 1), v.float().transpose(0, 1)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return torch.cat([(torch.softmax((part.float().transpose(0, 1) @ k.transpose(1, 2)) * scale, dim=-1) @ v).transpose(0, 1)
                      for part in q.split(chunk)], dim=0)

def relative(a, b):
    delta, norm = (a.float() - b.float()).norm(), b.float().norm()
    return float(delta / norm.clamp_min(1e-12))

def pack(ext, x, transpose=False, target=8):
    l, h, d = x.shape
    n = (l + 63) // 64
    packed = torch.empty((h, n, d, 64) if transpose else (h, n, 64, d), device=x.device, dtype=torch.uint8)
    scales = torch.empty((h, n), device=x.device, dtype=torch.float32)
    ext.fa_quant(x, packed, scales, 0, n, transpose, target)
    return packed, scales

def unpack(packed, scales, length, transpose=False):
    x = packed.view(torch.float8_e4m3fn).float()
    if transpose:
        x = x[..., VKEY.to(x.device)].transpose(-1, -2)
    return (x * scales[..., None, None]).flatten(1, 2).transpose(0, 1)[:length].contiguous()

def packed_attention(ext, packed, lq, lk, mode, scale=128 ** -0.5):
    (q, qs), (k, ks), (v, vs) = packed
    return ext.fa_attention(q, k, v, qs, ks, vs, lq, lk, scale, *mode)

def check(q, k, v, mode, sage=False, label=None):
    ext = ops.fa()
    packed = (pack(ext, q), pack(ext, k), pack(ext, v, True, 1))
    out = packed_attention(ext, packed, q.shape[0], k.shape[0], mode)
    assert out.shape == q.shape and out.dtype == torch.float16
    assert torch.isfinite(out).all()
    ref = reference(q, k, v)
    dq, dk, dv = (unpack(p, s, x.shape[0], i == 2) for i, ((p, s), x) in enumerate(zip(packed, (q, k, v))))
    qref = reference(dq, dk, dv)
    quant_error, kernel_error, total_error = relative(qref, ref), relative(out, qref), relative(out, ref)
    row = {"kind": "fa_correctness", "case": label, "Lq": q.shape[0], "Lk": k.shape[0], "H": q.shape[1],
           "load_mode": mode[0], "exp_mode": mode[1], "rescale_threshold": mode[2],
           "rel_quant_fp32": quant_error, "rel_kernel_quant_fp32": kernel_error, "rel_fa_fp32": total_error,
           "max_abs_fa_fp32": float((out.float() - ref).abs().max())}
    if sage:
        from sageattention import sageattn
        sout = sageattn(q[None], k[None], v[None], tensor_layout="NHD", is_causal=False)[0]
        row.update(rel_sage_fp32=relative(sout, ref), rel_fa_sage=relative(out, sout))
        assert torch.isfinite(sout).all()
        assert row["rel_fa_sage"] <= QUANT_REL + row["rel_sage_fp32"]
    print(json.dumps(row), flush=True)
    assert quant_error <= QUANT_REL, row
    assert kernel_error <= KERNEL_REL, row
    assert total_error <= QUANT_REL + KERNEL_REL, row
    return out

@pytest.mark.parametrize("transpose,target", [(False, 8), (True, 1)])
def test_bf16_source_matches_fp16(transpose, target):
    ext = ops.fa()
    x = (tensors(193, 193)[0].float() * 3).bfloat16()
    a, sa = pack(ext, x, transpose, target)
    b, sb = pack(ext, x.half(), transpose, target)
    assert torch.equal(a, b) and torch.equal(sa, sb)

def test_bf16_model_inputs_need_no_fp16_copies():
    q, k, v = (x.bfloat16()[None] for x in tensors(65, 194))
    fa = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)
    out = fa(0, q, k, v)
    ref = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)(0, q.half(), k.half(), v.half())
    assert torch.equal(out, ref.to(out.dtype))
    assert not any(name in fa.state[0] for name in ("kf", "vf"))

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("lq,lk,kind", [(1, 1, "random"), (17, 65, "random"), (65, 193, "random"),
                                        (33, 129, "constant"), (63, 127, "zero")])
def test_fp8_small(mode, lq, lk, kind):
    out = check(*tensors(lq, lk, kind=kind), mode, label=kind)
    if kind == "zero":
        assert torch.count_nonzero(out) == 0
    if kind == "constant":
        torch.testing.assert_close(out, torch.full_like(out, 0.375), atol=0.001, rtol=0)

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp16_direct_n2_bitwise(dtype):
    q, k, v = (x.to(dtype)[None] for x in tensors(65, 193))
    fa = FAttention(mode="fp16")
    out = fa(0, q, k, v)
    ref = ops.n2().n2_attention(*(x[0].half().contiguous() for x in (q, k, v)))[None].to(dtype)
    assert torch.equal(out, ref)
    assert fa.stats["hist_quant"] == 0

@pytest.mark.parametrize("transpose,target", [(False, 8), (True, 1)])
def test_quant_blocks_and_padding(transpose, target):
    ext = ops.fa()
    x = tensors(65, 65)[0]
    packed, scales = pack(ext, x, transpose, target)
    padded = torch.zeros((128, x.shape[1], 128), device="cuda", dtype=torch.float32)
    padded[:65] = x.float()
    blocks = padded.reshape(2, 64, x.shape[1], 128).permute(2, 0, 1, 3).contiguous()
    expected_scales = blocks.abs().amax(dim=(-1, -2)) / target
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=1e-6)
    expected = (blocks / scales[..., None, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    actual = packed[..., VKEY.to(packed.device)].transpose(-1, -2).contiguous() if transpose else packed
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual[:, 1, 1:]) == 0
    before, sbefore = packed.clone(), scales.clone()
    x[64:].mul_(2)
    ext.fa_quant(x, packed, scales, 1, 2, transpose, target)
    assert torch.equal(packed[:, :1], before[:, :1])
    assert torch.equal(scales[:, :1], sbefore[:, :1])
    full, sfull = pack(ext, x, transpose, target)
    assert torch.equal(packed, full) and torch.equal(scales, sfull)

@pytest.mark.parametrize("transpose,target", [(False, 8), (True, 1)])
def test_quant_zero_scale(transpose, target):
    packed, scales = pack(ops.fa(), tensors(65, 65, kind="zero")[0], transpose, target)
    assert torch.count_nonzero(packed) == 0
    torch.testing.assert_close(scales, torch.full_like(scales, 1e-12), atol=0, rtol=0)

@pytest.mark.parametrize("mode", MODES)
def test_increasing_maxima(mode):
    q, k, v = tensors(65, 384)
    q.fill_(0.2)
    for block in range(6):
        k[block * 64:(block + 1) * 64].fill_(block * 0.06125)
    check(q, k, v, mode, label="increasing_maxima")

@pytest.mark.parametrize("mode", MODES)
def test_query_tile_parity(mode):
    ext = ops.fa()
    q, k, v = tensors(129, 385)
    (q8, qs), (k8, ks), (v8, vs) = pack(ext, q), pack(ext, k), pack(ext, v, True, 1)
    plan = ext.fa_plan(k8, v8) if mode[0] in (1, 3) else None
    outputs = []
    for tile in (64, 128):
        out = ext.fa_attention(q8, k8, v8, qs, ks, vs, 129, 385, 128 ** -0.5, *mode, plan, tile)
        integrated = FAttention(mode="fp8", load_mode=mode[0], exp_mode=mode[1], rescale_threshold=mode[2], q_tile=tile)
        assert torch.equal(out, integrated(0, q[None], k[None], v[None])[0])
        assert torch.isfinite(out).all()
        outputs.append(out)
    parity = relative(outputs[0], outputs[1])
    error = relative(outputs[0], reference(q, k, v))
    print(json.dumps({"kind": "fa_query_tile", "mode": mode, "rel_64_128": parity, "rel_64_fp32": error}), flush=True)
    assert parity <= KERNEL_REL and error <= QUANT_REL + KERNEL_REL

@pytest.mark.parametrize("mode", MODES)
def test_nondefault_stream(mode):
    ops.fa()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q, k, v = tensors(33, 129)
        q.mul_(0.75); k.mul_(1.25)
        fa = FAttention(mode="fp8", load_mode=mode[0], exp_mode=mode[1], rescale_threshold=mode[2])
        out = fa(0, q[None], k[None], v[None])[0]
        ref = reference(q, k, v)
    torch.cuda.current_stream().wait_stream(stream)
    error = relative(out, ref)
    print(json.dumps({"kind": "fa_stream", "mode": mode, "relative_error": error}), flush=True)
    assert torch.isfinite(out).all() and error <= QUANT_REL + KERNEL_REL

class QuantTrace:
    def __init__(self, ext):
        self.ext, self.calls = ext, []
    def fa_quant(self, x, packed, scales, start, stop, transpose, target):
        self.calls.append((x.data_ptr(), start, stop, transpose, target))
        return self.ext.fa_quant(x, packed, scales, start, stop, transpose, target)
    def __getattr__(self, name):
        return getattr(self.ext, name)

def test_history_reuse_epoch_and_boundary(monkeypatch):
    trace = QuantTrace(ops.fa())
    monkeypatch.setattr(ops, "fa", lambda: trace)
    fa = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)
    q, k, v = tensors(65, 194)
    fa(0, q[None], k[None], v[None])
    assert fa.stats["hist_quant"] == 1
    prefix = [(call[1], call[2], call[3], call[4]) for call in trace.calls if call[0] in (k.data_ptr(), v.data_ptr())]
    assert (0, 2, False, 8) in prefix and (0, 2, True, 1) in prefix
    k[129:].mul_(3); v[129:].add_(0.5)
    trace.calls.clear()
    out = fa(0, q[None], k[None], v[None])
    assert fa.stats["hist_quant"] == 1
    tail = [call for call in trace.calls if call[0] in (k.data_ptr(), v.data_ptr())]
    assert len(tail) == 2 and all(call[1:3] == (2, 4) for call in tail)
    fresh = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)(0, q[None], k[None], v[None])
    assert torch.equal(out, fresh)
    k[:129].mul_(-2); v[:129].add_(1)
    old_epoch = fa.epoch
    fa.new_epoch()
    assert fa.epoch == old_epoch + 1
    trace.calls.clear()
    out = fa(0, q[None], k[None], v[None])
    assert fa.stats["hist_quant"] == 2
    assert sum(call[1:3] == (0, 2) for call in trace.calls if call[0] in (k.data_ptr(), v.data_ptr())) == 2
    fresh = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)(0, q[None], k[None], v[None])
    assert torch.equal(out, fresh)
    fa(1, q[None], k[None], v[None])
    assert fa.stats["hist_quant"] == 3
    fa(0, q[:64][None], k[None], v[None])
    assert fa.stats["hist_quant"] == 4
    q2, k2, v2 = tensors(65, 258)
    fa(0, q2[None], k2[None], v2[None])
    assert fa.stats["hist_quant"] == 5
    qb, kb, vb = (x.bfloat16()[None] for x in (q2, k2, v2))
    fa(0, qb, kb, vb)
    assert fa.stats["hist_quant"] == 6
    kb[:, 193:].mul_(2); vb[:, 193:].add_(0.25)
    out = fa(0, qb, kb, vb)
    assert fa.stats["hist_quant"] == 6
    fresh = FAttention(mode="fp8", load_mode=0, exp_mode=0, rescale_threshold=0)(0, qb, kb, vb)
    assert torch.equal(out, fresh)

def test_multicast_rejected():
    with pytest.raises((ValueError, RuntimeError), match="(?i)multicast|unsupported|load.mode"):
        FAttention(mode="fp8", load_mode=2)
    ext = ops.fa()
    q, k, v = tensors(1, 64)
    packed = (pack(ext, q), pack(ext, k), pack(ext, v, True, 1))
    with pytest.raises((ValueError, RuntimeError), match="(?i)multicast|unsupported|load.mode"):
        packed_attention(ext, packed, 1, 64, (2, 0, 0))

@pytest.mark.expensive
@pytest.mark.skipif(os.getenv("FA_EXPENSIVE") != "1", reason="set FA_EXPENSIVE=1 for full target shapes and Sage2")
@pytest.mark.parametrize("lq,lk", SHAPES)
@pytest.mark.parametrize("mode", MODES)
def test_target_shapes_sage2(lq, lk, mode):
    pytest.importorskip("sageattention")
    check(*tensors(lq, lk, h=12), mode, sage=True, label="lingbot")
