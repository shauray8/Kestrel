import collections, fcntl, hashlib, json, os, random, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "8")

MODELS = os.environ.get("KESTREL_MODELS", "/workspace/models")
OUT = os.environ.get("KESTREL_OUT", "/workspace/samples/kestrel")
TAEHV = os.environ.get("KESTREL_TAEHV", "/workspace/taehv")
EXAMPLE = os.environ.get("EXAMPLE", "examples/05")
sys.argv = [sys.argv[0], "--task", "i2v-1.3B", "--size", "480*832",
            "--ckpt_dir", f"{MODELS}/lingbot13b", "--assets_dir", f"{MODELS}/lingbot_assets",
            "--local_attn_size", os.environ.get("WINDOW", "18"), "--sink_size", os.environ.get("SINK", "6"), "--offload_model", "False",
            "--image", f"{EXAMPLE}/image.jpg", "--action_path", EXAMPLE]
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
from PIL import Image
import wan
from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video
from generate import _parse_args, _init_logging
from kestrel import ops

ext = ops.n1()
N3 = os.environ.get("N3") == "1"
linear_ext = ops.n3() if N3 else ext
linear = linear_ext.n3_linear if N3 else ext.n1_linear
linear_gelu_f8 = linear_ext.n3_linear_gelu_f8 if N3 else ext.n1_linear_gelu_f8
linear_f8in = linear_ext.n3_linear_f8in if N3 else ext.n1_linear_f8in
linear_f8in_gelu_f8 = linear_ext.n3_linear_f8in_gelu_f8 if N3 else ext.n1_linear_f8in_gelu_f8
args = _parse_args()
_init_logging(0)
cfg = WAN_CONFIGS[args.task]
pipe = wan.WanI2VCausal(config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
                        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
                        convert_model_dtype=False, local_attn_size=args.local_attn_size,
                        sink_size=args.sink_size, infer_mode="causal_fast", assets_dir=args.assets_dir)

class T5Offload:
    def __init__(self, enc):
        self.enc = enc

    def __getattr__(self, name):
        return getattr(self.enc, name)

    def __call__(self, texts, device):
        out = self.enc(texts, device)
        self.enc.model.cpu()
        torch.cuda.empty_cache()
        return out

if os.environ.get("T5_CPU") != "1":
    pipe.t5_cpu = False
    pipe.text_encoder = T5Offload(pipe.text_encoder)
if os.environ.get("SPEC") == "1":
    from spec_forcing import install as spec_install
    spec_install(pipe)
E4 = torch.float8_e4m3fn
MODE = {"n1": False, "calib": False}
PATHS = collections.Counter()
TARGET = float(os.environ.get("AMAX_TARGET", "16"))

class N1Linear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin
        w = lin.weight.detach().float()
        sw = (w.abs().amax(1) / TARGET).clamp(min=1e-12)
        self.register_buffer("wq", (w / sw[:, None]).to(E4).contiguous(), persistent=False)
        self.register_buffer("sw", sw.float().contiguous(), persistent=False)
        self.bias_bf16 = None if lin.bias is None else lin.bias.detach().bfloat16().contiguous()
        self.amax = 0.0
        self.ok = (lin.in_features % 64 == 0) and (lin.out_features % 128 == 0)

    def forward(self, x):
        if MODE["calib"]:
            self.amax = max(self.amax, float(x.detach().abs().max()))
        if not (MODE["n1"] and self.ok and self.amax > 0):
            PATHS["linear_bf16"] += 1
            return self.lin(x)
        PATHS["linear_n3" if N3 else "linear_n1"] += 1
        xb = x.to(torch.bfloat16).contiguous()
        y = linear(xb, self.wq, self.sw, self.bias_bf16, self.amax / TARGET)
        return y.to(x.dtype)

mods = []
for blk in pipe.model.blocks:
    for name, mod in list(blk.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                n1 = N1Linear(child); setattr(mod, cname, n1); mods.append(n1)
print(json.dumps({"exp": "n1_e2e", "linears": len(mods), "eligible": sum(m.ok for m in mods)}), flush=True)

class FusedFFN(nn.Module):

    def __init__(self, seq):
        super().__init__()
        self.l0, self.act, self.l2 = seq[0], seq[1], seq[2]

    def forward(self, x):
        if MODE["n1"] and MODE.get("fused_ffn") and self.l0.ok and self.l2.ok and self.l0.amax > 0 and self.l2.amax > 0:
            PATHS["ffn_fused"] += 1
            s_mid = self.l2.amax / TARGET
            y8 = linear_gelu_f8(x.to(torch.bfloat16).contiguous(), self.l0.wq, self.l0.sw, self.l0.bias_bf16, self.l0.amax / TARGET, s_mid)
            return linear_f8in(y8, self.l2.wq, self.l2.sw, self.l2.bias_bf16, s_mid).to(x.dtype)
        PATHS["ffn_unfused"] += 1
        return self.l2(self.act(self.l0(x)))

    def forward_fp8(self, x):
        PATHS["ffn_fp8in"] += 1
        s_mid = self.l2.amax / TARGET
        y8 = linear_f8in_gelu_f8(x, self.l0.wq, self.l0.sw, self.l0.bias_bf16, self.l0.amax / TARGET, s_mid)
        return linear_f8in(y8, self.l2.wq, self.l2.sw, self.l2.bias_bf16, s_mid).to(torch.bfloat16)

if os.environ.get("FUSED_FFN") == "1":
    for blk in pipe.model.blocks:
        blk.ffn = FusedFFN(blk.ffn)
    MODE["fused_ffn"] = True

if os.environ.get("FP4_FFN") == "1":
    fp4 = ops.nvfp4()

    class FP4FFN(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.l0, self.act, self.l2 = (inner.l0, inner.act, inner.l2) if isinstance(inner, FusedFFN) else (inner[0], inner[1], inner[2])
            self.ready = False

        def prepare(self):
            def qw(lin):
                w = lin.lin.weight.detach().to(torch.bfloat16).contiguous()
                gw = float(w.float().abs().max()) / (6 * 448)
                wq, ws = fp4.nvfp4_quant(w, gw)
                return wq, ws, gw
            self.w0q, self.w0s, self.gw0 = qw(self.l0)
            self.w2q, self.w2s, self.gw2 = qw(self.l2)
            self.g0 = self.l0.amax / (6 * 448)
            self.gm = self.l2.amax / (6 * 448)
            self.b0 = self.l0.lin.bias.detach().bfloat16().contiguous()
            self.b2 = self.l2.lin.bias.detach().bfloat16().contiguous()
            self.ready = True

        def forward(self, x):
            if not (MODE.get("fp4") and self.ready):
                return self.inner(x)
            shp = x.shape
            x2 = x.reshape(-1, shp[-1]).to(torch.bfloat16).contiguous()
            xq, xs = fp4.nvfp4_quant(x2, self.g0)
            y = torch._scaled_mm(xq.view(torch.float4_e2m1fn_x2), self.w0q.view(torch.float4_e2m1fn_x2).t(),
                                 scale_a=xs.view(torch.float8_e4m3fn), scale_b=self.w0s.view(torch.float8_e4m3fn), out_dtype=torch.bfloat16)
            hq, hs = fp4.nvfp4_quant(y, self.gm, self.g0 * self.gw0, self.b0, True)
            o = torch._scaled_mm(hq.view(torch.float4_e2m1fn_x2), self.w2q.view(torch.float4_e2m1fn_x2).t(),
                                 scale_a=hs.view(torch.float8_e4m3fn), scale_b=self.w2s.view(torch.float8_e4m3fn), out_dtype=torch.bfloat16)
            o = torch.add(self.b2, o, alpha=self.gm * self.gw2)
            return o.reshape(*shp[:-1], -1).to(x.dtype)

    FP4_MODS = []
    for blk in pipe.model.blocks:
        blk.ffn = FP4FFN(blk.ffn); FP4_MODS.append(blk.ffn)

CAM = {"start": None, "hits": 0, "miss": 0}
if os.environ.get("CAM_HOIST") == "1":
    class ChunkCached(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.key, self.val = inner, None, None
            CAM.setdefault("mods", []).append(self)

        def forward(self, x):
            if self.key is not None and self.key == CAM["start"] and self.val is not None:
                CAM["hits"] += 1
                return self.val
            CAM["miss"] += 1
            self.val = self.inner(x)
            self.key = CAM["start"]
            return self.val

    _cam_fwd = pipe.model.forward

    def _cam_model_fwd(*a, **kw):
        CAM["start"] = (kw.get("current_start"), id(kw.get("dit_cond_dict", {}).get("c2ws_plucker_emb", [None])[0]) if kw.get("dit_cond_dict") else None)
        return _cam_fwd(*a, **kw)

    pipe.model.forward = _cam_model_fwd
    pm = pipe.model
    for nm in ("patch_embedding_wancamctrl", "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"):
        setattr(pm, nm, ChunkCached(getattr(pm, nm)))
    for blk in pm.blocks:
        for nm in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"):
            setattr(blk, nm, ChunkCached(getattr(blk, nm)))

if os.environ.get("FUSED_GLUE") == "1":
    fmod = ops.fused_mod()
    import wan.modules.model_fast as _mf_glue
    GLUE2 = os.environ.get("GLUE2") == "1"

    def _cam_at(self, emb, current_start):
        key = CAM["start"] if CAM["start"] is not None else (current_start, id(emb), emb.data_ptr())
        if getattr(self, "_g2cam", (None,))[0] != key:
            h = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(emb))) + emb
            self._g2cam = (key, (1.0 + self.cam_scale_layer(h)).float().contiguous(), self.cam_shift_layer(h).float().contiguous())
        return self._g2cam[1:]

    def _g2_ln(self):
        if getattr(self, "_g2ln", None) is None:
            n = self.norm3
            self._g2ln = (n.weight.float().contiguous(), n.bias.float().contiguous(), n.eps) \
                if isinstance(n, nn.LayerNorm) and n.weight is not None and n.bias is not None else False
        return self._g2ln

    _orig_xattn = _mf_glue.WanCrossAttention.forward

    def _g2_xattn_forward(self, x, context, context_lens, crossattn_cache=None, cross_attn_first_call=None):
        if x.dtype != torch.bfloat16 or getattr(self.norm_q, "weight", None) is None or self.norm_q.weight.dtype != torch.bfloat16:
            return _orig_xattn(self, x, context, context_lens, crossattn_cache, cross_attn_first_call)
        b, n, d = x.size(0), self.num_heads, self.head_dim
        quant = isinstance(self.q, N1Linear) and MODE["n1"] and self.q.ok and self.q.amax > 0
        q = fmod.rms_bf16(self.q(x).contiguous(), self.norm_q.weight, self.norm_q.eps, not quant).view(b, -1, n, d)
        if crossattn_cache is not None:
            first = crossattn_cache["is_init"].item() == 0 if cross_attn_first_call is None else cross_attn_first_call
            if first:
                crossattn_cache["is_init"].fill_(1)
                crossattn_cache["k"].copy_(self.norm_k(self.k(context)).view(b, -1, n, d))
                crossattn_cache["v"].copy_(self.v(context).view(b, -1, n, d))
            k, v = crossattn_cache["k"], crossattn_cache["v"]
        else:
            k, v = self.norm_k(self.k(context)).view(b, -1, n, d), self.v(context).view(b, -1, n, d)
        PATHS["xattn_glue2"] += 1
        return self.o(_mf_glue.flash_attention(q, k, v, k_lens=context_lens).flatten(2))

    if GLUE2:
        _mf_glue.WanCrossAttention.forward = _g2_xattn_forward

    def _glue_block_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None, kv_cache=None,
                            crossattn_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None,
                            cross_attn_first_call=None, seq_lens_int=None):
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [t.squeeze(2) for t in e]
        B, L, C = x.shape
        x = x.float().contiguous()
        x2 = x.view(B * L, C)
        ev = [t.reshape(-1, C) if t.shape[1] != 1 else t.reshape(1, C) for t in e]
        y = self.self_attn(fmod.mod_norm(x2, ev[0].contiguous(), ev[1].contiguous(), self.norm1.eps).view(B, L, C),
                           seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                           frame_seqlen=frame_seqlen, seq_lens_int=seq_lens_int)
        fmod.gated_residual(x2, y.reshape(B * L, C).to(torch.bfloat16).contiguous(), ev[2].contiguous())
        ln = _g2_ln(self) if GLUE2 else False
        a = t = None
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            if ln:
                a, t = _cam_at(self, dit_cond_dict["c2ws_plucker_emb"], current_start)
            else:
                c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
                h = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
                h = h + c2ws_plucker_emb
                x = (1.0 + self.cam_scale_layer(h)) * x + self.cam_shift_layer(h)
                x = x.float().contiguous()
                x2 = x.view(B * L, C)
        xn = fmod.cam_ln(x2, a, t, ln[0], ln[1], ln[2]).view(B, L, C) if ln else self.norm3(x)
        ca = self.cross_attn(xn, context, context_lens, crossattn_cache=crossattn_cache,
                             cross_attn_first_call=cross_attn_first_call)
        if (os.environ.get("FP8_GLUE") == "1" and isinstance(self.ffn, FusedFFN) and MODE["n1"] and
                self.ffn.l0.ok and self.ffn.l2.ok and self.ffn.l0.amax > 0 and self.ffn.l2.amax > 0):
            hin = fmod.residual_then_mod_norm_fp8(x2, ca.reshape(B * L, C).to(torch.bfloat16).contiguous(), None,
                                                  ev[3].contiguous(), ev[4].contiguous(), self.norm2.eps, self.ffn.l0.amax / TARGET)
            y = self.ffn.forward_fp8(hin.view(B, L, C))
        else:
            hin = fmod.residual_then_mod_norm(x2, ca.reshape(B * L, C).to(torch.bfloat16).contiguous(), None,
                                              ev[3].contiguous(), ev[4].contiguous(), self.norm2.eps)
            y = self.ffn(hin.view(B, L, C))
        fmod.gated_residual(x2, y.reshape(B * L, C).to(torch.bfloat16).contiguous(), ev[5].contiguous())
        return x

    _mf_glue.CausalWanAttentionBlock.forward = _glue_block_forward

if os.environ.get("VM_FFN") == "1" or os.environ.get("VM_BLOCK") == "1":
    if os.environ.get("VM_BLOCK") == "1":
        from vm_block import install as vm_block_install
        vm_block_install(globals())
    else:
        from vm_hook import install as vm_ffn_install
        vm_ffn_install(globals())

ATTN_IMPL = os.environ.get("ATTN", "n2")
if ATTN_IMPL in ("sage2", "kvq"):
    from sageattention import sageattn
if ATTN_IMPL == "sage3":
    from sageattn3 import sageattn3_blackwell
KVQ = FA = None
if ATTN_IMPL == "kvq":
    from kestrel.kvq import KVQAttention
    KVQ = KVQAttention()
if ATTN_IMPL == "fa":
    from kestrel.fa import FAttention
    FA = FAttention()
    ops.fa()
if ATTN_IMPL in ("kvq", "fa"):
    ATTN_STATE = {"layer": 0, "start": None}
    _orig_model_fwd = pipe.model.forward

    def _cached_attn_model_fwd(*a, **kw):
        cs = kw.get("current_start")
        if cs != ATTN_STATE["start"]:
            ATTN_STATE["start"] = cs
            (KVQ if KVQ is not None else FA).new_epoch()
        ATTN_STATE["layer"] = 0
        return _orig_model_fwd(*a, **kw)

    pipe.model.forward = _cached_attn_model_fwd

import wan.modules.model_fast as mf
ext2 = ops.n2()
_orig_attention = mf.attention

def n2_attention(q, k, v, *a, **kw):
    if not MODE.get("n2") or ATTN_IMPL == "sdpa" or q.shape[0] != 1 or q.shape[-1] != 128 or a or kw:
        PATHS["attn_stock_" + ("off" if not MODE.get("n2") else "args" if a or kw else "sdpa" if ATTN_IMPL == "sdpa" else "shape")] += 1
        return _orig_attention(q, k, v, *a, **kw)
    if ATTN_IMPL == "fa" and q.ndim == 4 and k.shape == v.shape and k.shape[0] == 1 and q.shape[2:] == k.shape[2:] and k.shape[1] >= q.shape[1]:
        PATHS["attn_fa"] += 1
        layer = ATTN_STATE["layer"]; ATTN_STATE["layer"] += 1
        return FA(layer, q, k, v)
    if ATTN_IMPL == "kvq" and k.shape[1] > q.shape[1]:
        PATHS["attn_kvq"] += 1
        layer = ATTN_STATE["layer"]; ATTN_STATE["layer"] += 1
        return KVQ(layer, q, k, v).to(q.dtype)
    if ATTN_IMPL in ("sage2", "kvq"):
        PATHS["attn_sage2"] += 1
        return sageattn(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), tensor_layout="NHD", is_causal=False).to(q.dtype)
    if ATTN_IMPL == "sage3":
        PATHS["attn_sage3"] += 1
        return sageattn3_blackwell(q.transpose(1, 2).to(torch.bfloat16),
                                  k.transpose(1, 2).to(dtype=torch.bfloat16, copy=True, memory_format=torch.contiguous_format),
                                  v.transpose(1, 2).to(torch.bfloat16), is_causal=False).transpose(1, 2).to(q.dtype)
    PATHS["attn_n2"] += 1
    o = ext2.n2_attention(q[0].half().contiguous(), k[0].half().contiguous(), v[0].half().contiguous())
    return o[None].to(q.dtype)

if os.environ.get("TAE_LIVE") == "1":
    sys.path.insert(0, TAEHV)
    from taehv import TAEHV as TAEHVModel, StreamingTAEHV
    _tae = TAEHVModel(f"{TAEHV}/taew2_1.pth").cuda().half()
    _stream = StreamingTAEHV(_tae)
    _orig_fwd = pipe.model.forward
    TAE_STATS = {"n": 0, "ms": 0.0, "frames": [], "t_start": [], "t_pix": [], "last_start": None}

    def _fwd_with_decode(*a, **kw):
        t = kw.get("t")
        cs = kw.get("current_start")
        is_kv = t is not None and float(t.float().abs().max()) == 0.0
        if MODE.get("tae") and not is_kv and cs != TAE_STATS["last_start"]:
            torch.cuda.synchronize()
            TAE_STATS["last_start"] = cs
            TAE_STATS["t_start"].append(time.perf_counter())
        if MODE.get("tae") and is_kv:
            x0 = kw["x"][0]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            first = None
            with torch.no_grad():
                for f in range(x0.shape[1]):
                    fr = _stream.decode(x0[:, f:f + 1].permute(1, 0, 2, 3)[None].half())
                    while fr is not None:
                        if first is None:
                            torch.cuda.synchronize(); first = time.perf_counter()
                        TAE_STATS["frames"].append(fr[0, :, :, ::2, ::2].float().cpu())
                        fr = _stream.decode()
            torch.cuda.synchronize()
            TAE_STATS["t_pix"].append(first if first is not None else time.perf_counter())
            TAE_STATS["n"] += x0.shape[1]; TAE_STATS["ms"] += 1e3 * (time.perf_counter() - t0)
        return _orig_fwd(*a, **kw)

    pipe.model.forward = _fwd_with_decode

mf.attention = n2_attention

if os.environ.get("KV_FP16") == "1":
    _orig_init_kv = pipe._initialize_self_kv_cache
    pipe._initialize_self_kv_cache = lambda num_layers, shape, dtype, device: _orig_init_kv(num_layers=num_layers, shape=shape, dtype=torch.half, device=device)
    RUNS_OVERRIDE = [("n2_calib_kvfp16", False, True, True), ("n2_kvfp16", False, False, True), ("n1n2_kvfp16", True, False, True)]
else:
    RUNS_OVERRIDE = None

if os.environ.get("ROPE32") == "1":
    _rope_cache = {}

    def rope32(x, grid_sizes, freqs, start_frame=0):
        n, c = x.size(2), x.size(3) // 2
        f, h, w = (int(v) for v in grid_sizes[0].tolist()) if not isinstance(grid_sizes, tuple) else grid_sizes
        key = (f, h, w, start_frame, c, x.device)
        if key not in _rope_cache:
            fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            fi = torch.cat([fs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                            fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                            fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
            _rope_cache.clear()
            _rope_cache[key] = (fi.real.float().to(x.device), fi.imag.float().to(x.device))
        cos, sin = _rope_cache[key]
        L = f * h * w
        xr = x[:, :L].float().reshape(x.shape[0], L, n, c, 2)
        a, b = xr[..., 0], xr[..., 1]
        out = torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(3)
        return torch.cat([out, x[:, L:].float()], 1).type_as(x)

    _orig_rope = mf.causal_rope_apply
    _xt = torch.randn(1, 2 * 30 * 52, 12, 128, device="cuda", dtype=torch.bfloat16)
    _gs = torch.tensor([[2, 30, 52]])
    _fr = pipe.model.freqs.to("cuda")
    _ref = _orig_rope(_xt, _gs, _fr, 5).float()
    print(json.dumps({"exp": "rope32_check", "max_abs_diff": (rope32(_xt, _gs, _fr, 5).float() - _ref).abs().max().item(),
                      "ref_amax": _ref.abs().max().item()}), flush=True)
    mf.causal_rope_apply = rope32
    RUNS_OVERRIDE = [(t + "_rope32", a, b, c) for t, a, b, c in (RUNS_OVERRIDE or [("n2_calib", False, True, True), ("n1n2", True, False, True)])]

KV_RESET = lambda: None

def _qkv(self, x):
    ls = (self.q, self.k, self.v)
    if not (N3 and MODE["n1"] and x.dtype == torch.bfloat16 and all(isinstance(l, N1Linear) and l.ok and l.amax > 0 for l in ls)
            and ls[0].amax == ls[1].amax == ls[2].amax and len({l.bias_bf16 is None for l in ls}) == 1):
        return None
    st = getattr(self, "_qkv_cat", None)
    if st is None or st[0] != ls[0].amax:
        bias = None if ls[0].bias_bf16 is None else torch.cat([l.bias_bf16 for l in ls]).contiguous()
        st = self._qkv_cat = (ls[0].amax, torch.cat([l.wq for l in ls]).contiguous(), torch.cat([l.sw for l in ls]).contiguous(), bias)
    PATHS["qkv_fused"] += 1
    y = linear_ext.n3_linear_split(x.contiguous(), st[1], st[2], st[3], st[0] / TARGET, ls[0].lin.out_features)
    return y[0], y[1], y[2]

if os.environ.get("ROPE_KV") == "1":
    from ropekv_hook import install as ropekv_install
    _fuse = _qkv if os.environ.get("QKV_FUSE") == "1" else None
    if os.environ.get("VM_ATTN") == "1":
        assert ATTN_IMPL == "n2", "VM_ATTN=1 requires ATTN=n2"
        from kestrel import vm as vm_runtime
        KV_RESET = ropekv_install(mf, ops, vm_runtime, PATHS, lambda: MODE.get("n2"), fused_norm=os.environ.get("RMS_ROPE") == "1", qkv=_fuse)
    else:
        KV_RESET = ropekv_install(mf, ops, paths=PATHS, fused_norm=os.environ.get("RMS_ROPE") == "1", qkv=_fuse)
if os.environ.get("XATTN") == "1":
    from xattn import install as xattn_install
    xattn_install(mf)

GR = None
if os.environ.get("GRAPHS"):
    from graphs import Graphs
    GR = Graphs(pipe.model, lambda: not MODE["calib"])

os.makedirs(OUT, exist_ok=True)
prompt = open(f"{EXAMPLE}/prompt.txt").read().strip()
img = Image.open(f"{EXAMPLE}/image.jpg").convert("RGB")
RUNS = RUNS_OVERRIDE or [("bf16_calib", False, True, False), ("bf16", False, False, False), ("n1", True, False, False),
                         ("n2", False, False, True), ("n1n2", True, False, True)]
if os.environ.get("FP4_FFN") == "1":
    RUNS = list(RUNS) + [(RUNS[-1][0] + "_fp4", True, False, True)]
if os.environ.get("ONLY"):
    RUNS = [r for r in RUNS if r[0] in os.environ["ONLY"].split(",")]
if os.environ.get("KESTREL_GPU_LOCK"):
    _gpu_lock = open(os.environ["KESTREL_GPU_LOCK"], "w")
    fcntl.flock(_gpu_lock, fcntl.LOCK_EX)

def after_calib():
    if os.environ.get("FP4_FFN") == "1":
        for m in FP4_MODS:
            m.prepare()
    if os.environ.get("FREE_BF16_W", "1") == "1":
        for m in mods:
            if m.ok and m.amax > 0 and m.lin.weight.numel():
                m.lin.weight.data = torch.empty(0, device=m.lin.weight.device, dtype=m.lin.weight.dtype)
        torch.cuda.empty_cache()
    if "VM_WARMUP" in globals():
        VM_WARMUP()

CALIB_FILE = None
if os.environ.get("CALIB_CACHE"):
    _keys = ("ATTN", "ROPE32", "FUSED_GLUE", "KESTREL_E_BCAST", "CAM_HOIST", "CHUNK", "STEPS_IDX", "WINDOW", "SINK", "EXAMPLE", "FRAMES", "SEED", "AMAX_TARGET", "SPEC", "ROPE_KV", "VM_ATTN", "VM_BLOCK", "XATTN", "RMS_ROPE", "GLUE2")
    if ATTN_IMPL == "fa":
        _keys += ("FA_MODE", "FA_LOAD_MODE", "FA_EXP_MODE", "FA_RESCALE_THRESHOLD", "FA_Q_TILE", "FA_MX", "FA_BAL", "FA_QK")
    _key = hashlib.sha1(json.dumps({k: os.environ.get(k) for k in _keys}, sort_keys=True).encode()).hexdigest()[:16]
    CALIB_FILE = os.path.join(os.environ["CALIB_CACHE"], _key + ".json")
    if os.path.exists(CALIB_FILE):
        for m, a in zip(mods, json.load(open(CALIB_FILE))):
            m.amax = a
        RUNS = [r for r in RUNS if not r[2]]
        CALIB_FILE = None
        after_calib()
        print(json.dumps({"exp": "calib_cache", "hit": True, "key": _key}), flush=True)
for tag, n1, calib, n2 in RUNS:
    MODE.update(n1=n1, calib=calib, n2=n2, tae=not calib, fp4=tag.endswith("_fp4"))
    if FA is not None:
        FA.new_epoch()
        ATTN_STATE.update(layer=0, start=None)
    for m in CAM.get("mods", []):
        m.key, m.val = None, None
    CAM.update(hits=0, miss=0)
    if os.environ.get("TAE_LIVE") == "1":
        TAE_STATS.update(n=0, ms=0.0, frames=[], t_start=[], t_pix=[], last_start=None)
        _stream.reset()
    prof, prof_t = None, []
    if os.environ.get("PROFILE") == "1" and tag == os.environ.get("PROFILE_TAG", RUNS[-1][0]):
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        if os.environ.get("PROFILE_DIT") == "1":
            _vae_decode, _model_fwd = pipe.vae.decode, pipe.model.forward

            def _forward_profiled(*a, **kw):
                if not prof_t:
                    torch.cuda.synchronize(); prof_t.append(time.perf_counter())
                    prof.__enter__()
                return _model_fwd(*a, **kw)

            def _decode_unprofiled(*a, **kw):
                if len(prof_t) == 1:
                    torch.cuda.synchronize(); prof_t.append(time.perf_counter())
                    prof.__exit__(None, None, None)
                return _vae_decode(*a, **kw)

            pipe.model.forward, pipe.vae.decode = _forward_profiled, _decode_unprofiled
        else:
            prof.__enter__()
    KV_RESET()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    chunk = int(os.environ.get("CHUNK", "3"))
    generate_fn = pipe.generate
    stream_stats = None
    if os.environ.get("STREAM") == "1":
        from streaming import generate as stream_generate
        stream_stats = {}
        generate_fn = lambda *a, **kw: stream_generate(pipe, *a, stats=stream_stats, **kw)
    video = generate_fn(prompt, img, action_path=EXAMPLE, chunk_size=chunk, max_area=MAX_AREA_CONFIGS[args.size],
                        frame_num=int(os.environ.get("FRAMES", "97")), shift=args.sample_shift, seed=int(os.environ.get("SEED", "42")), offload_model=False,
                        **({"timesteps_index": [int(v) for v in os.environ["STEPS_IDX"].split(",")]} if os.environ.get("STEPS_IDX") else {}))
    torch.cuda.synchronize(); t1 = time.perf_counter()
    if prof is not None:
        if os.environ.get("PROFILE_DIT") == "1":
            pipe.model.forward, pipe.vae.decode = _model_fwd, _vae_decode
            if len(prof_t) == 1:
                prof.__exit__(None, None, None); prof_t.append(t1)
        else:
            prof.__exit__(None, None, None); prof_t[:] = [t0, t1]
        from kestrel.timeline import summarize
        print(json.dumps({"exp": "timeline", "tag": tag, **summarize(prof, prof_t[1] - prof_t[0])}), flush=True)
    out = f"{OUT}/{os.path.basename(EXAMPLE)}_{tag}_c{chunk}.mp4"
    save_video(tensor=video[None], save_file=out, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
    if os.environ.get("TAE_LIVE") == "1" and TAE_STATS["n"]:
        live = torch.cat(TAE_STATS["frames"], 0)
        save_video(tensor=(live.permute(1, 0, 2, 3)[None] * 2 - 1), save_file=out.replace(".mp4", "_taelive.mp4"), fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        ts, tp = TAE_STATS["t_start"], TAE_STATS["t_pix"]
        k = min(len(ts), len(tp))
        s0 = min(4, k // 2)
        rng = random.Random(0)
        lo, hi = ts[s0], ts[k - 1]
        lat = []
        for _ in range(20000):
            e = rng.uniform(lo, hi)
            j = next(i for i in range(k) if ts[i] >= e)
            lat.append(tp[j] - e)
        lat.sort()
        gen = [tp[i] - ts[i] for i in range(k)]
        print(json.dumps({"exp": "tae_live", "latents": TAE_STATS["n"], "frames_out": int(live.shape[0]),
                          "tae_ms_per_latent": round(TAE_STATS["ms"] / TAE_STATS["n"], 2),
                          "chunk_start_to_first_pixel_ms_median": round(1e3 * sorted(gen[s0:])[len(gen[s0:]) // 2], 1),
                          "input_to_pixels_ms": {"mean": round(1e3 * sum(lat) / len(lat), 1), "p50": round(1e3 * lat[len(lat) // 2], 1),
                                                 "p95": round(1e3 * lat[int(0.95 * len(lat))], 1), "max": round(1e3 * lat[-1], 1)},
                          "fps_e2e": round(4 * TAE_STATS["n"] / (tp[k - 1] - ts[0]), 2)}), flush=True)
    print(json.dumps({"exp": "n1_e2e", "tag": tag, "generate_s": round(t1 - t0, 2), "out": out}), flush=True)
    if stream_stats is not None:
        print(json.dumps({"exp": "stream", "tag": tag, **stream_stats}), flush=True)
    if CAM.get("mods"):
        PATHS["cam_cache_hits"] = CAM["hits"]
        PATHS["cam_cache_misses"] = CAM["miss"]
    print(json.dumps({"exp": "paths", "tag": tag, **dict(sorted(PATHS.items()))}), flush=True)
    PATHS.clear()
    if GR:
        GR.report(tag); GR.reset()
    if not pipe.t5_cpu and next(pipe.text_encoder.model.parameters()).is_cuda:
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
    if calib:
        after_calib()
        if CALIB_FILE:
            os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
            json.dump([m.amax for m in mods], open(CALIB_FILE, "w"))
