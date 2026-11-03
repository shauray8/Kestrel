import os

import torch

from kestrel import vm

def install(g):
    assert os.environ.get("FUSED_GLUE") == "1" and os.environ.get("FUSED_FFN") == "1"
    assert os.environ.get("ROPE_KV") == "1" and os.environ.get("VM_ATTN") == "1" and os.environ.get("ATTN", "n2") == "n2"
    PATHS, MODE, TARGET, FusedFFN, fmod, orig, mf, pipe = (g[k] for k in
        ("PATHS", "MODE", "TARGET", "FusedFFN", "fmod", "_glue_block_forward", "_mf_glue", "pipe"))
    programs, scratch = {}, {}

    def lin_ok(l):
        return (MODE["n1"] and hasattr(l, "wq") and l.ok and l.amax > 0 and l.bias_bf16 is not None)

    def ready(blk):
        return (MODE["n2"] and isinstance(blk.ffn, FusedFFN) and isinstance(blk.norm3, torch.nn.Identity) and
                all(lin_ok(l) for l in (blk.self_attn.q, blk.self_attn.k, blk.self_attn.v, blk.self_attn.o,
                                       blk.cross_attn.q, blk.cross_attn.k, blk.cross_attn.v, blk.cross_attn.o,
                                       blk.ffn.l0, blk.ffn.l2)))

    def run(p):
        p.run()
        if not p.checked:
            torch.cuda.synchronize()
            st = p.status()
            assert st["err"] == "ok" and st["done"] == st["n"], st
            p.checked = True

    def self_qkv(blk, x, shift, scale):
        a = blk.self_attn
        M, C = x.shape
        sa = a.q.amax / TARGET
        key = ("self_qkv", id(blk), M, C, shift.size(0), sa, blk.norm1.eps)
        ent = programs.get(key)
        if ent is None:
            q, k, v = (torch.empty(M, C, dtype=torch.bfloat16, device=x.device) for _ in range(3))
            p = vm.self_qkv_program(x, shift, scale, blk.norm1.eps, sa, a.q.wq, a.q.sw, a.q.bias_bf16,
                                    a.k.wq, a.k.sw, a.k.bias_bf16, a.v.wq, a.v.sw, a.v.bias_bf16, q, k, v)
            p.checked = False
            ent = programs[key] = (p, q, k, v)
        p, q, k, v = ent
        p.bind("residual", x); p.bind("shift", shift); p.bind("scale", scale)
        run(p)
        return q, k, v

    def self_o_residual(blk, attn, x, gate):
        a, (M, C) = blk.self_attn, attn.shape
        key = ("self_o", id(blk), M, C, gate.size(0), a.o.amax / TARGET)
        ent = programs.get(key)
        if ent is None:
            y = torch.empty_like(attn)
            p = vm.o_residual_program(attn, a.o.amax / TARGET, a.o.wq, a.o.sw, a.o.bias_bf16, y, x, gate)
            p.checked = False
            ent = programs[key] = p
        p = ent
        p.bind("x", attn); p.bind("residual", x); p.bind("gate", gate)
        run(p)

    def cross_qkv(blk, x, context, first):
        a, (M, C), L = blk.cross_attn, x.shape, context.size(0)
        key = ("cross_qkv" if first else "cross_q", id(blk), M, C, L, a.q.amax / TARGET, a.k.amax / TARGET)
        ent = programs.get(key)
        if ent is None:
            q = torch.empty(M, C, dtype=torch.bfloat16, device=x.device)
            if first:
                k, v = (torch.empty(L, C, dtype=torch.bfloat16, device=x.device) for _ in range(2))
                p = vm.cross_qkv_program(x, context, a.q.amax / TARGET, a.k.amax / TARGET,
                                         a.q.wq, a.q.sw, a.q.bias_bf16, a.k.wq, a.k.sw, a.k.bias_bf16,
                                         a.v.wq, a.v.sw, a.v.bias_bf16, q, k, v)
            else:
                k = v = None
                p = vm.linear_program(x, a.q.amax / TARGET, a.q.wq, a.q.sw, a.q.bias_bf16, q)
            p.checked = False
            ent = programs[key] = (p, q, k, v)
        p, q, k, v = ent
        p.bind("qx" if first else "x", x)
        if first:
            p.bind("kx", context)
        run(p)
        return q, k, v

    def cross_attention(blk, q, k, v, first):
        a = blk.cross_attn
        key = ("cross_attn", id(blk), q.size(0), k.size(0), first, a.eps)
        ent = programs.get(key)
        if ent is None:
            nq = a.norm_q.weight.view(1, -1).contiguous()
            nk = a.norm_k.weight.view(1, -1).contiguous() if first else None
            p, out = vm.cross_attention_program(q, k, v, nq, nk, a.eps)
            p.checked = False
            ent = programs[key] = (p, out)
        p, out = ent
        p.bind("q", q); p.bind("k", k); p.bind("v", v)
        run(p)
        return out

    def cross_o(blk, attn):
        a, (M, C) = blk.cross_attn, attn.shape
        key = ("cross_o", id(blk), M, C, a.o.amax / TARGET)
        ent = programs.get(key)
        if ent is None:
            y = torch.empty_like(attn)
            p = vm.linear_program(attn, a.o.amax / TARGET, a.o.wq, a.o.sw, a.o.bias_bf16, y)
            p.checked = False
            ent = programs[key] = (p, y)
        p, y = ent
        p.bind("x", attn)
        run(p)
        return y

    def ffn_glue(blk, x, ca, mods):
        l0, l2 = blk.ffn.l0, blk.ffn.l2
        (M, C), FF, R = x.shape, l0.wq.size(0), mods[0].size(0)
        sa, sm = l0.amax / TARGET, l2.amax / TARGET
        key = ("ffn", id(blk), M, C, FF, R, sa, sm, blk.norm2.eps)
        ent = programs.get(key)
        if ent is None:
            bufs = scratch.get((M, C, FF))
            if bufs is None:
                E4 = torch.float8_e4m3fn
                bufs = scratch[(M, C, FF)] = (torch.empty(M, C, dtype=E4, device=x.device),
                                               torch.empty(M, FF, dtype=E4, device=x.device))
            h0, h8 = bufs
            p = vm.ffn_glue_program(x, ca, *mods, h0, sa, l0.wq, l0.sw, l0.bias_bf16,
                                    h8, sm, l2.wq, l2.sw, l2.bias_bf16, blk.norm2.eps)
            p.checked = False
            ent = programs[key] = p
        p = ent
        for name, t in zip(("x", "ca", "shift", "scale", "gate"), (x, ca, *mods)):
            p.bind(name, t)
        run(p)
        PATHS["ffn_vm"] += 1

    def block_forward(blk, x, x2, ev, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict,
                      kv_cache, crossattn_cache, current_start, max_attention_size, frame_seqlen,
                      cross_attn_first_call, seq_lens_int):
        B, L, C = x.shape
        if context_lens is not None and not (len(context_lens) == B and all(int(v) == context.size(1) for v in context_lens)):
            return None
        q, k, v = self_qkv(blk, x2, ev[0].contiguous(), ev[1].contiguous())
        a = blk.self_attn
        out = a.vm_attend(q.view(B, L, a.num_heads, a.head_dim), k.view(B, L, a.num_heads, a.head_dim),
                          v.view(B, L, a.num_heads, a.head_dim), seq_lens, grid_sizes, freqs, kv_cache,
                          current_start, max_attention_size, frame_seqlen, seq_lens_int, True)
        if out is None:
            return None
        self_o_residual(blk, out.reshape(B * L, C).contiguous(), x2, ev[2].contiguous())
        PATHS["block_vm_self"] += 1
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            emb = dit_cond_dict["c2ws_plucker_emb"]
            h = blk.cam_injector_layer2(torch.nn.functional.silu(blk.cam_injector_layer1(emb)))
            h = h + emb
            x = (1.0 + blk.cam_scale_layer(h)) * x + blk.cam_shift_layer(h)
            x = x.float().contiguous()
            x2 = x.view(B * L, C)
        first = cross_attn_first_call
        if first is None:
            first = crossattn_cache is None or int(crossattn_cache["is_init"].item()) == 0
        first = bool(first or crossattn_cache is None)
        context2 = context.reshape(-1, C).to(torch.bfloat16).contiguous()
        q, k, v = cross_qkv(blk, x2, context2, first)
        if not first:
            k = crossattn_cache["k"].reshape(-1, C)
            v = crossattn_cache["v"].reshape(-1, C)
        out = cross_attention(blk, q, k, v, first)
        if crossattn_cache is not None and first:
            crossattn_cache["k"].copy_(k.view_as(crossattn_cache["k"]))
            crossattn_cache["v"].copy_(v.view_as(crossattn_cache["v"]))
            crossattn_cache["is_init"].fill_(1)
        ca = cross_o(blk, out)
        mods = [t.contiguous() for t in ev[3:6]]
        ffn_glue(blk, x2, ca, mods)
        PATHS["block_vm_cross"] += 1
        PATHS["block_vm"] += 1
        return x

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None, kv_cache=None,
                crossattn_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None,
                cross_attn_first_call=None, seq_lens_int=None):
        if not ready(self):
            PATHS["block_vm_fallback"] += 1
            return orig(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, kv_cache, crossattn_cache,
                        current_start, max_attention_size, frame_seqlen, cross_attn_first_call, seq_lens_int)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [t.squeeze(2) for t in e]
        B, L, C = x.shape
        x = x.float().contiguous()
        x2 = x.view(B * L, C)
        ev = [t.reshape(-1, C) if t.shape[1] != 1 else t.reshape(1, C) for t in e]
        out = block_forward(self, x, x2, ev, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict,
                            kv_cache, crossattn_cache, current_start, max_attention_size, frame_seqlen,
                            cross_attn_first_call, seq_lens_int)
        if out is None:
            PATHS["block_vm_fallback"] += 1
            return orig(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, kv_cache, crossattn_cache,
                        current_start, max_attention_size, frame_seqlen, cross_attn_first_call, seq_lens_int)
        return out

    mf.CausalWanAttentionBlock.forward = forward
    print('{"exp": "vm_block_hook", "installed": true}', flush=True)
