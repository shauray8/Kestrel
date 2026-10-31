import os

import torch

from kestrel import vm

def install(g):
    if os.environ.get("VM_FFN") != "1":
        return
    assert os.environ.get("FUSED_GLUE") == "1" and os.environ.get("FUSED_FFN") == "1", "VM_FFN=1 needs FUSED_GLUE=1 FUSED_FFN=1"
    PATHS, MODE, TARGET, FusedFFN, fmod, orig, mf = g["PATHS"], g["MODE"], g["TARGET"], g["FusedFFN"], g["fmod"], g["_glue_block_forward"], g["_mf_glue"]
    progs, scratch, tabs = {}, {}, {}
    DYN = ("x", "ca", "shift", "scale", "gate")

    def ready(blk):
        f = blk.ffn
        return (MODE["n1"] and isinstance(f, FusedFFN) and f.l0.ok and f.l2.ok and f.l0.amax > 0 and f.l2.amax > 0
                and f.l0.bias_bf16 is not None and f.l2.bias_bf16 is not None)

    def setup(blk, x2, ca, mods, key, amax):
        l0, l2 = blk.ffn.l0, blk.ffn.l2
        (M, C), FF = x2.shape, l0.wq.size(0)
        sa, s_mid = amax[0] / TARGET, amax[1] / TARGET
        bufs = scratch.get((M, C, FF))
        if bufs is None:
            E4 = torch.float8_e4m3fn
            bufs = scratch[(M, C, FF)] = (torch.empty(M, C, dtype=E4, device=x2.device), torch.empty(M, FF, dtype=E4, device=x2.device))
        h0, h8 = bufs
        p = progs.get(key)
        if p is None:
            p = vm.ffn_glue_program(x2, ca, *mods, h0, sa, l0.wq, l0.sw, l0.bias_bf16, h8, s_mid, l2.wq, l2.sw, l2.bias_bf16, key[-1])
            p.keep = []
            p.checked = False
            progs[key] = p
        for k, t in zip(DYN, (x2, ca, *mods)):
            p.bind(k, t)
        p.bind("b0", l0.bias_bf16.view(1, -1)); p.bind("b2", l2.bias_bf16.view(1, -1))
        p.bind("h0", h0, sa); p.bind("h8", h8, s_mid); p.bind("w0", l0.wq, vscale=l0.sw); p.bind("w2", l2.wq, vscale=l2.sw)
        ent = tabs[(key, id(blk))] = (amax, p, p.table(), [5 * p.slots[k] for k in DYN])
        return ent

    def run_vm(blk, x2, ca, mods):
        key, amax = (x2.size(0), x2.size(1), mods[0].size(0), blk.norm2.eps), (blk.ffn.l0.amax, blk.ffn.l2.amax)
        ent = tabs.get((key, id(blk)))
        if ent is None or ent[0] != amax:
            ent = setup(blk, x2, ca, mods, key, amax)
        _, p, tab, slots = ent
        t = tab[:]
        for i, v in zip(slots, (x2, ca, *mods)):
            t[i] = v.data_ptr()
        p.run(table=t)
        if not p.checked:
            torch.cuda.synchronize()
            st = p.status()
            assert st["err"] == "ok" and st["done"] == st["n"], st
            p.checked = True
        PATHS["ffn_vm"] += 1

    warmed = False

    def warmup():
        nonlocal warmed
        if warmed:
            return
        M = int(os.environ.get("CHUNK", "3")) * 30 * 52
        made, bufs = set(), {}
        for blk in pipe.model.blocks:
            if not isinstance(blk.ffn, FusedFFN):
                continue
            C = blk.ffn.l0.wq.size(1)
            for R in (1, M):
                shape = (M, C, R)
                if shape not in bufs:
                    x = torch.empty(M, C, dtype=torch.float32, device=blk.ffn.l0.wq.device)
                    ca = torch.empty(M, C, dtype=torch.bfloat16, device=x.device)
                    bufs[shape] = (x, ca, [torch.empty(R, C, dtype=torch.float32, device=x.device) for _ in range(3)])
                x, ca, mods = bufs[shape]
                key = (M, C, R, blk.norm2.eps)
                amax = (blk.ffn.l0.amax or TARGET, blk.ffn.l2.amax or TARGET)
                setup(blk, x, ca, mods, key, amax)
                made.add(key)
        warmed = True
        print(f'{{"exp": "vm_warmup", "programs": {len(made)}}}', flush=True)

    pipe = g["pipe"]
    g["VM_WARMUP"] = warmup

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None, kv_cache=None,
                crossattn_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None,
                cross_attn_first_call=None, seq_lens_int=None):
        if not ready(self):
            return orig(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, kv_cache, crossattn_cache,
                        current_start, max_attention_size, frame_seqlen, cross_attn_first_call, seq_lens_int)
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
        if dit_cond_dict is not None and "c2ws_plucker_emb" in dit_cond_dict:
            c2ws_plucker_emb = dit_cond_dict["c2ws_plucker_emb"]
            h = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(c2ws_plucker_emb)))
            h = h + c2ws_plucker_emb
            x = (1.0 + self.cam_scale_layer(h)) * x + self.cam_shift_layer(h)
            x = x.float().contiguous()
            x2 = x.view(B * L, C)
        ca = self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache,
                             cross_attn_first_call=cross_attn_first_call)
        ca = ca.reshape(B * L, C).to(torch.bfloat16).contiguous()
        mods = [t.contiguous() for t in ev[3:6]]
        R = mods[0].size(0)
        if (C in (1024, 1536, 2048) and all(t.dtype == torch.float32 and t.size(0) == R for t in mods) and (B * L) % R == 0):
            run_vm(self, x2, ca, mods)
        else:
            PATHS["ffn_vm_fallback"] += 1
            hin = fmod.residual_then_mod_norm(x2, ca, None, mods[0], mods[1], self.norm2.eps)
            y = self.ffn(hin.view(B, L, C))
            fmod.gated_residual(x2, y.reshape(B * L, C).to(torch.bfloat16).contiguous(), mods[2])
        return x

    mf.CausalWanAttentionBlock.forward = forward
    print('{"exp": "vm_ffn_hook", "installed": true}', flush=True)
