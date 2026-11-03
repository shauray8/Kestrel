import json
import math
import os
import numpy as np

import torch

from kestrel import ops, vm
from lingbot.vm_schedule import cache_plan, vm_advance_eligible

def cross_norm(blk):
    n = blk.norm3
    if isinstance(n, torch.nn.Identity):
        return None
    C, dev = n.normalized_shape[0], blk.modulation.device
    w = n.weight.detach().float().view(1, C) if n.weight is not None else torch.ones(1, C, device=dev)
    b = n.bias.detach().float().view(1, C).contiguous() if n.bias is not None else torch.zeros_like(w)
    return b, (w - 1).contiguous(), n.eps



def block_parts(blk, x, e, ev, context, cross, target, buffers):
    M, C = x.shape
    mods = [ev[:, i * C:(i + 1) * C] for i in range(6)]
    mp = vm.modulation_program(e, blk.modulation.view(1, -1).float(), ev, build=False)
    q, k, v, out, projected, cq, ca, h0, h8 = buffers
    a = blk.self_attn
    linear = lambda l: (l.wq, l.sw, l.bias_bf16)
    sq = vm.self_qkv_program(x, mods[0], mods[1], blk.norm1.eps, a.q.amax / target,
                              *linear(a.q), *linear(a.k), *linear(a.v), q, k, v, build=False)
    so = vm.o_residual_program(out, a.o.amax / target, *linear(a.o), projected, x, mods[2], build=False)
    a = blk.cross_attn
    norm = cross_norm(blk)
    phases = []
    for first in (True, False):
        ck, cv = cross["k"].view(-1, C), cross["v"].view(-1, C)
        if first:
            cp = vm.cross_qkv_program(x, context, a.q.amax / target, a.k.amax / target,
                                       *linear(a.q), *linear(a.k), *linear(a.v), cq, ck, cv, norm=norm, build=False)
        else:
            cp = vm.linear_program(x, a.q.amax / target, *linear(a.q), cq, norm=norm, build=False)
        ap, ao = vm.cross_attention_program(cq, ck, cv, a.norm_q.weight.view(1, -1),
                                            a.norm_k.weight.view(1, -1) if first else None, a.eps, build=False)
        op = vm.linear_program(ao, a.o.amax / target, *linear(a.o), ca, build=False)
        l0, l2 = blk.ffn.l0, blk.ffn.l2
        fp = vm.ffn_glue_program(x, ca, *mods[3:], h0, l0.amax / target, *linear(l0),
                                 h8, l2.amax / target, *linear(l2), blk.norm2.eps, build=False)
        phases.append([("cq.", cp), ("ca.", ap), ("co.", op), ("ff.", fp)])
    return mp, sq, so, phases


def install(g):
    assert os.environ.get("FUSED_GLUE") == "1" and os.environ.get("FUSED_FFN") == "1"
    assert os.environ.get("ROPE_KV") == "1" and os.environ.get("VM_ATTN") == "1" and os.environ.get("ATTN", "n2") == "n2"
    ops.vm()
    PATHS, MODE, TARGET, FusedFFN, orig, mf, pipe = (g[k] for k in
        ("PATHS", "MODE", "TARGET", "FusedFFN", "_glue_block_forward", "_mf_glue", "pipe"))
    plans, parts, workspaces, ropetabs, prepared = {}, {}, {}, {}, set()
    coverage = {}
    active, sample = None, None
    frames = min(int(os.environ.get("FRAMES", "97")), np.load(f'{g["EXAMPLE"]}/poses.npy', mmap_mode="r").shape[0])
    latent_frames = (frames - 1) // pipe.vae_stride[0] + 1


    def ready(blk):
        return (MODE["n1"] and MODE.get("n2") and isinstance(blk.ffn, FusedFFN) and
                isinstance(blk.norm3, (torch.nn.Identity, torch.nn.LayerNorm)) and all(
                    hasattr(l, "wq") and l.ok and l.amax > 0 and l.bias_bf16 is not None for l in
                    (blk.self_attn.q, blk.self_attn.k, blk.self_attn.v, blk.self_attn.o,
                     blk.cross_attn.q, blk.cross_attn.k, blk.cross_attn.v, blk.cross_attn.o, blk.ffn.l0, blk.ffn.l2)))
    assert len({(b.self_attn.local_attn_size, b.self_attn.sink_size) for b in pipe.model.blocks}) == 1

    init_self, init_cross = pipe._initialize_self_kv_cache, pipe._initialize_crossattn_cache

    def self_cache(*a, **kw):
        caches = init_self(*a, **kw)
        for c in caches:
            c["_vm_indices"] = (0, 0)
        return caches

    def cross_cache(*a, **kw):
        caches = init_cross(*a, **kw)
        for c in caches:
            c["_vm_first"] = True
        return caches

    pipe._initialize_self_kv_cache, pipe._initialize_crossattn_cache = self_cache, cross_cache

    def rope_table(shape, start, device):
        key = (shape, start, device)
        if key not in ropetabs:
            f, h, w = shape
            c = pipe.model.freqs.shape[1]
            fs = pipe.model.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            fi = torch.cat([fs[0][start:start + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                            fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                            fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], -1).reshape(f * h * w, -1)
            ropetabs[key] = torch.stack((fi.real.float(), fi.imag.float()), -1).flatten(1).to(device).contiguous()
        return ropetabs[key]

    def prepare(shape, R, context_len, caches, crosses, max_attention, camera, starts):
        M, C = shape[0] * shape[1] * shape[2], pipe.model.dim
        device = caches[0]["k"].device
        key = (M, C, R, context_len, device)
        if key not in workspaces:
            bf = lambda rows=M: torch.empty(rows, C, dtype=torch.bfloat16, device=device)
            x = torch.empty(M, C, dtype=torch.float32, device=device)
            e = torch.empty(R, 6 * C, dtype=torch.float32, device=device)
            ev = torch.empty_like(e)
            bufs = [bf() for _ in range(7)]
            bufs += [torch.empty(M, C, dtype=torch.float8_e4m3fn, device=device),
                     torch.empty(M, pipe.model.blocks[0].ffn.l0.wq.size(0), dtype=torch.float8_e4m3fn, device=device)]
            workspaces[key] = x, e, ev, bf(context_len), bufs
        x, e, ev, context, bufs = workspaces[key]
        fs = shape[1] * shape[2]
        oldg = oldl = 0
        for start in starts:
            a = pipe.model.blocks[0].self_attn
            plan = cache_plan(start, M, oldg, oldl, caches[0]["k"].size(1), a.sink_size * fs, max_attention, a.local_attn_size != -1)
            variants = [plan, cache_plan(start, M, plan.global_end, plan.local_end, caches[0]["k"].size(1),
                                         a.sink_size * fs, max_attention, a.local_attn_size != -1)]
            cs = rope_table(shape, start // fs, device)
            for blk, kv, cross in zip(pipe.model.blocks, caches, crosses):
                pk = (id(blk), key)
                if pk not in parts:
                    parts[pk] = block_parts(blk, x, e, ev, context, cross, TARGET, bufs)
                mp, sq, so, cps = parts[pk]
                for p in variants:
                    for first in (True, False) if start == 0 else (False,):
                        ck = (id(blk), key, p.global_end, p.write_offset, p.lo, p.local_end, p.advance, first, camera)
                        if ck in plans:
                            continue
                        eligible = vm_advance_eligible(p.advance)
                        coverage[(M, p.write_offset, p.lo, p.local_end, p.advance)] = {
                            "tokens": M, "write_offset": p.write_offset, "lo": p.lo, "hi": p.local_end,
                            "advance": p.advance, "path": "vm" if eligible else "fallback"}
                        if not eligible:
                            plans[ck] = None
                            continue
                        ap, ao = vm.self_attention_program(*bufs[:3], cs, kv["k"].view(-1, C), kv["v"].view(-1, C),
                                                           p.lo, p.local_end, p.write_offset, p.advance,
                                                           blk.self_attn.norm_q.weight.view(1, -1),
                                                           blk.self_attn.norm_k.weight.view(1, -1), blk.self_attn.eps, build=False)
                        ap.keep = [t for t in ap.keep if t is not ao]
                        ap.bind("out", bufs[3])
                        before = [("mod.", mp), ("sq.", sq), ("sa.", ap), ("so.", so)]
                        after = cps[0 if first else 1]
                        group = ([vm.Program.compose(before, build=False), vm.Program.compose(after, build=False)] if camera else
                                 [vm.Program.compose(before + after, build=False)])
                        group[-1].cache_commit(kv["global_end_index"], kv["local_end_index"], cross["is_init"],
                                              p.global_end, p.local_end, first)
                        plans[ck] = [program.build() for program in group]
            oldg, oldl = plan.global_end, plan.local_end
        return key

    inner = pipe.model.forward

    def model_forward(*args, **kw):
        nonlocal active, sample
        if args or kw.get("kv_cache") is None or kw.get("crossattn_cache") is None:
            return inner(*args, **kw)
        if not all(ready(b) for b in pipe.model.blocks):
            sample = kw
            out = inner(**kw)
            for cache in kw["kv_cache"]:
                cache.pop("_vm_indices", None)
            for cache in kw["crossattn_cache"]:
                cache.pop("_vm_first", None)
            return out
        caches, crosses = kw["kv_cache"], kw["crossattn_cache"]
        raw = kw["x"][0]
        patch = pipe.model.patch_size
        shape = tuple(raw.shape[i + 1] // patch[i] for i in range(3))
        M, fs = shape[0] * shape[1] * shape[2], shape[1] * shape[2]
        R = 1 if os.environ.get("KESTREL_E_BCAST") == "1" and kw["t"].dim() == 1 else M
        first = kw.get("cross_attn_first_call")
        if first is None:
            first = crosses[0].get("_vm_first")
            if first is None:
                first = not bool(crosses[0]["is_init"].item())
        start = kw.get("current_start", 0)
        camera = kw.get("dit_cond_dict") is not None and "c2ws_plucker_emb" in kw["dit_cond_dict"]
        max_attention = kw.get("max_attention_size", 1_000_000)
        old = caches[0].get("_vm_indices")
        if old is None:
            old = tuple(int(caches[0][name].item()) for name in ("global_end_index", "local_end_index"))
        a = pipe.model.blocks[0].self_attn
        p = cache_plan(start, M, *old, caches[0]["k"].size(1), a.sink_size * fs, max_attention, a.local_attn_size != -1)
        starts = range(0, latent_frames // shape[0] * M, M)
        signature = (shape, R, max_attention, camera)
        if signature not in prepared:
            key = prepare(shape, R, pipe.model.text_len, caches, crosses, max_attention, camera, starts)
            prepared.add(signature)
        else:
            key = (M, pipe.model.dim, R, pipe.model.text_len, caches[0]["k"].device)
        sample = None
        cs = rope_table(shape, start // fs, raw.device)
        active = key, p, bool(first), camera, cs
        for cache in caches:
            cache["_vm_plan"] = p
        try:
            out = inner(**{**kw, "cross_attn_first_call": bool(first), "frame_seqlen": fs})
        finally:
            active = None
            for cache in caches:
                cache.pop("_vm_plan", None)
        for kv, cross in zip(caches, crosses):
            kv["_vm_indices"] = (p.global_end, p.local_end)
            cross["_vm_first"] = False
        return out

    pipe.model.forward = model_forward
    def warmup():
        nonlocal sample
        vm.grid()
        if sample is None:
            h, w = g["img"].size[1], g["img"].size[0]
            area = g["MAX_AREA_CONFIGS"][g["args"].size]
            aspect = h / w
            patch, stride = pipe.model.patch_size, pipe.vae_stride
            lat_h = round(math.sqrt(area * aspect) // stride[1] // patch[1] * patch[1])
            lat_w = round(math.sqrt(area / aspect) // stride[2] // patch[2] * patch[2])
            chunk = int(os.environ.get("CHUNK", "3"))
            total = latent_frames // chunk * chunk
            fs = lat_h * lat_w // (patch[1] * patch[2])
            a = pipe.model.blocks[0].self_attn
            capacity = fs * (a.local_attn_size if a.local_attn_size != -1 else total)
            device = pipe.model.blocks[0].ffn.l0.wq.device
            def cache(rows, cross=False):
                values = {name: torch.empty(1, 1, a.num_heads, a.head_dim, dtype=torch.bfloat16, device=device).expand(1, rows, -1, -1)
                          for name in ("k", "v")}
                if cross:
                    values["is_init"] = torch.empty((), dtype=torch.int32, device=device)
                else:
                    values.update({name: torch.empty(1, dtype=torch.int64, device=device)
                                   for name in ("global_end_index", "local_end_index")})
                return values
            sample = {"x": [torch.empty(1, 1, 1, 1).expand(16, chunk, lat_h, lat_w)], "t": torch.empty(1),
                      "kv_cache": [cache(capacity) for _ in pipe.model.blocks],
                      "crossattn_cache": [cache(pipe.model.text_len, True) for _ in pipe.model.blocks],
                      "dit_cond_dict": {"c2ws_plucker_emb": None}, "max_attention_size": capacity}
        raw = sample["x"][0]
        patch = pipe.model.patch_size
        shape = tuple(raw.shape[i + 1] // patch[i] for i in range(3))
        M = shape[0] * shape[1] * shape[2]
        R = 1 if os.environ.get("KESTREL_E_BCAST") == "1" and sample["t"].dim() == 1 else M
        camera = sample.get("dit_cond_dict") is not None and "c2ws_plucker_emb" in sample["dit_cond_dict"]
        maximum = sample.get("max_attention_size", 1_000_000)
        prepare(shape, R, pipe.model.text_len, sample["kv_cache"], sample["crossattn_cache"], maximum, camera,
                range(0, latent_frames // shape[0] * M, M))
        prepared.add((shape, R, maximum, camera))
        external = {c[k].untyped_storage().data_ptr() for caches in (sample["kv_cache"], sample["crossattn_cache"]) for c in caches
                    for k in c if isinstance(c[k], torch.Tensor)}
        for group in plans.values():
            for program in group or ():
                program.keep = [t for t in program.keep if t is None or t.untyped_storage().data_ptr() not in external]
        for mp, sq, so, groups in parts.values():
            for program in [mp, sq, so, *(p for group in groups for _, p in group)]:
                program.keep = [t for t in program.keep if t is None or t.untyped_storage().data_ptr() not in external]
        sample = None
        print(json.dumps({"exp": "vm_block_warmup", "programs": sum(len(group) for group in plans.values() if group),
                          "classes": list(coverage.values())}), flush=True)

    g["VM_WARMUP"] = warmup


    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict=None, kv_cache=None,
                crossattn_cache=None, current_start=0, max_attention_size=1_000_000, frame_seqlen=None,
                cross_attn_first_call=None, seq_lens_int=None):
        if active is None or context_lens is not None:
            PATHS["block_vm_fallback"] += 1
            return orig(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, kv_cache, crossattn_cache,
                        current_start, max_attention_size, frame_seqlen, cross_attn_first_call, seq_lens_int)
        key, p, first, camera, cs = active
        ck = (id(self), key, p.global_end, p.write_offset, p.lo, p.local_end, p.advance, first, camera)
        programs = plans[ck]
        if programs is None:
            PATHS["block_vm_fallback"] += 1
            return orig(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, dit_cond_dict, kv_cache, crossattn_cache,
                        current_start, max_attention_size, frame_seqlen, first, seq_lens_int)
        x = x.float().contiguous()
        values = {"mod.e": e.reshape(-1, 6 * x.size(-1)), "sq.residual": x.view(-1, x.size(-1)),
                  "sa.cs": cs, "sa.kc": kv_cache["k"].view(-1, x.size(-1)),
                  "sa.vc": kv_cache["v"].view(-1, x.size(-1)),
                  "sa.ak": kv_cache["k"].view(-1, x.size(-1))[p.lo:p.local_end],
                  "sa.av": kv_cache["v"].view(-1, x.size(-1))[p.lo:p.local_end],
                  "so.residual": x.view(-1, x.size(-1))}
        def launch(program, bindings):
            for name, tensor in bindings.items():
                program.bind(name, tensor)
            program.run()
        if camera:
            launch(programs[0], values)
            emb = dit_cond_dict["c2ws_plucker_emb"]
            h = self.cam_injector_layer2(torch.nn.functional.silu(self.cam_injector_layer1(emb))) + emb
            x = ((1.0 + self.cam_scale_layer(h)) * x + self.cam_shift_layer(h)).float().contiguous()
            values = {}
        x2 = x.view(-1, x.size(-1))
        input_name = ("q" if first else "") + ("x" if isinstance(self.norm3, torch.nn.Identity) else "residual")
        values.update({"cq." + input_name: x2, "ff.x": x2})
        ck2, cv2 = crossattn_cache["k"].view(-1, x.size(-1)), crossattn_cache["v"].view(-1, x.size(-1))
        values.update({"ca.k": ck2, "ca.v": cv2})
        values.update({"global_index": kv_cache["global_end_index"].view(1, 1),
                       "local_index": kv_cache["local_end_index"].view(1, 1),
                       "initialized": crossattn_cache["is_init"].view(1, 1)})
        if first:
            values.update({"cq.ky": ck2, "cq.vy": cv2})
        if first:
            values["cq.kx"] = context.view(-1, x.size(-1)).to(torch.bfloat16).contiguous()
        launch(programs[-1], values)
        PATHS["block_vm"] += 1
        PATHS["ffn_vm"] += 1
        PATHS["attn_vm"] += 1
        return x

    mf.CausalWanAttentionBlock.forward = forward
    print('{"exp": "vm_block_hook", "installed": true, "merged": true}', flush=True)
