import math
import torch

def install(mf, ops, vm=None, paths=None, enabled=lambda: True):
    ext = ops.rope_kv()
    original = mf.CausalWanSelfAttention.forward
    cache, programs = {}, {}

    def tables(x, grid_sizes, freqs, start_frame):
        n, c = x.size(2), x.size(3) // 2
        f, h, w = (int(v) for v in grid_sizes[0].tolist()) if not isinstance(grid_sizes, tuple) else grid_sizes
        key = (f, h, w, start_frame, c, x.device)
        if key not in cache:
            fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
            fi = torch.cat([fs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                            fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                            fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)
            cache.clear()
            cos, sin = fi.real.float().to(x.device).contiguous(), fi.imag.float().to(x.device).contiguous()
            cache[key] = (cos, sin, torch.stack((cos[:, 0], sin[:, 0]), -1).flatten(1).contiguous())
        return (*cache[key], f * h * w)
    def run_vm(q, k, v, cs, kc, vc, write_offset, lo, hi, advance):
        s, n, d = q.shape[1:]
        q2, k2, v2 = (t.view(s, n * d) for t in (q, k, v))
        kc2, vc2 = kc.view(kc.size(1), n * d), vc.view(vc.size(1), n * d)
        ak, av = kc2[lo:hi], vc2[lo:hi]
        key = (s, n, d, kc2.size(0), hi - lo, write_offset, advance)
        ent = programs.get(key)
        if ent is None:
            p = vm.Program()
            qi, ki, vi, ci, kci, vci, aki, avi = (p.tensor(t, name=name) for t, name in
                                                   zip((q2, k2, v2, cs, kc2, vc2, ak, av),
                                                       ("q", "k", "v", "cs", "kc", "vc", "ak", "av")))
            shift_done = None
            if advance is not None:
                shift_done = p.join(p.kv_advance(kci, vci, *advance))
            app = p.rope_kv_append(qi, ki, vi, ci, kci, vci, write_offset,
                                   after=(lambda m: [shift_done]) if shift_done is not None else None)
            append_done = p.join(app)
            out = torch.empty_like(q2)
            p.attn_window(qi, aki, avi, p.tensor(out, name="out"), after=lambda m, h: [append_done])
            p.build()
            p.keep = []
            p.checked = False
            ent = programs[key] = (p, out)
        p, out = ent
        for name, t in zip(("q", "k", "v", "cs", "kc", "vc", "ak", "av"), (q2, k2, v2, cs, kc2, vc2, ak, av)):
            p.bind(name, t)
        p.run()
        if not p.checked:
            torch.cuda.synchronize()
            st = p.status()
            assert st["err"] == "ok" and st["done"] == st["n"], st
            p.checked = True
        if paths is not None:
            paths["attn_vm"] += 1
        return out.view_as(q)


    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000,
                frame_seqlen=None, seq_lens_int=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        if frame_seqlen is None:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        start_frame = current_start // frame_seqlen
        cos, sin, cs, rope_len = tables(q, grid_sizes, freqs, start_frame)
        eligible = (b == 1 and d == 128 and rope_len == s and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and
                    v.dtype == torch.bfloat16 and kv_cache["k"].dtype == torch.bfloat16 and kv_cache["v"].dtype == torch.bfloat16 and
                    q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
                    kv_cache["k"].is_contiguous() and kv_cache["v"].is_contiguous())
        if not eligible:
            return original(self, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                            frame_seqlen, seq_lens_int)
        current_end = current_start + s
        sink_tokens = self.sink_size * frame_seqlen
        cache_size = kv_cache["k"].shape[1]
        old_global = int(kv_cache["global_end_index"].item())
        old_local = int(kv_cache["local_end_index"].item())
        advance = None
        use_vm = vm is not None and enabled()
        if self.local_attn_size == -1:
            local_end_index = current_end
            local_start_index = current_start
        elif current_end > old_global and s + old_local > cache_size:
            num_evicted_tokens = s + old_local - cache_size
            num_rolled_tokens = old_local - num_evicted_tokens - sink_tokens
            advance = (sink_tokens, num_evicted_tokens, old_local) if num_rolled_tokens > 0 else None
            use_vm = use_vm and (advance is None or all(v % 16 == 0 for v in advance))
            if not use_vm:
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = old_local + current_end - old_global - num_evicted_tokens
            local_start_index = local_end_index - s
        else:
            local_end_index = old_local + current_end - old_global
            local_start_index = local_end_index - s
        lo = max(0, local_end_index - max_attention_size)
        if use_vm:
            out = run_vm(q, k, v, cs, kv_cache["k"], kv_cache["v"], local_start_index, lo, local_end_index, advance)
        else:
            roped_query = ext.rope_qk_append(q, k, v, cos, sin, kv_cache["k"], kv_cache["v"], local_start_index)
            out = mf.attention(roped_query, kv_cache["k"][:, lo:local_end_index], kv_cache["v"][:, lo:local_end_index])
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return self.o(out.flatten(2))

    mf.CausalWanSelfAttention.forward = forward
