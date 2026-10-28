import math
import torch

def install(mf, ops):
    ext = ops.rope_kv()
    original = mf.CausalWanSelfAttention.forward
    cache = {}

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
            cache[key] = (fi.real.float().to(x.device).contiguous(), fi.imag.float().to(x.device).contiguous())
        return (*cache[key], f * h * w)

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, max_attention_size=1_000_000,
                frame_seqlen=None, seq_lens_int=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        if frame_seqlen is None:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
        start_frame = current_start // frame_seqlen
        cos, sin, rope_len = tables(q, grid_sizes, freqs, start_frame)
        eligible = (rope_len == s and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16 and
                    kv_cache["k"].dtype == torch.bfloat16 and kv_cache["v"].dtype == torch.bfloat16 and
                    q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and
                    kv_cache["k"].is_contiguous() and kv_cache["v"].is_contiguous())
        if not eligible:
            return original(self, x, seq_lens, grid_sizes, freqs, kv_cache, current_start, max_attention_size,
                            frame_seqlen, seq_lens_int)
        current_end = current_start + s
        sink_tokens = self.sink_size * frame_seqlen
        cache_size = kv_cache["k"].shape[1]
        if self.local_attn_size == -1:
            local_end_index = current_end
            local_start_index = current_start
        elif current_end > kv_cache["global_end_index"].item() and s + kv_cache["local_end_index"].item() > cache_size:
            num_evicted_tokens = s + kv_cache["local_end_index"].item() - cache_size
            num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item() - num_evicted_tokens
            local_start_index = local_end_index - s
        else:
            local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - s
        roped_query = ext.rope_qk_append(q, k, v, cos, sin, kv_cache["k"], kv_cache["v"], local_start_index)
        kc = kv_cache["k"][:, max(0, local_end_index - max_attention_size):local_end_index]
        vc = kv_cache["v"][:, max(0, local_end_index - max_attention_size):local_end_index]
        out = mf.attention(roped_query, kc, vc)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return self.o(out.flatten(2))

    mf.CausalWanSelfAttention.forward = forward
