import torch

def install(mf):
    stock = mf.flash_attention

    def cross_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                        softmax_scale=None, q_scale=None, causal=False,
                        window_size=(-1, -1), deterministic=False,
                        dtype=torch.bfloat16, version=None):
        full = k_lens is None
        if k_lens is not None:
            try:
                full = len(k_lens) == q.shape[0] and all(int(x) == k.shape[1] for x in k_lens)
            except TypeError:
                full = False
        if (full and q_lens is None and dropout_p == 0. and softmax_scale is None
                and q_scale is None and not causal and not deterministic
                and version is None and window_size == (-1, -1)
                and q.is_cuda and q.shape[0] == 1 and q.shape[-1] == 128
                and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
                and v.dtype == torch.bfloat16):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            ).transpose(1, 2)
        return stock(q, k, v, q_lens=q_lens, k_lens=k_lens,
                     dropout_p=dropout_p, softmax_scale=softmax_scale,
                     q_scale=q_scale, causal=causal, window_size=window_size,
                     deterministic=deterministic, dtype=dtype, version=version)

    mf.flash_attention = cross_attention
    return cross_attention
