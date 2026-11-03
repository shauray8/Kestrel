import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lingbot.spec_compact import bucket_capacity, compact, distance_bank, expand_rows, redundancy
from lingbot.ropekv_hook import install as install_rope


class SpecCompactTest(unittest.TestCase):
    def check_compaction(self, active, capacity):
        idx, count, writes, rank, model_active = compact(active, capacity)
        expected = active.flatten().nonzero()[:, 0]
        total = active.numel()
        dense = expected.numel() > total - 16
        expected_order = torch.cat((expected, (~active.flatten()).nonzero()[:, 0])) if dense else expected
        self.assertEqual(count.item(), expected.numel())
        self.assertEqual(writes.item(), expected_order.numel())
        self.assertEqual(idx.shape, (capacity,))
        self.assertEqual(count.dtype, torch.int64)
        self.assertTrue(torch.equal(idx[:writes], expected_order))
        self.assertTrue(torch.equal(model_active, active.flatten() | dense))
        if writes:
            self.assertTrue(torch.equal(idx[writes:], expected_order[-1].expand(capacity - writes)))
            self.assertEqual(idx[:writes].unique().numel(), writes.item())
        else:
            self.assertTrue(torch.equal(idx, torch.zeros(capacity, dtype=torch.long)))
        source = torch.arange(total * 3).reshape(total, 3)
        rows = source[idx]
        result = expand_rows(rows, rank, model_active)
        self.assertTrue(torch.equal(result[active.flatten()], source[active.flatten()]))
        expected_full = torch.where(model_active[:, None], source, 0)
        self.assertTrue(torch.equal(result, expected_full))

    def test_bucket_boundaries(self):
        for count, expected in ((0, 0), (1, 128), (127, 128), (128, 128), (129, 256),
                                (3104, 3200), (3105, 3200), (3120, 3200)):
            self.assertEqual(bucket_capacity(count, 3120), expected)
        self.assertEqual(bucket_capacity(122, 130), 256)

    def test_masks_and_padded_rows(self):
        generator = torch.Generator().manual_seed(91)
        for total in (129, 256, 3120):
            for count in (0, 1, 16, 127, 128, total - 16, total - 15, total):
                with self.subTest(total=total, count=count):
                    active = torch.zeros(total, dtype=torch.bool)
                    active[torch.randperm(total, generator=generator)[:count]] = True
                    capacity = bucket_capacity(count, total) or 128
                    self.check_compaction(active, capacity)

    def test_capacity_stays_safe_as_active_set_shrinks(self):
        generator = torch.Generator().manual_seed(12)
        active = torch.rand(2, 30, 52, generator=generator) > 0.3
        capacity = bucket_capacity(int(active.sum()), active.numel())
        for _ in range(8):
            self.check_compaction(active, capacity)
            active &= torch.rand(active.shape, generator=generator) > 0.4
        active.zero_()
        self.check_compaction(active, capacity)

    def test_reconstruction_matches_legacy_padding(self):
        active = torch.arange(3120) % 3 == 0
        idx, _, _, rank, model_active = compact(active, 1152)
        selected = active.nonzero()[:, 0]
        legacy_idx = torch.cat((selected, selected[-1:].expand((-selected.numel()) % 16)))
        source = torch.randn(3120, 7, generator=torch.Generator().manual_seed(3))
        legacy = torch.zeros_like(source)
        legacy[legacy_idx] = source[legacy_idx]
        self.assertTrue(torch.equal(expand_rows(source[idx], rank, model_active), legacy))

    def test_cached_cdist_decomposition(self):
        generator = torch.Generator().manual_seed(5)
        query = torch.randn(33, 64, generator=generator)
        bank = torch.randn(8201, 64, generator=generator)
        expected = torch.full((33,), float("inf"))
        for rows in bank.split(8192):
            expected = torch.minimum(expected, torch.cdist(query, rows).amin(1))
        torch.testing.assert_close(redundancy(query, distance_bank(bank)), expected, rtol=0, atol=0)

    def test_dense_cache_mirror_never_reads_tensor_scalars(self):
        offsets = []
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads, self.head_dim, self.sink_size, self.local_attn_size = 1, 128, 0, 2
                self.q, self.k, self.v, self.o, self.norm_q, self.norm_k = [torch.nn.Identity() for _ in range(6)]
            def forward(self, *args, **kw):
                raise AssertionError("unexpected dense fallback")
        def append(q, k, v, cos, sin, kc, vc, offset):
            offsets.append(offset)
            kc[:, offset:offset + k.shape[1]] = k
            vc[:, offset:offset + v.shape[1]] = v
            return q
        mf = SimpleNamespace(CausalWanSelfAttention=Attention, attention=lambda q, k, v: q)
        install_rope(mf, SimpleNamespace(rope_kv=lambda: SimpleNamespace(rope_qk_append=append)))
        layer = Attention()
        cache = {"k": torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16),
                 "v": torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16),
                 "global_end_index": torch.zeros(1, dtype=torch.long),
                 "local_end_index": torch.zeros(1, dtype=torch.long), "_spec_end": (0, 0)}
        x = torch.ones(1, 4, 128, dtype=torch.bfloat16)
        freqs = torch.ones(16, 64, dtype=torch.complex128)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("device scalar readback")):
            for start in (0, 0, 4, 8):
                layer(x, None, (1, 2, 2), freqs, kv_cache=cache, current_start=start, frame_seqlen=4)
        self.assertEqual(offsets, [0, 0, 4, 4])
        self.assertEqual(cache["_spec_end"], (12, 8))

    def test_sparse_hook_bypasses_dense_projections(self):
        class Attention(torch.nn.Module):
            _spec_sparse = staticmethod(lambda: True)
            def forward(self, *args, **kw):
                return "sparse"
        mf = SimpleNamespace(CausalWanSelfAttention=Attention)
        install_rope(mf, SimpleNamespace(rope_kv=lambda: None))
        self.assertEqual(Attention()(None, None, None, None), "sparse")


if __name__ == "__main__":
    unittest.main()
