import math
import unittest

import torch

from bench.attn_mass import mass_metrics


class AttentionMassTest(unittest.TestCase):
    def test_random_dense_reference(self):
        generator = torch.Generator().manual_seed(42)
        q = torch.randn(1, 131, 2, 5, generator=generator)
        k = torch.randn(1, 19, 2, 5, generator=generator)
        v = torch.randn(1, 19, 2, 7, generator=generator)
        fractions, budgets = (0.1, 0.5, 1.0), (0.25, 0.5, 0.75)
        got = mass_metrics(q, k, v, 5, 4, cdf_fractions=fractions, budgets=budgets)
        self.assertEqual(got["region_mass"].shape, (2, 2, 3))
        for qi, start in enumerate((0, 128)):
            for head in range(2):
                logits = torch.einsum("qd,kd->qk", q[0, start:start + 128, head], k[0, :, head]) / math.sqrt(5)
                p = torch.softmax(logits.double(), -1)
                values = v[0, :, head].double()
                dense = p @ values
                regions = torch.tensor([p[:, :5].sum(-1).mean(), p[:, 5:15].sum(-1).mean(), p[:, 15:].sum(-1).mean()])
                torch.testing.assert_close(got["region_mass"][qi, head].double(), regions, atol=2e-7, rtol=2e-6)
                for size in (4, 16, 64):
                    blocks = [list(range(i, min(i + size, 19))) for i in range(0, 19, size)]
                    mass = [float(p[:, block].sum(-1).mean()) for block in blocks]
                    order = sorted(range(len(blocks)), key=lambda i: mass[i], reverse=True)
                    cdf = [sum(mass[i] for i in order[:math.ceil(fraction * len(blocks))]) for fraction in fractions]
                    torch.testing.assert_close(got[f"b{size}_cdf"][qi, head], torch.tensor(cdf), atol=2e-7, rtol=2e-6)
                    for bi, budget in enumerate(budgets):
                        limit = math.floor(19 * budget)
                        selected = []
                        for block in order:
                            if len(selected) + len(blocks[block]) > limit:
                                break
                            selected.extend(blocks[block])
                        recall = []
                        for row in p:
                            top = sorted(range(19), key=lambda key: float(row[key]), reverse=True)[:limit]
                            recall.append(len(set(top) & set(selected)) / limit)
                        restricted = torch.zeros_like(p)
                        restricted[:, selected] = p[:, selected]
                        output = restricted @ values
                        retained = restricted.sum(-1, keepdim=True)
                        renorm = output / retained.clamp_min(torch.finfo(torch.float64).tiny)
                        expected = {"mass": retained.mean(), "recall": sum(recall) / len(recall),
                                    "key_fraction": len(selected) / 19,
                                    "error": (output - dense).norm() / dense.norm(),
                                    "error_renorm": (renorm - dense).norm() / dense.norm()}
                        for name, value in expected.items():
                            self.assertAlmostEqual(float(got[f"b{size}_{name}"][qi, head, bi]), float(value), places=5)
                        self.assertLessEqual(float(got[f"b{size}_key_fraction"][qi, head, bi]), budget)
        torch.testing.assert_close(got["region_mass"].sum(-1), torch.ones(2, 2))
        for size in (4, 16, 64):
            torch.testing.assert_close(got[f"b{size}_cdf"][..., -1], torch.ones(2, 2))
            self.assertTrue(bool((got[f"b{size}_cdf"].diff(dim=-1) >= -1e-7).all()))

    def test_full_budget_recovers_dense(self):
        generator = torch.Generator().manual_seed(7)
        q, k, v = (torch.randn(9, 2, 5, generator=generator) for _ in range(3))
        got = mass_metrics(q, k, v, 3, 2, budgets=(1.0,))
        for size in (4, 16, 64):
            for name in ("mass", "recall", "key_fraction"):
                torch.testing.assert_close(got[f"b{size}_{name}"], torch.ones(1, 2, 1))
            torch.testing.assert_close(got[f"b{size}_error"], torch.zeros(1, 2, 1))
            torch.testing.assert_close(got[f"b{size}_error_renorm"], torch.zeros(1, 2, 1), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
