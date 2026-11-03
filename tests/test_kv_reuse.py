import collections
from types import SimpleNamespace
import unittest

import torch

from lingbot.kv_reuse import install


class TinyModel:
    patch_size = (1, 1, 1)

    def __init__(self):
        self.calls = []

    def forward(self, **kw):
        x, t, start = kw["x"][0], float(kw["t"][0]), kw["current_start"]
        self.calls.append((start, t, x.clone()))
        n = x.shape[1]
        for layer, cache in enumerate(kw["kv_cache"]):
            end = start + n
            old_global, old_local = int(cache["global_end_index"]), int(cache["local_end_index"])
            size = cache["k"].shape[1]
            local_end = old_local + end - old_global
            if end > old_global and n + old_local > size:
                evicted = n + old_local - size
                rolled = old_local - evicted - 2
                for name in ("k", "v"):
                    cache[name][:, 2:2 + rolled] = cache[name][:, 2 + evicted:2 + evicted + rolled].clone()
                local_end -= evicted
            pos = torch.arange(start, end).reshape(1, n, 1, 1)
            cache["k"][:, local_end - n:local_end] = pos + 1 + 2 * t + layer
            cache["v"][:, local_end - n:local_end] = 3 * pos - 4 * t + layer
            cache["global_end_index"].fill_(end)
            cache["local_end_index"].fill_(local_end)
            cross = kw["crossattn_cache"][layer]
            if kw["cross_attn_first_call"]:
                cross["k"].fill_(layer + 7)
                cross["v"].fill_(layer + 9)
                cross["is_init"].fill_(1)
                cross["writes"] += 1
        return [x * 0.1 + t * 0.01]


class TinyPipe:
    def __init__(self, size=6):
        self.model = TinyModel()
        self.size = size
        self.scheduler = SimpleNamespace(timesteps=torch.tensor([9., 6., 2.]))
        self.completed = []

    def generate(self, seed=42, chunks=5, timesteps_index=(0, 1, 2)):
        self.cache = [{"k": torch.zeros(1, self.size, 1, 1), "v": torch.zeros(1, self.size, 1, 1),
                       "global_end_index": torch.tensor(0), "local_end_index": torch.tensor(0)} for _ in range(2)]
        self.cross = [{"k": torch.zeros(1), "v": torch.zeros(1), "is_init": torch.tensor(0), "writes": 0} for _ in range(2)]
        generator = torch.Generator().manual_seed(seed)
        output = []
        first = True
        self.completed = []
        for chunk in range(chunks):
            x = torch.randn(1, 2, 1, 1, generator=generator)
            common = {"kv_cache": self.cache, "crossattn_cache": self.cross, "current_start": chunk * 2}
            for i in timesteps_index:
                t = self.scheduler.timesteps[i:i + 1]
                pred = self.model.forward(x=[x], t=t, cross_attn_first_call=first, **common)[0]
                first = False
                x0 = x - t * pred
                x = x0 + torch.randn(x.shape, generator=generator)
            output.append(x0)
            self.model.forward(x=[x0], t=torch.zeros(1), cross_attn_first_call=False, **common)
            self.completed.append([{name: value.clone() for name, value in cache.items()} for cache in self.cache])
        return torch.cat(output, dim=1)


class KVReuseTest(unittest.TestCase):
    def test_last_preserves_output_indices_and_last_appended_values(self):
        for size in (6, 20):
            with self.subTest(size=size):
                baseline, pipe = TinyPipe(size), TinyPipe(size)
                expected = baseline.generate()
                paths = collections.Counter()
                install(pipe, paths, "last")
                observed = pipe.generate()
                self.assertTrue(torch.equal(expected, observed))
                self.assertEqual(len(baseline.model.calls), 20)
                self.assertEqual(len(pipe.model.calls), 15)
                self.assertEqual(paths["kv_forward_skipped"], 5)
                for chunk, (ref, got) in enumerate(zip(baseline.completed, pipe.completed)):
                    for layer, (a, b) in enumerate(zip(ref, got)):
                        self.assertTrue(torch.equal(a["global_end_index"], b["global_end_index"]))
                        self.assertTrue(torch.equal(a["local_end_index"], b["local_end_index"]))
                        end = min((chunk + 1) * 2, size)
                        pos = torch.arange(chunk * 2, (chunk + 1) * 2).reshape(1, 2, 1, 1)
                        self.assertTrue(torch.equal(b["k"][:, end - 2:end], pos + 5 + layer))
                        self.assertTrue(torch.equal(b["v"][:, end - 2:end], 3 * pos - 8 + layer))
                self.assertEqual([c["writes"] for c in pipe.cross], [1, 1])
                for a, b in zip(baseline.cross, pipe.cross):
                    for name in ("k", "v", "is_init"):
                        self.assertTrue(torch.equal(a[name], b[name]))

    def test_extrap_reaches_zero_for_affine_appended_kv(self):
        baseline, pipe = TinyPipe(), TinyPipe()
        expected = baseline.generate()
        paths = collections.Counter()
        install(pipe, paths, "extrap")
        observed = pipe.generate()
        self.assertTrue(torch.equal(expected, observed))
        for ref, got in zip(baseline.completed, pipe.completed):
            for a, b in zip(ref, got):
                for name in a:
                    self.assertTrue(torch.equal(a[name], b[name]), name)
        self.assertEqual(paths["kv_forward_skipped"], 5)

    def test_commit_callback_receives_exact_x0_and_resets_per_generate(self):
        pipe = TinyPipe()
        paths, committed = collections.Counter(), []
        pipe._kv_reuse_commit = lambda x0: committed.append(x0.clone())
        install(pipe, paths, "last")
        for _ in range(2):
            output = pipe.generate(17, 2)
            self.assertTrue(torch.equal(output, torch.cat(committed[-2:], dim=1)))
        self.assertEqual(paths["kv_forward_skipped"], 4)
        self.assertEqual([c["writes"] for c in pipe.cross], [1, 1])

    def test_single_denoise_step_last_and_extrap_requirement(self):
        pipe = TinyPipe()
        paths = collections.Counter()
        install(pipe, paths, "last")
        pipe.generate(chunks=1, timesteps_index=(2,))
        self.assertEqual(len(pipe.model.calls), 1)
        self.assertEqual(paths["kv_forward_skipped"], 1)
        pipe = TinyPipe()
        install(pipe, collections.Counter(), "extrap")
        with self.assertRaisesRegex(ValueError, "at least two"):
            pipe.generate(timesteps_index=(2,))

    def test_disabled_generation_forwards_directly(self):
        pipe = TinyPipe()
        install(pipe, collections.Counter(), "last")
        out = pipe.model.forward(x=[torch.ones(1, 2, 1, 1)], t=torch.ones(1), current_start=0,
                                 kv_cache=[], crossattn_cache=[], cross_attn_first_call=False)
        self.assertEqual(len(pipe.model.calls), 1)
        self.assertEqual(len(out), 1)

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "last or extrap"):
            install(TinyPipe(), collections.Counter(), "invalid")


if __name__ == "__main__":
    unittest.main()
