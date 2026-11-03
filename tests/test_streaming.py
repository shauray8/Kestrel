import unittest

import torch

from lingbot.streaming import StreamChunk, StreamResult, generate, iter_generate


class TinyModel:
    def __init__(self):
        self.cache = []
        self.calls = []
        self.decode_log = []

    def forward(self, **kw):
        x = kw["x"][0]
        t = kw["t"]
        self.calls.append((x.clone(), t.clone(), kw["y"][0].clone(),
                           kw["dit_cond_dict"]["c2ws_plucker_emb"][0].clone(),
                           kw["current_start"], kw["cross_attn_first_call"]))
        if bool((t == 0).all()):
            self.decode_log.append("tae")
            self.cache.append(x.clone())
            self.decode_log.append("cache")
            return [torch.zeros_like(x)]
        return [torch.full_like(x, t[0] / 10)]


class TinyPipe:
    infer_mode = "causal_fast"

    def __init__(self):
        self.model = TinyModel()
        self.conditions = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
        self.cameras = (torch.arange(4, dtype=torch.float32) + 10).reshape(4, 1, 1, 1)

    def generate(self, seed=42, chunk_size=2, timesteps_index=(0, 1, 2), **kwargs):
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(1, 4, 1, 1, generator=generator)
        timesteps = torch.tensor([4.0, 2.0, 0.5])[list(timesteps_index)]
        output = []
        initialized = False
        for chunk_id, initial in enumerate(noise.split(chunk_size, dim=1)):
            current = initial
            condition = self.conditions[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
            camera = self.cameras[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
            common = {"y": [condition], "dit_cond_dict": {"c2ws_plucker_emb": (camera,)},
                      "current_start": chunk_id * chunk_size * 7}
            for step, t in enumerate(timesteps):
                pred = self.model.forward(x=[current], t=t[None], cross_attn_first_call=not initialized, **common)[0]
                initialized = True
                x0 = current - t * pred
                if step + 1 < len(timesteps):
                    fresh = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
                    current = x0 + timesteps[step + 1] * fresh
            output.append(x0)
            self.model.forward(x=[x0], t=torch.zeros(1), cross_attn_first_call=False, **common)
        self.model.decode_log.append("vae")
        return torch.cat(output, dim=1) * 3


class StreamingTest(unittest.TestCase):
    def assert_calls_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            for x, y in zip(a[:4], b[:4]):
                self.assertTrue(torch.equal(x, y))
            self.assertEqual(a[4:], b[4:])

    def test_identical_sampling_conditioning_cache_and_final_decode(self):
        reference = TinyPipe()
        expected = reference.generate(seed=17)
        pipe = TinyPipe()
        original = pipe.model.forward
        observed = []
        stats = {}

        def on_chunk(chunk):
            observed.append((chunk.chunk_id, chunk.latent.clone(), len(pipe.model.cache)))
            pipe.model.decode_log.append("callback")

        video = generate(pipe, seed=17, on_chunk=on_chunk, stats=stats)
        self.assertEqual(pipe.model.forward, original)
        self.assertTrue(torch.equal(video, expected))
        self.assertEqual([(c[0], c[2]) for c in observed], [(0, 1), (1, 2)])
        for (_, latent, _), cached in zip(observed, reference.model.cache):
            self.assertTrue(torch.equal(latent, cached))
        self.assertEqual(pipe.model.decode_log,
                         ["tae", "cache", "callback", "tae", "cache", "callback", "vae"])
        self.assert_calls_equal(pipe.model.calls, reference.model.calls)
        self.assertEqual(stats, {"forwards": 8, "cache_updates": 2, "chunks": 2, "completed": 1})

    def test_optional_pixels_decode_after_cache_update(self):
        pipe = TinyPipe()
        events = []

        def decode(latent, chunk_id):
            self.assertEqual(len(pipe.model.cache), chunk_id + 1)
            return latent * 10

        video = generate(pipe, decode_chunk=decode, on_chunk=events.append)
        self.assertEqual([event.chunk_id for event in events], [0, 1])
        for event in events:
            self.assertTrue(torch.equal(event.pixels, event.latent * 10))
        self.assertTrue(torch.equal(video, torch.cat(pipe.model.cache, dim=1) * 3))

    def test_iterator_orders_chunks_before_result(self):
        reference = TinyPipe()
        expected = reference.generate(seed=23)
        pipe = TinyPipe()
        original = pipe.model.forward
        events = list(iter_generate(pipe, seed=23))
        self.assertEqual(len(events), 3)
        self.assertTrue(all(isinstance(event, StreamChunk) for event in events[:2]))
        self.assertIsInstance(events[-1], StreamResult)
        self.assertTrue(torch.equal(events[-1].video, expected))
        self.assertEqual([event.chunk_id for event in events[:2]], [0, 1])
        self.assertEqual(pipe.model.forward, original)

    def test_callback_failure_stops_before_next_chunk(self):
        pipe = TinyPipe()
        original = pipe.model.forward

        def fail(chunk):
            raise RuntimeError("stop")

        with self.assertRaisesRegex(RuntimeError, "stop"):
            generate(pipe, on_chunk=fail)
        self.assertEqual(len(pipe.model.cache), 1)
        self.assertNotIn("vae", pipe.model.decode_log)
        self.assertEqual(pipe.model.forward, original)

    def test_unobserved_call_is_direct(self):
        pipe = TinyPipe()
        original = pipe.model.forward
        video = generate(pipe)
        self.assertEqual(pipe.model.forward, original)
        self.assertTrue(torch.equal(video, torch.cat(pipe.model.cache, dim=1) * 3))


if __name__ == "__main__":
    unittest.main()
