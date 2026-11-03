import random
import unittest

from lingbot.vm_schedule import cache_plan, vm_advance_eligible


def upstream(start, tokens, global_end, local_end, capacity, sink, maximum, local=True):
    end = start + tokens
    advance = None
    if not local:
        hi, offset = end, start
    elif end > global_end and tokens + local_end > capacity:
        evicted = tokens + local_end - capacity
        rolled = local_end - evicted - sink
        if rolled > 0:
            advance = (sink, evicted, local_end)
        hi = local_end + end - global_end - evicted
        offset = hi - tokens
    else:
        hi = local_end + end - global_end
        offset = hi - tokens
    return end, hi, offset, max(0, hi - maximum), advance


class CacheScheduleTests(unittest.TestCase):
    def test_lingbot_denoise_and_update(self):
        for frames, chunk, maximum in ((9, 2, 28080), (97, 2, 28080), (257, 2, 15600), (97, 3, 28080)):
            tokens, capacity, sink = chunk * 1560, 28080, 9360
            old = (0, 0)
            chunks = ((frames - 1) // 4 + 1) // chunk
            for i in range(chunks):
                for step in range(5):
                    expected = upstream(i * tokens, tokens, *old, capacity, sink, maximum)
                    actual = cache_plan(i * tokens, tokens, *old, capacity, sink, maximum)
                    self.assertEqual(tuple(actual), expected)
                    self.assertTrue(0 <= actual.write_offset < actual.local_end <= capacity)
                    if step:
                        self.assertIsNone(actual.advance)
                    old = actual.global_end, actual.local_end

    def test_repeated_roll_preserves_sink(self):
        tokens, capacity, sink = 4, 16, 4
        old = (0, 0)
        cache = [None] * capacity
        for chunk in range(15):
            for step in range(5):
                p = cache_plan(chunk * tokens, tokens, *old, capacity, sink, capacity)
                if p.advance is not None:
                    begin, drop, used = p.advance
                    cache[begin:used - drop] = cache[begin + drop:used]
                cache[p.write_offset:p.local_end] = [(chunk, step)] * tokens
                old = p.global_end, p.local_end
                if chunk:
                    self.assertEqual(cache[:sink], [(0, 4)] * sink)
            self.assertEqual(cache[p.write_offset:p.local_end], [(chunk, 4)] * tokens)

    def test_global_and_attention_window(self):
        for start in (0, 3120, 24960, 28080, 31200):
            p = cache_plan(start, 3120, 999, 888, 100000, 9360, 4680, local=False)
            self.assertEqual(p, upstream(start, 3120, 999, 888, 100000, 9360, 4680, False))
            self.assertEqual(p.write_offset, start)
            self.assertIsNone(p.advance)

    def test_variable_chunks_and_rewinds(self):
        rng = random.Random(1895)
        for _ in range(200):
            capacity, sink, old = 128, 16, (0, 0)
            for _ in range(50):
                tokens = rng.choice((8, 16, 24, 32))
                start = old[0] if rng.randrange(3) else max(0, old[0] - tokens)
                maximum = rng.choice((16, 64, 128, 256))
                self.assertEqual(cache_plan(start, tokens, *old, capacity, sink, maximum),
                                 upstream(start, tokens, *old, capacity, sink, maximum))
                p = cache_plan(start, tokens, *old, capacity, sink, maximum)
                old = p.global_end, p.local_end

    def test_no_retained_tokens(self):
        p = cache_plan(16, 16, 16, 16, 16, 0, 16)
        self.assertEqual(p, (32, 16, 0, 0, None))

    def test_advance_alignment_classes(self):
        self.assertTrue(vm_advance_eligible(None))
        self.assertTrue(vm_advance_eligible((16, 16, 64)))
        for advance in ((8, 16, 64), (16, 17, 64), (16, 16, 65)):
            self.assertFalse(vm_advance_eligible(advance))

    def test_unaligned_eviction_then_vm_overwrite(self):
        for fs in (1560, 1561, 1564):
            old, paths = (0, 0), set()
            tokens, capacity, sink = 2 * fs, 18 * fs, 6 * fs
            for chunk in range(12):
                for step in range(5):
                    p = cache_plan(chunk * tokens, tokens, *old, capacity, sink, capacity)
                    self.assertEqual(p, upstream(chunk * tokens, tokens, *old, capacity, sink, capacity))
                    eligible = vm_advance_eligible(p.advance)
                    paths.add("vm" if eligible else "fallback")
                    if chunk >= 9:
                        self.assertEqual(eligible, step != 0 or fs == 1560)
                    old = p.global_end, p.local_end
            self.assertEqual(paths, {"vm"} if fs == 1560 else {"vm", "fallback"})


if __name__ == "__main__":
    unittest.main()
