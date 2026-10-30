import unittest
from pathlib import Path

from bench.matrix import LOCK, overrides, variant_config


class MatrixConfigTest(unittest.TestCase):
    def test_override_precedence_and_recorded_scope(self):
        presets = {"common": {"A": "common", "B": "common"}, "variants": {"fast": {"B": "variant", "C": "variant"}}}
        config = variant_config(presets, "fast", {"C": "extra", "D": "extra"}, Path("/tmp/out"))
        self.assertEqual(config, {"A": "common", "B": "variant", "C": "extra", "D": "extra",
                                  "KESTREL_OUT": "/tmp/out", "KESTREL_GPU_LOCK": LOCK})

    def test_lock_override(self):
        presets = {"common": {}, "variants": {"stock": {}}}
        config = variant_config(presets, "stock", {"KESTREL_GPU_LOCK": "/tmp/gpu.lock"}, "/tmp/out")
        self.assertEqual(config["KESTREL_GPU_LOCK"], "/tmp/gpu.lock")

    def test_override_validation(self):
        self.assertEqual(overrides(["A=1", "B=x=y"]), {"A": "1", "B": "x=y"})
        with self.assertRaisesRegex(ValueError, "KEY=VALUE"):
            overrides(["A"])
        with self.assertRaisesRegex(ValueError, "nonempty key"):
            overrides(["=value"])


if __name__ == "__main__":
    unittest.main()
