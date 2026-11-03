import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import hwprobe


class DriverTests(unittest.TestCase):
    def test_metric_contract(self):
        result = hwprobe.parse_metric('{"metric":"throughput","value":42.5,"unit":"TOP/s"}\n')
        self.assertEqual(result["value"], 42.5)

    def test_invalid_metrics(self):
        for text in ["", "{}", "not json", '{"metric":"x","value":NaN,"unit":"x"}',
                     '{"metric":"x","value":true,"unit":"x"}',
                     '{"metric":"x","value":1,"unit":null}',
                     '{"metric":"x","value":1,"unit":"x"}\n{}']:
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                hwprobe.parse_metric(text)

    def test_feature_selection(self):
        selected = hwprobe.sources(["mma_fp8_*"])
        self.assertEqual([p.stem for p in selected], ["mma_fp8_f16", "mma_fp8_f32"])
        with self.assertRaises(ValueError):
            hwprobe.sources(["no_such_probe"])

    def test_error_summary(self):
        log = "nvcc warning\nptxas error   : Feature 'x' not supported\nmore\n"
        self.assertEqual(hwprobe.summarize(log, True), "ptxas error   : Feature 'x' not supported")
        self.assertEqual(hwprobe.summarize("a\nlast line\n", False), "last line")
        self.assertIsNone(hwprobe.summarize("", False))

    def test_failed_compile_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = MagicMock(run_only=False, build_only=False, output_dir=Path(tmp), iterations=None)
            with patch.object(hwprobe, "invoke", return_value=(False, "", "ptxas error")) as invoke:
                record = hwprobe.probe(Path("test.cu"), args)
        self.assertEqual(invoke.call_count, 1)
        self.assertFalse(record["compiled"])
        self.assertFalse(record["ran"])
        self.assertIsNone(record["value"])
        self.assertEqual(record["error"], "ptxas error")

    def test_build_only_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = MagicMock(run_only=False, build_only=True, output_dir=Path(tmp), iterations=None)
            with patch.object(hwprobe, "invoke", return_value=(True, "", None)) as invoke:
                record = hwprobe.probe(Path("test.cu"), args)
        self.assertEqual(invoke.call_count, 1)
        self.assertTrue(record["compiled"])
        self.assertFalse(record["ran"])
        self.assertEqual(record["metric"], "build_only")
        self.assertIsNone(record["error"])

    def test_successful_run_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = MagicMock(run_only=False, build_only=False, output_dir=Path(tmp), iterations=7)
            outputs = [(True, "", None), (True, '{"metric":"m","value":2,"unit":"u"}\n', None)]
            with patch.object(hwprobe, "invoke", side_effect=outputs) as invoke:
                record = hwprobe.probe(Path("test.cu"), args)
        self.assertEqual(invoke.call_args_list[1].args[0][-1], "7")
        self.assertEqual(record, dict(feature="test", compiled=True, ran=True, metric="m", value=2, unit="u", error=None))

    def test_python_hosted_probe_builds_cubin(self):
        source = hwprobe.ROOT / "tma_encode.cu"
        args = MagicMock(nvcc="nvcc", iterations=None)
        binary, build, run = hwprobe.commands(source, Path("/tmp/x"), args)
        self.assertEqual(binary.name, "probe.cubin")
        self.assertIn("-cubin", build)
        self.assertEqual(run[:2], [sys.executable, str(hwprobe.ROOT / "tma_encode.py")])

    def test_failure_does_not_stop_later_probe(self):
        records = [dict(feature="bad", compiled=False, ran=False, metric="compile_failed", value=None, unit=None, error="e"),
                   dict(feature="good", compiled=True, ran=True, metric="throughput", value=1, unit="TOP/s", error=None)]
        stream = io.StringIO()
        with patch.object(hwprobe, "sources", return_value=[Path("bad.cu"), Path("good.cu")]), \
             patch.object(hwprobe, "probe", side_effect=records), contextlib.redirect_stdout(stream):
            result = hwprobe.main(["--output-dir", "/workspace/samples/unit"])
        self.assertEqual(result, 0)
        self.assertEqual([json.loads(line) for line in stream.getvalue().splitlines()], records)


if __name__ == "__main__":
    unittest.main()
