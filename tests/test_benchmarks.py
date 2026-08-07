from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from socialclaw.benchmarks.base import BenchmarkSample
from socialclaw.benchmarks.contextmath import answers_match, extract_boxed
from socialclaw.benchmarks.intphys2 import IntPhys2Benchmark, parse_binary_answer


class ContextMathParsingTests(unittest.TestCase):
    def test_extracts_last_boxed_integer(self) -> None:
        self.assertEqual(extract_boxed(r"work \boxed{12}, correction \boxed{007}"), "007")

    def test_numeric_answers_ignore_padding_and_commas(self) -> None:
        self.assertTrue(answers_match("007", "7"))
        self.assertTrue(answers_match("1,024", "1024"))
        self.assertFalse(answers_match(None, "7"))


class IntPhysParsingTests(unittest.TestCase):
    def test_binary_parser_is_strict(self) -> None:
        self.assertEqual(parse_binary_answer("1"), 1)
        self.assertEqual(parse_binary_answer("Answer: 0"), 0)
        self.assertIsNone(parse_binary_answer("plausible"))

    def test_evaluation(self) -> None:
        benchmark = IntPhys2Benchmark("unused")
        sample = BenchmarkSample(id="x", prompt="p", gold=1)
        self.assertTrue(benchmark.evaluate("1", sample).correct)
        self.assertFalse(benchmark.evaluate("0", sample).correct)

    def test_schema_query_excludes_label_metadata(self) -> None:
        benchmark = IntPhys2Benchmark("unused")
        sample = BenchmarkSample(
            id="x",
            prompt="judge",
            gold=1,
            metadata={
                "condition": "solidity",
                "camera": "cam1",
                "game": "blocks",
                "type": "Possible",
            },
        )
        query = benchmark.schema_query(sample)
        self.assertIn("solidity", query)
        self.assertNotIn("Possible", query)

    def test_main_sample_uses_pinned_sample_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Main").mkdir()
            sample = root / "Main" / "sample_300.csv"
            sample.write_text("file_name,type\nVideos/x.mp4,1_Possible\n")
            benchmark = IntPhys2Benchmark(root)
            self.assertEqual(benchmark._metadata_path("main_300"), sample)
            self.assertEqual(benchmark._videos_dir("main_300"), root / "Main/Videos")

    def test_main_sample_requires_preparation_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Main/Videos").mkdir(parents=True)
            (root / "Main/sample_300.csv").write_text(
                "file_name,type\nVideos/x.mp4,1_Possible\n"
            )
            (root / "Main/Videos/x.mp4").touch()
            benchmark = IntPhys2Benchmark(root)
            with self.assertRaises(FileNotFoundError):
                benchmark.load("main_300")

    def test_incomplete_debug_inventory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Debug/Videos").mkdir(parents=True)
            (root / "Debug/metadata.csv").write_text(
                "file_name,type\nVideos/missing.mp4,1_Impossible\n"
            )
            benchmark = IntPhys2Benchmark(root)
            with self.assertRaises(FileNotFoundError):
                benchmark.load("debug")


if __name__ == "__main__":
    unittest.main()
