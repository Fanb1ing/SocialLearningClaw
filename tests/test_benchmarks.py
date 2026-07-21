from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()

