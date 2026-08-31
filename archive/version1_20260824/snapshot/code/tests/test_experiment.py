from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

from socialclaw.experiment import (
    BASELINES,
    METHODS,
    AttemptResult,
    ExperimentBudget,
    ExperimentConfig,
    SampleResult,
    write_results,
)
from socialclaw.methods.controller import MethodController
from socialclaw.memory import MemoryBank, MemoryStore
from socialclaw.schema import SchemaManager


class ExperimentProtocolTests(unittest.TestCase):
    def test_method_registry_contains_all_requested_baselines(self) -> None:
        self.assertEqual(
            BASELINES,
            ("naive", "icl", "rag", "withrule", "reflexion", "expel", "amem", "tgm"),
        )
        self.assertEqual(METHODS[-1], "schema")

    def test_gold_feedback_is_rejected(self) -> None:
        config = ExperimentConfig(
            benchmark="contextmath",
            method="amem",
            model="test",
            feedback="gold",
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_memory_update_interface_has_no_gold_argument(self) -> None:
        signature = inspect.signature(MethodController.after_sample)
        self.assertNotIn("gold", signature.parameters)

    def test_schema_method_uses_shared_binary_feedback_lifecycle(self) -> None:
        manager = SchemaManager(memory=MemoryBank(MemoryStore()))
        controller = MethodController(
            method="schema",
            openai_client=None,
            model="test",
            schema_manager=manager,
        )
        controller.after_sample(
            task="If a lever is visible, choose ACTION1",
            response="ACTION1",
            correct=True,
            domain="arc_agi3",
        )
        self.assertEqual(len(manager.memory.store.list()), 1)
        self.assertEqual(len(manager.graph.list()), 1)
        self.assertIn("Retrieved world schemas", controller.context("lever"))

    def test_schema_keeps_memory_when_auxiliary_induction_fails(self) -> None:
        class BrokenGenerator:
            def propose(self, memory, candidates):
                raise RuntimeError("provider unavailable")

            def merge_description(self, left, right):
                raise RuntimeError("provider unavailable")

        manager = SchemaManager(
            memory=MemoryBank(MemoryStore()),
            generator=BrokenGenerator(),
        )
        controller = MethodController(
            method="schema",
            openai_client=None,
            model="test",
            schema_manager=manager,
        )
        controller.after_sample(
            task="task",
            response="response",
            correct=False,
            domain="contextmath",
        )
        records = manager.memory.store.list()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].metadata["induction_error"], "RuntimeError")
        self.assertEqual(manager.graph.list(), [])

    def test_results_report_first_and_final_accuracy(self) -> None:
        results = [
            SampleResult(
                sample_id="x",
                correct=True,
                prediction="2",
                attempts=[
                    AttemptResult(1, "1", False, "wrong"),
                    AttemptResult(2, "2", True, "right"),
                ],
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_results(Path(directory), results)
            payload = json.loads(path.read_text())
        self.assertEqual(payload["metrics"]["accuracy"], 1.0)
        self.assertEqual(payload["metrics"]["first_attempt_accuracy"], 0.0)

    def test_budget_validation(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentBudget(max_attempts=0).validate()


if __name__ == "__main__":
    unittest.main()
