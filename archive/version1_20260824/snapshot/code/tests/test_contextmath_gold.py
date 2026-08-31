from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "gold/contextmath/v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContextMathGoldTests(unittest.TestCase):
    def test_pilot_manifest_and_artifacts(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["scope"], "pilot")
        self.assertEqual(manifest["status"], "pilot_review_pending")
        self.assertEqual(manifest["totals"]["problems"], 6)
        self.assertEqual(manifest["totals"]["variants"], 12)
        self.assertEqual(manifest["totals"]["schemas"], 23)
        self.assertEqual(
            manifest["totals"]["witness_checks"],
            manifest["totals"]["witness_passed"],
        )
        for problem in manifest["problems"]:
            root = GOLD_ROOT / problem["directory"]
            for name in (
                "schemas.json",
                "surface_alignment.json",
                "witness.json",
                "coverage.json",
                "validation.json",
                "review.md",
            ):
                self.assertTrue((root / name).is_file(), f"missing {root / name}")
            validation = json.loads((root / "validation.json").read_text())
            self.assertEqual(validation["status"], "passed")
            self.assertTrue(validation["coverage_complete"])
            self.assertTrue(validation["answer_match"])
            self.assertEqual(validation["review"], "pending")

    def test_schemas_are_traceable_atomic_dags(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        required = set(json.loads((GOLD_ROOT / "schema_spec.json").read_text())["required"])
        for problem in manifest["problems"]:
            root = GOLD_ROOT / problem["directory"]
            schemas = json.loads((root / "schemas.json").read_text())["schemas"]
            witness = json.loads((root / "witness.json").read_text())
            witness_ids = {item["witness_id"] for item in witness["checks"]}
            seen: set[str] = set()
            for schema in schemas:
                self.assertFalse(required - set(schema))
                self.assertEqual(schema["benchmark"], "contextmath")
                self.assertEqual(schema["abstraction_level"], 3)
                self.assertNotIn("gold_answer", schema)
                self.assertNotIn("memory_ids", schema)
                self.assertTrue(schema["source_evidence"])
                self.assertTrue(schema["witness_ids"])
                self.assertTrue(set(schema["witness_ids"]) <= witness_ids)
                self.assertTrue(set(schema["relations"]["requires"]) <= seen)
                for evidence in schema["source_evidence"]:
                    source = PROJECT_ROOT / evidence["path"]
                    self.assertEqual(sha256_file(source), evidence["sha256"])
                    row = pd.read_parquet(source).loc[
                        lambda frame: frame["id"] == evidence["row_id"]
                    ]
                    self.assertEqual(len(row), 1)
                    value = str(row.iloc[0][evidence["field"]])
                    self.assertEqual(
                        hashlib.sha256(value.encode("utf-8")).hexdigest(),
                        evidence["value_sha256"],
                    )
                seen.add(schema["schema_id"])
            self.assertEqual(len(seen), len(schemas))

    def test_alignments_and_witnesses_are_complete(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        for problem in manifest["problems"]:
            root = GOLD_ROOT / problem["directory"]
            alignment = json.loads((root / "surface_alignment.json").read_text())
            witness = json.loads((root / "witness.json").read_text())
            self.assertEqual({item["split"][-2:] for item in alignment["variants"]}, {"sg", "cs"})
            for variant in alignment["variants"]:
                self.assertEqual(variant["coverage"], "complete")
                mapped = {item["canonical"] for item in variant["mappings"]}
                self.assertTrue(set(alignment["canonical_symbols"]) <= mapped)
            self.assertTrue(all(item["passed"] for item in witness["checks"]))
            self.assertEqual(
                witness["derived_answer_normalized"],
                witness["gold_answer_normalized"],
            )


if __name__ == "__main__":
    unittest.main()
