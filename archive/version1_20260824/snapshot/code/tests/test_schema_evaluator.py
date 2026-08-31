from __future__ import annotations

import unittest
from pathlib import Path

from socialclaw.schema.evaluation import (
    CanonicalSchema,
    LearnedGoldEvaluator,
    MatchRelation,
    StructuredARCJudge,
    canonicalize_gold,
)
from socialclaw.schema.gold_loader import load_accepted_arc_gold


def schema(
    schema_id: str,
    source: str,
    *,
    action: str = "ACTION1",
    concepts=frozenset({"canvas"}),
    effect: str = "effect",
    expectation: str = "Paint the upper half of the canvas",
    metadata=None,
) -> CanonicalSchema:
    return CanonicalSchema(
        schema_id=schema_id,
        source=source,
        game_id="fixture-game",
        kind="action_effect",
        levels={1, 2},
        actions={action} if action else set(),
        roles=set(),
        concepts=set(concepts),
        effect_class=effect if source == "learned" else "",
        trigger="same detailed trigger",
        expectation=expectation,
        text=expectation,
        evidence_ids={f"memory_{schema_id}"} if source == "learned" else set(),
        metadata=dict(metadata or {}),
    )


class SchemaEvaluatorTests(unittest.TestCase):
    def test_gold_canonicalizer_expands_action_families_and_roles(self) -> None:
        value = {
            "schema_id": "gold",
            "game_id": "fixture-game",
            "kind": "action_effect",
            "level_scope": [1],
            "title": "direction rule",
            "trigger": "",
            "expectation": "",
            "constraints": [],
            "exceptions": [],
            "action_sequence": [
                {"action": "ACTION6", "arguments": {"target_role": "chain_head"}},
                {"action": "movement_action", "arguments": {}},
            ],
        }
        canonical = canonicalize_gold(value)
        self.assertEqual(
            canonical.actions,
            {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"},
        )
        self.assertEqual(canonical.roles, {"chain_head"})

    def test_structured_judge_separates_equivalent_partial_and_unrelated(self) -> None:
        judge = StructuredARCJudge()
        gold = schema("gold", "gold")
        exact = judge.judge(schema("exact", "learned"), gold)
        generic = judge.judge(
            schema("generic", "learned", expectation="The canvas changes at medium scale"), gold
        )
        unrelated = judge.judge(schema("other", "learned", action="ACTION7"), gold)
        self.assertEqual(exact.relation, MatchRelation.EQUIVALENT)
        self.assertEqual(generic.relation, MatchRelation.PARTIAL)
        self.assertEqual(unrelated.relation, MatchRelation.UNRELATED)

    def test_same_condition_opposite_effect_is_a_contradiction(self) -> None:
        learned = schema(
            "learned",
            "learned",
            effect="no_effect",
            metadata={"condition_key": "same"},
        )
        gold = schema(
            "gold",
            "gold",
            metadata={"condition_key": "same", "expected_effect": "effect"},
        )
        result = StructuredARCJudge().judge(learned, gold)
        self.assertEqual(result.relation, MatchRelation.CONTRADICTION)

    def test_evaluator_labels_overmerge_split_and_evidence_traceability(self) -> None:
        broad = schema("broad", "learned", expectation="The canvas changes at medium scale")
        narrow = schema("narrow", "learned", expectation="The canvas changes at local scale")
        gold_one = schema("gold_one", "gold")
        gold_two = schema("gold_two", "gold")
        metrics, alignments = LearnedGoldEvaluator().evaluate(
            [broad, narrow],
            [gold_one, gold_two],
            memory_ids={"memory_broad", "memory_narrow"},
        )
        relations = {item["relation"] for item in alignments}
        self.assertIn(MatchRelation.LEARNED_BROADER.value, relations)
        self.assertEqual(metrics["split_gold_count"], 2)
        self.assertEqual(metrics["overmerged_learned_count"], 2)
        self.assertEqual(metrics["evidence_traceability"], 1.0)

        _, split_alignments = LearnedGoldEvaluator().evaluate(
            [broad, narrow], [gold_one], memory_ids={"memory_broad", "memory_narrow"}
        )
        self.assertEqual(
            {item["relation"] for item in split_alignments},
            {MatchRelation.LEARNED_NARROWER.value},
        )

    def test_gold_loader_accepts_the_three_reviewed_games(self) -> None:
        games = {"cd82-fb555c5d", "sk48-d8078629", "tu93-0768757b"}
        bundle = load_accepted_arc_gold("gold/arc_agi3/v1", game_ids=games)
        self.assertEqual(set(bundle.games), games)
        self.assertEqual(len(bundle.schemas), 37)

    def test_induction_modules_do_not_import_gold_or_evaluator(self) -> None:
        for path in (
            Path("socialclaw/schema/trajectory_pipeline.py"),
            Path("socialclaw/schema/window_induction.py"),
        ):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("gold_loader", source)
            self.assertNotIn("schema.evaluation", source)


if __name__ == "__main__":
    unittest.main()
