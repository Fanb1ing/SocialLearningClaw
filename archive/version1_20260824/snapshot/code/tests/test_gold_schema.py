from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "gold/arc_agi3/v1"
GAME_ROOT = GOLD_ROOT / "games/cd82-fb555c5d"
SOURCE_PATH = (
    PROJECT_ROOT / "third_party/arc_agi3_games/cd82/fb555c5d/cd82.py"
)
REVIEWED_GAMES = {
    "cd82-fb555c5d": {
        "source": PROJECT_ROOT / "third_party/arc_agi3_games/cd82/fb555c5d/cd82.py",
        "levels": list(range(1, 7)),
        "schemas": 18,
    },
    "sk48-d8078629": {
        "source": PROJECT_ROOT / "third_party/arc_agi3_games/sk48/d8078629/sk48.py",
        "levels": list(range(1, 9)),
        "schemas": 10,
    },
    "tu93-0768757b": {
        "source": PROJECT_ROOT / "third_party/arc_agi3_games/tu93/0768757b/tu93.py",
        "levels": list(range(1, 10)),
        "schemas": 9,
    },
}


class GoldSchemaArtifactTests(unittest.TestCase):
    def test_arc_full_manifest_and_all_game_artifacts(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        games = {game["game_id"]: game for game in manifest["games"]}
        inventory = json.loads(
            (PROJECT_ROOT / "third_party/arc_agi3_games/inventory.json").read_text()
        )
        inventory_games = {game["game_id"]: game for game in inventory["games"]}

        self.assertEqual(set(games), set(inventory_games))
        self.assertEqual(len(games), 25)
        self.assertEqual(manifest["totals"]["games"], 25)
        self.assertEqual(
            manifest["totals"]["levels"],
            sum(len(game["levels"]) for game in games.values()),
        )
        self.assertEqual(
            manifest["totals"]["schemas"],
            sum(game["schema_count"] for game in games.values()),
        )
        for game_id, game in games.items():
            game_root = GOLD_ROOT / "games" / game_id
            source = PROJECT_ROOT / inventory_games[game_id]["source"]
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            validation = json.loads((game_root / "validation.json").read_text())
            schemas = json.loads((game_root / "schemas.json").read_text())["schemas"]

            self.assertEqual(game["source_sha256"], source_hash)
            self.assertEqual(validation["source_sha256"], source_hash)
            self.assertEqual(validation["status"], "passed")
            self.assertEqual(validation["goal_levels_covered"], game["levels"])
            self.assertEqual(len(schemas), game["schema_count"])
            self.assertEqual(
                validation["runtime_passed"], validation["runtime_case_count"]
            )
            self.assertTrue(validation["coverage_complete"])
            for schema in schemas:
                self.assertEqual(schema["game_id"], game_id)
                self.assertTrue(schema["source_evidence"])
                self.assertNotIn("memory_index", schema)
                self.assertNotIn("memory_ids", schema)

        for game_id, expected in REVIEWED_GAMES.items():
            self.assertEqual(games[game_id]["levels"], expected["levels"])
            self.assertEqual(games[game_id]["schema_count"], expected["schemas"])

    def test_new_batch_covers_every_advertised_action(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        for game in manifest["games"]:
            game_root = GOLD_ROOT / "games" / game["game_id"]
            cases = json.loads((game_root / "runtime_cases.json").read_text())["cases"]
            smoke = next(
                (
                    case
                    for case in cases
                    if case["case_id"] == "all_levels_and_actions_smoke"
                ),
                None,
            )
            if smoke is None:
                continue
            advertised = {
                row["action"] for row in smoke["observed"]["advertised_actions"]
            }
            schemas = json.loads((game_root / "schemas.json").read_text())["schemas"]
            documented = " ".join(
                action["action"]
                for schema in schemas
                for action in schema["action_sequence"]
            )
            missing = {
                action_id
                for action_id in advertised
                if f"ACTION{action_id}" not in documented
            }
            self.assertFalse(missing, f"{game['game_id']} missing actions {missing}")

    def test_cross_game_schemas_are_traceable_and_provisional(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        cross_manifest = manifest["cross_game"]
        payload = json.loads(
            (GOLD_ROOT / cross_manifest["schemas"]).read_text()
        )
        validation = json.loads(
            (GOLD_ROOT / cross_manifest["validation"]).read_text()
        )
        schemas = payload["schemas"]
        cross_ids = {schema["schema_id"] for schema in schemas}
        concrete_ids = {
            schema["schema_id"]
            for game in manifest["games"]
            for schema in json.loads(
                (GOLD_ROOT / "games" / game["game_id"] / "schemas.json").read_text()
            )["schemas"]
        }

        self.assertEqual(cross_manifest["status"], "provisional")
        self.assertEqual(cross_manifest["schema_count"], 15)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(validation["level_counts"], {"0": 3, "1": 12})
        self.assertEqual(validation["covered_game_count"], 25)
        for schema in schemas:
            self.assertEqual(schema["game_id"], "__cross_game__")
            self.assertEqual(schema["derivation_status"], "provisional")
            self.assertTrue(schema["source_evidence"])
            self.assertTrue(schema["member_schema_ids"])
            expected = concrete_ids if schema["abstraction_level"] == 1 else cross_ids
            self.assertTrue(set(schema["member_schema_ids"]).issubset(expected))
            self.assertEqual(
                set(schema["relations"]["members"]),
                set(schema["member_schema_ids"]),
            )
            for evidence in schema["source_evidence"]:
                source = PROJECT_ROOT / evidence["path"]
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                    evidence["sha256"],
                )

    def test_cd82_pilot_is_source_pinned_and_fully_validated(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        validation = json.loads((GAME_ROOT / "validation.json").read_text())
        source_hash = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()

        cd_entry = next(
            game for game in manifest["games"] if game["game_id"] == "cd82-fb555c5d"
        )
        self.assertEqual(cd_entry["source_sha256"], source_hash)
        self.assertEqual(validation["source_sha256"], source_hash)
        self.assertEqual(validation["status"], "passed")
        self.assertEqual(
            validation["runtime_passed"], validation["runtime_case_count"]
        )
        self.assertTrue(validation["coverage_complete"])
        self.assertEqual(validation["goal_levels_covered"], [1, 2, 3, 4, 5, 6])

    def test_cd82_nodes_are_unique_atomic_records_without_memory_requirement(self) -> None:
        payload = json.loads((GAME_ROOT / "schemas.json").read_text())
        schemas = payload["schemas"]
        schema_ids = [schema["schema_id"] for schema in schemas]

        self.assertEqual(len(schemas), 18)
        self.assertEqual(len(schema_ids), len(set(schema_ids)))
        for schema in schemas:
            self.assertEqual(schema["game_id"], "cd82-fb555c5d")
            self.assertTrue(schema["source_evidence"])
            self.assertNotIn("memory_index", schema)
            self.assertNotIn("memory_ids", schema)
            self.assertEqual(schema["verification"]["static"], "passed")

        titles = {schema["title"] for schema in schemas}
        self.assertNotIn("不同关卡提供不同的可选颜色集合", titles)
        self.assertNotIn("最底行进度条显示 100 次 action 预算的剩余比例", titles)
        self.assertEqual(
            sum("相邻的 12 格区域" in title for title in titles),
            1,
        )

    def test_cd82_coverage_has_no_open_requirement(self) -> None:
        payload = json.loads((GAME_ROOT / "coverage.json").read_text())
        schemas = json.loads((GAME_ROOT / "schemas.json").read_text())["schemas"]
        schema_ids = {schema["schema_id"] for schema in schemas}

        for requirement in payload["requirements"]:
            self.assertEqual(requirement["status"], "covered")
            self.assertTrue(requirement["schema_ids"])
            self.assertTrue(set(requirement["schema_ids"]).issubset(schema_ids))


if __name__ == "__main__":
    unittest.main()
