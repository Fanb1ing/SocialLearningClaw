from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD_ROOT = PROJECT_ROOT / "gold/intphys2/v1"
TYPE_SET = {"1_Possible", "1_Impossible", "2_Possible", "2_Impossible"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IntPhys2GoldTests(unittest.TestCase):
    def test_manifest_covers_four_condition_pilot(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        self.assertEqual(manifest["scope"], "four_condition_pilot")
        self.assertEqual(manifest["status"], "pilot_review_pending")
        self.assertEqual(manifest["totals"]["scene_count"], 4)
        self.assertEqual(manifest["totals"]["unique_clip_count"], 16)
        self.assertEqual(manifest["totals"]["schema_count"], 8)
        self.assertEqual(manifest["totals"]["pair_assessment_count"], 8)
        self.assertEqual(
            {scene["condition"] for scene in manifest["scenes"]},
            {"permanence", "immutability", "continuity", "solidity"},
        )
        for scene in manifest["scenes"]:
            root = GOLD_ROOT / scene["directory"]
            self.assertTrue((root / "review.md").is_file())
            self.assertTrue((root / "scene_evidence.json").is_file())
            self.assertTrue((root / "contact_sheet.jpg").is_file())

    def test_scene_evidence_is_source_pinned_and_reproducible(self) -> None:
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        for scene in manifest["scenes"]:
            root = GOLD_ROOT / scene["directory"]
            evidence = json.loads((root / "scene_evidence.json").read_text())
            self.assertEqual({clip["type"] for clip in evidence["clips"]}, TYPE_SET)
            self.assertEqual(
                sorted(clip["gold_label"] for clip in evidence["clips"]),
                [0, 0, 1, 1],
            )
            sheet = root / evidence["contact_sheet"]["path"]
            self.assertEqual(sha256_file(sheet), evidence["contact_sheet"]["sha256"])
            metadata = PROJECT_ROOT / evidence["metadata_source"]["path"]
            self.assertEqual(sha256_file(metadata), evidence["metadata_source"]["sha256"])
            for alias in evidence["metadata_aliases"]:
                alias_path = PROJECT_ROOT / alias["path"]
                self.assertEqual(sha256_file(alias_path), alias["sha256"])
                self.assertTrue(alias["same_video_inventory"])
            for clip in evidence["clips"]:
                path = PROJECT_ROOT / clip["video_path"]
                self.assertEqual(sha256_file(path), clip["video_sha256"])
                cap = cv2.VideoCapture(str(path))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), clip["frame_count"])
                for sampled in clip["sampled_frames"]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, sampled["frame_index"])
                    ok, frame = cap.read()
                    self.assertTrue(ok)
                    self.assertEqual(
                        hashlib.sha256(frame.tobytes()).hexdigest(),
                        sampled["decoded_bgr_sha256"],
                    )
                cap.release()

    def test_schemas_are_linked_to_paired_visual_evidence(self) -> None:
        payload = json.loads((GOLD_ROOT / "schemas.json").read_text())
        schemas = payload["schemas"]
        required = set(json.loads((GOLD_ROOT / "schema_spec.json").read_text())["required"])
        schema_ids = {schema["schema_id"] for schema in schemas}
        manifest = json.loads((GOLD_ROOT / "manifest.json").read_text())
        evidence_ids = set()
        for scene in manifest["scenes"]:
            evidence = json.loads(
                (GOLD_ROOT / scene["directory"] / "scene_evidence.json").read_text()
            )
            evidence_ids.update(item["evidence_id"] for item in evidence["pair_assessments"])
        self.assertEqual(len(schemas), 8)
        for schema in schemas:
            self.assertFalse(required - set(schema))
            self.assertEqual(schema["benchmark"], "intphys2")
            self.assertTrue(set(schema["source_evidence"]) <= evidence_ids)
            self.assertEqual(len(schema["source_evidence"]), 2)
            self.assertTrue(set(schema["relations"]["parents"]) <= schema_ids)
            self.assertTrue(set(schema["relations"]["members"]) <= schema_ids)
            self.assertNotIn("gold_label", schema)
            self.assertEqual(schema["verification"]["visual"], "provisional")
            self.assertEqual(schema["verification"]["review"], "pending")


if __name__ == "__main__":
    unittest.main()
