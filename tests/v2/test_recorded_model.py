from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from socialclaw.v2.model import ModelImage, RecordedVisionModel


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RecordedVisionModelTests(unittest.TestCase):
    def test_replays_and_validates_logical_calls(self) -> None:
        instructions = "frozen instructions"
        payload = "frozen payload"
        image = ModelImage(
            label="current",
            artifact_id="artifact_1",
            sha256="a" * 64,
            relative_path="images/a.png",
            data_url="data:image/png;base64,AA==",
        )
        transcript = {
            "format_version": 1,
            "model": "recorded/model",
            "episode_created_at": "2026-08-30T00:00:00+00:00",
            "experiment_config": {"game_id": "fixture-v1"},
            "calls": [
                {
                    "method": "structured",
                    "instructions_sha256": _hash(instructions),
                    "payload_sha256": _hash(payload),
                    "image_sha256": [image.sha256],
                    "output": {"answer": 1},
                    "model": "recorded/model",
                    "usage": {"total_tokens": 3},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(json.dumps(transcript), encoding="utf-8")
            model = RecordedVisionModel(path)
            result = model.generate(
                instructions=instructions,
                payload=payload,
                images=[image],
            )
            self.assertEqual(result.data, {"answer": 1})
            self.assertEqual(result.usage["total_tokens"], 3)
            self.assertEqual(model.experiment_config["game_id"], "fixture-v1")
            model.assert_exhausted()

    def test_rejects_changed_runtime_input(self) -> None:
        transcript = {
            "format_version": 1,
            "model": "recorded/model",
            "calls": [
                {
                    "method": "text",
                    "instructions_sha256": _hash("instructions"),
                    "payload_sha256": _hash("expected"),
                    "image_sha256": [],
                    "output": "probe",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(json.dumps(transcript), encoding="utf-8")
            model = RecordedVisionModel(path)
            with self.assertRaisesRegex(ValueError, "model payload changed"):
                model.generate_text(
                    instructions="instructions",
                    payload="changed",
                    images=[],
                )


if __name__ == "__main__":
    unittest.main()
