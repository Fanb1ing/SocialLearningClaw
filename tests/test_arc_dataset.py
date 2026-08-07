import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from socialclaw.dataset.arc_agi3 import (
    ARCAGI3EnvWrapper,
    DEFAULT_ARC_ENVIRONMENTS_DIR,
    arc_environment_files,
    arc_environment_fingerprint,
)


class ARCAGI3DatasetTests(unittest.TestCase):
    def test_wrapper_uses_vendored_games_in_offline_mode(self) -> None:
        calls = {}
        fake_environment = object()

        class FakeArcade:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def make(self, game_id, render_mode=None):
                calls["make"] = (game_id, render_mode)
                return fake_environment

        fake_arc_agi = SimpleNamespace(
            Arcade=FakeArcade,
            OperationMode=SimpleNamespace(OFFLINE="offline"),
        )
        fake_arcengine = SimpleNamespace(GameState=object())
        with patch.dict(
            sys.modules,
            {"arc_agi": fake_arc_agi, "arcengine": fake_arcengine},
        ):
            wrapper = ARCAGI3EnvWrapper("sk48-d8078629")

        self.assertIs(wrapper.env, fake_environment)
        self.assertEqual(calls["init"]["operation_mode"], "offline")
        self.assertEqual(
            Path(calls["init"]["environments_dir"]),
            DEFAULT_ARC_ENVIRONMENTS_DIR,
        )
        self.assertEqual(calls["make"], ("sk48-d8078629", None))

    def test_environment_fingerprint_ignores_volatile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            game_dir = root / "ab12" / "deadbeef"
            game_dir.mkdir(parents=True)
            metadata = {
                "game_id": "ab12-deadbeef",
                "title": "AB12",
                "local_dir": "/host/one",
                "date_downloaded": "2026-01-01T00:00:00Z",
            }
            metadata_path = game_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            source_path = game_dir / "ab12.py"
            source_path.write_text("VALUE = 1\n", encoding="utf-8")
            first = arc_environment_fingerprint("ab12-deadbeef", root)
            metadata["local_dir"] = "/host/two"
            metadata["date_downloaded"] = "2026-02-02T00:00:00Z"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            second = arc_environment_fingerprint("ab12-deadbeef", root)
            source_path.write_text("VALUE = 2\n", encoding="utf-8")
            third = arc_environment_fingerprint("ab12-deadbeef", root)

        self.assertEqual(first, second)
        self.assertNotEqual(second, third)

    def test_environment_files_require_full_versioned_id(self) -> None:
        with self.assertRaises(ValueError):
            arc_environment_files("sk48")


if __name__ == "__main__":
    unittest.main()
