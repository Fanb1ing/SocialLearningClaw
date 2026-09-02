from __future__ import annotations

from tycho.harness.record_slim import slim_record


def test_slim_record_drops_harness_owned_animation_files():
    rec = {
        "trace": [{
            "frame": [[0]],
            "reasoning": {
                "workspace": {
                    "files": [
                        "notes/animation_evidence.md",
                        "level_0/animation_003_ACTION5/meta.json",
                        "level_0/animation_003_ACTION5/frame_000.txt",
                        "level_0/animation_003_ACTION5/frame_000.png",
                    ],
                    "contents": {
                        "notes/animation_evidence.md": "note",
                        "level_0/animation_003_ACTION5/meta.json": "{}",
                        "level_0/animation_003_ACTION5/frame_000.txt": "x: 00-07\n...",
                    },
                    "images": {},
                }
            },
        }]
    }

    slim_record(rec)
    ws = rec["trace"][0]["reasoning"]["workspace"]

    assert "notes/animation_evidence.md" in ws["files"]
    assert "level_0/animation_003_ACTION5/meta.json" not in ws["files"]
    assert "level_0/animation_003_ACTION5/frame_000.txt" not in ws["files"]
    assert "level_0/animation_003_ACTION5/frame_000.png" not in ws["files"]
    assert "level_0/animation_003_ACTION5/meta.json" not in ws["contents"]
    assert "level_0/animation_003_ACTION5/frame_000.txt" not in ws["contents"]


def test_slim_record_preserves_binary_descriptor_and_versions_unchanged_manifest():
    descriptor = {
        "sha256": "a" * 64,
        "size": 128,
        "kind": "binary",
        "stored": True,
    }
    rec = {
        "trace": [
            {
                "frame": [[0]],
                "reasoning": {
                    "workspace": {
                        "files": ["state.npy", "level_0/turn_000.txt"],
                        "file_versions": {
                            "state.npy": dict(descriptor),
                            "level_0/turn_000.txt": {
                                "sha256": "b" * 64,
                                "size": 1,
                                "kind": "text",
                                "stored": True,
                            },
                        },
                        "contents": {"level_0/turn_000.txt": "0"},
                    }
                },
            },
            {
                "frame": [[1]],
                "reasoning": {
                    "workspace": {
                        "files": ["state.npy"],
                        "file_versions": {"state.npy": dict(descriptor)},
                        "contents": {},
                    }
                },
            },
        ]
    }

    slim_record(rec)

    first = rec["trace"][0]["reasoning"]["workspace"]
    second = rec["trace"][1]["reasoning"]["workspace"]
    assert first["files"] == ["state.npy"]
    assert first["file_versions"] == {"state.npy": descriptor}
    assert second["file_versions"] == "\x00=0"
