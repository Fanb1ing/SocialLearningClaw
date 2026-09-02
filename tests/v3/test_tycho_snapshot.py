from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_vendored_tycho_matches_public_manifest_except_documented_hook() -> None:
    manifest = json.loads(
        (ROOT / "third_party" / "tycho" / "PUBLIC_RELEASE_MANIFEST.json").read_text()
    )["files"]
    mismatches = []
    for relative, descriptor in manifest.items():
        if relative in {
            "tycho/agent/agent.py",
            "tycho/harness/harness.py",
            "tycho/harness/run_parallel.py",
            "tycho/harness/run_spec.py",
            "tycho/workspace/sandbox.py",
            "tycho/config/run_config.py",
        }:
            continue
        if relative.startswith("tycho/"):
            target = ROOT / relative
        elif relative.startswith("tests/"):
            target = ROOT / "third_party" / "tycho" / relative
        else:
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else "<missing>"
        if actual != descriptor["sha256"]:
            mismatches.append(relative)
    assert mismatches == []


def test_documented_extension_is_registered_and_policy_hashed() -> None:
    from tycho.harness._run_extension import (
        APPROACH_MODULES,
        POLICY_PATHS,
        execution_extra_sources,
    )
    from tycho.harness.run_parallel import _run_spec_extra_sources

    assert APPROACH_MODULES["tycho_efps"] == "socialclaw.v3.agent"
    assert "socialclaw/v3" in POLICY_PATHS
    game = "cd82-fb555c5d"
    extension_sources = execution_extra_sources({game: [1]})
    assert len(extension_sources[f"environment/{game}"]) == 64
    assert _run_spec_extra_sources({game: [1]})[f"environment/{game}"] == (
        extension_sources[f"environment/{game}"]
    )
