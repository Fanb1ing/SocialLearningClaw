#!/usr/bin/env python3
"""Download every ARC-AGI-3 game visible to the configured API account."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.metadata import version as package_version
from pathlib import Path

import arc_agi

from socialclaw.utils import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS_DIR = PROJECT_ROOT / "third_party" / "arc_agi3_games"
INVENTORY_PATH = ENVIRONMENTS_DIR / "inventory.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    # The SDK writes a host-specific absolute path. Offline loading scans the
    # supplied environments directory and reconstructs local_dir at runtime.
    metadata.pop("local_dir", None)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _write_inventory() -> list[str]:
    entries = []
    failures = []
    for metadata_path in sorted(ENVIRONMENTS_DIR.rglob("metadata.json")):
        metadata = _canonicalize_metadata(metadata_path)
        game_id = str(metadata.get("game_id", ""))
        base_id, separator, version = game_id.partition("-")
        source_path = metadata_path.parent / f"{base_id}.py"
        if not separator or not base_id or not version or not source_path.exists():
            failures.append(game_id or str(metadata_path))
            continue
        entries.append(
            {
                "game_id": game_id,
                "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metadata_sha256": _sha256(metadata_path),
                "source": str(source_path.relative_to(PROJECT_ROOT)),
                "source_sha256": _sha256(source_path),
            }
        )
    if failures:
        raise RuntimeError(
            "Invalid local ARC game pairs: " + ", ".join(failures)
        )
    payload = {
        "format_version": 1,
        "source": "ARC Prize Foundation ARC-AGI-3 API",
        "arc_agi_sdk": package_version("arc-agi"),
        "games": entries,
    }
    INVENTORY_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [entry["game_id"] for entry in entries]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="Canonicalize and inventory already-downloaded games without API access",
    )
    args = parser.parse_args()

    if args.normalize_only:
        game_ids = _write_inventory()
        print(f"Verified {len(game_ids)} local games; inventory: {INVENTORY_PATH}")
        return

    load_dotenv(str(PROJECT_ROOT / ".env"))
    if os.environ.get("ARC_AGI_API_KEY") and not os.environ.get("ARC_API_KEY"):
        os.environ["ARC_API_KEY"] = os.environ["ARC_AGI_API_KEY"]
    if not os.environ.get("ARC_API_KEY"):
        raise SystemExit("Missing ARC API key in ARC_AGI_API_KEY or ARC_API_KEY")

    ENVIRONMENTS_DIR.mkdir(parents=True, exist_ok=True)
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.NORMAL,
        environments_dir=str(ENVIRONMENTS_DIR),
    )
    games = sorted(arcade.get_environments(), key=lambda item: item.game_id)
    if not games:
        raise SystemExit("The ARC API returned no available games")

    failures: list[str] = []
    for index, game in enumerate(games, start=1):
        base_id, version = game.game_id.split("-", 1)
        local_dir = ENVIRONMENTS_DIR / base_id / version
        if (local_dir / "metadata.json").exists() and list(local_dir.glob("*.py")):
            print(
                f"[{index:02d}/{len(games):02d}] {game.game_id} (already present)",
                flush=True,
            )
            continue
        print(f"[{index:02d}/{len(games):02d}] {game.game_id}", flush=True)
        if arcade.make(game.game_id) is None:
            failures.append(game.game_id)

    local_game_ids = _write_inventory()
    metadata_files = list(ENVIRONMENTS_DIR.rglob("metadata.json"))
    source_files = [
        path
        for path in ENVIRONMENTS_DIR.rglob("*.py")
        if path.name != "__init__.py"
    ]
    print(
        f"Downloaded inventory: {len(metadata_files)} metadata files, "
        f"{len(source_files)} source files in {ENVIRONMENTS_DIR}"
    )
    if failures:
        raise SystemExit(f"Failed to prepare {len(failures)} games: {', '.join(failures)}")
    if len(metadata_files) != len(games) or len(source_files) != len(games):
        raise SystemExit(
            f"Incomplete inventory: API={len(games)}, metadata={len(metadata_files)}, "
            f"sources={len(source_files)}"
        )
    remote_game_ids = sorted(game.game_id for game in games)
    if local_game_ids != remote_game_ids:
        raise SystemExit(
            "Local/remote game ID mismatch after refresh: "
            f"local={local_game_ids}, remote={remote_game_ids}"
        )


if __name__ == "__main__":
    main()
