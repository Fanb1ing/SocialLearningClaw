from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ACCEPTED_REVIEW_STATUSES = {"accepted", "accepted_and_revised"}


@dataclass(frozen=True)
class GoldSchemaBundle:
    benchmark: str
    version: int
    root: Path
    games: Dict[str, str]
    schemas: List[Dict[str, Any]]


def load_accepted_arc_gold(
    root: str | Path,
    *,
    game_ids: Optional[Iterable[str]] = None,
) -> GoldSchemaBundle:
    """Load human-accepted Gold in an evaluator-only module.

    Generation and runner modules must never import this file. The loader is
    deliberately read-only and returns detached dictionaries.
    """

    gold_root = Path(root)
    manifest = json.loads((gold_root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("format_version", 0)) != 1:
        raise ValueError("Unsupported ARC Gold manifest format")
    requested = set(game_ids or [])
    games: Dict[str, str] = {}
    schemas: List[Dict[str, Any]] = []
    for game in manifest.get("games", []):
        game_id = str(game["game_id"])
        if requested and game_id not in requested:
            continue
        status = str(game.get("review_status", "pending"))
        if status not in ACCEPTED_REVIEW_STATUSES:
            continue
        path = gold_root / "games" / game_id / "schemas.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = payload.get("schemas", [])
        if int(game.get("schema_count", -1)) != len(values):
            raise ValueError(f"Gold schema count mismatch for {game_id}")
        games[game_id] = status
        schemas.extend(dict(value) for value in values)
    missing = requested - set(games)
    if missing:
        raise ValueError(f"Requested games are absent or not accepted: {sorted(missing)}")
    return GoldSchemaBundle(
        benchmark=str(manifest.get("benchmark", "")),
        version=int(manifest["format_version"]),
        root=gold_root.resolve(),
        games=games,
        schemas=schemas,
    )
