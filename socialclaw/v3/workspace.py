"""Tycho workspace extension that adds EFPS helpers and evidence indexing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tycho.workspace.workspace import GameWorkspace

from .evidence import index_workspace


_WMLIB_EVIDENCE_EXTENSION = r'''

# ---- SocialLearningClaw V3 typed Evidence extension ----
def evidence_refs(ids=None, roles=None, root="."):
    """Return portable real-observation Evidence records from notes/evidence_index.json.

    This reads a harness-authored index only. World-model simulations never append to it.
    ``ids`` and ``roles`` are optional exact-match filters.
    """
    path = os.path.join(root, "notes", "evidence_index.json")
    if not os.path.exists(path):
        return []
    try:
        data = json.load(open(path))
    except Exception:
        return []
    wanted_ids = set(ids or [])
    wanted_roles = set(roles or [])
    return [
        item for item in data.get("evidence", [])
        if (not wanted_ids or item.get("evidence_id") in wanted_ids)
        and (not wanted_roles or item.get("role") in wanted_roles)
    ]
'''


_AUDIT_SCRIPT = r'''"""Export and validate the current executable EFPS view."""
import json

import efps_runtime
import wmlib
import world_model


refs = wmlib.evidence_refs()
state = wmlib.current_state(world_model)
entity_fn = getattr(world_model, "efps_entities", None)
entities = entity_fn(state) if callable(entity_fn) and state is not None else ()
manifest = efps_runtime.export_efps(
    known_evidence_ids=[item["evidence_id"] for item in refs],
    world_model_path="world_model.py",
    entities=entities,
    applicability=getattr(world_model, "EFPS_APPLICABILITY", "partial"),
    applicability_reason=getattr(
        world_model,
        "EFPS_APPLICABILITY_REASON",
        "No applicability decision has been recorded yet.",
    ),
)
with open("notes/efps_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(json.dumps({
    "status": "ok",
    "prototypes": len(manifest["prototypes"]),
    "schemas": len(manifest["schemas"]),
    "entities": len(manifest["entities"]),
    "world_model_sha256": manifest["world_model_sha256"],
}, sort_keys=True))
'''


_CONTRACT_NOTE = """# EFPS executable-world-model contract

`world_model.py` remains the sole source of transition dynamics. Use the local
`efps_runtime` module only when Prototype abstraction is useful:

- cite Evidence IDs returned by `wmlib.evidence_refs()`;
- decorate executable matchers with `@efps_runtime.prototype`;
- decorate actual transition handlers with `@efps_runtime.schema_rule`;
- let `efps_entities(state)` return current `EntityInstance` values for audit;
- call `REGISTRY.apply_rules(...)` from `transition()` for rules that drive it;
- never create Evidence from simulated states or planner rollouts.

Run `python efps_audit.py` after model edits. A manifest-only rule with no
callable handler is impossible by construction, and unknown Evidence IDs fail
the audit. Set `EFPS_APPLICABILITY` to `full`, `partial`, or `not_applicable`
and explain partial/bypass choices in `EFPS_APPLICABILITY_REASON`.
"""


class EFPSGameWorkspace(GameWorkspace):
    """A behavior-preserving Tycho workspace with append-only EFPS support."""

    def _seed_worldmodel(
        self,
        available: list,
        resume: bool = False,
        seed_world_model: bool = True,
    ) -> None:
        super()._seed_worldmodel(
            available,
            resume=resume,
            seed_world_model=seed_world_model,
        )
        if resume or not seed_world_model:
            return
        source = Path(__file__).with_name("efps_runtime.py").read_text()
        (self.dir / "efps_runtime.py").write_text(source)
        (self.dir / "efps_audit.py").write_text(_AUDIT_SCRIPT)
        (self.dir / "notes" / "efps_contract.md").write_text(_CONTRACT_NOTE)
        wmlib_path = self.dir / "wmlib.py"
        wmlib_path.write_text(wmlib_path.read_text() + _WMLIB_EVIDENCE_EXTENSION)
        self.refresh_evidence_index()

    @property
    def evidence_run_id(self) -> str:
        return os.environ.get("SC_V3_RUN_ID", "unbound-dev")

    def refresh_evidence_index(self) -> None:
        index = index_workspace(
            self.dir,
            run_id=self.evidence_run_id,
            game_id=self.dir.name,
        )
        index.write(self.dir / "notes" / "evidence_index.json")

    def record(self, grid, **kwargs: Any) -> None:
        super().record(grid, **kwargs)
        self.refresh_evidence_index()

    def record_terminal(self, level: int, terminal_grid, **kwargs: Any) -> None:
        super().record_terminal(level, terminal_grid, **kwargs)
        self.refresh_evidence_index()

    def record_solved(self, level: int, action: str, row=None, col=None) -> None:
        super().record_solved(level, action, row=row, col=col)
        self.refresh_evidence_index()

    def record_game_over(self, **kwargs: Any) -> dict | None:
        level = int(kwargs["level"])
        attempts_dir = self.dir / "attempts"
        attempt = len(list(attempts_dir.glob(f"level_{level}_attempt_*"))) if attempts_dir.exists() else 0
        event = super().record_game_over(**kwargs)
        if event is not None:
            event["efps_attempt"] = attempt
            path = self.dir / f"level_{level}" / f"{event['stem']}.json"
            path.write_text(json.dumps(event))
        self.refresh_evidence_index()
        return event

    def record_animation_event(self, **kwargs: Any) -> dict | None:
        event = super().record_animation_event(**kwargs)
        self.refresh_evidence_index()
        return event

    def reset_level(self, level: int, *, reason: str = "reset") -> dict | None:
        archived = super().reset_level(level, reason=reason)
        self.refresh_evidence_index()
        return archived
