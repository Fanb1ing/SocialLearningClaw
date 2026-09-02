from __future__ import annotations

import json

import pytest

from socialclaw.v3.efps_runtime import (
    EFPSConflictError,
    EFPSEvidenceError,
    EFPSRegistry,
    EntityInstance,
    applied_schema_ids,
)


OBS = "evi_observation"


def _registry() -> EFPSRegistry:
    registry = EFPSRegistry()
    registry.add_prototype(
        "moving_piece",
        description="A colored piece that moves on the board.",
        evidence_ids=[OBS],
        matcher=lambda entity: entity.attributes.get("color") == 2,
    )
    return registry


def test_classification_executes_schema_handler_and_attributes_rule() -> None:
    registry = _registry()

    def move(state, entity, action):
        result = dict(state)
        result[entity.entity_id] = action["col"]
        return result

    registry.add_schema_rule(
        "move_right",
        prototype_id="moving_piece",
        action={"action": "ACTION6", "row": 1},
        output="The selected moving piece appears in the clicked column.",
        evidence_ids=[OBS],
        counter_evidence_ids=[],
        handler=move,
    )
    entities = registry.classify([
        EntityInstance("piece_1", {"color": 2}, (OBS,)),
        EntityInstance("wall_1", {"color": 5}, (OBS,)),
    ])

    result = registry.apply_rules(
        {},
        {"action": "ACTION6", "row": 1, "col": 7},
        entities,
    )

    assert entities[0].prototype_ids == ("moving_piece",)
    assert entities[1].prototype_ids == ()
    assert result.state == {"piece_1": 7}
    assert result.schema_ids == ("move_right",)
    # Dict states cannot carry attributes, but the explicit result remains canonical.
    assert applied_schema_ids(result.state) == ()


def test_conflicting_output_is_rejected_atomically() -> None:
    registry = _registry()
    handler = lambda state, entity, action: state
    registry.add_schema_rule(
        "first",
        prototype_id="moving_piece",
        action="ACTION1",
        output="moves up",
        evidence_ids=[OBS],
        counter_evidence_ids=[],
        handler=handler,
    )

    with pytest.raises(EFPSConflictError):
        registry.add_schema_rule(
            "conflict",
            prototype_id="moving_piece",
            action="ACTION1",
            output="moves down",
            evidence_ids=[OBS],
            counter_evidence_ids=[],
            handler=handler,
        )

    assert [rule.schema_id for rule in registry.schemas] == ["first"]


def test_export_binds_world_model_and_enforces_evidence_closure(tmp_path) -> None:
    registry = _registry()
    model = tmp_path / "world_model.py"
    model.write_text("VALUE = 1\n")

    manifest = registry.export_manifest(
        known_evidence_ids=[OBS],
        world_model_path=model,
        entities=[EntityInstance("piece_1", {"color": 2}, (OBS,))],
        applicability="partial",
        applicability_reason="Only the movable piece is object-centric.",
    )

    assert manifest["schema"] == 1
    assert len(manifest["world_model_sha256"]) == 64
    assert manifest["entities"][0]["prototype_ids"] == ("moving_piece",)
    json.dumps(manifest)

    with pytest.raises(EFPSEvidenceError, match="unknown Evidence IDs"):
        registry.export_manifest(
            known_evidence_ids=[],
            world_model_path=model,
        )
