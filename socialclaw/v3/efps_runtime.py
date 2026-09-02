"""Executable EFPS primitives for agent-authored Tycho world models.

This module intentionally depends only on the Python standard library because a
copy is seeded into each sandboxed game workspace. It is not a second world
model: registered schema rules are callables used by ``transition`` and the
manifest is a read-only projection of those callables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EFPS_MANIFEST_SCHEMA = 1
_APPLIED_ATTR = "__efps_applied_schema_ids__"


class EFPSError(ValueError):
    """Base error for an invalid executable EFPS definition."""


class EFPSConflictError(EFPSError):
    """Two rules make incompatible claims for one Prototype/action pattern."""


class EFPSEvidenceError(EFPSError):
    """An EFPS definition has missing or unknown durable Evidence."""


def _ids(values: Iterable[str], *, field: str, required: bool = True) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    if required and not out:
        raise EFPSEvidenceError(f"{field} must cite at least one Evidence ID")
    for value in out:
        if not value.startswith("evi_"):
            raise EFPSEvidenceError(f"invalid Evidence ID {value!r} in {field}")
    return out


def _nonempty(value: str, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise EFPSError(f"{field} must be non-empty")
    return text


def _action_pattern(value: Mapping[str, Any] | str) -> dict[str, Any]:
    pattern = {"action": value} if isinstance(value, str) else dict(value)
    if not pattern.get("action"):
        raise EFPSError("an action pattern must include a non-empty public action name")
    allowed = {"action", "row", "col"}
    unknown = sorted(set(pattern) - allowed)
    if unknown:
        raise EFPSError("unsupported action-pattern fields: " + ", ".join(unknown))
    return {key: pattern[key] for key in ("action", "row", "col") if key in pattern}


def _action_key(pattern: Mapping[str, Any]) -> str:
    return json.dumps(dict(pattern), sort_keys=True, separators=(",", ":"))


def _matches(pattern: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    return all(action.get(key) == expected for key, expected in pattern.items())


@dataclass(frozen=True)
class EntityInstance:
    entity_id: str
    attributes: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    prototype_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _nonempty(self.entity_id, "entity_id"))
        object.__setattr__(self, "attributes", dict(self.attributes))
        object.__setattr__(
            self,
            "evidence_ids",
            _ids(self.evidence_ids, field=f"Entity {self.entity_id} evidence_ids"),
        )
        object.__setattr__(
            self,
            "prototype_ids",
            tuple(dict.fromkeys(_nonempty(value, "prototype_id") for value in self.prototype_ids)),
        )


@dataclass(frozen=True)
class PrototypeDef:
    prototype_id: str
    description: str
    evidence_ids: tuple[str, ...]
    matcher: Callable[[EntityInstance], bool]


@dataclass(frozen=True)
class SchemaRule:
    schema_id: str
    prototype_id: str
    action_pattern: Mapping[str, Any]
    output: str
    evidence_ids: tuple[str, ...]
    counter_evidence_ids: tuple[str, ...]
    handler: Callable[[Any, EntityInstance, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class RuleApplication:
    state: Any
    schema_ids: tuple[str, ...]


class EFPSRegistry:
    """Registry scoped to one imported ``world_model.py`` module."""

    def __init__(self) -> None:
        self._prototypes: dict[str, PrototypeDef] = {}
        self._schemas: dict[str, SchemaRule] = {}

    @property
    def prototypes(self) -> tuple[PrototypeDef, ...]:
        return tuple(self._prototypes[key] for key in sorted(self._prototypes))

    @property
    def schemas(self) -> tuple[SchemaRule, ...]:
        return tuple(self._schemas[key] for key in sorted(self._schemas))

    def add_prototype(
        self,
        prototype_id: str,
        *,
        description: str,
        evidence_ids: Iterable[str],
        matcher: Callable[[EntityInstance], bool],
    ) -> PrototypeDef:
        pid = _nonempty(prototype_id, "prototype_id")
        if pid in self._prototypes:
            raise EFPSError(f"duplicate Prototype ID {pid}")
        if not callable(matcher):
            raise EFPSError(f"Prototype {pid} matcher must be callable")
        item = PrototypeDef(
            prototype_id=pid,
            description=_nonempty(description, "Prototype description"),
            evidence_ids=_ids(evidence_ids, field=f"Prototype {pid} evidence_ids"),
            matcher=matcher,
        )
        self._prototypes[pid] = item
        return item

    def add_schema_rule(
        self,
        schema_id: str,
        *,
        prototype_id: str,
        action: Mapping[str, Any] | str,
        output: str,
        evidence_ids: Iterable[str],
        counter_evidence_ids: Iterable[str],
        handler: Callable[[Any, EntityInstance, Mapping[str, Any]], Any],
    ) -> SchemaRule:
        sid = _nonempty(schema_id, "schema_id")
        pid = _nonempty(prototype_id, "prototype_id")
        if sid in self._schemas:
            raise EFPSError(f"duplicate Schema ID {sid}")
        if pid not in self._prototypes:
            raise EFPSError(f"Schema {sid} cites unknown Prototype {pid}")
        if not callable(handler):
            raise EFPSError(f"Schema {sid} handler must be callable")
        item = SchemaRule(
            schema_id=sid,
            prototype_id=pid,
            action_pattern=_action_pattern(action),
            output=_nonempty(output, "Schema output"),
            evidence_ids=_ids(evidence_ids, field=f"Schema {sid} evidence_ids"),
            counter_evidence_ids=_ids(
                counter_evidence_ids,
                field=f"Schema {sid} counter_evidence_ids",
                required=False,
            ),
            handler=handler,
        )
        self._schemas[sid] = item
        try:
            self.validate()
        except Exception:
            del self._schemas[sid]
            raise
        return item

    def validate(self, known_evidence_ids: Iterable[str] | None = None) -> None:
        claims: dict[tuple[str, str], tuple[str, str]] = {}
        refs: set[str] = set()
        for prototype_def in self.prototypes:
            refs.update(prototype_def.evidence_ids)
        for rule in self.schemas:
            if rule.prototype_id not in self._prototypes:
                raise EFPSError(
                    f"Schema {rule.schema_id} cites unknown Prototype {rule.prototype_id}"
                )
            key = (rule.prototype_id, _action_key(rule.action_pattern))
            prior = claims.get(key)
            if prior is not None:
                relation = "different outputs" if prior[0] != rule.output else "a duplicate rule"
                raise EFPSConflictError(
                    f"Schemas {prior[1]} and {rule.schema_id} claim {relation} "
                    f"for Prototype {rule.prototype_id} and action {dict(rule.action_pattern)}"
                )
            claims[key] = (rule.output, rule.schema_id)
            refs.update(rule.evidence_ids)
            refs.update(rule.counter_evidence_ids)
        if known_evidence_ids is not None:
            missing = sorted(refs - set(known_evidence_ids))
            if missing:
                raise EFPSEvidenceError("unknown Evidence IDs: " + ", ".join(missing))

    def classify(self, entities: Iterable[EntityInstance]) -> tuple[EntityInstance, ...]:
        out = []
        for entity in entities:
            unknown = sorted(set(entity.prototype_ids) - set(self._prototypes))
            if unknown:
                raise EFPSError(
                    f"Entity {entity.entity_id} cites unknown Prototypes: {', '.join(unknown)}"
                )
            memberships = list(entity.prototype_ids)
            for prototype_def in self.prototypes:
                if prototype_def.matcher(entity) and prototype_def.prototype_id not in memberships:
                    memberships.append(prototype_def.prototype_id)
            out.append(replace(entity, prototype_ids=tuple(memberships)))
        return tuple(out)

    def apply_rules(
        self,
        state: Any,
        action: Mapping[str, Any],
        entities: Sequence[EntityInstance],
    ) -> RuleApplication:
        """Execute every matching rule and return state plus exact attribution.

        The world model remains responsible for choosing whether these rules
        constitute all or only part of ``transition``.
        """

        current = state
        applied: list[str] = []
        for rule in self.schemas:
            if not _matches(rule.action_pattern, action):
                continue
            for entity in entities:
                if rule.prototype_id not in entity.prototype_ids:
                    continue
                current = rule.handler(current, entity, action)
                applied.append(rule.schema_id)
        result = RuleApplication(current, tuple(applied))
        record_applied_schema_ids(result.state, result.schema_ids)
        return result

    def export_manifest(
        self,
        *,
        known_evidence_ids: Iterable[str],
        world_model_path: str | Path,
        entities: Iterable[EntityInstance] = (),
        applicability: str = "full",
        applicability_reason: str = "",
    ) -> dict[str, Any]:
        applicability = str(applicability)
        if applicability not in {"full", "partial", "not_applicable"}:
            raise EFPSError("applicability must be full, partial, or not_applicable")
        self.validate(known_evidence_ids)
        classified = self.classify(entities)
        known = set(known_evidence_ids)
        for entity in classified:
            missing = sorted(set(entity.evidence_ids) - known)
            if missing:
                raise EFPSEvidenceError(
                    f"Entity {entity.entity_id} cites unknown Evidence IDs: {', '.join(missing)}"
                )
        model_path = Path(world_model_path)
        if not model_path.is_file():
            raise EFPSError(f"world model does not exist: {model_path}")
        return {
            "schema": EFPS_MANIFEST_SCHEMA,
            "world_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "applicability": applicability,
            "applicability_reason": str(applicability_reason),
            "prototypes": [
                {
                    "prototype_id": item.prototype_id,
                    "description": item.description,
                    "evidence_ids": list(item.evidence_ids),
                    "matcher": _callable_name(item.matcher),
                }
                for item in self.prototypes
            ],
            "schemas": [
                {
                    "schema_id": item.schema_id,
                    "prototype_id": item.prototype_id,
                    "action": dict(item.action_pattern),
                    "output": item.output,
                    "evidence_ids": list(item.evidence_ids),
                    "counter_evidence_ids": list(item.counter_evidence_ids),
                    "handler": _callable_name(item.handler),
                }
                for item in self.schemas
            ],
            "entities": [asdict(entity) for entity in classified],
        }


def _callable_name(value: Callable[..., Any]) -> str:
    module = inspect.getmodule(value)
    prefix = module.__name__ + "." if module is not None else ""
    return prefix + getattr(value, "__qualname__", getattr(value, "__name__", "<callable>"))


REGISTRY = EFPSRegistry()


def prototype(
    prototype_id: str,
    *,
    description: str,
    evidence_ids: Iterable[str],
    registry: EFPSRegistry = REGISTRY,
) -> Callable[[Callable[[EntityInstance], bool]], Callable[[EntityInstance], bool]]:
    def decorate(matcher: Callable[[EntityInstance], bool]) -> Callable[[EntityInstance], bool]:
        registry.add_prototype(
            prototype_id,
            description=description,
            evidence_ids=evidence_ids,
            matcher=matcher,
        )
        return matcher
    return decorate


def schema_rule(
    schema_id: str,
    *,
    prototype_id: str,
    action: Mapping[str, Any] | str,
    output: str,
    evidence_ids: Iterable[str],
    counter_evidence_ids: Iterable[str] = (),
    registry: EFPSRegistry = REGISTRY,
) -> Callable[[Callable[[Any, EntityInstance, Mapping[str, Any]], Any]], Callable[..., Any]]:
    def decorate(handler: Callable[[Any, EntityInstance, Mapping[str, Any]], Any]) -> Callable[..., Any]:
        registry.add_schema_rule(
            schema_id,
            prototype_id=prototype_id,
            action=action,
            output=output,
            evidence_ids=evidence_ids,
            counter_evidence_ids=counter_evidence_ids,
            handler=handler,
        )
        return handler
    return decorate


def record_applied_schema_ids(state: Any, schema_ids: Iterable[str]) -> None:
    values = tuple(dict.fromkeys(str(value) for value in schema_ids))
    try:
        setattr(state, _APPLIED_ATTR, values)
    except (AttributeError, TypeError):
        try:
            object.__setattr__(state, _APPLIED_ATTR, values)
        except (AttributeError, TypeError):
            pass


def applied_schema_ids(state: Any) -> tuple[str, ...]:
    return tuple(getattr(state, _APPLIED_ATTR, ()))


def export_efps(**kwargs: Any) -> dict[str, Any]:
    return REGISTRY.export_manifest(**kwargs)
