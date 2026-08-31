from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ...trajectory import Action


def bounded_probability(value: Any, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def legal_action(
    payload: Any,
    action_contracts: List[Dict[str, Any]],
) -> Action:
    if not isinstance(payload, dict):
        raise ValueError("selected action must be an object")
    name = str(payload.get("name") or "")
    contract = next(
        (item for item in action_contracts if item.get("name") == name), None
    )
    if contract is None:
        raise ValueError(f"Action {name!r} is not currently available")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("action arguments must be an object")
    schema = contract.get("arguments_schema") or {}
    properties = schema.get("properties") or {}
    allowed = set(properties)
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"Unexpected action arguments: {sorted(unknown)}")
    normalized: Dict[str, Any] = {}
    for key, value in arguments.items():
        spec = properties.get(key) or {}
        if spec.get("type") == "integer":
            if isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
            try:
                value = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} must be an integer") from error
            if "minimum" in spec and value < int(spec["minimum"]):
                raise ValueError(f"{key} is below its public minimum")
            if "maximum" in spec and value > int(spec["maximum"]):
                raise ValueError(f"{key} is above its public maximum")
        normalized[str(key)] = value
    return Action(name=name, arguments=normalized)


def known_ids(values: Iterable[Any], allowed: Iterable[str]) -> List[str]:
    valid = set(allowed)
    return sorted({str(item) for item in values if str(item) in valid})


__all__ = ["bounded_probability", "known_ids", "legal_action"]
