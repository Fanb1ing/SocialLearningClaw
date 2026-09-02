"""Deterministic inference-budget accounting over recorded action traces."""

from __future__ import annotations

from dataclasses import dataclass

from tycho.serving.pricing import PRICE_SCHEDULE, usage_cost_usd


@dataclass(frozen=True)
class TraceBudgetResult:
    trace: list[dict]
    cost_usd: float
    original_cost_usd: float
    cap_triggered: bool
    stop_reason: str | None


def _step_cost_usd(step: dict, default_model: str = "") -> float:
    calls = ((step.get("reasoning") or {}).get("llm_calls") or [])
    total = 0.0
    for call in calls:
        model = str(call.get("model") or default_model)
        value = usage_cost_usd(call, model)
        if value is None:
            raise ValueError(
                f"cannot price trace call for model {model!r} under {PRICE_SCHEDULE}"
            )
        total += value
    return total


def cap_trace_by_inference_cost(
    trace: list[dict],
    *,
    max_cost_per_game_usd: float = 0.0,
    max_cost_per_level_usd: float = 0.0,
    default_model: str = "",
) -> TraceBudgetResult:
    """Apply the runtime's post-action soft budget to a recorded trace.

    The action that reaches a threshold remains committed, then the trace stops. A completing action
    resets the level meter before the per-level threshold is tested, matching ``TychoAgent.is_done``
    on the next harness iteration. The game threshold never resets.
    """

    if max_cost_per_game_usd < 0 or max_cost_per_level_usd < 0:
        raise ValueError("inference cost limits must be >= 0")
    step_costs = [_step_cost_usd(step, default_model) for step in trace]
    original_cost = sum(step_costs)
    kept: list[dict] = []
    game_cost = 0.0
    level_cost = 0.0
    completed = 0
    reason = None
    for step, cost in zip(trace, step_costs):
        kept.append(step)
        game_cost += cost
        level_cost += cost
        next_completed = int(step.get("levels_completed") or 0)
        level_advanced = next_completed > completed
        if level_advanced:
            completed = next_completed
            level_cost = 0.0
        if max_cost_per_game_usd > 0 and game_cost >= max_cost_per_game_usd:
            reason = "inference_cost_game_limit"
            break
        if not level_advanced and max_cost_per_level_usd > 0 \
                and level_cost >= max_cost_per_level_usd:
            reason = "inference_cost_level_limit"
            break
    return TraceBudgetResult(
        trace=kept,
        cost_usd=game_cost,
        original_cost_usd=original_cost,
        cap_triggered=len(kept) < len(trace),
        stop_reason=reason if len(kept) < len(trace) else None,
    )
