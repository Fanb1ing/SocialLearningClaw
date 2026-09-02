from __future__ import annotations

from types import SimpleNamespace

from tycho.agent.agent import TychoAgent


def _agent(*, game_cap: float = 0, level_cap: float = 0) -> TychoAgent:
    agent = TychoAgent.__new__(TychoAgent)
    agent.max_inference_cost_per_game = game_cap
    agent.max_inference_cost_per_level = level_cap
    agent.cfg = SimpleNamespace(model="claude-opus-4-8")
    agent._inference_price = (5.0, 25.0, 0.5, 6.25)
    agent.token_stats = {
        "tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cache_write": 0,
        "latency_ms": 0,
    }
    agent._budget_level = 0
    agent._budget_level_start_usd = 0.0
    agent._done_reason = None
    agent.calls = 0
    agent.max_calls = 10_000
    return agent


def test_game_cost_budget_stops_at_next_action_boundary() -> None:
    agent = _agent(game_cap=750)
    agent.token_stats["tokens_out"] = 30_000_000  # 30M * $25/M = $750

    assert agent.is_done([], SimpleNamespace(levels_completed=0))
    assert agent.done_reason == "inference_cost_game_limit"


def test_level_completion_resets_level_cost_before_stop_check() -> None:
    agent = _agent(level_cap=500)
    agent.token_stats["tokens_out"] = 24_000_000  # $600 spent while solving level 0

    assert not agent.is_done([], SimpleNamespace(levels_completed=1))
    assert agent.inference_budget_state()["level_cost_usd"] == 0

    agent.token_stats["tokens_out"] += 20_000_000  # another $500 on level 1
    assert agent.is_done([], SimpleNamespace(levels_completed=1))
    assert agent.done_reason == "inference_cost_level_limit"


def test_builder_usage_merges_into_actor_budget_meter() -> None:
    agent = _agent()
    agent.builder_invocations = 0
    agent.level = 0
    agent._current_available_actions = ["ACTION1"]
    agent._turn_builder_runs = []

    class Builder:
        calls = 2

        def build(self, *args, **kwargs):
            return "confidence: high", []

        def take_token_stats(self):
            return {
                "tokens_in": 100, "tokens_out": 200, "cache_read": 300,
                "cache_write": 400, "latency_ms": 500,
            }

    agent.builder = Builder()
    agent._invoke_builder("test")

    assert agent.calls == 2
    assert agent.token_stats["tokens_out"] == 200
    assert agent.token_stats["cache_write"] == 400
    assert agent.builder.calls == 0
