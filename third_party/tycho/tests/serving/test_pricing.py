from __future__ import annotations

from tycho.serving.pricing import price_for, reply_usage, usage_cost_usd


def test_opus_cache_aware_cost() -> None:
    usage = {
        "tokens_in": 1_000_000,
        "tokens_out": 1_000_000,
        "cache_read": 1_000_000,
        "cache_write": 1_000_000,
    }
    assert usage_cost_usd(usage, "claude-opus-4-8") == 36.75
    assert usage_cost_usd(usage, "claude-opus-5") == 36.75


def test_fable_cache_aware_cost() -> None:
    usage = {
        "tokens_in": 1_000_000,
        "tokens_out": 1_000_000,
        "cache_read": 1_000_000,
        "cache_write": 1_000_000,
    }
    expected = (10.0, 50.0, 1.0, 12.5)
    assert price_for("claude-fable-5") == expected
    assert usage_cost_usd(usage, "claude-fable-5") == 73.5


def test_unknown_proprietary_model_has_no_silent_price() -> None:
    assert price_for("future-proprietary-model") is None
    assert usage_cost_usd({}, "future-proprietary-model") is None


def test_reply_usage_normalizes_client_keys() -> None:
    assert reply_usage({
        "usage": {"in": 1, "out": 2, "cache_read": 3, "cache_write": 4},
        "latency_ms": 5,
    }) == {
        "tokens_in": 1, "tokens_out": 2, "cache_read": 3, "cache_write": 4,
        "latency_ms": 5,
    }
