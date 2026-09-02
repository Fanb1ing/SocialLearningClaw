"""Public list-price accounting for inference budgets and reporting.

The runtime budget and post-run analysis must price the same token buckets identically. Prices are
versioned constants rather than provider billing lookups so a recorded budget remains reproducible
if a vendor later changes its rate card.
"""

from __future__ import annotations

from typing import Mapping


USAGE_KEYS = ("tokens_in", "tokens_out", "cache_read", "cache_write")

# (fresh input, output, cache read, five-minute cache write), USD per million tokens.
PRICE_PER_1M = {
    "fable-5": (10.0, 50.0, 1.0, 12.50),
    "opus-5": (5.0, 25.0, 0.50, 6.25),
    "opus-4-8": (5.0, 25.0, 0.50, 6.25),
    "opus-4-7": (5.0, 25.0, 0.50, 6.25),
    # OpenAI GPT-5.6 short-context public list (in, out, cached-in, cache-write), USD/1M.
    "gpt-5.6-sol": (5.0, 30.0, 0.50, 6.25),
    "gpt-5.6-terra": (2.5, 15.0, 0.25, 3.125),
    "gpt-5.6-luna": (1.0, 6.0, 0.10, 1.25),
    "gpt-5.6": (5.0, 30.0, 0.50, 6.25),   # fallback for an un-suffixed gpt-5.6 id (= sol)
    "gpt-5.5": (5.0, 30.0, 0.50, 5.0),
    "gpt-5.4": (2.5, 15.0, 0.25, 2.5),
}
PRICE_SCHEDULE = "public-list-2026-07-13"


def price_for(model: str) -> tuple[float, float, float, float] | None:
    """Return the frozen public rate vector for a model id.

    Open-weight models return a zero vector because no portable per-token API price describes
    self-hosted execution. Unknown proprietary models return ``None`` so a requested dollar budget
    fails loudly instead of silently becoming unlimited.
    """

    name = (model or "").lower()
    if "gemma" in name or "qwen" in name:
        return (0.0, 0.0, 0.0, 0.0)
    if "fable-5" in name:
        return PRICE_PER_1M["fable-5"]
    if "opus-5" in name or "opus-5.0" in name:
        return PRICE_PER_1M["opus-5"]
    if "opus-4-8" in name or "opus-4.8" in name:
        return PRICE_PER_1M["opus-4-8"]
    if "opus-4-7" in name or "opus-4.7" in name:
        return PRICE_PER_1M["opus-4-7"]
    if "gpt-5.6-sol" in name:
        return PRICE_PER_1M["gpt-5.6-sol"]
    if "gpt-5.6-terra" in name:
        return PRICE_PER_1M["gpt-5.6-terra"]
    if "gpt-5.6-luna" in name:
        return PRICE_PER_1M["gpt-5.6-luna"]
    if "gpt-5.6" in name:
        return PRICE_PER_1M["gpt-5.6"]
    if "gpt-5.5" in name:
        return PRICE_PER_1M["gpt-5.5"]
    if "gpt-5.4" in name:
        return PRICE_PER_1M["gpt-5.4"]
    return None


def usage_cost_usd(
    usage: Mapping[str, int | float],
    model: str | None = None,
    *,
    price: tuple[float, float, float, float] | None = None,
) -> float | None:
    """Price normalized usage buckets with either ``model`` or an explicit rate vector."""

    rates = price if price is not None else price_for(model or "")
    if rates is None:
        return None
    fresh, output, cache_read, cache_write = rates
    return (
        float(usage.get("tokens_in") or 0) / 1_000_000 * fresh
        + float(usage.get("tokens_out") or 0) / 1_000_000 * output
        + float(usage.get("cache_read") or 0) / 1_000_000 * cache_read
        + float(usage.get("cache_write") or 0) / 1_000_000 * cache_write
    )


def reply_usage(reply: Mapping) -> dict[str, int]:
    """Normalize one ``chat_tools`` reply into the cumulative meter's bucket names."""

    raw = reply.get("usage") or {}
    return {
        "tokens_in": int(raw.get("in") or 0),
        "tokens_out": int(raw.get("out") or 0),
        "cache_read": int(raw.get("cache_read") or 0),
        "cache_write": int(raw.get("cache_write") or 0),
        "latency_ms": int(reply.get("latency_ms") or 0),
    }
