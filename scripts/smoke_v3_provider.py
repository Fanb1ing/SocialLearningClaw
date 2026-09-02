#!/usr/bin/env python3
"""One-call paid smoke test for the V3 OpenAI-compatible transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from socialclaw.utils import load_dotenv
from tycho.serving.llm_client import LLMConfig, chat_tools


def _image() -> bytes:
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((64, 64, 191, 191), fill=(30, 90, 220))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _configure_openrouter(model: str) -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key and not os.environ.get("LLM_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY (or LLM_API_KEY) is not set")
    os.environ["LLM_MODEL"] = model
    os.environ.setdefault("LLM_BACKEND", "openai")
    os.environ.setdefault("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    os.environ.setdefault("LLM_API_KEY", key)
    os.environ.setdefault("LLM_RETRY_BUDGET_S", "30")
    os.environ["TYCHO_PROMPT_CACHING"] = "0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confirm-paid-call", action="store_true")
    args = parser.parse_args()
    if not args.confirm_paid_call:
        raise SystemExit("refusing paid request without --confirm-paid-call")

    root = Path(__file__).resolve().parents[1]
    load_dotenv(str(root / ".env"))
    _configure_openrouter(args.model)
    cfg = LLMConfig.from_env()
    if cfg.backend != "openai":
        raise SystemExit(f"V3 OpenRouter smoke requires LLM_BACKEND=openai, got {cfg.backend!r}")

    tool = {
        "name": "report_transport_ok",
        "description": "Report that text, image, and tool calling were received.",
        "schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["ok"]}},
            "required": ["status"],
        },
    }
    reply = chat_tools(
        [{
            "role": "user",
            "content": [
                {"text": "Inspect the attached blue-square test image, then call report_transport_ok."},
                {"image_png": _image()},
            ],
        }],
        [tool],
        cfg,
        system="This is a transport test. Call the provided tool exactly once.",
        max_tokens=256,
        timeout=int(os.environ.get("LLM_HTTP_TIMEOUT", "180")),
        effort="off",
        call_type="provider_smoke",
    )
    calls = reply.get("tool_calls") or []
    ok = any(
        call.get("name") == "report_transport_ok"
        and call.get("input", {}).get("status") == "ok"
        for call in calls
    )
    receipt = {
        "schema": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": cfg.backend,
        "model": cfg.public_model or cfg.model,
        "expected_tool_call": ok,
        "latency_ms": reply.get("latency_ms"),
        "usage": reply.get("usage"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
