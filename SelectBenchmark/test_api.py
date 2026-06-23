#!/usr/bin/env python3
"""
API connectivity test: verify each model is actually being called via OpenRouter.
Shows model name, response, token usage, and cost info from the API response.

Usage:
    cd /data5/fanbingbing/SocialLearningClaw
    .venv/bin/python SelectBenchmark/test_api.py
"""

import os, json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODELS = {
    "claude-opus-4.8":  "anthropic/claude-opus-4.8",
    "gemini-2.5-pro":   "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gpt-4o":           "openai/gpt-4o",
}

# Simple unambiguous question with a known answer
QUESTION = (
    "What is 17 × 23? "
    "Reply in exactly this format: 'The answer is X' where X is the number. "
    "Do not show any working."
)
EXPECTED = "391"

print(f"Test question: {QUESTION}")
print(f"Expected answer: {EXPECTED}\n")
print("=" * 70)

for model_name, model_id in MODELS.items():
    print(f"\nModel: {model_name}  ({model_id})")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": QUESTION}],
            max_tokens=64,
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        finish  = resp.choices[0].finish_reason
        usage   = resp.usage

        # OpenRouter returns model actually used in response
        actual_model = getattr(resp, "model", "unknown")

        print(f"  OpenRouter model field : {actual_model}")
        print(f"  Finish reason          : {finish}")
        print(f"  Tokens (prompt/compl)  : {usage.prompt_tokens}/{usage.completion_tokens}" if usage else "  Tokens: N/A")
        print(f"  Response               : {content.strip()!r}")
        print(f"  Contains {EXPECTED}?    : {'YES ✓' if EXPECTED in content else 'NO ✗'}")

        # OpenRouter extra: check if there's a model info header
        extra = {}
        if hasattr(resp, "_hidden_params"):
            extra = resp._hidden_params
        if extra:
            print(f"  Extra                  : {json.dumps(extra)[:200]}")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("Done. If 'Contains 391?' is YES ✓ for all models, the API calls are working correctly.")
