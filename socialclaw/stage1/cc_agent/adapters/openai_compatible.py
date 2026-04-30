from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from ..base import CCAgentResult, ToolCallStats, Usage
from .safe_tools import ToolError, ToolRegistry


_CONF_RE = re.compile(r"confidence\s*[:：]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_confidence(text: str) -> Optional[float]:
    m = _CONF_RE.search(text or "")
    if not m:
        return None
    try:
        v = float(m.group(1))
    except Exception:
        return None
    if v < 0:
        v = 0.0
    if v > 1:
        v = 1.0
    return v


def _try_load_image_data_url(*, meta: Dict[str, Any]) -> Optional[str]:
    """Load image referenced by problem_meta.image as data URL.

    Expected meta from pipeline:
      - meta["prepared_dataset_path"]: path to prepared all.jsonl
      - meta["problem_meta"]["image"]: relative path like "images/av_000.jpg"
    """

    problem_meta = meta.get("problem_meta") or {}
    if not isinstance(problem_meta, dict):
        return None
    image_rel = problem_meta.get("image")
    if not isinstance(image_rel, str) or not image_rel:
        return None

    prepared_dataset_path = meta.get("prepared_dataset_path")
    if not isinstance(prepared_dataset_path, str) or not prepared_dataset_path:
        return None

    prepared_dir = os.path.dirname(prepared_dataset_path)
    image_abs = os.path.join(prepared_dir, image_rel)
    if not os.path.exists(image_abs):
        return None

    try:
        with open(image_abs, "rb") as f:
            b = f.read()
    except Exception:
        return None

    if not b:
        return None

    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class OpenAICompatibleAgent:
    base_url: str
    api_key: str
    model: str
    timeout_s: int = 120
    enable_safe_tools: bool = True
    max_tool_rounds: int = 4

    def answer(self, *, prompt: str, meta: Dict[str, Any]) -> CCAgentResult:
        """Call an OpenAI-compatible /chat/completions endpoint.

        Designed for:
        - OpenRouter
        - SiliconFlow (硅基流动) OpenAI-compatible endpoint
        - Any OpenAI-compatible gateway

        Stage1 safety: only enable safe tools (calculator/noop) and never run shell.
        """

        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        registry = ToolRegistry.safe_default()

        # We keep a message list; prompt_builder already includes system-like content.
        image_data_url = _try_load_image_data_url(meta=meta)
        if image_data_url:
            # OpenAI-compatible multimodal content format
            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        all_tool_names: List[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        tools_payload: Optional[List[Dict[str, Any]]] = None
        if self.enable_safe_tools:
            tools_payload = [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Evaluate a basic math expression.",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "No-op tool for testing tool calling.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ]

        with httpx.Client(timeout=self.timeout_s) as client:
            for _round in range(1 + (self.max_tool_rounds if self.enable_safe_tools else 0)):
                payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2}
                if tools_payload is not None:
                    payload["tools"] = tools_payload
                    payload["tool_choice"] = "auto"

                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                try:
                    data = r.json()
                except Exception as e:
                    body = (r.text or "")
                    snippet = body[:2000]
                    raise RuntimeError(
                        "Failed to parse JSON from OpenAI-compatible response. "
                        f"status={r.status_code}, content-type={r.headers.get('content-type')}, "
                        f"body_snippet=\n{snippet}"
                    ) from e

                usage = data.get("usage") or {}
                total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                total_completion_tokens += int(usage.get("completion_tokens") or 0)

                choice0 = (data.get("choices") or [{}])[0]
                msg = choice0.get("message") or {}

                # If model returned tool calls, execute and append tool results.
                tc = msg.get("tool_calls")
                if self.enable_safe_tools and isinstance(tc, list) and tc:
                    messages.append(msg)  # assistant message that requests tool(s)

                    for call in tc:
                        fn = (call.get("function") or {})
                        name = fn.get("name")
                        arguments = fn.get("arguments")
                        call_id = call.get("id")

                        if name:
                            all_tool_names.append(str(name))

                        try:
                            result = registry.call(str(name), arguments)
                            tool_content = result
                        except ToolError as e:
                            tool_content = {"error": str(e)}
                        except Exception as e:
                            tool_content = {"error": f"tool execution failed: {e}"}

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_content,
                        }
                        messages.append(tool_msg)

                    # continue loop to let model consume tool results
                    continue

                # Otherwise, treat as final answer.
                content = msg.get("content") or ""
                conf = _parse_confidence(content)

                total_tokens = int(data.get("usage", {}).get("total_tokens") or (total_prompt_tokens + total_completion_tokens))
                return CCAgentResult(
                    answer_text=content,
                    confidence=conf,
                    usage=Usage(
                        input_tokens=total_prompt_tokens,
                        output_tokens=total_completion_tokens,
                        total_tokens=total_tokens,
                    ),
                    tool_calls=ToolCallStats(count=len(all_tool_names), tools=all_tool_names),
                    raw={"response": data, "meta": meta, "messages": messages},
                )

        # Should not reach here
        return CCAgentResult(
            answer_text="",
            confidence=None,
            usage=Usage(input_tokens=total_prompt_tokens, output_tokens=total_completion_tokens, total_tokens=total_prompt_tokens + total_completion_tokens),
            tool_calls=ToolCallStats(count=len(all_tool_names), tools=all_tool_names),
            raw={"meta": meta, "messages": messages},
        )
