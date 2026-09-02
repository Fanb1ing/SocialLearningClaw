"""Public model protocols and transports used by Tycho.

The harness speaks a small provider-neutral tool protocol.  This module translates it to
Anthropic Messages, OpenAI Responses, or an OpenAI-compatible Chat Completions endpoint.  It is
deliberately transport-only: orchestration, history retention, and action policy remain in the
agent.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import urllib.error
import urllib.request
from typing import Callable


Retry = Callable[[Callable[[], dict]], dict]


def strict_tool_specs(tools: list[dict]) -> list[dict]:
    """Return Responses-compatible schemas without changing optional-field semantics."""

    def strict_schema(schema: dict) -> dict:
        if not isinstance(schema, dict):
            return schema
        out = copy.deepcopy(schema)
        if out.get("type") != "object":
            return out
        properties = out.get("properties") or {}
        originally_required = set(out.get("required") or [])
        for name, prop in list(properties.items()):
            prop = strict_schema(prop)
            if name not in originally_required and isinstance(prop, dict) and "type" in prop:
                types = prop["type"] if isinstance(prop["type"], list) else [prop["type"]]
                if "null" not in types:
                    prop["type"] = [*types, "null"]
            properties[name] = prop
        out["properties"] = properties
        out["required"] = list(properties)
        out["additionalProperties"] = False
        return out

    normalized = copy.deepcopy(tools)
    for tool in normalized:
        tool["schema"] = strict_schema(tool.get("schema") or {})
    return normalized


def normalize_usage(usage: dict | None, *, input_tokens_include_cache: bool = False) -> dict:
    """Normalize provider usage into fresh input, output, cache-read, and cache-write tokens."""
    usage = usage or {}
    cache_read = int(
        usage.get("cacheReadInputTokens")
        or usage.get("cache_read_input_tokens")
        or ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
        or 0
    )
    cache_write = int(
        usage.get("cacheWriteInputTokens") or usage.get("cache_creation_input_tokens") or 0
    )
    if "inputTokens" in usage:
        total = int(usage.get("inputTokens") or 0)
        fresh = max(0, total - cache_read - cache_write) if input_tokens_include_cache else total
    elif "input_tokens" in usage:
        fresh = int(usage.get("input_tokens") or 0)
    else:
        fresh = max(0, int(usage.get("prompt_tokens") or 0) - cache_read)
    return {
        "in": fresh,
        "out": int(
            usage.get("outputTokens")
            or usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        ),
        "cache_read": cache_read,
        "cache_write": cache_write,
    }


def _post_json(url: str, body: dict, headers: dict, timeout: int, retry) -> dict:
    data = json.dumps(body).encode()

    def send() -> dict:
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            exc._tycho_body = detail  # type: ignore[attr-defined]
            if detail:
                exc.msg = f"{exc.reason}; provider response: {detail[:2000]}"
            raise

    return retry(send)


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


# Anthropic Messages ---------------------------------------------------------------------------


def _anthropic_part(part: dict) -> dict:
    if "text" in part:
        return {"type": "text", "text": part["text"]}
    if "image_png" in part:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(part["image_png"]).decode(),
            },
        }
    return {"type": "text", "text": ""}


def _anthropic_content(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = [_anthropic_part(part) for part in content]
    return blocks or [{"type": "text", "text": "(elided)"}]


def _anthropic_cache_control() -> dict:
    control = {"type": "ephemeral"}
    ttl = os.environ.get("ANTHROPIC_CACHE_TTL", "")
    if ttl:
        control["ttl"] = ttl
    return control


def _anthropic_cache_on() -> bool:
    from tycho.agent.context_config import prompt_caching_on

    return prompt_caching_on()


def rolling_cachepoint_index(n_messages: int, stable_upto: int | None = None) -> int | None:
    if not _anthropic_cache_on() or n_messages < 2:
        return None
    from tycho.agent.context_config import cache_rolling_every

    every = cache_rolling_every()
    if every <= 0:
        return None
    latest = n_messages - 2 if stable_upto is None else min(stable_upto, n_messages - 2)
    if latest < 0:
        return None
    return latest if every == 1 else max(0, (latest // every) * every)


def _anthropic_history(history: list[dict], cache_anchor_index: int | None) -> list[dict]:
    messages = []
    for message in history:
        if message["role"] == "tool":
            results = []
            for result in message["results"]:
                content = [{"type": "text", "text": str(result.get("output", ""))}]
                if result.get("image_png") is not None:
                    content.append(_anthropic_part({"image_png": result["image_png"]}))
                results.append(
                    {"type": "tool_result", "tool_use_id": result["id"], "content": content}
                )
            messages.append({"role": "user", "content": results})
        elif message.get("tool_calls"):
            content = list(message.get("raw_reasoning") or [])
            if message.get("content"):
                content.append({"type": "text", "text": message["content"]})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call["id"],
                    "name": call["name"],
                    "input": call.get("input", {}),
                }
                for call in message["tool_calls"]
            )
            messages.append({"role": "assistant", "content": content})
        else:
            role = "assistant" if message["role"] == "assistant" else "user"
            content = list(message.get("raw_reasoning") or []) if role == "assistant" else []
            content.extend(_anthropic_content(message.get("content", "")))
            messages.append({"role": role, "content": content})
    index = rolling_cachepoint_index(len(messages), cache_anchor_index)
    if index is not None:
        for message in reversed(messages[: index + 1]):
            for block in reversed(message.get("content") or []):
                if block.get("type") in ("text", "image"):
                    block["cache_control"] = _anthropic_cache_control()
                    return messages
    return messages


def _anthropic_thinking(body: dict, effort: str, max_tokens: int) -> None:
    effort = (os.environ.get("ANTHROPIC_EFFORT") or effort or "").lower()
    if effort and effort not in ("off", "none", "0", "false"):
        body["output_config"] = {"effort": effort}
    mode = os.environ.get("ANTHROPIC_THINKING", "off").lower()
    if mode in ("", "off", "none", "0", "false"):
        return
    if mode == "enabled":
        budget = int(
            os.environ.get(
                "ANTHROPIC_THINKING_BUDGET_TOKENS",
                str(min(max(max_tokens // 2, 1024), 16000)),
            )
        )
        body["max_tokens"] = max(body["max_tokens"], budget + 1024)
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    else:
        body["thinking"] = {"type": "adaptive"}


def anthropic_tools(history, tools, cfg, system, max_tokens, timeout, effort, retry,
                    cache_anchor_index=None) -> dict:
    mapped_tools = [
        {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
        for t in tools
    ]
    if mapped_tools and _anthropic_cache_on():
        mapped_tools[-1]["cache_control"] = _anthropic_cache_control()
    body = {
        "model": cfg.model,
        "messages": _anthropic_history(history, cache_anchor_index),
        "max_tokens": max_tokens,
        "tools": mapped_tools,
    }
    if system:
        body["system"] = (
            [{"type": "text", "text": system, "cache_control": _anthropic_cache_control()}]
            if _anthropic_cache_on()
            else system
        )
    _anthropic_thinking(body, effort, max_tokens)
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        "x-api-key": cfg.api_key,
    }
    beta = os.environ.get("ANTHROPIC_BETA", "")
    if beta:
        headers["anthropic-beta"] = beta
    response = _post_json(
        f"{cfg.base_url}/messages", body, headers, timeout, retry
    )
    text, reasoning, calls, raw_reasoning = "", [], [], []
    for block in response.get("content", []):
        kind = block.get("type")
        if kind == "text":
            text += block.get("text", "")
        elif kind == "tool_use":
            calls.append(
                {"id": block["id"], "name": block["name"], "input": block.get("input", {})}
            )
        elif kind in ("thinking", "redacted_thinking"):
            raw_reasoning.append(block)
            if block.get("thinking"):
                reasoning.append(block["thinking"])
    return {
        "text": text,
        "tool_calls": calls,
        "stop": response.get("stop_reason", ""),
        "reasoning": "\n".join(reasoning),
        "raw_reasoning": raw_reasoning or None,
        "usage": normalize_usage(response.get("usage")),
    }


# OpenAI Responses ------------------------------------------------------------------------------


def _responses_part(part: dict) -> dict:
    if "text" in part:
        return {"type": "input_text", "text": part["text"]}
    if "image_png" in part:
        return {"type": "input_image", "image_url": _data_url(part["image_png"])}
    return {"type": "input_text", "text": ""}


def responses_input(history: list[dict], *, since: int = 0, delta: bool = False) -> list[dict]:
    """Translate Tycho history to Responses items; also used by transport contract tests."""
    source = history[since:] if delta and since else history
    items = []
    for message in source:
        role = message["role"]
        if role == "tool":
            for result in message["results"]:
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": result.get("id", ""),
                        "output": str(result.get("output", "")),
                    }
                )
                if result.get("image_png") is not None:
                    items.append(
                        {"role": "user", "content": [_responses_part({"image_png": result["image_png"]})]}
                    )
        elif message.get("tool_calls"):
            items.extend(message.get("reasoning_items") or [])
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message["tool_calls"]:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": call["name"],
                        "arguments": json.dumps(call.get("input", {})),
                    }
                )
        elif message.get("reasoning_items"):
            items.extend(message["reasoning_items"])
            if message.get("content"):
                content = message["content"]
                items.append(
                    {
                        "role": role,
                        "content": [_responses_part(p) for p in content]
                        if isinstance(content, list)
                        else content,
                    }
                )
        elif isinstance(message.get("content"), list):
            items.append(
                {"role": role, "content": [_responses_part(p) for p in message["content"]]}
            )
        else:
            items.append({"role": role, "content": message.get("content", "")})
    return items


def responses_request(history, tools, cfg, system, max_tokens, effort,
                      prev_response_id=None, since=0) -> dict:
    effort = (effort or "").lower()
    if effort in ("off", "none", "0", "false"):
        effort = ""
    stateless = os.environ.get("OPENAI_STATELESS_REASONING", "") in ("1", "on", "true")
    delta = not stateless and prev_response_id is not None
    budget = max(max_tokens, 16000)
    if effort:
        budget = max(max_tokens, 4000) + {
            "low": 8000,
            "medium": 16000,
            "high": 24000,
            "xhigh": 40000,
            "max": 56000,
        }.get(effort, 24000)
    body = {
        "model": cfg.model,
        "input": responses_input(history, since=since, delta=delta),
        "max_output_tokens": budget,
        "tools": [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["schema"],
                "strict": True,
            }
            for tool in strict_tool_specs(tools)
        ],
        "tool_choice": "auto",
    }
    if not tools:
        body.pop("tools")
        body.pop("tool_choice")
    if system:
        body["instructions"] = system
    if effort:
        body["reasoning"] = {"effort": effort}
    if stateless:
        body["store"] = False
        body["include"] = ["reasoning.encrypted_content"]
        has_prior = any(message.get("reasoning_items") for message in history)
        body.setdefault("reasoning", {})["context"] = "all_turns" if has_prior else "current_turn"
    elif prev_response_id is not None:
        body["store"] = True
        body["previous_response_id"] = prev_response_id
    elif os.environ.get("OPENAI_REASONING_CONTINUITY", "") in ("1", "on", "true"):
        body["store"] = True
    return body


def parse_responses(response: dict, *, keep_reasoning: bool) -> dict:
    text, calls, reasoning_items = "", [], []
    for item in response.get("output", []):
        kind = item.get("type")
        if kind == "function_call":
            calls.append(
                {
                    "id": item.get("call_id", item.get("id", "")),
                    "name": item["name"],
                    "input": json.loads(item.get("arguments") or "{}"),
                }
            )
        elif kind == "message":
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    text += block.get("text", "")
        elif kind == "reasoning" and keep_reasoning:
            reasoning_items.append(item)
    usage = response.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    cache_read = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    total_input = int(usage.get("input_tokens") or 0)
    return {
        "text": text,
        "tool_calls": calls,
        "stop": response.get("status", ""),
        "reasoning": "",
        "raw_reasoning": None,
        "usage": {
            "in": max(0, total_input - cache_read - cache_write),
            "out": int(usage.get("output_tokens") or 0),
            "cache_read": cache_read,
            "cache_write": cache_write,
        },
        "response_id": response.get("id"),
        "reasoning_items": reasoning_items,
    }


def openai_responses_tools(history, tools, cfg, system, max_tokens, timeout, effort, retry,
                           prev_response_id=None, since=0) -> dict:
    body = responses_request(
        history, tools, cfg, system, max_tokens, effort,
        prev_response_id=prev_response_id, since=since,
    )
    response = _post_json(
        f"{cfg.base_url}/responses",
        body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        timeout,
        retry,
    )
    stateless = os.environ.get("OPENAI_STATELESS_REASONING", "") in ("1", "on", "true")
    return parse_responses(response, keep_reasoning=stateless)


# OpenAI-compatible Chat Completions -------------------------------------------------------------


def openai_chat_tools(history, tools, cfg, system, max_tokens, timeout, effort, retry) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for message in history:
        if message["role"] == "tool":
            for result in message["results"]:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["id"],
                        "content": str(result.get("output", "")),
                    }
                )
                if result.get("image_png") is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Current frame:"},
                                {"type": "image_url", "image_url": {"url": _data_url(result["image_png"])}},
                            ],
                        }
                    )
        elif message.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call.get("input", {})),
                            },
                        }
                        for call in message["tool_calls"]
                    ],
                }
            )
        elif isinstance(message.get("content"), list):
            content = []
            for part in message["content"]:
                if "text" in part:
                    content.append({"type": "text", "text": part["text"]})
                elif "image_png" in part:
                    content.append({"type": "image_url", "image_url": {"url": _data_url(part["image_png"])}})
            messages.append({"role": message["role"], "content": content})
        else:
            messages.append({"role": message["role"], "content": message.get("content", "")})
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["schema"],
                },
            }
            for tool in tools
        ],
    }
    if not tools:
        body.pop("tools")
    if effort:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    response = _post_json(
        f"{cfg.base_url}/chat/completions",
        body,
        {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"},
        timeout,
        retry,
    )
    message = response["choices"][0]["message"]
    calls = []
    for index, call in enumerate(message.get("tool_calls") or []):
        try:
            arguments = json.loads(call["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        calls.append(
            {
                "id": call.get("id", f"call_{index}"),
                "name": call["function"]["name"],
                "input": arguments,
            }
        )
    return {
        "text": message.get("content") or "",
        "tool_calls": calls,
        "stop": response["choices"][0].get("finish_reason", ""),
        "reasoning": message.get("reasoning") or "",
        "usage": normalize_usage(response.get("usage")),
    }
