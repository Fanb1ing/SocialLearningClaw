"""Provider-neutral LLM client for Tycho.

Tycho supports Anthropic Messages, OpenAI Responses, and OpenAI-compatible Chat Completions.
Additional transports can be loaded through ``TYCHO_LLM_PLUGIN`` without changing agent policy.
"""

from __future__ import annotations

import importlib
import json
import os
import random
import threading
import time
import urllib.error
from dataclasses import MISSING, asdict, dataclass, field

from .public_backends import (
    anthropic_tools,
    normalize_usage,
    openai_chat_tools,
    openai_responses_tools,
    rolling_cachepoint_index,
    strict_tool_specs,
)


PUBLIC_BACKENDS = {"anthropic", "openai", "openai_responses"}


@dataclass
class LLMConfig:
    model: str
    backend: str = "openai"
    base_url: str = ""
    api_key: str = "EMPTY"
    region: str = ""
    reasoning_effort: str = ""
    provider_options: dict = field(default_factory=dict)
    api_protocol: str = ""
    public_model: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        backend = os.environ.get("LLM_BACKEND", "openai").lower()
        model = os.environ.get("LLM_MODEL", "")
        effort = os.environ.get("LLM_EFFORT", "")
        if not model:
            raise SystemExit("set LLM_MODEL")
        extension = _extension()
        extension_handles = False
        if extension is not None:
            if hasattr(extension, "handles_backend"):
                extension_handles = bool(extension.handles_backend(backend))
            else:
                extension_handles = backend in getattr(extension, "OVERRIDE_BACKENDS", set())
        if extension_handles:
            return _public_config(extension.LLMConfig.from_env(), cls)
        if backend == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY", "")
            if not key:
                raise SystemExit("anthropic backend needs ANTHROPIC_API_KEY (or LLM_API_KEY)")
            base = (
                os.environ.get("ANTHROPIC_BASE_URL")
                or os.environ.get("LLM_BASE_URL")
                or "https://api.anthropic.com/v1"
            )
            return cls(
                model,
                backend,
                base.rstrip("/"),
                key,
                reasoning_effort=effort,
                api_protocol="anthropic",
                public_model=model,
            )
        if backend == "openai_responses":
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
            if not key:
                raise SystemExit("openai_responses backend needs OPENAI_API_KEY (or LLM_API_KEY)")
            base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") \
                or "https://api.openai.com/v1"
            return cls(
                model,
                backend,
                base.rstrip("/"),
                key,
                reasoning_effort=effort,
                api_protocol="openai_responses",
                public_model=model,
            )
        if backend == "openai":
            base = os.environ.get("LLM_BASE_URL", "")
            if not base:
                raise SystemExit("openai backend needs LLM_BASE_URL")
            return cls(
                model, backend, base.rstrip("/"), os.environ.get("LLM_API_KEY", "EMPTY"),
                reasoning_effort=effort,
                api_protocol="openai",
                public_model=model,
            )
        if extension is None:
            raise SystemExit(
                f"unknown LLM_BACKEND={backend!r}; use anthropic, openai_responses, openai, "
                "or configure TYCHO_LLM_PLUGIN"
            )
        external = extension.LLMConfig.from_env()
        return _public_config(external, cls)


def _public_config(external, config_class=LLMConfig) -> LLMConfig:
    """Project a transport extension's config onto the stable core facade."""
    cls = config_class
    known = {
        name: getattr(external, name)
        for name in cls.__dataclass_fields__
        if name != "provider_options" and hasattr(external, name)
    }
    known["provider_options"] = {
        name: getattr(external, name)
        for name in getattr(external, "__dataclass_fields__", {})
        if name not in cls.__dataclass_fields__
    }
    return cls(**known)


def public_identity(cfg: LLMConfig | None = None) -> dict[str, str]:
    resolved = cfg or LLMConfig.from_env()
    protocol = resolved.api_protocol or (
        resolved.backend if resolved.backend in PUBLIC_BACKENDS else "external"
    )
    return {
        "api_protocol": protocol,
        "model": resolved.public_model or resolved.model,
    }


_REC = threading.local()


def _extension():
    module_name = os.environ.get("TYCHO_LLM_PLUGIN", "")
    if module_name:
        return importlib.import_module(module_name)
    # Source deployments may provide an in-package extension instead of an environment plugin.
    try:
        return importlib.import_module("tycho.serving._provider_extension")
    except ModuleNotFoundError:
        return None


def _external_config(cfg: LLMConfig, extension):
    fields = extension.LLMConfig.__dataclass_fields__
    if hasattr(cfg, "__dataclass_fields__"):
        values = asdict(cfg)
    else:
        values = {name: getattr(cfg, name) for name in fields if hasattr(cfg, name)}
    options = values.pop("provider_options", {}) or {}
    required = {}
    for name, definition in fields.items():
        if name in values:
            required[name] = values[name]
        elif name in options:
            required[name] = options[name]
        elif definition.default is not MISSING:
            required[name] = definition.default
        elif definition.default_factory is not MISSING:
            required[name] = definition.default_factory()
        else:
            required[name] = ""
    return extension.LLMConfig(**required)


def start_recording() -> None:
    _REC.buf = []
    _REC.system = ""
    extension = _extension()
    if extension is not None:
        extension.start_recording()


def set_system_prompt(text: str) -> None:
    _REC.system = text or ""
    extension = _extension()
    if extension is not None:
        extension.set_system_prompt(text)


def take_recording() -> list:
    out = list(getattr(_REC, "buf", None) or [])
    system = getattr(_REC, "system", "")
    if out and system and not out[0].get("system_prompt"):
        out[0] = {**out[0], "system_prompt": system}
    if hasattr(_REC, "buf") and _REC.buf is not None:
        _REC.buf.clear()
    extension = _extension()
    if extension is not None:
        out.extend(extension.take_recording())
    return out


def stop_recording() -> None:
    _REC.buf = None
    extension = _extension()
    if extension is not None:
        extension.stop_recording()


_CONTEXT_MARKERS = (
    "context length", "context window", "maximum context", "max context", "input length",
    "too many tokens", "token limit", "prompt is too long", "prompt too long",
    "exceeds maximum", "exceeded maximum", "max_model_len", "request too large",
)


def is_context_limit_error(error: Exception) -> bool:
    body = getattr(error, "_tycho_body", "") or ""
    text = f"{type(error).__name__}: {error} {body}".lower()
    return any(marker in text for marker in _CONTEXT_MARKERS)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 409, 429) or 500 <= error.code < 600
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    extension = _extension()
    if extension is not None and hasattr(extension, "_is_retryable"):
        return bool(extension._is_retryable(error))
    return False


def _with_retry(fn, *, base: float = 4.0, cap: float = 30.0, is_retryable=None):
    retryable = is_retryable or _is_retryable
    budget = float(os.environ.get("LLM_RETRY_BUDGET_S", "1200"))
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as error:  # noqa: BLE001 - provider-neutral retry boundary
            elapsed = time.monotonic() - started
            if not retryable(error) or elapsed >= budget:
                raise
            delay = min(cap, base * (2 ** min(attempt, 5))) * random.uniform(0.75, 1.25)
            if elapsed + delay > budget:
                raise
            time.sleep(delay)
            attempt += 1


def _latest_input_digest(history: list[dict]) -> str:
    if not history:
        return ""
    message = history[-1]
    if message.get("role") == "tool":
        return "\n".join(
            f"[{result.get('name', 'tool')} result]\n{result.get('output', '')}"
            for result in message.get("results", [])
        )
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "[image]") for part in content)


def _latest_has_image(history: list[dict]) -> bool:
    if not history:
        return False
    message = history[-1]
    if message.get("role") == "tool":
        return any(result.get("image_png") is not None for result in message.get("results", []))
    content = message.get("content")
    return isinstance(content, list) and any("image_png" in part for part in content)


def _calls_digest(calls: list[dict]) -> str:
    if not calls:
        return ""
    return "\n" + "\n".join(
        f"[tool] {call['name']}({json.dumps(call.get('input', {}), sort_keys=True)})" for call in calls
    )


def reasoning_kind_for(backend: str) -> str:
    if backend == "openai":
        return "trace"
    if backend == "anthropic":
        return "summary"
    if backend == "openai_responses":
        return "none"
    extension = _extension()
    if extension is not None:
        return extension.reasoning_kind_for(backend)
    return "none"


def response_id_continuity_enabled(cfg: LLMConfig) -> bool:
    """Whether this transport should retain opaque model state through response IDs."""
    if cfg.backend == "openai_responses":
        return os.environ.get("OPENAI_REASONING_CONTINUITY", "") in ("1", "on", "true")
    extension = _extension()
    if extension is not None and hasattr(extension, "response_id_continuity_enabled"):
        return bool(extension.response_id_continuity_enabled(_external_config(cfg, extension)))
    return False


def _record_call(history, tools, cfg, system, effort, call_type, reply, latency_ms) -> None:
    buf = getattr(_REC, "buf", None)
    if buf is None:
        return
    usage = reply.get("usage") or {}
    recorded_model = cfg.public_model or cfg.model
    entry = {
        "call_type": call_type,
        "model": recorded_model,
        "effort": effort or "",
        "prompt": _latest_input_digest(history),
        "system_prompt": system or "",
        "reasoning_text": reply.get("reasoning", ""),
        "reasoning_kind": reasoning_kind_for(cfg.backend),
        "reasoning_summary": reply.get("reasoning", ""),
        "response": (reply.get("text", "") or "") + _calls_digest(reply.get("tool_calls", [])),
        "has_image": _latest_has_image(history),
        "latency_ms": latency_ms,
        "tokens_in": usage.get("in", 0),
        "tokens_out": usage.get("out", 0),
        "cache_read": usage.get("cache_read", 0),
        "cache_write": usage.get("cache_write", 0),
    }
    if not any(item.get("tools") and item.get("call_type") == call_type for item in buf):
        entry["tools"] = [
            {"name": tool["name"], "description": tool.get("description", ""), "schema": tool.get("schema")}
            for tool in tools
        ]
    buf.append(entry)


def _note_status(start: float, call_type: str, cfg: LLMConfig, reply: dict | None, error=None) -> None:
    try:
        from tycho.harness.run_status import process_status_store
        from tycho.serving.pricing import reply_usage, usage_cost_usd

        store = process_status_store()
        if store is None:
            return
        if error is not None:
            store.note_llm_failed(call_type=call_type or "llm", error=type(error).__name__)
        elif reply is not None:
            usage = reply_usage(reply)
            store.note_llm_finished(
                call_type=call_type or "llm",
                response=(reply.get("text", "") or "") + _calls_digest(reply.get("tool_calls", [])),
                usage=usage,
                model=cfg.public_model or cfg.model,
                cost_usd=usage_cost_usd(usage, cfg.public_model or cfg.model),
            )
    except Exception:  # noqa: BLE001 - telemetry must never affect inference
        return


def chat_tools(history: list[dict], tools: list[dict], cfg: LLMConfig | None = None, *,
               system: str = "", max_tokens: int = 4096, timeout: int = 600,
               effort: str | None = None, call_type: str = "", prev_response_id: str | None = None,
               since: int = 0, cache_anchor_index: int | None = None) -> dict:
    cfg = cfg or LLMConfig.from_env()
    effort = effort or cfg.reasoning_effort
    timeout = max(timeout, int(os.environ.get("LLM_HTTP_TIMEOUT", "0") or 0))
    started = time.monotonic()
    try:
        from tycho.harness.run_status import process_status_store
        store = process_status_store()
        if store is not None:
            store.note_llm_started(call_type=call_type or "llm")
    except Exception:  # noqa: BLE001
        pass
    try:
        extension = _extension()
        override = extension is not None and cfg.backend in getattr(
            extension, "OVERRIDE_BACKENDS", set()
        )
        if override:
            return extension.chat_tools(
                history, tools, _external_config(cfg, extension), system=system,
                max_tokens=max_tokens, timeout=timeout, effort=effort, call_type=call_type,
                prev_response_id=prev_response_id, since=since,
                cache_anchor_index=cache_anchor_index,
            )
        if cfg.backend == "anthropic":
            reply = anthropic_tools(
                history, tools, cfg, system, max_tokens, timeout, effort, _with_retry,
                cache_anchor_index,
            )
        elif cfg.backend == "openai_responses":
            reply = openai_responses_tools(
                history, tools, cfg, system, max_tokens, timeout, effort, _with_retry,
                prev_response_id, since,
            )
        elif cfg.backend == "openai":
            reply = openai_chat_tools(
                history, tools, cfg, system, max_tokens, timeout, effort, _with_retry
            )
        else:
            if extension is None:
                raise RuntimeError(f"no provider plugin for backend {cfg.backend!r}")
            return extension.chat_tools(
                history, tools, _external_config(cfg, extension), system=system,
                max_tokens=max_tokens, timeout=timeout, effort=effort, call_type=call_type,
                prev_response_id=prev_response_id, since=since,
                cache_anchor_index=cache_anchor_index,
            )
    except Exception as error:
        _note_status(started, call_type, cfg, None, error)
        raise
    latency_ms = int((time.monotonic() - started) * 1000)
    reply["latency_ms"] = latency_ms
    _record_call(history, tools, cfg, system, effort, call_type, reply, latency_ms)
    _note_status(started, call_type, cfg, reply)
    return reply


def chat(messages: list[dict], cfg: LLMConfig | None = None, *, max_tokens: int = 4096,
         timeout: int = 600, effort: str | None = None, call_type: str = "") -> str:
    history = [message for message in messages if message.get("role") != "system"]
    system = "\n".join(
        str(message.get("content", "")) for message in messages if message.get("role") == "system"
    )
    return chat_tools(
        history, [], cfg, system=system, max_tokens=max_tokens, timeout=timeout,
        effort=effort, call_type=call_type,
    )["text"]


def extract_code(text: str) -> str:
    for fence in ("```python", "```py", "```"):
        if fence in text:
            return text.split(fence, 1)[1].split("```", 1)[0].strip()
    return text.strip()


# Stable helper names used by existing tests and context configuration.
_norm_usage = normalize_usage
_strict_tool_specs = strict_tool_specs
_rolling_cachepoint_index = rolling_cachepoint_index


def __getattr__(name: str):
    """Forward optional attributes supplied by a configured transport extension."""
    extension = _extension()
    if extension is not None and hasattr(extension, name):
        return getattr(extension, name)
    raise AttributeError(name)
