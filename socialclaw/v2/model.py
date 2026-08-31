from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

import httpx


@dataclass(frozen=True)
class ModelImage:
    """One image sent to a model plus its durable, non-secret audit identity."""

    label: str
    artifact_id: str
    sha256: str
    relative_path: str
    data_url: str

    def audit_dict(self) -> Dict[str, str]:
        return {
            "label": self.label,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class ModelToolResult:
    """Deterministic tool text plus optional stored public images."""

    text: str
    images: List[ModelImage] = field(default_factory=list)


@dataclass(frozen=True)
class ModelTool:
    """One bounded local tool exposed to a model during a single call."""

    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable[[Dict[str, Any]], str | ModelToolResult] = field(
        repr=False, compare=False
    )

    def api_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True)
class ModelResult:
    data: Dict[str, Any]
    model: str
    usage: Dict[str, int]
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    usage_rounds: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TextModelResult:
    text: str
    model: str
    usage: Dict[str, int]
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    usage_rounds: List[Dict[str, Any]] = field(default_factory=list)


class StructuredVisionModel(Protocol):
    """Dependency-injected model boundary used by every V2 cognitive Agent."""

    @property
    def model_name(self) -> str:
        ...

    def generate(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> ModelResult:
        ...

    def generate_text(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> TextModelResult:
        ...


class RecordedVisionModel:
    """Deterministic model boundary backed by a frozen logical-call transcript.

    This mode re-executes the public environment and all cognition/validation
    code while replaying previously audited model outputs and usage. It exists
    for exact experiment artifact reproduction; it is not a fresh model trial.
    """

    def __init__(self, transcript_path: str | Path) -> None:
        self.path = Path(transcript_path)
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format_version") != 1:
            raise ValueError(f"Unsupported recorded transcript: {self.path}")
        calls = payload.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"Recorded transcript has no calls: {self.path}")
        self._payload = payload
        self._calls = calls
        self._index = 0

    @property
    def model_name(self) -> str:
        return str(self._payload["model"])

    @property
    def experiment_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self._payload.get("experiment_config") or {})

    @property
    def episode_created_at(self) -> str | None:
        value = self._payload.get("episode_created_at")
        return str(value) if value else None

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _payload_text(payload: str | Dict[str, Any]) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def _next(
        self,
        *,
        method: str,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
    ) -> Dict[str, Any]:
        if self._index >= len(self._calls):
            raise ValueError(
                f"Recorded transcript exhausted before {method} call {self._index + 1}"
            )
        call = self._calls[self._index]
        logical_index = self._index
        self._index += 1
        actual_payload_hash = self._sha256(self._payload_text(payload))
        actual_instruction_hash = self._sha256(instructions)
        actual_images = [item.sha256 for item in images]
        mismatches = []
        if call.get("method") != method:
            mismatches.append(f"method={method!r}, expected={call.get('method')!r}")
        if call.get("payload_sha256") != actual_payload_hash:
            mismatches.append("model payload changed")
        if call.get("instructions_sha256") != actual_instruction_hash:
            mismatches.append("model instructions changed")
        if list(call.get("image_sha256") or []) != actual_images:
            mismatches.append("model image sequence changed")
        if mismatches:
            raise ValueError(
                f"Recorded call {logical_index} no longer matches runtime: "
                + "; ".join(mismatches)
            )
        return copy.deepcopy(call)

    def generate(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> ModelResult:
        call = self._next(
            method="structured",
            instructions=instructions,
            payload=payload,
            images=images,
        )
        output = call.get("output")
        if not isinstance(output, dict):
            raise ValueError("Recorded structured call output is not an object")
        return ModelResult(
            data=output,
            model=str(call.get("model") or self.model_name),
            usage=dict(call.get("usage") or {}),
            tool_trace=list(call.get("tool_trace") or []),
            usage_rounds=list(call.get("usage_rounds") or []),
        )

    def generate_text(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> TextModelResult:
        call = self._next(
            method="text",
            instructions=instructions,
            payload=payload,
            images=images,
        )
        output = call.get("output")
        if not isinstance(output, str):
            raise ValueError("Recorded text call output is not text")
        return TextModelResult(
            text=output,
            model=str(call.get("model") or self.model_name),
            usage=dict(call.get("usage") or {}),
            tool_trace=list(call.get("tool_trace") or []),
            usage_rounds=list(call.get("usage_rounds") or []),
        )

    def assert_exhausted(self) -> None:
        if self._index != len(self._calls):
            raise ValueError(
                f"Recorded transcript has {len(self._calls) - self._index} unused calls"
            )


def _parse_json_object(text: str) -> Dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response did not contain a JSON object")
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be one JSON object")
    return parsed


class OpenAICompatibleVisionModel:
    """Minimal multimodal JSON/text client for OpenAI-compatible chat endpoints.

    It deliberately has no ARC- or game-specific behavior. Agent prompts and
    public inputs are supplied by callers; the client only transports them.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: int = 180,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        if not api_key:
            raise ValueError("An API key is required for the live vision model")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self.model

    def generate(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> ModelResult:
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        all_tool_trace: List[Dict[str, Any]] = []
        all_rounds: List[Dict[str, Any]] = []
        parse_error: ValueError | json.JSONDecodeError | None = None
        provider_model = self.model
        for attempt in range(3):
            retry_note = ""
            if attempt:
                retry_note = (
                    "\n\nYour previous response could not be parsed because its JSON was "
                    "incomplete or invalid. Regenerate the answer as one complete, compact "
                    "JSON object. Close every string, array, and object."
                )
                if attempt == 2:
                    retry_note += (
                        " This is the final transport-repair attempt: do not call tools and "
                        "return only the smallest valid JSON object that satisfies the schema."
                    )
            text, provider_model, usage, tool_trace, usage_rounds = self._request(
                instructions=instructions + retry_note,
                payload=payload,
                images=images,
                json_mode=True,
                tools=[] if attempt == 2 else (tools or []),
            )
            all_tool_trace.extend(tool_trace)
            all_rounds.extend(usage_rounds)
            for key in total_usage:
                total_usage[key] += int(usage.get(key) or 0)
            try:
                data = _parse_json_object(text)
            except (ValueError, json.JSONDecodeError) as error:
                parse_error = error
                continue
            return ModelResult(
                data=data,
                model=provider_model,
                usage=total_usage,
                tool_trace=all_tool_trace,
                usage_rounds=all_rounds,
            )
        raise ValueError(
            "Model returned invalid JSON on the initial request and both repair retries"
        ) from parse_error

    def generate_text(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        tools: List[ModelTool] | None = None,
    ) -> TextModelResult:
        text, provider_model, usage, tool_trace, usage_rounds = self._request(
            instructions=instructions,
            payload=payload,
            images=images,
            json_mode=False,
            tools=tools or [],
        )
        value = text.strip()
        if not value:
            raise ValueError("Model response did not contain text")
        return TextModelResult(
            text=value,
            model=provider_model,
            usage=usage,
            tool_trace=tool_trace,
            usage_rounds=usage_rounds,
        )

    def _request(
        self,
        *,
        instructions: str,
        payload: str | Dict[str, Any],
        images: List[ModelImage],
        json_mode: bool,
        tools: List[ModelTool],
    ) -> tuple[
        str,
        str,
        Dict[str, int],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        payload_text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": payload_text,
            }
        ]
        for item in images:
            content.append(
                {
                    "type": "text",
                    "text": f"Image label: {item.label}; artifact_id={item.artifact_id}",
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": item.data_url, "detail": "high"},
                }
            )
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ]
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        if tools:
            request["tools"] = [item.api_dict() for item in tools]
            request["tool_choice"] = "auto"
            request["parallel_tool_calls"] = False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        available_tools = {item.name: item for item in tools}
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        tool_trace: List[Dict[str, Any]] = []
        usage_rounds: List[Dict[str, Any]] = []
        provider_model = self.model
        for model_round in range(3):
            request["messages"] = messages
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=self.timeout_s) as client:
                        response = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=headers,
                            json=request,
                        )
                        response.raise_for_status()
                        raw = response.json()
                    if raw.get("error"):
                        raise RuntimeError(f"Model provider error: {raw['error']}")
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            else:  # pragma: no cover
                raise RuntimeError("Model request failed") from last_error

            provider_model = str(raw.get("model") or self.model)
            choice = (raw.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = raw.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            round_usage = {
                "round": model_round + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(
                    usage.get("total_tokens") or input_tokens + output_tokens
                ),
                "finish_reason": choice.get("finish_reason"),
            }
            usage_rounds.append(round_usage)
            for key in total_usage:
                total_usage[key] += int(round_usage[key])

            raw_tool_calls = message.get("tool_calls") or []
            if not raw_tool_calls:
                text = message.get("content") or message.get("reasoning_content") or ""
                return str(text), provider_model, total_usage, tool_trace, usage_rounds
            if model_round == 2:
                raise ValueError(
                    "Model requested a cognition tool after the bounded tool budget was closed"
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": raw_tool_calls,
                }
            )
            for raw_call in raw_tool_calls:
                function = raw_call.get("function") or {}
                name = str(function.get("name") or "")
                if name not in available_tools:
                    raise ValueError(f"Model requested unknown tool: {name}")
                raw_arguments = function.get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid arguments for tool {name}") from error
                if not isinstance(arguments, dict):
                    raise ValueError(f"Tool {name} arguments must be an object")
                executed = available_tools[name].execute(arguments)
                tool_result = (
                    executed
                    if isinstance(executed, ModelToolResult)
                    else ModelToolResult(text=str(executed))
                )
                trace = {
                    "round": model_round + 1,
                    "tool_call_id": str(raw_call.get("id") or ""),
                    "name": name,
                    "arguments": arguments,
                    "result": tool_result.text,
                    "result_characters": len(tool_result.text),
                    "returned_images": [
                        item.audit_dict() for item in tool_result.images
                    ],
                }
                tool_trace.append(trace)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": trace["tool_call_id"],
                        "name": name,
                        "content": tool_result.text,
                    }
                )
                if tool_result.images:
                    image_content: List[Dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": (
                                "Exact stored public image(s) returned by the preceding "
                                f"{name} command:"
                            ),
                        }
                    ]
                    for image in tool_result.images:
                        image_content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Image label: {image.label}; "
                                        f"artifact_id={image.artifact_id}"
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image.data_url,
                                        "detail": "high",
                                    },
                                },
                            ]
                        )
                    messages.append({"role": "user", "content": image_content})
            if model_round == 1:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The bounded cognition-read budget is now exhausted. Use the "
                            "catalog and exact records already provided to complete your final "
                            "answer. No further tool is available in this call."
                        ),
                    }
                )
                request.pop("tools", None)
                request.pop("tool_choice", None)
                request.pop("parallel_tool_calls", None)
        raise RuntimeError("Model tool loop did not return a final response")  # pragma: no cover


__all__ = [
    "ModelImage",
    "ModelResult",
    "ModelTool",
    "ModelToolResult",
    "OpenAICompatibleVisionModel",
    "RecordedVisionModel",
    "StructuredVisionModel",
    "TextModelResult",
]
