from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tycho.serving import public_backends as backends
from tycho.serving import llm_client


def _cfg(**overrides):
    values = {
        "model": "test-model",
        "base_url": "https://example.invalid/v1",
        "api_key": "test-key",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


TOOLS = [
    {
        "name": "take_action",
        "description": "Commit one action.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "row": {"type": "integer"},
            },
            "required": ["action"],
        },
    }
]


def test_strict_tool_specs_preserve_optional_semantics() -> None:
    strict = backends.strict_tool_specs(TOOLS)
    schema = strict[0]["schema"]

    assert schema["required"] == ["action", "row"]
    assert schema["properties"]["row"]["type"] == ["integer", "null"]
    assert schema["additionalProperties"] is False
    assert TOOLS[0]["schema"]["required"] == ["action"]


def test_anthropic_transport_maps_tool_calls_and_usage(monkeypatch) -> None:
    captured = {}

    def fake_post(url, body, headers, timeout, retry):
        captured.update(url=url, body=body, headers=headers, timeout=timeout)
        return {
            "content": [
                {"type": "thinking", "thinking": "inspect"},
                {"type": "text", "text": "acting"},
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "take_action",
                    "input": {"action": "ACTION1"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 3,
            },
        }

    monkeypatch.setattr(backends, "_post_json", fake_post)
    monkeypatch.setenv("TYCHO_PROMPT_CACHING", "0")
    reply = backends.anthropic_tools(
        [{"role": "user", "content": "frame"}],
        TOOLS,
        _cfg(),
        "system",
        100,
        30,
        "high",
        lambda fn: fn(),
    )

    assert captured["url"] == "https://example.invalid/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["body"]["tools"][0]["name"] == "take_action"
    assert reply["text"] == "acting"
    assert reply["reasoning"] == "inspect"
    assert reply["tool_calls"] == [
        {"id": "tool-1", "name": "take_action", "input": {"action": "ACTION1"}}
    ]
    assert reply["usage"] == {"in": 11, "out": 7, "cache_read": 5, "cache_write": 3}


def test_openai_responses_delta_chain_and_tool_parsing(monkeypatch) -> None:
    captured = {}
    history = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "new"},
    ]

    def fake_post(url, body, headers, timeout, retry):
        captured.update(url=url, body=body, headers=headers)
        return {
            "id": "resp-next",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "take_action",
                    "arguments": json.dumps({"action": "ACTION2"}),
                }
            ],
            "usage": {
                "input_tokens": 19,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 9,
            },
        }

    monkeypatch.setattr(backends, "_post_json", fake_post)
    monkeypatch.delenv("OPENAI_STATELESS_REASONING", raising=False)
    reply = backends.openai_responses_tools(
        history,
        TOOLS,
        _cfg(),
        "system",
        100,
        30,
        "high",
        lambda fn: fn(),
        prev_response_id="resp-old",
        since=1,
    )

    assert captured["url"] == "https://example.invalid/v1/responses"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["body"]["previous_response_id"] == "resp-old"
    assert captured["body"]["input"] == [{"role": "user", "content": "new"}]
    assert reply["response_id"] == "resp-next"
    assert reply["tool_calls"][0]["input"] == {"action": "ACTION2"}
    assert reply["usage"] == {"in": 15, "out": 9, "cache_read": 4, "cache_write": 0}


def test_openai_responses_omits_tool_controls_for_toolless_calls() -> None:
    body = backends.responses_request(
        [{"role": "user", "content": "summarize"}],
        [],
        _cfg(),
        "",
        100,
        "medium",
    )

    assert "tools" not in body
    assert "tool_choice" not in body


def test_openai_responses_omits_disabled_reasoning_effort() -> None:
    body = backends.responses_request(
        [{"role": "user", "content": "ping"}],
        TOOLS,
        _cfg(),
        "",
        100,
        "off",
    )

    assert "reasoning" not in body
    assert body["max_output_tokens"] == 16000


def test_post_json_round_trip_over_local_http() -> None:
    captured = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            size = int(self.headers.get("Content-Length", "0"))
            captured.update(
                path=self.path,
                authorization=self.headers.get("Authorization"),
                body=json.loads(self.rfile.read(size)),
            )
            payload = json.dumps({"ok": True, "request_id": "local"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("local socket binding is disabled by the execution sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = backends._post_json(
            f"http://127.0.0.1:{server.server_port}/v1/responses",
            {"model": "offline", "input": "ping"},
            {"Content-Type": "application/json", "Authorization": "Bearer local-key"},
            5,
            lambda fn: fn(),
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response == {"ok": True, "request_id": "local"}
    assert captured == {
        "path": "/v1/responses",
        "authorization": "Bearer local-key",
        "body": {"model": "offline", "input": "ping"},
    }


def test_post_json_preserves_provider_error_body() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            payload = json.dumps({"error": {"message": "unsupported field"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("local socket binding is disabled by the execution sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError, match="unsupported field") as caught:
            backends._post_json(
                f"http://127.0.0.1:{server.server_port}/v1/messages",
                {"model": "offline"},
                {"Content-Type": "application/json"},
                5,
                lambda fn: fn(),
            )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert "unsupported field" in getattr(caught.value, "_tycho_body", "")


@pytest.mark.parametrize(
    ("status", "expected"),
    [(400, False), (408, True), (429, True), (500, True), (520, True), (599, True)],
)
def test_native_transport_retryable_http_statuses(status: int, expected: bool) -> None:
    error = urllib.error.HTTPError("https://example.invalid", status, "failure", {}, None)

    assert llm_client._is_retryable(error) is expected


@pytest.mark.parametrize(
    ("backend", "key_name", "expected_base"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1"),
        ("openai_responses", "OPENAI_API_KEY", "https://api.openai.com/v1"),
    ],
)
def test_native_provider_config_from_environment(
    monkeypatch, backend, key_name, expected_base
) -> None:
    monkeypatch.setattr(llm_client, "_extension", lambda: None)
    for name in (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL",
        "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.setenv(key_name, "offline-key")

    cfg = llm_client.LLMConfig.from_env()

    assert cfg.backend == backend
    assert cfg.model == "offline-model"
    assert cfg.api_key == "offline-key"
    assert cfg.base_url == expected_base


def test_native_provider_config_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "_extension", lambda: None)
    monkeypatch.setenv("LLM_BACKEND", "openai_responses")
    monkeypatch.setenv("LLM_MODEL", "offline-model")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        llm_client.LLMConfig.from_env()


def test_recording_uses_public_model_identity(monkeypatch) -> None:
    monkeypatch.setattr(llm_client, "_extension", lambda: None)
    cfg = llm_client.LLMConfig(
        model="deployment-model",
        backend="openai_responses",
        api_protocol="openai_responses",
        public_model="published-model",
    )
    llm_client.start_recording()
    try:
        llm_client._record_call(
            [{"role": "user", "content": "hello"}],
            [],
            cfg,
            "",
            "",
            "actor",
            {"text": "world", "usage": {"in": 1, "out": 1}},
            10,
        )
        record = llm_client.take_recording()
    finally:
        llm_client.stop_recording()

    assert record[0]["model"] == "published-model"
    assert llm_client.public_identity(cfg) == {
        "api_protocol": "openai_responses",
        "model": "published-model",
    }
