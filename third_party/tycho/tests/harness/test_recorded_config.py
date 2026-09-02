from __future__ import annotations

import json

from tycho.config.run_config import CONFIG_KEYS, PUBLIC_CONFIG_KEYS, recorded_config


def test_recorded_config_contains_only_stable_public_fields(monkeypatch) -> None:
    marker = "operator-local-marker"
    monkeypatch.setenv("LLM_BACKEND", "local-transport")
    monkeypatch.setenv("LLM_MODEL", "deployment-model-id")
    monkeypatch.setenv("LLM_BASE_URL", f"https://{marker}.invalid/v1")
    monkeypatch.setenv("RUN_HARDWARE", marker)
    public_names = {key.env for key in PUBLIC_CONFIG_KEYS}
    extension_names = {key.env for key in CONFIG_KEYS} - public_names
    for name in extension_names:
        monkeypatch.setenv(name, marker)

    snapshot = recorded_config(
        api_protocol="openai_responses",
        model="published-model",
    )
    encoded = json.dumps(snapshot, sort_keys=True)
    model = snapshot["by_section"]["model"]

    assert marker not in encoded
    assert all(name not in encoded for name in extension_names)
    assert "LLM_BASE_URL" not in encoded
    assert "RUN_HARDWARE" not in encoded
    assert model["LLM_BACKEND"]["value"] == "openai_responses"
    assert model["LLM_MODEL"]["value"] == "published-model"
