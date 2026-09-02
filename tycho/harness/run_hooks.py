"""Optional operational hooks for benchmark workers.

Hooks may alter process placement or attach an adapter fingerprint, but they cannot alter prompts,
actions, scoring, or agent state. The core runner works without a hook module.
"""

from __future__ import annotations

import importlib
import os


def _module():
    name = os.environ.get("TYCHO_RUNNER_PLUGIN", "")
    if name:
        return importlib.import_module(name)
    try:
        return importlib.import_module("tycho.harness._run_extension")
    except ModuleNotFoundError:
        return None


def worker_environment(game: str, rank: int, game_count: int) -> tuple[dict[str, str], dict]:
    """Return extra child environment and non-policy status metadata for one worker."""
    module = _module()
    if module is None or not hasattr(module, "worker_environment"):
        return {}, {}
    env, metadata = module.worker_environment(game=game, rank=rank, game_count=game_count)
    return {str(k): str(v) for k, v in (env or {}).items()}, dict(metadata or {})


def adapter_fingerprint() -> str | None:
    module = _module()
    if module is None or not hasattr(module, "adapter_fingerprint"):
        return None
    return module.adapter_fingerprint()
