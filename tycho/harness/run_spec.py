"""Immutable score-affecting identity for a long-running Tycho benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional


RUN_SPEC_SCHEMA = 2

# These knobs control transport recovery or observability, not the policy presented to the agent.
# They may be adjusted while a long run is active without changing its experimental identity.
OPERATIONAL_CONFIG_KEYS = {
    "LLM_HTTP_TIMEOUT",
    "LLM_RETRY_BUDGET_S",
    "RUN_HARDWARE",
    "LLM_LOG",
}

_POLICY_TREES = (
    "tycho/agent",
    "tycho/prompts",
    "tycho/workspace",
    "tycho/serving",
)
_POLICY_FILES = (
    "tycho/config/settings.py",
    "tycho/config/run_config.py",
    "tycho/config/__init__.py",
    "tycho/harness/actions.py",
    "tycho/harness/agent.py",
    "tycho/harness/animation_evidence.py",
    "tycho/harness/harness.py",
    "tycho/harness/inference_budget.py",
    "tycho/harness/resume.py",
    "tycho/harness/scoring.py",
    "tycho/harness/run.py",
)
_SOURCE_SUFFIXES = {".py", ".j2", ".tmpl"}


def _extension_policy_paths() -> tuple[str, ...]:
    """Additional score-affecting trees registered by an optional runner extension."""
    try:
        from tycho.harness._run_extension import POLICY_PATHS
    except (ImportError, AttributeError):
        return ()
    return tuple(str(path) for path in POLICY_PATHS)


class RunSpecError(RuntimeError):
    """The requested continuation differs from the immutable benchmark policy."""


def policy_config(resolved: dict) -> dict:
    """Strip provenance and operational knobs from a resolved-config snapshot."""
    out: dict[str, dict[str, str]] = {}
    for section, entries in (resolved.get("by_section") or {}).items():
        kept = {}
        for key, descriptor in entries.items():
            if key in OPERATIONAL_CONFIG_KEYS:
                continue
            value = descriptor.get("value") if isinstance(descriptor, dict) else descriptor
            kept[key] = value
        if kept:
            out[section] = kept
    return out


def _policy_paths(repo: Path, include_paths: Optional[Iterable[str]] = None) -> list[Path]:
    if include_paths is not None:
        candidates = [repo / value for value in include_paths]
    else:
        candidates = [
            repo / value
            for value in (*_POLICY_TREES, *_extension_policy_paths(), *_POLICY_FILES)
        ]
    out: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            out.update(
                path for path in candidate.rglob("*")
                if path.is_file()
                and path.suffix in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
                and not path.name.startswith(("test_", "smoke_", "probe_"))
                and path.name != "toy_eval.py"
            )
        elif candidate.is_file():
            out.add(candidate)
    return sorted(out)


def execution_source_manifest(
    repo: str | Path,
    *,
    include_paths: Optional[Iterable[str]] = None,
    extra_sources: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    root = Path(repo).resolve()
    manifest = {}
    for path in _policy_paths(root, include_paths):
        manifest[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, source in sorted((extra_sources or {}).items()):
        manifest[f"<callable>/{name}"] = hashlib.sha256(source.encode()).hexdigest()
    return manifest


def build_run_spec(
    *,
    repo: str | Path,
    approach: str,
    games: dict[str, list[int]],
    seed: int,
    viz: bool,
    operation_mode: str,
    resolved_config: dict,
    git_version: str,
    recorded_config: dict | None = None,
    extra_sources: Optional[dict[str, str]] = None,
    include_paths: Optional[Iterable[str]] = None,
    experiment_limits: Optional[dict[str, int]] = None,
) -> dict:
    sources = execution_source_manifest(
        repo, include_paths=include_paths, extra_sources=extra_sources
    )
    sources_sha256 = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    fingerprint_payload = {
        "approach": approach,
        "games": {game: list(baselines) for game, baselines in sorted(games.items())},
        "seed": int(seed),
        "viz": bool(viz),
        "operation_mode": operation_mode,
        "config": policy_config(resolved_config),
        "experiment_limits": dict(sorted((experiment_limits or {}).items())),
        "sources": sources,
    }
    digest = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy = {
        "approach": approach,
        "games": {game: list(baselines) for game, baselines in sorted(games.items())},
        "seed": int(seed),
        "viz": bool(viz),
        "operation_mode": operation_mode,
        "config": policy_config(recorded_config or resolved_config),
        "experiment_limits": dict(sorted((experiment_limits or {}).items())),
        "sources_sha256": sources_sha256,
    }
    return {
        "schema": RUN_SPEC_SCHEMA,
        "fingerprint": digest,
        "initial_git_version": git_version,
        "policy": policy,
        "operational_config_keys": sorted(OPERATIONAL_CONFIG_KEYS),
    }


def ensure_run_spec(path: str | Path, candidate: dict, *, resume: bool) -> dict:
    target = Path(path)
    if target.exists():
        try:
            saved = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RunSpecError(f"cannot read immutable run spec {target}: {exc}") from exc
        if saved.get("fingerprint") != candidate.get("fingerprint"):
            saved_policy = saved.get("policy") or {}
            new_policy = candidate.get("policy") or {}
            changed = [
                key for key in sorted(set(saved_policy) | set(new_policy))
                if saved_policy.get(key) != new_policy.get(key)
            ]
            raise RunSpecError(
                "score-affecting execution policy changed; refusing continuation "
                f"(changed: {', '.join(changed) or 'unknown'})"
            )
        return saved
    if resume:
        raise RunSpecError(
            f"{target.parent} has no run_spec.json; exact continuation cannot be proven"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    tmp.replace(target)
    return candidate
