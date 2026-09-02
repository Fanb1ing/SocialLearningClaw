"""Small, process-safe status/event store for long-running per-game workers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import socket
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


class WorkerLeaseError(RuntimeError):
    """Another live process already owns this game's worker lease."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _node_token() -> str:
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _pid_alive(pid: object) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


def _recent_heartbeat(status: dict, *, max_age_s: float = 120.0) -> bool:
    value = status.get("heartbeat_at")
    if not value:
        return False
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return False
    return time.time() - stamp <= max_age_s


class GameStatusStore:
    """Atomic ``status.json`` plus append-only ``events.jsonl`` for one game."""

    def __init__(self, game_dir: str | Path):
        self.dir = Path(game_dir)
        self.status_path = self.dir / "status.json"
        self.events_path = self.dir / "events.jsonl"
        self.lock_path = self.dir / ".status.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        with self.lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_unlocked(self) -> dict:
        try:
            return json.loads(self.status_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def update(self, **fields) -> dict:
        with self._locked():
            status = self._read_unlocked()
            status.update(fields)
            status["updated_at"] = utc_now()
            _atomic_json(self.status_path, status)
            return status

    def event(
        self,
        event_type: str,
        *,
        status_fields: Optional[dict] = None,
        status_increments: Optional[dict] = None,
        **fields,
    ) -> dict:
        event = {"ts": utc_now(), "time": time.time(), "type": event_type, **fields}
        with self._locked():
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if status_fields or status_increments:
                status = self._read_unlocked()
                status.update(status_fields or {})
                for key, increment in (status_increments or {}).items():
                    status[key] = float(status.get(key) or 0.0) + float(increment)
                status["updated_at"] = event["ts"]
                _atomic_json(self.status_path, status)
        return event

    def claim(self, *, game: str, attempt: int, resume: bool) -> dict:
        host = _node_token()
        pid = os.getpid()
        with self._locked():
            old = self._read_unlocked()
            same_host_live = (
                old.get("host") == host
                and old.get("pid") != pid
                and _pid_alive(old.get("pid"))
            )
            remote_host_fresh = old.get("host") != host and _recent_heartbeat(old)
            if old.get("state") in {"running", "in_llm"} and (same_host_live or remote_host_fresh):
                raise WorkerLeaseError(
                    f"{game} is already owned by live pid {old.get('pid')} on {old.get('host')}"
                )
            prior_attempt_actions = int(old.get("attempt_start_action_count") or 0)
            prior_last_actions = int(old.get("action_count") or 0)
            actionless = int(old.get("actionless_resume_count") or 0)
            if resume and int(old.get("attempt") or 0) > 0 and prior_last_actions <= prior_attempt_actions:
                actionless += 1
            now = utc_now()
            status = {
                **old,
                "game": game,
                "state": "running",
                "attempt": int(attempt),
                "pid": pid,
                "host": host,
                "resume": bool(resume),
                "started_at": old.get("started_at") or now,
                "attempt_started_at": now,
                "heartbeat_at": now,
                "attempt_start_action_count": prior_last_actions,
                "actionless_resume_count": actionless,
                "error": None,
                "next_resume_at": None,
                "updated_at": now,
            }
            _atomic_json(self.status_path, status)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "ts": now,
                    "time": time.time(),
                    "type": "worker_started",
                    "attempt": int(attempt),
                    "pid": pid,
                    "resume": bool(resume),
                    "action_count": prior_last_actions,
                }, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return status

    def heartbeat(self) -> None:
        self.update(heartbeat_at=utc_now())

    def note_action(
        self,
        *,
        action_count: int,
        action: str,
        level: int,
        level_action_count: int,
        levels_completed: int,
        rhae: float,
        inference_budget: Optional[dict] = None,
    ) -> None:
        now = utc_now()
        budget_fields = {}
        if inference_budget:
            budget_fields = {
                "inference_cost_usd": inference_budget.get("cost_usd"),
                "inference_level_cost_usd": inference_budget.get("level_cost_usd"),
                "max_inference_cost_per_game_usd": inference_budget.get(
                    "max_cost_per_game_usd"
                ),
                "max_inference_cost_per_level_usd": inference_budget.get(
                    "max_cost_per_level_usd"
                ),
            }
        self.event(
            "action_committed",
            action_count=action_count,
            action=action,
            level=level,
            level_action_count=level_action_count,
            levels_completed=levels_completed,
            rhae=round(float(rhae), 6),
            status_fields={
                "state": "running",
                "last_action_at": now,
                "last_action": action,
                "action_count": int(action_count),
                "level": int(level),
                "level_action_count": int(level_action_count),
                "levels_completed": int(levels_completed),
                "current_rhae": round(float(rhae), 6),
                **budget_fields,
            },
        )

    def note_llm_started(self, *, call_type: str = "actor") -> None:
        now = utc_now()
        self.event(
            "llm_started",
            call_type=call_type,
            status_fields={"state": "in_llm", "last_llm_started_at": now},
        )

    def note_llm_finished(
        self,
        *,
        call_type: str = "actor",
        response: str = "",
        usage: Optional[dict] = None,
        model: str = "",
        cost_usd: Optional[float] = None,
    ) -> None:
        now = utc_now()
        compact = " ".join(str(response).split())[-1000:]
        event_fields = {"call_type": call_type, "model": model}
        if usage:
            event_fields["usage"] = dict(usage)
        if cost_usd is not None:
            event_fields["cost_usd"] = round(float(cost_usd), 9)
        self.event(
            "llm_finished",
            **event_fields,
            status_fields={
                "state": "running",
                "last_llm_completed_at": now,
                "last_llm_response": compact,
            },
            status_increments=(
                {"inference_cost_usd": cost_usd} if cost_usd is not None else None
            ),
        )

    def note_llm_failed(self, *, call_type: str = "actor", error: str = "") -> None:
        now = utc_now()
        self.event(
            "llm_failed",
            call_type=call_type,
            error=str(error)[:1000],
            status_fields={
                "state": "running",
                "last_llm_completed_at": now,
                "last_llm_error": str(error)[:1000],
            },
        )

    def events(self) -> list[dict]:
        out = []
        try:
            lines = self.events_path.read_text().splitlines()
        except OSError:
            return out
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
        return out


_PROCESS_STORE: Optional[GameStatusStore] = None


def process_status_store() -> Optional[GameStatusStore]:
    """Return the current worker's store, configured by the supervisor."""
    global _PROCESS_STORE
    path = os.environ.get("TYCHO_GAME_STATUS_DIR")
    if not path:
        return None
    if _PROCESS_STORE is None or _PROCESS_STORE.dir != Path(path):
        _PROCESS_STORE = GameStatusStore(path)
    return _PROCESS_STORE


def _epoch(value: object) -> Optional[float]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _recent_rate(
    events: list[dict],
    *,
    event_type: str,
    now: float,
    started_at: Optional[float],
    value_key: Optional[str] = None,
    window_s: float = 24 * 3600,
) -> float:
    """Trailing event rate per hour, retaining slow multi-hour model calls."""
    start = max(now - window_s, started_at or now - window_s)
    selected = [
        event for event in events
        if event.get("type") == event_type
        and isinstance(event.get("time"), (int, float))
        and (value_key is None or isinstance(event.get(value_key), (int, float)))
        and start <= float(event["time"]) <= now
    ]
    if not selected:
        return 0.0
    total = (
        sum(float(event.get(value_key) or 0.0) for event in selected)
        if value_key else float(len(selected))
    )
    duration = max(60.0, now - start)
    return total * 3600.0 / duration


def _action_rate_series(
    action_times: list[float], *, start: float, end: float, target_bins: int = 96
) -> tuple[list[dict], float]:
    """Bin a complete run trajectory into a compact actions/hour series."""
    duration = max(60.0, end - start)
    candidates = (900, 1800, 3600, 7200, 14400, 21600, 43200, 86400)
    desired = duration / max(1, target_bins)
    bin_s = float(next((value for value in candidates if value >= desired), candidates[-1]))
    n_bins = max(1, int(math.ceil(duration / bin_s)))
    counts = [0] * n_bins
    for stamp in action_times:
        if start <= stamp <= end:
            counts[min(int((stamp - start) // bin_s), n_bins - 1)] += 1
    points = []
    for index, count in enumerate(counts):
        bin_start = start + index * bin_s
        bin_end = min(end, bin_start + bin_s)
        observed_s = max(60.0, bin_end - bin_start)
        points.append({
            "time": round(bin_start, 3),
            "actions": count,
            "actions_per_hour": round(count * 3600.0 / observed_s, 4),
        })
    return points, bin_s


def collect_run_status(run_dir: str | Path) -> dict:
    """Return one structured monitoring snapshot for CLI or web status consumers."""
    run = Path(run_dir).resolve()
    now = time.time()
    manifest = {}
    spec = {}
    try:
        manifest = json.loads((run / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    try:
        spec = json.loads((run / "run_spec.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass

    baselines: dict[str, list[int]] = {}
    for full, values in (spec.get("policy", {}).get("games") or {}).items():
        baselines[full.split("-")[0]] = list(values)
    requested = set(baselines)
    requested.update(manifest.get("requested_games") or [])
    status_root = run / "status"
    if status_root.is_dir():
        requested.update(path.name for path in status_root.iterdir() if path.is_dir())
    requested.update(path.stem.replace("game_", "") for path in run.glob("game_*.json"))
    requested.update(path.stem.replace(".partial_", "") for path in run.glob(".partial_*.json"))

    reasoning_cfg = spec.get("policy", {}).get("config", {}).get("reasoning", {})
    configured_game_cap = _as_float(reasoning_cfg.get("TYCHO_MAX_INFERENCE_COST_PER_GAME"))

    games = []
    run_action_times: list[float] = []
    run_started_epochs: list[float] = []
    run_stopped_epochs: list[float] = []
    run_active = False
    for game in sorted(requested):
        store = GameStatusStore(status_root / game)
        live = store.read()
        record = (manifest.get("per_game") or {}).get(game) or {}
        record_path = run / f"game_{game}.json"
        partial_path = run / f".partial_{game}.json"

        if live.get("state"):
            state = live["state"]
        elif record_path.exists() and not str(record.get("mode", "")).startswith(("ERROR", "PARTIAL")):
            state = "finished"
        elif str(record.get("mode", "")).startswith(("ERROR", "PARTIAL")):
            state = "error"
        elif partial_path.exists():
            state = "running"
        else:
            state = "queued"
        rhae = live.get("current_rhae")
        if state == "finished" or rhae is None:
            rhae = record.get("rhae", rhae)
        rhae = float(rhae or 0.0)
        action_count = int(live.get("action_count") or record.get(
            "total_actions_including_unfinished", record.get("total_actions", 0)
        ) or 0)
        levels = int(live.get("levels_completed") or record.get("levels") or 0)

        events = store.events()
        started_epoch = _epoch(live.get("started_at"))
        if started_epoch is not None:
            run_started_epochs.append(started_epoch)
        event_times = [
            float(event["time"])
            for event in events
            if isinstance(event.get("time"), (int, float))
        ]
        run_action_times.extend(
            float(event["time"])
            for event in events
            if event.get("type") == "action_committed"
            and isinstance(event.get("time"), (int, float))
        )
        stopped_epoch = _epoch(live.get("finished_at") or live.get("updated_at"))
        if stopped_epoch is not None:
            run_stopped_epochs.append(stopped_epoch)
        elif event_times:
            run_stopped_epochs.append(max(event_times))
        if state in {"queued", "running", "in_llm", "backoff"}:
            run_active = True
        actions_per_hour = _recent_rate(
            events,
            event_type="action_committed",
            now=now,
            started_at=started_epoch,
        )
        inference_cost_per_hour = _recent_rate(
            events,
            event_type="llm_finished",
            value_key="cost_usd",
            now=now,
            started_at=started_epoch,
        )

        level = int(live.get("level") or levels)
        level_actions = int(live.get("level_action_count") or 0)
        eta_s = None
        game_baselines = baselines.get(game) or []
        if actions_per_hour > 0 and 0 <= level < len(game_baselines):
            try:
                from tycho.harness.harness import ACTION_BUDGET_MULTIPLIER
                remaining = max(0, ACTION_BUDGET_MULTIPLIER * game_baselines[level] - level_actions)
                eta_s = remaining / actions_per_hour * 3600.0
            except Exception:  # noqa: BLE001
                eta_s = None

        record_budget = record.get("inference_budget") or {}
        live_result_budget = ((live.get("result") or {}).get("inference_budget") or {})
        inference_cost = _as_float(live.get("inference_cost_usd"))
        if inference_cost is None:
            inference_cost = _as_float(
                live_result_budget.get("cost_usd", record_budget.get("cost_usd"))
            )
        game_cap = _as_float(live.get("max_inference_cost_per_game_usd"))
        if game_cap is None:
            game_cap = _as_float(
                live_result_budget.get(
                    "max_cost_per_game_usd",
                    record_budget.get("max_cost_per_game_usd", configured_game_cap),
                )
            )
        game_cap_pct = None
        eta_game_cap_s = None
        if game_cap is not None and game_cap > 0 and inference_cost is not None:
            game_cap_pct = 100.0 * inference_cost / game_cap
            if inference_cost_per_hour > 0 and state in {"running", "in_llm", "backoff"}:
                remaining_cost = max(0.0, game_cap - inference_cost)
                eta_game_cap_s = remaining_cost / inference_cost_per_hour * 3600.0

        last_action_epoch = _epoch(live.get("last_action_at"))
        last_llm_epoch = _epoch(live.get("last_llm_completed_at") or live.get("last_llm_started_at"))
        heartbeat_epoch = _epoch(live.get("heartbeat_at"))
        games.append({
            "game": game,
            "state": state,
            "attempt": int(live.get("attempt") or 0),
            "rhae": round(rhae, 6),
            "levels_completed": levels,
            "n_levels": len(game_baselines),
            "level_completion_pct": (
                round(100.0 * levels / len(game_baselines), 3) if game_baselines else None
            ),
            "level": level,
            "level_action_count": level_actions,
            "action_count": action_count,
            "last_action": live.get("last_action"),
            "actions_per_hour": round(actions_per_hour, 2),
            "eta_to_level_cap_s": round(eta_s, 1) if eta_s is not None else None,
            "inference_cost_usd": (
                round(inference_cost, 6) if inference_cost is not None else None
            ),
            "inference_game_cap_usd": round(game_cap, 6) if game_cap is not None else None,
            "inference_game_cap_pct": (
                round(game_cap_pct, 3) if game_cap_pct is not None else None
            ),
            "inference_cost_per_hour": round(inference_cost_per_hour, 4),
            "eta_to_inference_game_cap_s": (
                round(eta_game_cap_s, 1) if eta_game_cap_s is not None else None
            ),
            "seconds_since_action": round(now - last_action_epoch, 1) if last_action_epoch else None,
            "seconds_since_llm": round(now - last_llm_epoch, 1) if last_llm_epoch else None,
            "seconds_since_heartbeat": round(now - heartbeat_epoch, 1) if heartbeat_epoch else None,
            "last_llm_response": live.get("last_llm_response", ""),
            "last_llm_error": live.get("last_llm_error"),
            "error": live.get("error"),
            "actionless_resume_count": int(live.get("actionless_resume_count") or 0),
            "next_resume_at": live.get("next_resume_at"),
        })

    scores = [game["rhae"] for game in games]
    finished_scores = [game["rhae"] for game in games if game["state"] == "finished"]
    if run_started_epochs:
        run_started = min(run_started_epochs)
    else:
        try:
            run_started = (run / "run_spec.json").stat().st_mtime
        except OSError:
            run_started = now
    run_ended = now if run_active else max(run_stopped_epochs or [now])
    run_ended = max(run_started, run_ended)
    run_elapsed_s = max(0.0, run_ended - run_started)
    total_actions = sum(game["action_count"] for game in games)
    avg_actions_per_hour = (
        total_actions * 3600.0 / max(60.0, run_elapsed_s) if total_actions else 0.0
    )
    action_rate_series, action_rate_bin_s = _action_rate_series(
        run_action_times, start=run_started, end=run_ended
    )
    return {
        "run": run.name,
        "path": str(run),
        "model": manifest.get("model") or spec.get("policy", {}).get("config", {}).get(
            "model", {}
        ).get("LLM_MODEL"),
        "mode": manifest.get("mode") or spec.get("policy", {}).get("config", {}).get(
            "orchestration", {}
        ).get("TYCHO_MODE"),
        "n_requested": len(games),
        "n_finished": len(finished_scores),
        "floor_rhae": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "mean_finished_rhae": (
            round(sum(finished_scores) / len(finished_scores), 6) if finished_scores else None
        ),
        "total_actions": total_actions,
        "started_at": datetime.fromtimestamp(run_started, timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(run_elapsed_s, 1),
        "average_actions_per_hour": round(avg_actions_per_hour, 4),
        "action_rate_bin_s": action_rate_bin_s,
        "action_rate_series": action_rate_series,
        "games": games,
        "generated_at": utc_now(),
    }
