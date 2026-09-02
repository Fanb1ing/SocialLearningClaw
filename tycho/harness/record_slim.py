"""Lossless write-time compaction for Tycho game records.

An `EnvRunRecord` references causal workspace state on every trace step. Text and path-to-blob
manifests often remain unchanged for many turns, so storing each snapshot literally creates large
redundant records. This module applies three transforms at write time:

  1. per-frame workspace files (turn_*.txt, diff_*.txt, and animation keyframe payloads)
     are dropped from the embedded
     snapshot — they live on disk in the workspace dir and as the `grid` field already, so no
     information is lost (and they accumulate O(turns) per snapshot, the original blow-up).
  2. text VERSIONING: a causal file whose text is unchanged from the previous step it
     appeared in is replaced by the sentinel "\x00=<step_index>" pointing back to the step that
     holds the real text. Trace readers can resolve the chain exactly.
  3. manifest VERSIONING: an unchanged path-to-content-addressed-blob manifest is replaced by
     the same kind of frame-step backreference. Binary bodies remain in the run's blob store.

The `EnvRunRecord` shape is preserved, so resume, scorecard export, and analysis use the same record.
Tycho does not impose a scene graph; the agent chooses its own state representation.
"""

from __future__ import annotations

from tycho.workspace.version_store import is_causal_workspace_path

def _is_authored(path: str) -> bool:
    """Compatibility name for the shared causal-workspace path predicate."""
    return is_causal_workspace_path(path)


def _slim_step_workspace(reasoning: dict) -> None:
    """In place: drop the accumulating per-frame bloat from one step's workspace snapshot —
    the file TREE listing, the embedded per-frame file contents, and extra images. Each grows
    O(turns) and is re-embedded every step (the file `files` list alone was ~56 KB/step = the
    dominant cost on long games). All captured causal content remains reconstructable."""
    if not isinstance(reasoning, dict):
        return
    ws = reasoning.get("workspace")
    if not isinstance(ws, dict):
        return
    c = ws.get("contents")
    if isinstance(c, dict):
        ws["contents"] = {k: v for k, v in c.items() if _is_authored(k)}
    versions = ws.get("file_versions")
    if isinstance(versions, dict):
        ws["file_versions"] = {k: v for k, v in versions.items() if _is_authored(k)}
    # `files` is the full workspace tree (every turn_/diff_/level_ path) re-listed each step —
    # the single biggest field on long games. Trace consumers need causal files; per-frame grids
    # remain available through the `grid` and `cur_grid` fields.
    fl = ws.get("files")
    if isinstance(fl, list):
        ws["files"] = [p for p in fl if _is_authored(p)]
    imgs = ws.get("images")
    if isinstance(imgs, dict) and len(imgs) > 1:
        ws["images"] = {}   # the current frame is already stored by the trace


def _dedup_trace_contents(trace: list) -> None:
    """In place: content-version causal text files across the trace. An unchanged repeat becomes
    "\\x00=<idx>" pointing at the step that last held the real text.

    Back-reference indices address the frame-bearing step stream, not every raw trace entry. Readers
    use the same stream, so entries without a frame must not advance the index. The transform is
    idempotent."""
    last_val: dict = {}
    last_idx: dict = {}
    step_idx = -1
    for t in trace:
        if t.get("frame") is None:
            continue  # not part of the frame-bearing step stream
        step_idx += 1
        ws = (t.get("reasoning") or {}).get("workspace")
        if not isinstance(ws, dict):
            continue
        c = ws.get("contents")
        if not isinstance(c, dict):
            continue
        for k in list(c.keys()):
            v = c[k]
            if isinstance(v, str) and v.startswith("\x00="):
                continue  # already a back-ref (idempotent)
            if k in last_val and last_val[k] == v:
                c[k] = f"\x00={last_idx[k]}"
            else:
                last_val[k] = v
                last_idx[k] = step_idx


def _dedup_trace_versions(trace: list) -> None:
    """Replace an unchanged full path-to-blob manifest with a frame-step backreference."""
    last_value = None
    last_idx = None
    step_idx = -1
    for step in trace:
        if step.get("frame") is None:
            continue
        step_idx += 1
        workspace = (step.get("reasoning") or {}).get("workspace")
        if not isinstance(workspace, dict):
            continue
        value = workspace.get("file_versions")
        if isinstance(value, str) and value.startswith("\x00="):
            continue
        if not isinstance(value, dict):
            continue
        if last_value is not None and value == last_value:
            workspace["file_versions"] = f"\x00={last_idx}"
        else:
            last_value = value
            last_idx = step_idx


def slim_record(rec: dict) -> dict:
    """Slim an EnvRunRecord dict IN PLACE (and return it) for write-time persistence. Preserves
    every top-level field and the trace shape; only shrinks each trace step's workspace snapshot
    (drop per-frame files), versions causal text, and versions content-addressed manifests.
    Marks
    `_slim=1` so a reader knows the record is already slimmed + dedup back-refs are present."""
    trace = rec.get("trace")
    if isinstance(trace, list) and trace:
        for t in trace:
            _slim_step_workspace(t.get("reasoning"))
        _dedup_trace_contents(trace)
        _dedup_trace_versions(trace)
    rec["_slim"] = 1
    return rec
