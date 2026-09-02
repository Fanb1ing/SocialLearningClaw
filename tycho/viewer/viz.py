"""Build a self-contained HTML replay viewer from Tycho run records.

The core player renders frames, actions, level boundaries, run status, and captured model calls. A
small overlay registry adds executable-model predictions, verification diagnostics, plans, and
workspace history when those fields are present. Unknown reasoning fields remain inspectable through
a JSON fallback rather than being discarded.

Usage:
  .venv/bin/python -m tycho.viewer.viz results/<run>.json [--game <id>] [--out viz.html]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tycho.serving.pricing import PRICE_PER_1M as _PRICE_PER_1M
from tycho.serving.pricing import price_for as _price_for

REPO = Path(__file__).resolve().parent.parent.parent  # viewer/ -> tycho/ -> repo root
# Official ARC-AGI-3 palette (arc_agi/rendering.py), same colors as arcprize.org/tasks.
ARC16 = ["#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
         "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
         "#FF851B", "#921231", "#4FCC30", "#A356D6"]
def _slim_workspace(reasoning: dict) -> dict:
    """Per-step cleanup: keep only causal files in the workspace snapshot,
    and at most the current frame's image. Older records may embed the accumulating workspace and
    many images at every step; this keeps generated viewer data bounded."""
    if not isinstance(reasoning, dict):
        return reasoning
    ws = reasoning.get("workspace")
    if not isinstance(ws, dict):
        return reasoning
    from tycho.harness.record_slim import _is_authored  # shared filter (single source of truth)
    c = ws.get("contents")
    if isinstance(c, dict):
        ws["contents"] = {k: v for k, v in c.items() if _is_authored(k)}
    versions = ws.get("file_versions")
    if isinstance(versions, dict):
        ws["file_versions"] = {k: v for k, v in versions.items() if _is_authored(k)}
    fl = ws.get("files")   # prune the accumulating file-tree listing to causal files (the
    if isinstance(fl, list):   # per-frame turn_/diff_ paths bloat O(turns) and aren't useful tree nodes)
        ws["files"] = [p for p in fl if _is_authored(p)]
    imgs = ws.get("images")
    if isinstance(imgs, dict) and len(imgs) > 1:
        ws["images"] = {}  # old multi-image embed: drop (the canvas shows the current frame)
    return reasoning


def _dedup_contents(steps: list) -> None:
    """Content-versioning (in place): causal text barely changes turn-to-turn, but each step
    re-embeds their FULL text (a large trace can be ~25% duplicated file text). Emit a file's content only on steps where
    it CHANGED; replace an unchanged repeat with the sentinel "\\x00=" + the step index that
    last held it. The viewer carries the last value forward. Cheap, lossless, ~no logic in JS."""
    last_val: dict = {}; last_idx: dict = {}
    for i, st in enumerate(steps):
        ws = (st.get("reasoning") or {}).get("workspace")
        if not isinstance(ws, dict):
            continue
        c = ws.get("contents")
        if not isinstance(c, dict):
            continue
        for k in list(c.keys()):
            v = c[k]
            if k in last_val and last_val[k] == v:
                c[k] = f"\x00={last_idx[k]}"   # back-ref to the step that holds the real text
            else:
                last_val[k] = v; last_idx[k] = i
    # store the marker so the viewer knows to resolve back-refs
    if steps:
        steps[0].setdefault("_dedup", True)


def _dedup_versions(steps: list) -> None:
    """Version identical path-to-blob manifests in legacy/static viewer builds."""
    last_value = None
    last_idx = None
    for i, step in enumerate(steps):
        ws = (step.get("reasoning") or {}).get("workspace")
        if not isinstance(ws, dict):
            continue
        value = ws.get("file_versions")
        if isinstance(value, str) and value.startswith("\x00="):
            continue
        if not isinstance(value, dict):
            continue
        if last_value is not None and value == last_value:
            ws["file_versions"] = f"\x00={last_idx}"
        else:
            last_value = value
            last_idx = i


def build_steps(env_rec: dict) -> list:
    """One viewer step per trace entry that carries a frame. `level` = level being
    played at the step start; `just_completed` flags the observable goal-reached.

    Tycho forms task-specific abstractions in its own workspace, so this generic viewer does not
    impose an object segmentation. Records already slimmed at write time skip the re-dedup pass
    because their back-references are already correct."""
    already_slim = bool(env_rec.get("_slim"))
    # A run launched WITHOUT --viz captures no frames (frame=None on every step) but DOES carry the
    # reasoning/tool-trace/workspace per step. Don't drop those — emit a frame-optional step so the
    # transcript, beliefs, and world-model panels are still viewable (the grid pane just hides). Only
    # fall back to "frame-bearing only" when at least one frame exists (the normal --viz case), so a
    # mixed record isn't padded with the inter-frame cognitive turns.
    trace = env_rec.get("trace", [])
    has_any_frame = any(t.get("frame") is not None for t in trace)
    steps = []
    prev_completed = 0
    for t in trace:
        if has_any_frame and t.get("frame") is None:
            continue
        completed = t["levels_completed"]
        step = {
            "turn": t["turn"], "action": t["action"], "x": t.get("x"), "y": t.get("y"),
            "state": t["state"], "levels": completed, "level": prev_completed,
            "just_completed": completed > prev_completed or t["state"] == "GameState.WIN",
            "game_over": t["state"] == "GameState.GAME_OVER",
            "auto_reset": t["action"] == "RESET",
            "changed": t["frame_changed"], "reasoning": _slim_workspace(t.get("reasoning")),
            "grid": t.get("frame"),   # None on no-viz runs → client renders reasoning-only
        }
        # _bake_wm_predictions stores the planned-trajectory under reasoning.wm_pred, but the client
        # (renderWmPred / the col-1 audit) reads step.wm_pred at the TOP LEVEL — lift it here so the
        # planner overlay + prediction-vs-reality panel actually render.
        wmp = (t.get("reasoning") or {}).get("wm_pred")
        if wmp:
            step["wm_pred"] = wmp
        steps.append(step)
        prev_completed = completed
    if not already_slim:
        _dedup_contents(steps)
        _dedup_versions(steps)
    elif steps:
        steps[0].setdefault("_dedup", True)  # back-refs already present (write-time); flag for parity
    return steps


HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>ARC-AGI-3 replay — __TITLE__</title>
<style>
 /* ===== Polished research console palette ===== */
 :root{
   --bg:#0d1017; --bg-elev:#141923; --bg-panel:#0f141c; --bg-soft:#161c26;
   --line:#222c3a; --line-soft:#1a212c;
   --tx:#d2dce8; --tx-dim:#9aa7b8; --tx-faint:#6e7a89;
   --accent:#5ad1c0; --accent-dim:#2e6f67; --accent2:#7aa2ff;
   --good:#5ff0a0; --good-bg:#10301d; --warn:#ffb454; --bad:#ff7a85; --bad-bg:#3a1518;
   --gold:#ffd24a; --scribe:#6cf;
 }
 *,*::before,*::after{box-sizing:border-box}
 html,body{height:100%;min-width:0}
 body{font-family:'Inter',system-ui,-apple-system,sans-serif;margin:0;background:var(--bg);color:var(--tx);display:flex;flex-direction:column;font-size:13px;line-height:1.5;overflow:hidden;
   font-feature-settings:'tnum' 1;-webkit-font-smoothing:antialiased}
 code,pre,.mono{font-family:'JetBrains Mono','SF Mono',ui-monospace,Menlo,monospace}
 /* ===== two-tier header ===== */
 #brand{display:flex;align-items:center;gap:10px 14px;padding:9px 16px;background:linear-gradient(180deg,#141a25,#0f141d);flex-wrap:wrap;
   border-bottom:1px solid var(--line);flex:0 0 auto}
 #brand .logo{font-weight:700;font-size:16px;letter-spacing:.02em;color:#fff;white-space:nowrap}
 #brand .logo .a3{color:var(--accent)}
 #brand .tag{font-size:12px;color:var(--tx-dim);border-left:1px solid var(--line);padding-left:14px;min-width:0}
 #brand .tag select{background:var(--bg-soft);color:var(--tx);border:1px solid var(--line);border-radius:5px;padding:2px 6px;font-size:12px;font-family:inherit;cursor:pointer;max-width:min(420px,100%)}
 #brand .tag select:hover{border-color:#3a4757}
 #brand .spacer{flex:1 1 auto}
 #brand .metric{text-align:right;line-height:1.1}
 #brand .metric .num{font-size:23px;font-weight:700;color:var(--accent);font-variant-numeric:tabular-nums}
 #brand .metric .lbl{font-size:11px;color:var(--tx-dim);text-transform:uppercase;letter-spacing:.08em}
 #brand .modechip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;font-weight:600;
   background:var(--accent-dim);color:#cdeee8;border:1px solid var(--accent);letter-spacing:.02em;vertical-align:middle}
 #brand .cfgicon{cursor:pointer;font-size:13px;opacity:.7;padding:0 2px;vertical-align:middle}
 #brand .cfgicon:hover{opacity:1}
 #cfgpanel{position:fixed;top:54px;right:14px;z-index:50;width:min(480px,calc(100vw - 28px));max-height:78vh;overflow:auto;
   background:var(--bg-panel);border:1px solid var(--accent);border-radius:8px;padding:0 0 8px;
   box-shadow:0 8px 30px rgba(0,0,0,.5);font-size:12px}
 #cfgpanel .cfghd{position:sticky;top:0;background:linear-gradient(180deg,#141a25,#0f141d);padding:8px 12px;
   font-weight:600;color:#fff;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:12px;overflow-wrap:anywhere}
 #cfgpanel .cfgx{cursor:pointer;opacity:.7} #cfgpanel .cfgx:hover{opacity:1}
 #cfgpanel .cfgsub{padding:8px 12px 2px;color:var(--accent);font-weight:600;font-size:11px;letter-spacing:.03em;text-transform:uppercase}
 #cfgpanel .cfgtbl{width:100%;border-collapse:collapse}
 #cfgpanel .cfgtbl td{padding:2px 12px;vertical-align:top;border-bottom:1px solid rgba(255,255,255,.04)}
 #cfgpanel .ck{color:var(--tx-dim);white-space:nowrap;width:40%} #cfgpanel .cv{color:var(--tx);font-family:ui-monospace,monospace;word-break:break-word}
 #cfgpanel .src{font-size:9px;font-weight:700;padding:0 5px;border-radius:7px;text-transform:uppercase;letter-spacing:.04em}
 #cfgpanel .src-env{background:var(--accent-dim);color:#cdeee8}      /* explicitly set via env */
 #cfgpanel .src-file{background:#3a3168;color:#d6c8ff}              /* set by the --config file */
 #cfgpanel .src-default{background:transparent;color:var(--tx-faint)} /* left at default */
 #brand .sub{font-size:12px;color:var(--tx-dim);text-align:right;border-left:1px solid var(--line);padding-left:14px;min-width:0;overflow-wrap:anywhere}
 #brand .sub b{color:var(--tx);font-weight:600}
 #top{padding:7px 16px;background:var(--bg-elev);display:flex;gap:8px 10px;align-items:center;flex-wrap:wrap;flex:0 0 auto;border-bottom:1px solid var(--line);min-width:0}
 #top select{max-width:100%}
 /* FOUR COLUMNS: grid | transcript | file-tree | file-view */
 #wrap{display:grid;grid-template-columns:minmax(400px,576px) minmax(340px,1.2fr) minmax(190px,234px) minmax(260px,.85fr);grid-template-areas:'grid chat tree file';flex:1 1 auto;min-height:0;min-width:0;overflow:hidden}
 #col-grid{grid-area:grid;padding:16px;overflow:auto;border-right:1px solid var(--line);background:var(--bg);min-width:0}
 #wmpanel{max-width:100%}
 .wmhdr{display:flex;align-items:center;gap:8px;padding:6px 8px;background:var(--bg-soft);
   border:1px solid var(--line);border-radius:6px;flex-wrap:wrap}
 .wmacc{font-size:11px;font-weight:600;color:var(--good);background:var(--good-bg);border-radius:8px;padding:1px 7px;font-variant-numeric:tabular-nums}
 .wmok{font-size:11px;font-weight:600;color:var(--good);background:var(--good-bg);border-radius:8px;padding:1px 7px}
 .wmbad{font-size:11px;font-weight:600;color:var(--bad);background:var(--bad-bg);border-radius:8px;padding:1px 7px}
 .fchg{color:var(--gold);font-weight:700;margin-left:4px}  /* file-tree "edited this turn" marker */
 #col-chat{grid-area:chat;padding:16px;overflow:auto;min-width:0}
 #col-tree{grid-area:tree;padding:12px;overflow:auto;border-left:1px solid var(--line);background:var(--bg-panel);min-width:0}
 #col-file{grid-area:file;padding:12px;overflow:auto;border-left:1px solid var(--line);background:var(--bg-panel);min-width:0}
 canvas{display:block;width:min(576px,100%);height:auto;aspect-ratio:1;background:#05070b;image-rendering:pixelated;border:1px solid var(--line);border-radius:6px;box-shadow:0 2px 12px rgba(0,0,0,.45)}
 .canvwrap{font-size:11px;color:var(--tx-dim);text-align:center;max-width:576px;min-width:0}
 .canvwrap>div{margin-top:6px;letter-spacing:.03em}
 .k{color:var(--tx-dim)} .v{color:var(--tx);font-weight:600}
 /* the "no prediction / planning skipped" reason is a long single string — wrap it so it doesn't
    overflow the prediction-panel header row and distort the layout. */
 .noplan{display:block;white-space:normal;overflow-wrap:anywhere;margin-top:3px;line-height:1.4}
 button{background:var(--bg-soft);color:var(--tx);border:1px solid var(--line);padding:4px 10px;cursor:pointer;border-radius:5px;font-size:12px;transition:background .12s,border-color .12s}
 button:hover{background:#1f2937;border-color:#3a4757}
 button:active{background:#263244}
 select{background:var(--bg-soft);color:var(--tx);border:1px solid var(--line);border-radius:5px;padding:3px 7px;font-size:12px;font-family:inherit;cursor:pointer}
 select:hover{border-color:#3a4757}
 #bar{font-size:12px;color:var(--tx-dim);font-variant-numeric:tabular-nums}
 pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--bg-panel);padding:9px 11px;border-radius:6px;max-height:320px;overflow:auto;font-size:12px;margin:3px 0;border:1px solid var(--line-soft);line-height:1.45}
 .chip{display:inline-block;padding:1px 7px;border-radius:10px;background:var(--bg-soft);border:1px solid var(--line);margin:1px;font-size:11px}
 .sing{background:#4a3a00;color:var(--gold);border-color:#6a5300}
 label{font-size:12px;color:var(--tx-dim)}
 .panel{margin-top:13px;border-top:1px solid var(--line);padding-top:9px}
 .panel h4{margin:0 0 5px;font-size:11px;color:var(--accent);text-transform:uppercase;letter-spacing:.06em}
 #lvlbar{margin:0 4px;font-size:12px}
 #note{max-width:560px;font-size:11px;color:var(--tx-faint);margin-top:8px;line-height:1.5;overflow-wrap:anywhere}
 .win{background:var(--good-bg);color:var(--good);padding:5px 10px;border-radius:6px;margin-bottom:9px;font-weight:600;border:1px solid var(--accent-dim)}
 .lose{background:var(--bad-bg);color:var(--bad);padding:5px 10px;border-radius:6px;margin-bottom:9px;font-weight:600;border:1px solid #5a252b}
 .resetnote{background:#241f16;color:var(--warn);padding:5px 10px;border-radius:6px;margin-bottom:9px;font-weight:600;border:1px solid #5b4520}
 ol{margin:2px 0;padding-left:20px} ol li.done{color:var(--tx-faint);text-decoration:line-through}
 .obj{padding:3px 0;border-bottom:1px solid var(--line-soft);font-size:12px}
 .err{background:var(--bad-bg);color:var(--bad);padding:1px 7px;border-radius:4px;font-size:11px}
 details{margin:5px 0;background:var(--bg-panel);border:1px solid var(--line-soft);border-radius:6px;padding:5px 9px;min-width:0;overflow-wrap:anywhere}
 details>summary{cursor:pointer;font-size:12px;color:var(--accent);font-weight:600}
 details b,.lbl{display:block;margin-top:7px;color:var(--tx-dim);font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
 details pre{max-height:340px;margin:2px 0}
 .fullctx pre{max-height:600px;white-space:pre-wrap;word-break:break-word}
 .agf{cursor:pointer;padding:1px 7px;border:1px solid var(--line);border-radius:10px;font-size:11px;color:var(--tx-dim)}
 .agf.on{background:var(--accent-dim);color:var(--tx);border-color:var(--accent)}
 /* per-action chips WRAP (flex) so a CLICK game's many ACTION6(r,c) candidates don't widen the column */
 .agfwrap{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;max-width:100%;min-width:0}
 .call.bcall{border-left:3px solid var(--gold);background:rgba(255,210,74,0.04)}
 .call.scall{border-left:3px solid var(--scribe);background:rgba(102,204,255,0.04)}
 .agtag{display:inline-block;padding:1px 7px;margin-right:4px;border-radius:9px;font-size:11px;font-weight:600;border:1px solid var(--line)}
 .agtag.actor{background:var(--accent-dim);color:var(--tx)}
 .agtag.builder{background:rgba(255,210,74,0.12);color:var(--gold);border-color:#6a5300}
 .agtag.scribe{background:rgba(102,204,255,0.12);color:var(--scribe);border-color:#246}
 /* transcript */
 .call{border:1px solid var(--line-soft);border-radius:7px;margin:9px 0;padding:7px 10px;background:var(--bg-panel);min-width:0;overflow-wrap:anywhere}
 .call>.hd{font-size:11px;color:var(--accent);font-weight:600;margin-bottom:3px}
 .toolres{background:var(--good-bg);border-left:3px solid var(--accent-dim);padding:5px 9px;margin:3px 0;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;border-radius:0 5px 5px 0}
 .turnsep{margin:14px 0 4px;border-top:2px solid var(--accent-dim);padding-top:5px;color:var(--accent2);font-weight:600;font-size:13px}
 /* workspace file browser (tree col + file col) */
 .ftree{font-size:12px;line-height:1.75}
 .frow{cursor:pointer;padding:1px 5px;border-radius:4px;white-space:normal;overflow-wrap:anywhere;word-break:break-word}
 .frow:hover{background:var(--bg-soft)} .frow.sel{background:#1c3a36;color:var(--accent)}
 .fdir{color:var(--accent2);cursor:pointer;font-weight:600} .fdir:hover{background:var(--bg-soft)}
 .shortcut{display:inline-block;margin:2px 4px 6px 0;padding:2px 9px;background:#1c3a36;color:var(--accent);border-radius:10px;cursor:pointer;font-size:11px;border:1px solid var(--accent-dim)}
 .shortcut:hover{background:#234c46}
 #col-file pre{max-height:none}
 ::-webkit-scrollbar{width:10px;height:10px} ::-webkit-scrollbar-track{background:transparent}
 ::-webkit-scrollbar-thumb{background:#28323f;border-radius:6px;border:2px solid var(--bg)} ::-webkit-scrollbar-thumb:hover{background:#36424f}
 /* per-game progress strip in the brand bar */
 #gamestrip{display:flex;gap:3px;align-items:center;border-left:1px solid var(--line);padding-left:14px;flex-wrap:wrap;min-width:0}
 .gpip{width:8px;height:8px;border-radius:2px;background:#26303d;cursor:pointer}
 .gpip.cur{outline:1.5px solid var(--accent);outline-offset:1px}
 .gpip.live{background:var(--gold);border-radius:50%;animation:pulse 1.4s ease-in-out infinite}
 @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
 .livebadge{color:var(--gold);font-weight:600;font-size:11px;margin-left:8px;animation:pulse 1.4s ease-in-out infinite}
 #bar,#lvlbar,#runstats,#brandmeta,#brandsub,.panel,.obj,li{min-width:0;overflow-wrap:anywhere}
 @media(max-width:1600px){
   #wrap{grid-template-columns:minmax(360px,576px) minmax(340px,1fr);grid-template-rows:minmax(610px,auto) minmax(240px,auto);grid-template-areas:'grid chat' 'tree file';overflow:auto}
   #col-tree,#col-file{border-top:1px solid var(--line)}
 }
 @media(max-width:900px){
   html,body{height:auto;min-height:100%}
   body{overflow:auto}
   #brand .spacer{display:none}
   #brand .sub{text-align:left;border-left:0;padding-left:0;flex-basis:100%}
   #wrap{display:block;overflow:visible}
   #col-grid,#col-chat,#col-tree,#col-file{overflow:visible;border:0;border-bottom:1px solid var(--line);padding:12px;min-width:0}
   #col-tree,#col-file{background:var(--bg-panel)}
 }
 @media(max-width:560px){
   #brand,#top{padding-left:10px;padding-right:10px}
   #brand .tag{border-left:0;padding-left:0;flex-basis:100%}
   #brand .tag select{width:100%}
   #brand .metric{text-align:left}
   #gamestrip{border-left:0;padding-left:0;flex-basis:100%}
   #cfgpanel{top:8px;right:8px;width:calc(100vw - 16px);max-height:calc(100vh - 16px)}
 }
</style></head><body>
<div id="brand">
 <span class="logo"><span class="a3">ARC-AGI-3</span> · Tycho Freeform</span>
 <span class="tag">run <select id="run" onchange="loadRun()"></select></span>
 <span class="tag" id="brandmeta"></span>
 <span id="gamestrip"></span>
 <span class="spacer"></span>
 <span class="sub" id="brandsub"></span>
 <span class="metric"><div class="num" id="brandrhae">—</div><div class="lbl">mean RHAE</div></span>
</div>
<div id="top">
 <button onclick="step(-10)">⏪ 10</button><button onclick="step(-1)">◀</button>
 <span id="bar"></span>
 <button onclick="step(1)">▶</button><button onclick="step(10)">10 ⏩</button>
 <button onclick="play()">▶ play</button>
 <button onclick="jumpLevel(-1)">◀ lvl</button><button onclick="jumpLevel(1)">lvl ▶</button>
 <span id="lvlbar"></span>
 <span id="layers"></span>
 <select id="game" onchange="loadGame()"></select>
 <span id="runstats" class="k"></span>
</div>
<div id="wrap">
 <div id="col-grid">
  <div class="canvwrap"><canvas id="cv" width="576" height="576"></canvas></div>
  <div class="canvwrap" id="predwrap" style="display:none;margin-top:10px"><canvas id="cvp" width="576" height="576"></canvas>
   <div>agent's predicted frame (red = mismatch vs actual)</div></div>
  <!-- WORLD-MODEL prediction overlay (--wm-predict): plan-to-level-complete trajectory scrubber (Case A)
       or per-action prediction picker (Case B). Hidden unless the step carries s.wm_pred. -->
  <div class="canvwrap" id="wmwrap" style="display:none;margin-top:10px">
   <div id="wmctl" class="lbl" style="text-transform:none"></div>
   <canvas id="cvw" width="576" height="576"></canvas>
   <div id="wmcap" class="k" style="font-size:11px;margin-top:4px;white-space:normal;overflow-wrap:anywhere;max-width:576px;line-height:1.4"></div></div>
  <!-- Executable-model verification and planning diagnostics for this step. -->
  <div id="wmpanel" style="margin-top:12px"></div>
 </div>
 <div id="col-chat"></div>
 <div id="col-tree"></div>
 <div id="col-file"></div>
</div>
<script>
// LAZY-LOADED + MULTI-RUN: the shell embeds RUNS = {runId: manifest[]}. Each run's game
// data lives in <runId>/game_<id>.js, <script>-loaded on demand (works under file:// too).
// Game ids collide across runs (every run shares the same id set), so data is namespaced by run:
// window.GAMES[runId][gameId]. The run <select> switches the active manifest.
// RUNS = {runId: {games: manifest[], meta: {model, effort, run_time}}}.
const RUNS = __RUNS__, PAL = __PAL__, UNKNOWN = -1;
// runs sorted NEWEST-FIRST by finish time (meta.run_time = the run dir's mtime) so the most
// recent run is at the top of the dropdown and is the default active run.
const RUNIDS = Object.keys(RUNS).sort((a,b)=>((RUNS[b].meta||{}).run_time||'').localeCompare((RUNS[a].meta||{}).run_time||''));
const MAN = (rid)=>RUNS[rid].games;             // games manifest for a run
const META = (rid)=>RUNS[rid].meta||{};         // run-level header info
// Prefer the per-call reasoning kind recorded by the transport. Older records did not capture this
// field, so their text is conservatively labelled as a summary instead of guessing from provider or
// model names.
function reasoningKind(opts){
  if(opts && opts.kind) return opts.kind;
  return 'summary';
}
const KIND_LABEL = {trace:'reasoning', summary:'reasoning summary', none:'reasoning'};
window.GAMES = {};                              // runId -> {gameId -> steps}
let R = RUNIDS[0];                              // active run
let MANIFEST = MAN(R);                          // active run's games (re-pointed on run switch)
let G = MANIFEST[0].id, i = 0, timer = null;
function DATA(){return (window.GAMES[R]||{})[G]||[];}   // current run+game's steps

// ---- CANVAS LAYER REGISTRY: each draws on the main grid; toggleable. ----
// (Tycho agents form their own beliefs, so the CV "scene/parser lens" overlay and agent-
// highlight layers were no-ops here and were dropped. Kept: the ACTION6 click marker and
// the cell grid lines, which are the two overlays that actually apply to Tycho replays.)
const LAYERS = [
 {id:'gridlines', label:'grid lines', on:true, fn:()=>{}},  // toggles faint cell lines in drawGrid
 {id:'action', label:'action marker', on:true, fn:(ctx,s,CELL)=>{
    if(s.action==='ACTION6'&&s.x!=null){ctx.strokeStyle='#ff2d55';ctx.lineWidth=2;
     ctx.beginPath();ctx.arc(s.x*CELL+CELL/2,s.y*CELL+CELL/2,CELL,0,7);ctx.stroke();}}},
 // hovered per-action click cell (set on chip mouseenter): highlight WHERE a candidate ACTION6 clicks
 // on the live grid. Reuses the draw() pass — no new render path. Always on; no-op when not hovering.
 {id:'hovercell', label:'hover click cell', on:true, fn:(ctx,s,CELL)=>{
    const h=window._hoverClick; if(!h||h.row==null||h.col==null) return;
    ctx.fillStyle='rgba(122,162,255,0.45)'; ctx.fillRect(h.col*CELL,h.row*CELL,CELL,CELL);
    ctx.strokeStyle='#7aa2ff'; ctx.lineWidth=2; ctx.strokeRect(h.col*CELL,h.row*CELL,CELL,CELL);}},
];
function hoverClick(row,col){window._hoverClick=(row!=null&&col!=null)?{row,col}:null; draw();}

// ---- PANEL REGISTRY: each returns side-panel HTML for a step (or ''). ----
function isGrid(v){return Array.isArray(v)&&Array.isArray(v[0]);}
// keys rendered by dedicated panels below; everything else -> "(other)" JSON.
const KNOWN_KEYS = new Set(['predicted_frame','world_model_code','code','plan','goal','outcome',
  'goal_hypothesis','goal_confidence','outcome_hypothesis','outcome_confidence','cells_highlight','objects','mechanics','stats','llm_calls',
  't','calls','src','planner_error',
  'tool_choice','tracker','cognitive_trace','predicted_effect',
  'tool_trace','beliefs','workspace']);                           // Tycho
const PANELS = [
 s=>s.just_completed?`<div class="win">✓ LEVEL ${s.level+1} COMPLETED (observable goal-reached)</div>`:'',
 s=>s.game_over?`<div class="lose">✕ GAME OVER after ${esc(s.action)} — terminal frame; automatic RESET follows</div>`:'',
 s=>s.auto_reset?`<div class="resetnote">↻ automatic RESET after GAME_OVER — this frame is a fresh playable state, not ordinary action dynamics</div>`:'',
 // frame shown is the RESULT of the action (harness records post-action); label it so
 // turn 1 doesn't read as "an action already happened on the initial frame".
 s=>`<div><span class="k">level</span> <span class="v">${s.level+1}</span>  <span class="k">frame after</span> <span class="v">${s.action}</span>`
    +(s.x!=null?` @(${s.x},${s.y})`:'')+`  <span class="k">state</span> <span class="v">${s.state.replace('GameState.','')}</span>`
    +`  <span class="k">changed</span> <span class="v">${s.changed}</span>`
    +(s.reasoning&&s.reasoning.planner_error?` <span class="err">planner_error</span>`:'')+`</div>`,
 // GOAL / OUTCOME hypothesis (+ confidence)
 s=>{const r=s.reasoning||{}; const g=r.outcome_hypothesis||r.goal||r.goal_hypothesis; if(!g)return'';
     const c=(r.outcome_confidence!=null||r.goal_confidence!=null)
       ?` <span class="k">conf</span> ${r.outcome_confidence!=null?r.outcome_confidence:r.goal_confidence}`:'';
     return `<div class="panel"><h4>${r.outcome_hypothesis?'outcome':'goal'} hypothesis${c}</h4>${typeof g==='string'?esc(g):'<pre>'+esc(JSON.stringify(g,null,1))+'</pre>'}</div>`;},
 // OBJECT HYPOTHESES (the agent's OWN — first-class, not the parser lens)
 s=>{const o=s.reasoning&&s.reasoning.objects; if(!o||!o.length)return'';
     let h=`<div class="panel"><h4>object hypotheses (agent)</h4>`;
     for(const x of o.slice(0,30)) h+=`<div class="obj"><b>${esc(x.id||'')}</b> <span class="k">${esc(x.role||'')}</span> `
       +`c${JSON.stringify(x.colors||[])} <span class="k">@${esc(x.where||'')}</span> ${esc(x.note||'')}</div>`;
     return h+`</div>`;},
 // MECHANICS + STATS
 s=>{const r=s.reasoning||{}; if(!r.mechanics&&!r.stats)return''; let h=`<div class="panel"><h4>mechanics + stats</h4>`;
     if(r.mechanics)h+=`<div>${esc(r.mechanics)}</div>`;
     if(r.stats&&Object.keys(r.stats).length)h+=`<pre>${esc(JSON.stringify(r.stats,null,1))}</pre>`;
     return h+`</div>`;},
 // PLAN
 s=>{const p=s.reasoning&&s.reasoning.plan; if(!p||!p.length)return'';
     return `<div class="panel"><h4>plan (next ${p.length})</h4><ol>`+p.map(x=>`<li>${esc(typeof x==='string'?x:JSON.stringify(x))}</li>`).join('')+`</ol></div>`;},
 // WRITTEN PLANNER CODE
 s=>{const c=s.reasoning&&(s.reasoning.world_model_code||s.reasoning.code); if(!c)return'';
     return `<div class="panel"><h4>written planner code</h4><pre>${esc(c)}</pre></div>`;},
 // LLM CALLS (full I/O this step — collapsible; only present with --viz)
 s=>{const calls=s.reasoning&&s.reasoning.llm_calls; if(!calls||!calls.length)return'';
     let h=`<div class="panel"><h4>LLM calls this step (${calls.length})</h4>`;
     calls.forEach((c,i)=>{h+=`<details><summary>${esc(c.call_type||'call')} `
       +`<span class="k">${esc(c.model||'')}${c.effort?' · effort '+esc(c.effort):''}</span></summary>`
       +(c.reasoning_summary?`<b>reasoning summary</b><pre>${esc(c.reasoning_summary)}</pre>`:'')
       +`<b>prompt</b><pre>${esc(c.prompt||'')}</pre>`
       +`<b>response</b><pre>${esc(c.response||'')}</pre></details>`;});
     return h+`</div>`;},
 // ORCHESTRATOR TOOL CHOICE: which tool, rationale, and target uncertainty.
 s=>{const tc=s.reasoning&&s.reasoning.tool_choice; if(!tc)return'';
     return `<div class="panel"><h4>orchestrator → ${esc(tc.tool||'')}</h4>`
       +(tc.target_uncertainty?`<div><span class="k">targeting</span> ${esc(tc.target_uncertainty)}</div>`:'')
       +(tc.rationale?`<div><i>${esc(tc.rationale)}</i></div>`:'')
       +(s.reasoning.predicted_effect?`<div><span class="k">predicts</span> ${esc(s.reasoning.predicted_effect)}</div>`:'')
       +`</div>`;},
 // BELIEF TRACKER: natural-language journal entries or a structured evidence ledger.
 s=>{const t=s.reasoning&&s.reasoning.tracker; if(!t||t.kind==='none')return'';
     if(t.kind==='nl_journal'){let h=`<div class="panel"><h4>belief journal (${t.n_total||0} notes)</h4>`;
       for(const e of (t.entries||[]).slice(-20)) h+=`<div class="obj"><span class="k">t${e.turn}</span> ${esc(e.text||'')}</div>`;
       return h+`</div>`;}
     if(t.kind==='evidence_ledger'){let h=`<div class="panel"><h4>evidence ledger</h4>`;
       for(const x of (t.hypotheses||[]).slice(0,30)){const c=x.confidence!=null?` ${Math.round(x.confidence*100)}%`:'';
         h+=`<div class="obj ${x.status==='retired'?'k':''}"><b>${esc(x.id||'')}</b><span class="k">[${esc(x.uncertainty||'')}]${c}</span> `
           +`${esc(x.statement||'')} <span class="k">(${x.n_evidence||0} ev)</span></div>`;}
       return h+`</div>`;}
     return '';},
 // COGNITIVE TRACE: intermediate tool use before the committed action.
 s=>{const tr=s.reasoning&&s.reasoning.cognitive_trace; if(!tr||!tr.length)return'';
     let h=`<div class="panel"><h4>cognitive trace (${tr.length} steps before acting)</h4><ol>`;
     for(const c of tr){const a=c.args?' '+esc(JSON.stringify(c.args)).slice(0,80):'';
       h+=`<li><b>${esc(c.tool||'')}</b>${a}${c.committed?' <span class="k">→ committed</span>':''}`
         +(c.result?`<br><span class="k">${esc(c.result)}</span>`:'')+`</li>`;}
     return h+`</ol></div>`;},
 // Durable actor beliefs captured in notes/actor_beliefs.md.
 s=>{const b=s.reasoning&&s.reasoning.beliefs; if(!b||!b.trim())return'';
     return `<div class="panel"><h4>actor beliefs (notes/actor_beliefs.md)</h4><pre>${esc(b)}</pre></div>`;},
 // (other) — genuinely unhandled keys only
 s=>{if(!s.reasoning)return `<div class="panel k"><i>no agent reasoning (model-free agent)</i></div>`;
     const rest={}; for(const k in s.reasoning) if(!KNOWN_KEYS.has(k)) rest[k]=s.reasoning[k];
     if(!Object.keys(rest).length)return'';
     return `<div class="panel"><h4>other</h4><pre>${esc(JSON.stringify(rest,null,1))}</pre></div>`;},
];
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function attr(t){return esc(t).replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
window._argMap = window._argMap || {}; window._argSeq = window._argSeq || 0;
function bindArg(v){const id=++window._argSeq; window._argMap[id]=String(v); return id;}
function argVal(id){return window._argMap[id] || '';}
function fmtTok(n){n=n||0; return n>=1000?(n/1000).toFixed(n>=10000?0:1)+'k':String(n);}
function promptHtml(text, s){  // The prompt's [image] marker stays a plain marker:
 // the actual frame is the grid on the canvas (← left panel), and the call header notes
 // "📎 frame image attached", so re-embedding the thumbnail here is redundant.
 return `<pre>${esc(text)}</pre>`;
}

// ---- CENTER COLUMN: sequential append-style model transcript. One block per
// LLM call in order — prompt -> reasoning -> response (with tool calls shown inline
// in the response). Tool RESULTS appear in the NEXT call's prompt (that's how the
// append loop works), so we don't duplicate them in a separate "flow" panel. ----
// Reconstruct the FULL rolling conversation the model had accumulated by the END of step
// `upto`. The recorder stores per-call deltas only; the real sent prompt = system prompt +
// every (delta -> response) pair in order. We replay all steps 0..upto and, within each,
// all llm_calls, emitting the captured delta then the assistant response. This mirrors the
// append-only self.history the agent maintains (minus the wire-time heavy-payload elision).
function fullContextUpTo(upto){
 const D=DATA(); if(!D.length) return '(no data)';
 // system prompt once at the top (captured on the run's first call)
 let sysp='';
 outer: for(const st of D){ for(const c of ((st.reasoning&&st.reasoning.llm_calls)||[])){ if(c.system_prompt){sysp=c.system_prompt; break outer;} } }
 const parts=[];
 if(sysp) parts.push('========== SYSTEM PROMPT ==========\n'+sysp);
 // The agent WIPES self.history at every level boundary (_reset_history_for_new_level) — each
 // level is faced fresh, carry-forward lives on disk (notes/). So the real context at step `upto`
 // starts at the most-recent level boundary (turn_in_level===0), NOT at k=0. Find it.
 let start=0;
 for(let k=Math.min(upto,D.length-1);k>=0;k--){
   const r=D[k].reasoning||{};
   if(r.turn_in_level===0){ start=k; break; }
 }
 if(start>0) parts.push(`\n(history cleared at the level boundary — context below is level ${(D[start].reasoning||{}).level} only; prior levels' carry-forward is on disk in notes/)`);
 let n=0;
 for(let k=start;k<=upto && k<D.length;k++){
   const st=D[k]; const cs=(st.reasoning&&st.reasoning.llm_calls)||[];
   for(const c of cs){
     n++;
     const tag=`turn ${st.turn} · call ${n}${c.call_type?' ('+c.call_type+')':''}`;
     parts.push(`\n———————————— ${tag} ————————————\n[INPUT DELTA]\n`+(c.prompt||'(none)'));
     const resp=(c.response||'').trim();
     if(resp) parts.push('[ASSISTANT RESPONSE]\n'+resp);
   }
 }
 return parts.join('\n');
}

// The system prompt that REALLY governed a call. Newer runs record it per-call (entry
// .system_prompt). Older runs (pre-fix) attached ONE batch prompt to the first call, which was
// whichever set_system_prompt() ran last (the builder's) — so an actor call wrongly carries the
// BUILDER prompt. Reconstruct from call_type: builder calls → builder prompt, others → actor.
function sysPromptFor(calls, c){
 // return a per-call prompt of the RIGHT kind for this call's agent role, or null. NEVER
 // return a builder prompt for an actor call (or vice-versa) — that mislabeling was the bug.
 const isB=c.call_type==='builder';
 const wantB=p=>/WORLD-MODEL BUILDER/.test(p||'');
 for(const x of calls){ const p=x.system_prompt; if(p && wantB(p)===isB) return p; }
 return null;   // none of the right kind recorded this turn (legacy data) → caller shows a note
}
function renderTranscript(s){
 const calls=(s.reasoning&&s.reasoning.llm_calls)||[];
 if(!calls.length) return '';
 let h='';
 // Actor system prompt + tool definitions are collapsible. Show only a genuine actor (non-
 // builder) prompt here; never fall back to the builder prompt (that bug made every actor turn
 // look like a BUILDER prompt). Legacy runs that didn't record an actor prompt → show a note.
 const actorSys=calls.find(c=>c.system_prompt && !/WORLD-MODEL BUILDER/.test(c.system_prompt));
 // tools are recorded on the first call of EACH agent role — pick the actor's (non-builder) and
 // the builder's separately so BOTH tool sets are inspectable (the builder's was invisible before).
 const actorTools=(calls.find(c=>c.tools && c.call_type!=='builder')||{}).tools;
 const builderTools=(calls.find(c=>c.tools && c.call_type==='builder')||{}).tools;
 const toolsBlock=(label, tools)=>{
   if(!tools) return '';
   let s=`<span class="lbl">${label} tools (${tools.length}) — structured definitions sent to the model</span>`;
   for(const t of tools){const wire={name:t.name, description:t.description, inputSchema:t.schema||null};
     s+=`<details><summary>${esc(t.name)}</summary><pre>${esc(JSON.stringify(wire,null,2))}</pre></details>`;}
   return s;
 };
 // Per-turn instrumentation: sum latency and tokens across this turn's calls.
 const tot=calls.reduce((a,c)=>({ms:a.ms+(c.latency_ms||0),ti:a.ti+(c.tokens_in||0),to:a.to+(c.tokens_out||0)}),{ms:0,ti:0,to:0});
 const secs=(tot.ms/1000).toFixed(1);
 // PER-TURN AGENT BANNER (orchestration framework): which AGENTS were active this turn. Derived
 // from the call_type set (freeform/builder/level_summary) — actor always; 🔧 builder if it fired
 // (orchestrator pull or harness trigger); 📝 scribe on level-completion turns. builder_runs (emitted
 // by the harness) carries the trigger reason when present.
 const ctypes=new Set(calls.map(c=>c.call_type||'freeform'));
 const nBcall=calls.filter(c=>c.call_type==='builder').length;
 const nScribe=calls.filter(c=>c.call_type==='level_summary').length;
 const bruns=(s.reasoning&&s.reasoning.builder_runs)||[];
 const agentParts=[`<span class="agtag actor">actor</span>`];
 if(nBcall||bruns.length) agentParts.push(`<span class="agtag builder">🔧 builder ×${nBcall||bruns.length}</span>`);
 if(nScribe) agentParts.push(`<span class="agtag scribe">📝 scribe (level summary)</span>`);
 h+=`<div class="lbl" style="text-transform:none">agents active this turn: ${agentParts.join(' ')}`
   +(bruns.length&&bruns[0].reason?` <span class="k">— builder fired: ${esc((bruns[0].reason||'').slice(0,80))}</span>`:'')+`</div>`;
 // AGENT FILTER + BUILDER NAVIGATION (orchestrator runs): the actor and the world-model
 // BUILDER subagent (call_type=='builder') interleave in the transcript, but the builder is
 // invoked RARELY (a few turns per game), so its turns are hard to find by scrubbing. This
 // control is ALWAYS shown for runs that ever used a builder, with ◀▶ jumps to builder turns.
 const nB=calls.filter(c=>c.call_type==='builder').length;
 const builderTurns=DATA().map((st,j)=>({j,has:((st.reasoning&&st.reasoning.llm_calls)||[]).some(c=>c.call_type==='builder')})).filter(x=>x.has).map(x=>x.j);
 if(builderTurns.length){
   const F=window._agentF||'all';
   const btn=(v,lab)=>`<span class="agf ${F===v?'on':''}" onclick="setAgentF('${v}')">${lab}</span>`;
   const prevB=builderTurns.filter(j=>j<i).pop(), nextB=builderTurns.find(j=>j>i);
   const jb=(j,lab)=> j!=null?`<span class="agf" onclick="setStep(${j})">${lab}</span>`:`<span class="agf" style="opacity:.4">${lab}</span>`;
   const hasScribe=DATA().some(st=>((st.reasoning&&st.reasoning.llm_calls)||[]).some(c=>c.call_type==='level_summary'));
   h+=`<div class="lbl" style="text-transform:none">agent view: ${btn('all','all')}${btn('actor','actor')}${btn('builder','🔧 builder')}${hasScribe?btn('scribe','📝 scribe'):''}`
     +`  ·  builder invoked on ${builderTurns.length} turn(s): ${jb(prevB,'◀ prev')}${jb(nextB,'next ▶')}`
     +(nB?` <b style="color:var(--gold)">— THIS turn has ${nB} builder call(s)</b>`
         :` <span class="k">— not this turn; jump to one</span>`)+`</div>`;
 }
 // System prompts + tool definitions (per agent). Placed AFTER the agent banner so it's clear
 // these refer to BOTH agents — actor's prompt + tools and (if used) builder's prompt + tools.
 // Switching the 'agent view' filter above changes the transcript filter below, but the prompts
 // shown here always include both — they're static per-run, not per-turn.
 const sysp=(actorSys||{}).system_prompt;
 if(sysp||actorTools||builderTools){
   h+=`<details><summary>system prompt & tools (per agent — actor + builder)</summary>`;
   if(sysp) h+=`<span class="lbl">actor system prompt</span><pre>${esc(sysp)}</pre>`;
   else h+=`<span class="lbl">actor system prompt</span><div class="k" style="font-size:11px">(actor prompt not recorded in this run — see per-call prompts below; builder prompt shown on 🔧 builder calls)</div>`;
   h+=toolsBlock('actor', actorTools);
   h+=toolsBlock('🔧 builder', builderTools);   // the world-model subagent's toolset (when invoked this turn)
   h+=`</details>`;
 }
 h+=`<div class="lbl">LLM transcript this turn (${calls.length} call${calls.length>1?'s':''}`
   +(nB?` · <b style="color:var(--gold)">${nB} builder</b>`:'')
   +` · ${secs}s · ${fmtTok(tot.ti)} in / ${fmtTok(tot.to)} out)</div>`;
 // FULL ROLLING CONTEXT (on demand): col-2 normally shows only each call's prompt DELTA
 // (the new user/tool input since the last assistant turn) — the actual sent prompt is the
 // whole growing conversation. Reconstruct it from captured deltas+responses so the full
 // context the model accumulated through THIS turn is inspectable for prompt-design work.
 // NOTE: this is the LOGICAL conversation; at send time _truncate_old_grids strips heavy
 // payloads (images, grid-embeds, big text >1500 chars) older than TYCHO_GRID_KEEP turns, so
 // the real wire prompt is SMALLER for old frames. Use this to study content/flow; the
 // per-call token counts above reflect what was actually billed.
 h+=`<details class="fullctx"><summary>▸ full rolling context sent this turn (reconstructed — click)</summary>`
   +`<div class="k" style="font-size:11px;margin:4px 0">Concatenation of every prior prompt-delta + response (the append-only history). `
   +`Pre-truncation: old frames appear in full here, but on the wire heavy payloads (images, big grids) older than the keep-window (~3 turns) are elided.</div>`
   +`<pre>${esc(fullContextUpTo(i))}</pre></details>`;
 const AF=window._agentF||'all';
 // three-way agent filter: builder (call_type=='builder'), scribe (=='level_summary'), actor (rest).
 const roleOf=c=>c.call_type==='builder'?'builder':(c.call_type==='level_summary'?'scribe':'actor');
 const keep=c=>AF==='all'||roleOf(c)===AF;
 calls.forEach((c,idx)=>{
   if(!keep(c)) return;            // agent filter (actor / builder / scribe)
   const isB=c.call_type==='builder';
   const isScribe=c.call_type==='level_summary';
   // Label by the reasoning kind recorded with this call; do not infer it from provider names.
   const rkind=reasoningKind({kind:c.reasoning_kind});
   const rlabel=KIND_LABEL[rkind]||'reasoning';
   const meta=[]; if(c.latency_ms!=null)meta.push((c.latency_ms/1000).toFixed(1)+'s');
   if(c.tokens_in||c.tokens_out)meta.push(fmtTok(c.tokens_in||0)+'→'+fmtTok(c.tokens_out||0)+' tok');
   // builder calls get a gold left-border + 🔧 tag so the subagent's work is visually obvious.
   h+=`<div class="call${isB?' bcall':''}${isScribe?' scall':''}"><div class="hd">▸ call ${idx+1}: `
      +(isB?`<span style="color:var(--gold)">🔧 ${esc(c.call_type)} (world-model subagent)</span> `
       :isScribe?`<span style="color:var(--scribe,#6cf)">📝 scribe (boundary consolidation)</span> `
       :`${esc(c.call_type||'call')} `)
     +`<span class="k">${esc(c.model||'')}${c.effort?' · effort '+esc(c.effort):''}${meta.length?' · '+meta.join(' · '):''}</span>`
     +(c.has_image?` <span class="k">📎 frame image attached (shown in the grid panel ←)</span>`:``)+`</div>`;
   // show the agent's OWN system prompt (collapsed) — for builder calls this is the BUILDER
   // prompt, for actor calls the actor prompt. Reconstructed from call_type for legacy runs.
   const sp=sysPromptFor(calls,c);
   if(sp) h+=`<details><summary>${isB?'🔧 builder':'actor'} system prompt</summary><pre>${esc(sp)}</pre></details>`;
   // Keep the [image] marker in the prompt; the corresponding frame is visible in the grid panel.
   h+=`<span class="lbl">prompt</span>`+promptHtml(c.prompt||'(not recorded)', s);
   const rtext=(c.reasoning_text!=null?c.reasoning_text:c.reasoning_summary)||'';  // new field, old fallback
   if(rtext.trim()) h+=`<span class="lbl">${rlabel}</span><pre>${esc(rtext)}</pre>`;
   else h+=`<span class="lbl">${rlabel}</span><div class="k" style="font-size:11px">(none this call)</div>`;
   h+=`<span class="lbl">response (tool calls shown as name(args))</span><pre>${esc(c.response||'')}</pre></div>`;
 });
 const tr=(s.reasoning&&s.reasoning.tool_trace)||[];
 const act=tr.find(x=>x.committed);
 if(act) h+=`<div class="toolres"><b>→ committed action:</b> ${esc(JSON.stringify(act.args||{}))}</div>`;
 return h;
}

// ---- COL 3: level-nested file tree with collapsible folders and current-grid/
// current-diff shortcuts. Click a file -> shown in col 4. ----
window._wsOpen = window._wsOpen || {};      // folder -> open?
function _fileChangedThisStep(path){  // Did `path` differ from the previous step?
 const D=DATA(); if(i<=0) return false;
 const curVersions=_wsVersions(D[i]), prevVersions=_wsVersions(D[i-1]);
 if(Object.keys(curVersions).length||Object.keys(prevVersions).length){
   if(!(path in curVersions)) return false;
   return JSON.stringify(curVersions[path])!==JSON.stringify(prevVersions[path]||null);
 }
 const cur=_wsContents(D[i]), prev=_wsContents(D[i-1]);
 if(!(path in cur)) return false;
 const a=resolveContent(path, cur[path]); const b=(path in prev)?resolveContent(path, prev[path]):null;
 return a!==b;
}
function renderTree(s){
 const w=s.reasoning&&s.reasoning.workspace;
 if(!w||!w.files) return '<div class="k" style="font-size:12px">workspace (Tycho only)</div>';
 // did the BUILDER run this turn? (orchestrator pull or harness trigger) — attribute world_model.py edits.
 const builderRan=((s.reasoning&&s.reasoning.llm_calls)||[]).some(c=>c.call_type==='builder')
                  ||((s.reasoning&&s.reasoning.builder_runs)||[]).length>0;
 // Shortcuts to the current grid and diff; synthetic paths are handled by selFile.
 const warnings=Array.isArray(w.snapshot_warnings)?w.snapshot_warnings:[];
 let h=`<div class="lbl">workspace · level ${w.level} / turn ${w.turn}`
   +(warnings.length?` <span class="bad" title="${attr(warnings.join('\n'))}">· ${warnings.length} snapshot warning${warnings.length===1?'':'s'}</span>`:'')
   +`</div>`;
 h+=`<span class="shortcut" onclick="selSpecial('grid')">▦ current grid</span>`
   +`<span class="shortcut" onclick="selSpecial('diff')">Δ current diff</span>`;
 const tree={};
 for(const f of w.files){let cur=tree; const parts=f.split('/');
   parts.forEach((p,j)=>{ if(j===parts.length-1){ (cur.__files=cur.__files||[]).push({name:p,path:f}); }
     else { cur=cur[p]=cur[p]||{}; } });}
 function render(node,path){let out='';
   for(const k in node){ if(k==='__files')continue;
     const fp=path?path+'/'+k:k;
     const isOpen=window._wsOpen[fp]===true;   // default collapsed; toggleDir opens
     out+=`<div class="frow fdir" onclick="toggleDirArg(${bindArg(fp)})">${isOpen?'▾':'▸'} ${esc(k)}/</div>`;
     if(isOpen) out+=`<div style="margin-left:12px">${render(node[k],fp)}</div>`;}
   for(const f of (node.__files||[])){ const sel=(window._wsSel===f.path)?'sel':'';
     const changed=_fileChangedThisStep(f.path);
     const chg=changed?`<span class="fchg" title="edited this turn (content differs from the previous step)">✎</span>`:'';
     // attribute a world_model.py edit to the builder when the builder ran this turn (orchestrator/trigger)
     const byB=(changed && builderRan && /(^|\/)world_model\.py$/.test(f.path))
       ?`<span class="fchg" style="color:var(--gold)" title="the world-model builder ran this turn and world_model.py changed → builder-edited">🔧</span>`:'';
     out+=`<div class="frow ${sel}" onclick="selFileArg(${bindArg(f.path)})">📄 ${esc(f.name)}${chg}${byB}</div>`;}
   return out;}
 // The snapshot tree above is the AUTHORED files at THIS turn. Below it, the FULL on-disk
 // workspace (notes/ + every level_<L>/turn_*.txt + diffs) — lazy-loaded from the durable ws dir
 // via serve.py, so cross-level browsing works even though per-frame files are pruned from the
 // record (O(turns²) bloat). Static builds (no server) just won't expand it; that's fine.
 return h+`<div class="ftree">${render(tree,'')}</div>`+renderDiskTree();
}
// ON-DISK workspace browser. Lazy-fetches /<run>/ws/<game>/tree.json once per game; clicking a
// file selects '__disk:<rel>' which renderFile fetches via the file endpoint. window._diskTree
// caches per "run/game". A leading collapsible row keeps it out of the way until clicked.
window._diskTree = window._diskTree || {};   // "run/game" -> tree | 'loading' | 'none'
function _diskKey(){return R+'/'+G;}
function renderDiskTree(){
 const key=_diskKey(); const t=window._diskTree[key];
 const open=window._wsOpen['__disk']===true;
 let head=`<div class="frow fdir" style="margin-top:8px;border-top:1px solid var(--line);padding-top:6px" `
   +`onclick="toggleDiskTree()">${open?'▾':'▸'} 📂 full workspace (on disk)</div>`;
 if(!open) return head;
 if(t===undefined){ loadDiskTree(); return head+`<div class="k" style="font-size:11px;margin-left:12px">loading…</div>`; }
 if(t==='loading') return head+`<div class="k" style="font-size:11px;margin-left:12px">loading…</div>`;
 if(t==='none'||!t) return head+`<div class="k" style="font-size:11px;margin-left:12px">(no durable workspace on disk for this run)</div>`;
 function render(node,path){let out='';
   for(const k in node){ if(k==='__files__')continue;
     const fp='__disk/'+(path?path+'/'+k:k); const isOpen=window._wsOpen[fp]===true;
     out+=`<div class="frow fdir" onclick="toggleDirArg(${bindArg(fp)})">${isOpen?'▾':'▸'} ${esc(k)}/</div>`;
     if(isOpen) out+=`<div style="margin-left:12px">${render(node[k],path?path+'/'+k:k)}</div>`;}
   for(const nm of (node.__files__||[])){ const rel=(path?path+'/':'')+nm;
     const ssel=('__disk:'+rel===window._wsSel)?'sel':'';
     out+=`<div class="frow ${ssel}" onclick="selDiskArg(${bindArg(rel)})">📄 ${esc(nm)}</div>`;}
   return out;}
 return head+`<div class="ftree" style="margin-left:12px">${render(t,'')}</div>`;
}
// Config panel: a floating popover listing every recorded config variable for the active run
// (orchestration mode, backend, context/caching config, vision profile, git, hardware, throughput).
// Source = META(R).manifest (the full manifest minus the per-game lists), passed through by serve.py.
function toggleConfig(ev){ if(ev)ev.stopPropagation();
 let el=document.getElementById('cfgpanel');
 if(el){ el.remove(); return; }
 const mt=META(R); const man=mt.manifest||{};
 let body='';
 // (a) run_config: stable policy fields + provenance (env|file|default).
 // Rendered first, grouped by section, with a colored source badge so you can see at a glance which
 // values were explicitly set vs left at default. This is what answers "what config shipped?".
 const rc=man.run_config;
 if(rc&&rc.by_section){
   const badge=(s)=>`<span class="src src-${s}" title="${s==='env'?'set via environment variable (launch script / CLI)':s==='file'?'set by the --config file':'subsystem default (not explicitly set)'}">${s}</span>`;
   if(rc.config_file) body+=`<div class="cfgsub">config file</div><div class="cv" style="padding:2px 12px">${esc(rc.config_file)}</div>`;
   for(const sec of Object.keys(rc.by_section).sort()){
     const keys=rc.by_section[sec];
     const rows=Object.keys(keys).sort().map(k=>{const o=keys[k];
       return `<tr><td class="ck">${esc(k)}</td><td class="cv">${esc(o.value)}</td><td>${badge(o.source)}</td></tr>`;}).join('');
     body+=`<div class="cfgsub">${esc(sec)}</div><table class="cfgtbl">${rows}</table>`;
   }
 }
 // (b) the rest of the manifest (vision profile note, throughput, git, hardware, mean_rhae …).
 const skip=new Set(['run_config','context_config']);  // context_config is covered by run_config's `context` section
 const rows=[]; const subs=[];
 for(const k of Object.keys(man).sort()){
   if(skip.has(k)) continue;
   const v=man[k];
   if(v&&typeof v==='object'&&!Array.isArray(v)) subs.push([k,v]);
   else rows.push([k, Array.isArray(v)?v.join(', '):String(v)]);
 }
 const r2=(pairs)=>pairs.map(([k,v])=>`<tr><td class="ck">${esc(k)}</td><td class="cv" colspan="2">${esc(v)}</td></tr>`).join('');
 if(rows.length){ body+=`<div class="cfgsub">run</div><table class="cfgtbl">${r2(rows)}</table>`; }
 for(const [name,obj] of subs){
   const pairs=Object.keys(obj).sort().map(k=>[k, (obj[k]&&typeof obj[k]==='object')?JSON.stringify(obj[k]):String(obj[k])]);
   body+=`<div class="cfgsub">${esc(name)}</div><table class="cfgtbl">${r2(pairs)}</table>`;
 }
 if(!body) body='<div class="k">no manifest config recorded for this run (legacy run?)</div>';
 el=document.createElement('div'); el.id='cfgpanel';
 el.innerHTML=`<div class="cfghd">run config · ${esc(R)}`
   +`<span class="cfgx" onclick="toggleConfig()">✕</span></div>`+body;
 document.body.appendChild(el);
}
function toggleDiskTree(){window._wsOpen['__disk']=!(window._wsOpen['__disk']===true); draw();}
function loadDiskTree(){
 const key=_diskKey(); window._diskTree[key]='loading';
 fetch(encodeURIComponent(R)+'/ws/'+encodeURIComponent(G)+'/tree.json')
   .then(r=>r.ok?r.json():null).then(j=>{window._diskTree[key]=j||'none'; draw();})
   .catch(()=>{window._diskTree[key]='none'; draw();});
}
function selDisk(rel){window._wsSel='__disk:'+rel; window._diskFile=window._diskFile||{};
 const k=_diskKey()+'/'+rel;
 if(window._diskFile[k]===undefined){ window._diskFile[k]='loading';
   fetch(encodeURIComponent(R)+'/ws/'+encodeURIComponent(G)+'/file?path='+encodeURIComponent(rel))
     .then(r=>r.ok?r.text():'(file not found)').then(tx=>{window._diskFile[k]=tx; draw();})
     .catch(()=>{window._diskFile[k]='(fetch failed)'; draw();}); }
 draw();
}
function toggleDirArg(id){toggleDir(argVal(id));}
function selFileArg(id){selFile(argVal(id));}
function selDiskArg(id){selDisk(argVal(id));}
function toggleDir(fp){window._wsOpen[fp]=!(window._wsOpen[fp]===true); draw();}
function setAgentF(v){window._agentF=v; draw();}   // actor/builder transcript filter
function setStep(j){i=Math.max(0,Math.min(DATA().length-1,j));window._wmIdx=0;draw();}  // jump to a step (builder nav)

// WORLD-MODEL prediction overlay. s.wm_pred (baked by --wm-predict) is either:
//   {plan:[{action,row,col,grid},...]}  — predicted trajectory toward level_complete (Case A), or
//   {per_action:[{action,grid},...]}    — predicted next grid per available action (Case B).
// Grids are drawn on #cvw with red = cells that differ from the CURRENT frame (what the model
// says will change). A scrubber (plan) / picker (per-action) navigates the predictions.
window._wmIdx=0;
function setWmIdx(n){window._wmIdx=n; draw();}
window._wmModel='final';  // 'final' | 'current' — which model's predictions the panel shows
function setWmModel(m){window._wmModel=m; window._wmIdx=0; draw();}
function renderWmPred(s){
 const wrap=document.getElementById('wmwrap'); const top=s.wm_pred;
 const hasFinal=top&&(top.plan||top.per_action||top.no_plan_reason);
 const hasCur=top&&top.current&&(top.current.plan||top.current.per_action||top.current.no_plan_reason);
 if(!hasFinal&&!hasCur){wrap.style.display='none'; return;}
 wrap.style.display='block';
 const ctl=document.getElementById('wmctl'), cap=document.getElementById('wmcap');
 const fmtA=p=>p.action+(p.row!=null?`(${p.row},${p.col})`:'');
 // Two model views (when both available): FINAL = the world_model.py the agent settled on at GAME END,
 // replayed retrospectively; CURRENT = the model AS IT EXISTED at THIS turn (the per-turn snapshot).
 // A header toggle picks which; default 'final', fall back to whichever exists.
 let view=window._wmModel; if(view==='current'&&!hasCur)view='final'; if(view==='final'&&!hasFinal)view='current';
 const wp = (view==='current') ? top.current : top;
 const mkbtn=(m,lbl,on)=>`<span class="agf ${on?'on':''}" onclick="setWmModel('${m}')" title="${m==='final'?'world_model.py at GAME END, replayed on this frame':'world_model.py AS IT EXISTED at this turn'}">${lbl}</span>`;
 const toggle = (hasFinal&&hasCur)
   ? `<div style="margin:2px 0 6px"><span class="k">model view: </span>`
       +mkbtn('current','current'+(top.current&&top.current.sim_acc!=null?` (sim ${(+top.current.sim_acc).toFixed(2)})`:''),view==='current')
       +mkbtn('final','final',view==='final')
       +`</div>`
   : `<div class="k" style="margin:2px 0 6px" title="${view==='final'?'world_model.py at GAME END, replayed on this frame — NOT the model as of this turn':'the model as it existed at this turn'}">`
       +`model view: ${view==='final'?'final (game-end, retrospective)':'current (this turn)'}</div>`;
 if(wp.plan){
   const n=wp.plan.length; let k=Math.min(window._wmIdx, n-1); if(k<0)k=0;
   const step=wp.plan[k];
   ctl.innerHTML=toggle+`🧭 predicted plan to level_complete — <b>${n}</b> step${n>1?'s':''} `
     +`<button onclick="setWmIdx(${Math.max(0,k-1)})">◀</button> ${k+1}/${n} `
     +`<button onclick="setWmIdx(${Math.min(n-1,k+1)})">▶</button>`;
   drawGrid(document.getElementById('cvw'), step.grid, s.grid);
   cap.innerHTML=`predicted grid after planned action <b>${esc(fmtA(step))}</b> (red outline = claimed cells changed vs THIS frame; red diagonal = UNKNOWN/unclaimed). `
     +`Full plan: ${wp.plan.map(fmtA).join(' → ')}`;
 } else if(wp.per_action){
   const n=wp.per_action.length; let k=Math.min(window._wmIdx, n-1); if(k<0)k=0;
   const pa=wp.per_action[k];
   const why=wp.no_plan_reason?`<span class="k noplan"> — ${esc(wp.no_plan_reason)}</span>`:'';
   // chips wrap (flex) so a CLICK game's many ACTION6(r,c) candidates don't blow out the column width.
   // hover a chip → highlight that candidate's click cell on the live grid (the 'hovercell' LAYER).
   ctl.innerHTML=toggle+`🔮 per-action prediction${why}<div class="agfwrap">`
     +wp.per_action.map((p,j)=>`<span class="agf ${j===k?'on':''}" onclick="setWmIdx(${j})" title="${attr(fmtA(p))}"`
        +(p.row!=null?` onmouseenter="hoverClick(${p.row},${p.col})" onmouseleave="hoverClick()"`:'')
        +`>${esc(fmtA(p))}</span>`).join('')
     +`</div>`;
   drawGrid(document.getElementById('cvw'), pa.grid, s.grid);
   cap.innerHTML=`predicted grid after <b>${esc(fmtA(pa))}</b> (red outline = claimed cells this prediction changes vs THIS frame; red diagonal = UNKNOWN/unclaimed).`;
 } else {  // no_plan_reason only (no plan AND no per-action prediction)
   ctl.innerHTML=toggle+`🔮 no prediction here${wp.no_plan_reason?`<span class="k"> — ${esc(wp.no_plan_reason)}</span>`:''}`;
   cap.innerHTML='';
 }
}

// ---- COL 4: selected file content. notes/* have full content in the snapshot;
// the current-grid/diff shortcuts read the snapshot's cur_grid/cur_diff. ----
function _versionMeta(sel, meta){
 if(!meta||typeof meta!=='object') return '';
 const bits=[];
 if(meta.kind)bits.push(meta.kind);
 if(meta.size!=null)bits.push(Number(meta.size).toLocaleString()+' bytes');
 if(meta.sha256)bits.push('sha256 '+meta.sha256);
 if(meta.status)bits.push(meta.status);
 let h=`<div class="k" style="font-size:11px;margin:2px 0 8px">${esc(bits.join(' · '))}</div>`;
 if(meta.stored===true&&meta.sha256){
   const url=encodeURIComponent(R)+'/ws/'+encodeURIComponent(G)+'/blob/'+meta.sha256;
   h+=`<a href="${url}" target="_blank" rel="noopener" class="shortcut">open exact historical blob</a>`;
 }else if(meta.stored===false){
   h+=`<div class="bad" style="font-size:11px">This file body was not stored; see the snapshot status above.</div>`;
 }
 return h;
}
function renderFile(s){
 const w=s.reasoning&&s.reasoning.workspace; if(!w) return '';
 const sel=window._wsSel;
 // grid/diff content uses white-space:pre (NO wrap) so a 64-wide row stays one line.
 const pre=(t)=>`<pre style="white-space:pre;overflow:auto">${esc(t||'')}</pre>`;
 if(sel==='__grid') return `<span class="lbl">current grid (level ${w.level} turn ${w.turn})</span>`+pre(w.cur_grid);
 if(sel==='__diff') return `<span class="lbl">current diff</span>`+pre(w.cur_diff);
 // on-disk file (fetched lazily from the durable workspace via the file endpoint)
 if(typeof sel==='string' && sel.indexOf('__disk:')===0){
   const rel=sel.slice('__disk:'.length); const k=_diskKey()+'/'+rel;
   const body=(window._diskFile||{})[k];
   const wideD=/turn_\d+\.txt$|diff_|\.txt$/.test(rel);
   const lbl=`<span class="lbl">${esc(rel)} <span class="k">(on disk)</span></span>`;
   if(body===undefined||body==='loading') return lbl+'<div class="k" style="font-size:11px">loading…</div>';
   return lbl+(wideD?pre(body):`<pre>${esc(body)}</pre>`);
 }
 if(!sel) return '<div class="k" style="font-size:11px">click a file (or a shortcut) to view it</div>';
 // Causal text is embedded exactly; binary bodies live in the content-addressed blob store.
 const contents=w.contents||w.notes||{};
 const versions=_wsVersions(s), meta=versions[sel];
 if(!(sel in contents)){
   if(meta){
     const what=meta.kind==='binary'?'binary file':'historical file';
     return `<span class="lbl">${esc(sel)} <span class="k">(${what})</span></span>`+_versionMeta(sel,meta);
   }
   // Render the actual frame PNG inline when it is embedded in the snapshot.
   if(/\.png$/.test(sel)){
     const img=(w.images||{})[sel];
     if(img) return `<span class="lbl">${esc(sel)} (the frame image sent to the model)</span>`
       +`<img src="${img}" style="image-rendering:pixelated;width:100%;max-width:512px;border:1px solid var(--line);border-radius:6px">`;
     return `<span class="lbl">${esc(sel)}</span><div class="k" style="font-size:11px">(image not embedded — older than the viewer keep window)</div>`;
   }
   return `<span class="lbl">${esc(sel)}</span><div class="k" style="font-size:11px">(content not captured)</div>`;
 }
 // grids/diffs are wide → no-wrap; prose/code can wrap normally.
 const wide=/turn_\d+\.txt$|diff_|cur_grid/.test(sel);
 const body=resolveContent(sel, contents[sel]);
 return `<span class="lbl">${esc(sel)}</span>`+_versionMeta(sel,meta)
   +(wide?pre(body):`<pre>${esc(body)}</pre>`);
}
// content de-dup (build_steps): an unchanged file is stored as "\x00=<stepIndex>" pointing
// at the step that holds the real text. Walk back to resolve.
function resolveContent(sel, v){
 let guard=0;
 while(typeof v==='string' && v.charCodeAt(0)===0 && v[1]==='=' && guard++<5000){
   const idx=parseInt(v.slice(2),10); const st=DATA()[idx];
   const c=st&&st.reasoning&&st.reasoning.workspace&&st.reasoning.workspace.contents;
   v=(c&&c[sel]!=null)?c[sel]:'(unresolved)';
 }
 return v;
}
function resolveVersions(v){
 let guard=0;
 while(typeof v==='string' && v.charCodeAt(0)===0 && v[1]==='=' && guard++<5000){
   const idx=parseInt(v.slice(2),10); const st=DATA()[idx];
   const w=st&&st.reasoning&&st.reasoning.workspace;
   v=w&&w.file_versions;
 }
 return (v&&typeof v==='object'&&!Array.isArray(v))?v:{};
}
function _wsVersions(s){
 const w=s&&s.reasoning&&s.reasoning.workspace;
 return resolveVersions(w&&w.file_versions);
}
function selFile(p){window._wsSel=p; draw();}
function selSpecial(kind){window._wsSel = kind==='grid'?'__grid':'__diff'; draw();}

function drawGrid(cv, grid, diffAgainst){
 // No-frame runs (launched without --viz) carry grid=null; clear the canvas and bail so the
 // reasoning/transcript panels still render instead of throwing on grid.length.
 if(!isGrid(grid)){const ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);return cv.width/64;}
 // The 576px backing canvas gives a 64x64 grid nine pixels per cell and scales down responsively.
 const ctx=cv.getContext('2d'), N=grid.length, W=grid[0].length, CELL=cv.width/Math.max(N,W);
 for(let r=0;r<N;r++)for(let c=0;c<W;c++){
  const v=grid[r][c], x=c*CELL, y=r*CELL;
  if(v===UNKNOWN){
   ctx.fillStyle='#111827';ctx.fillRect(x,y,CELL+1,CELL+1);
   ctx.strokeStyle='#ff2d55';ctx.lineWidth=Math.max(1,CELL/8);
   ctx.beginPath();ctx.moveTo(x+1,y+1);ctx.lineTo(x+CELL-1,y+CELL-1);ctx.stroke();
   ctx.beginPath();ctx.moveTo(x+CELL-1,y+1);ctx.lineTo(x+1,y+CELL-1);ctx.stroke();
  }else{
   ctx.fillStyle=PAL[v]||'#f0f';ctx.fillRect(x,y,CELL+1,CELL+1);
  }
 }
 // faint cell grid lines (like the official arcprize player) — toggle with window._lines
 if(window._lines!==false && CELL>=4){ctx.strokeStyle='rgba(128,128,128,0.25)';ctx.lineWidth=0.5;
  for(let i=0;i<=N;i++){ctx.beginPath();ctx.moveTo(0,i*CELL);ctx.lineTo(W*CELL,i*CELL);ctx.stroke();}
  for(let j=0;j<=W;j++){ctx.beginPath();ctx.moveTo(j*CELL,0);ctx.lineTo(j*CELL,N*CELL);ctx.stroke();}}
 if(diffAgainst){ctx.strokeStyle='#ff2d55';ctx.lineWidth=1.5;
  for(let r=0;r<N;r++)for(let c=0;c<W;c++) if(grid[r][c]!==UNKNOWN && (diffAgainst[r]||[])[c]!==grid[r][c]) ctx.strokeRect(c*CELL,r*CELL,CELL,CELL);}
 return CELL;
}
function runsel(){const sel=document.getElementById('run');sel.innerHTML='';
 for(const rid of RUNIDS){const o=document.createElement('option');o.value=rid;
  const man=MAN(rid); const haveR=man.some(m=>m.rhae!=null);
  const mean=haveR?(man.reduce((a,m)=>a+(m.rhae||0),0)/man.length).toFixed(1):'?';
  const rt=(META(rid).run_time||'');   // finish date/time (run dir mtime); newest-first ordering
  o.text=`${rt?rt+'  ·  ':''}${rid}  (${man.length}g · RHAE ${mean})`;sel.add(o);} sel.value=R;}
function gsel(){const sel=document.getElementById('game');sel.innerHTML='';
 for(const m of MANIFEST){const o=document.createElement('option');o.value=m.id;
  const r=(m.rhae!=null)?` · RHAE ${m.rhae}`:'';
  // m.live = an in-flight game (only a level-partial record so far, no final result yet).
  o.text=`${m.live?'● LIVE  ':''}${m.id} (${m.steps} steps, ${m.levels} levels${r})`;sel.add(o);} sel.value=G;
 // ===== brand header: headline metric + per-game progress strip (run id is in the dropdown) =====
 const haveR=MANIFEST.some(m=>m.rhae!=null);
 if(haveR){const mt0=META(R);
  // Headline RHAE: prefer the manifest's CLEAN mean (reportable games only — excludes PARTIAL/ERROR,
  // whose truncated scores understate RHAE). Fall back to the raw all-rows mean for legacy manifests
  // that predate the clean metric. Recomputing over all per-game rows (the old behavior) silently
  // averaged in infra-killed games.
  const _cleanMean=(mt0.manifest||{}).mean_rhae_clean;
  const _rawMean=(MANIFEST.reduce((a,m)=>a+(m.rhae||0),0)/MANIFEST.length);
  const mean=(_cleanMean!=null)?_cleanMean:_rawMean;
  const fmtH=s=>s>=3600?(s/3600).toFixed(1)+'h':(s/60).toFixed(0)+'m';
  const compute=MANIFEST.reduce((a,m)=>a+(m.wall_s||0),0);  // SUM of per-game compute (not elapsed)
  const lv=MANIFEST.reduce((a,m)=>a+(m.levels||0),0);
  document.getElementById('brandrhae').textContent=mean.toFixed(2)
    +((_cleanMean!=null && (mt0.manifest||{}).n_partial)?` (clean; ${(mt0.manifest).n_partial}P excl)`:'');
  // Prefer true elapsed time when recorded, and show summed per-game compute with worker count.
  // Older records may contain only the sum, which is labelled as compute rather than elapsed time.
  let rt; if(mt0.wall_clock_s){rt=`${fmtH(mt0.wall_clock_s)} elapsed`
    +(mt0.workers?` · ${mt0.workers}-way · ${fmtH(compute)} compute`:'');}
  else{rt=`${fmtH(compute)} compute`+(mt0.workers?` (${mt0.workers}-way)`:'')+`<span class="k"> · elapsed n/a</span>`;}
  // Token spend and public-list USD-equivalent cost. Unknown model prices omit the cost field.
  const tin=MANIFEST.reduce((a,m)=>a+(m.tok_in||0),0), tout=MANIFEST.reduce((a,m)=>a+(m.tok_out||0),0);
  const haveCost=MANIFEST.some(m=>m.cost!=null);
  const cost=MANIFEST.reduce((a,m)=>a+(m.cost||0),0);
  // cache-aware: if any call recorded cache reads/writes, the cost is the real cached price;
  // otherwise it's the no-cache upper bound (prompt caching was off for that run). Label honestly.
  const cached=MANIFEST.some(m=>(m.cache_read||0)+(m.cache_write||0)>0);
  const costCO=MANIFEST.reduce((a,m)=>a+(m.cost_cacheon||0),0);
  const haveCO=MANIFEST.some(m=>m.cost_cacheon!=null);
  let tokpart='';
  if(tin||tout){tokpart=` · <b>${fmtTok(tin)}</b> in / <b>${fmtTok(tout)}</b> out`
    +(haveCost?` · <b title="published API list-price equivalent. ${cached?'Cache-priced from recorded cache usage.':'NO-CACHE upper bound; the cache-on estimate assumes prefix reuse.'}">~$${cost.toFixed(2)}</b> <span class="k">list${cached?'':' (no-cache max)'}</span>`:'')
    // cache-on equivalent: what the SAME run would cost with prompt caching on (per-turn delta priced
    // at base, repeated prefix at cache-read 0.1×). Only shown when it differs from the actual (i.e.
    // the run was no-cache) — the comparable-to-leaderboard number. Labeled an estimate.
    +((haveCO&&!cached&&Math.abs(costCO-cost)>0.01)?` <span class="k" title="estimate for the same recorded calls with per-turn new tokens at base price and repeated prefixes at the model's cache-read price">(~$${costCO.toFixed(2)} cache-on est)</span>`:'');}
  // WM/PLANNING COVERAGE: over all COMPLETED levels across games, the share with a correct final-model
  // world model (sim_accuracy==1.0) and the share the planner found a path for. Summed client-side from
  // each game's wm activity so it updates live (partial runs included). Omitted if no game carries it
  // (e.g. legacy static export, or no level completed yet). See run_parallel._coverage_agg.
  let wmpart='';
  const wmA=MANIFEST.map(m=>m.wm).filter(Boolean);
  const cLv=wmA.reduce((a,w)=>a+(w.completed_levels||0),0);
  if(cLv){const wmOK=wmA.reduce((a,w)=>a+(w.wm_correct_levels||0),0);
    const plOK=wmA.reduce((a,w)=>a+(w.plan_levels||0),0);
    wmpart=` · <span class="k" title="of all levels completed across games: share with an accepted final-model world model (simulation_accuracy==1.0 and prediction_coverage above the guard threshold), and share the planner found a path for. Measures the model AVAILABLE at game end, not a causal attribution of the solve.">WM correct <b>${wmOK}/${cLv}</b> (${Math.round(100*wmOK/cLv)}%) · planned <b>${plOK}/${cLv}</b> (${Math.round(100*plOK/cLv)}%)</span>`;}
  document.getElementById('brandsub').innerHTML=`<b>${MANIFEST.length}</b> games · <b>${lv}</b> levels · ${rt}${tokpart}${wmpart}`;}
 // Run date/time, model, reasoning effort, and orchestration mode in the header.
 const mt=META(R);
 // Reasoning details are labelled per call from recorded metadata. The header only reports an effort
 // setting when the run manifest recorded one; it does not guess behavior from a provider name.
 const eff = mt.effort ? ('effort '+mt.effort) : 'reasoning details per call';
 // orchestration pattern chip: four mutually exclusive experiment treatments.
 // TYCHO_MODE 'orchestrator' is the actor-pull subagent design.
 const MODEDESC={no_world_model:['direct reasoning','observations, notes, and lightweight tool use'],
   single:['single WM','actor writes the world model inline'],
   orchestrator:['builder WM','actor pulls a world-model builder subagent'],
   trigger:['triggered WM','harness fires the builder on a verify-accuracy trigger'],
   'trigger+subagent':['trigger+subagent','triggered builder subagent']};
 const md=MODEDESC[(mt.mode||'single')]||[mt.mode,''];
 const modeChip=`<span class="modechip" title="${esc(md[1])}">${esc(md[0])}</span>`;
 document.getElementById('brandmeta').innerHTML=
   (mt.run_time?`<span class="k">${esc(mt.run_time)}</span> · `:'')
   +modeChip+' '
   +`<span class="k">${esc(mt.model||'?')} · ${esc(eff)}</span>`
   +(mt.hardware?` · <span class="k">${esc(mt.hardware)}</span>`:'')
   +(mt.git_version?` · <span class="k">git ${esc(mt.git_version)}</span>`:'')
   +` <span class="cfgicon" title="show all config variables for this run" onclick="toggleConfig(event)">⚙</span>`;
 // per-game pips: green-tinted by RHAE; click to jump to that game
 const strip=document.getElementById('gamestrip'); strip.innerHTML='';
 const mx=Math.max(1,...MANIFEST.map(m=>m.rhae||0));
 for(const m of MANIFEST){const p=document.createElement('span');p.className='gpip'+(m.live?' live':'');
  p.title=`${m.id}${m.live?' · IN PROGRESS (live)':''} · RHAE ${m.rhae!=null?m.rhae:'?'}`
    +((m.tok_in||m.tok_out)?` · ${fmtTok(m.tok_in)}→${fmtTok(m.tok_out)} tok`:'')
    +(m.cost!=null?` · ~$${m.cost.toFixed(2)} list`:'');
  const f=(m.rhae||0)/mx; if(f>0&&!m.live)p.style.background=`rgba(95,240,160,${0.18+0.72*f})`;
  p.onclick=()=>{document.getElementById('game').value=m.id;loadGame();updateStrip();};strip.appendChild(p);}
 updateStrip();}
function updateStrip(){const pips=document.getElementById('gamestrip').children;
 for(let k=0;k<MANIFEST.length;k++)pips[k]&&pips[k].classList.toggle('cur',MANIFEST[k].id===G);}
function layerToggles(){const el=document.getElementById('layers');el.innerHTML='';
 for(const L of LAYERS){const id='ly_'+L.id;el.innerHTML+=`<label><input type="checkbox" id="${id}" ${L.on?'checked':''} onchange="tog('${L.id}')"> ${L.label}</label> `;}}
function tog(id){const L=LAYERS.find(x=>x.id===id);L.on=document.getElementById('ly_'+id).checked;draw();}
function ensureLoaded(id, cb){
 window.GAMES[R]=window.GAMES[R]||{};
 if(window.GAMES[R][id]){cb();return;}
 document.getElementById('col-chat').innerHTML='<div class="panel k">loading '+id+'…</div>';
 // data files live under <runId>/game_<id>.js and assign window.GAMES[runId][gameId]
 const sc=document.createElement('script'); sc.src=encodeURIComponent(R)+'/game_'+id+'.js';
 sc.onload=cb; sc.onerror=()=>{document.getElementById('col-chat').innerHTML='<div class="err">failed to load '+R+'/game_'+id+'.js (serve over http if file:// blocks it)</div>';};
 document.head.appendChild(sc);
}
function loadGame(){G=document.getElementById('game').value;i=0;window._wsSel=null;window._agentF='all';ensureLoaded(G,draw);}
function loadRun(){R=document.getElementById('run').value;MANIFEST=MAN(R);i=0;window._wsSel=null;window._agentF='all';
 G=MANIFEST[0].id;gsel();ensureLoaded(G,draw);}
function step(d){const n=DATA().length; i=Math.max(0,Math.min(n-1,i+d));window._wmIdx=0;draw();}
function jumpLevel(d){const D=DATA(); const want=D[i].level+d;const idx=D.findIndex(s=>s.level===want);if(idx>=0){i=idx;draw();}}
function play(){if(timer){clearInterval(timer);timer=null;return;}timer=setInterval(()=>{if(i>=DATA().length-1){clearInterval(timer);timer=null;}else step(1);},250);}

// World-model impact panel under the grid. world_model.py itself lives in the
// workspace browser, cols 3+4). This surfaces the model's IMPACT: (1) verify accuracy this step;
// (2) the prediction-vs-reality audit — where the model's predicted next grid DIFFERED from what we
// then actually observed (the actionable signal for latent-state / transition-function inference).
function _wsContents(s){const w=s&&s.reasoning&&s.reasoning.workspace; return (w&&w.contents)||{};}
function parseGridText(txt){
 if(typeof txt!=='string') return null;
 const rows=[];
 for(const line of txt.split(/\r?\n/)){
   const m=line.match(/^\s*y\d+:\s*(.*)$/);
   if(!m) continue;
   const vals=m[1].split(/\s+/).filter(tok=>/^[0-9a-f]$/i.test(tok)).map(tok=>parseInt(tok,16));
   if(vals.length) rows.push(vals);
 }
 if(!rows.length) return null;
 const w=rows[0].length;
 return rows.every(r=>r.length===w) ? rows : null;
}
function terminalGridForBoundary(s){
 const r=s.reasoning||{}; const prevLevel=(r.level||0)-1;
 if(prevLevel<0) return null;
 if(isGrid(s.boundary_terminal_grid)) return s.boundary_terminal_grid;
 const c=_wsContents(s);
 const txtKey=`level_${prevLevel}/terminal.txt`;
 if(txtKey in c){
   const grid=parseGridText(resolveContent(txtKey,c[txtKey]));
   if(isGrid(grid)) return grid;
 }
 const key=`level_${prevLevel}/terminal.json`;
 if(!(key in c)) return null;
 try{
   const body=resolveContent(key,c[key]);
   const obj=JSON.parse(body);
   return isGrid(obj.terminal_grid) ? obj.terminal_grid : null;
 }catch(e){ return null; }
}
function terminalGridForCompletedStep(s){
 const lvl=s.level;
 if(!Number.isInteger(lvl) || lvl<0) return null;
 if(isGrid(s.boundary_terminal_grid)) return s.boundary_terminal_grid;
 const c=_wsContents(s);
 const txtKey=`level_${lvl}/terminal.txt`;
 if(txtKey in c){
   const grid=parseGridText(resolveContent(txtKey,c[txtKey]));
   if(isGrid(grid)) return grid;
 }
 const key=`level_${lvl}/terminal.json`;
 if(!(key in c)) return null;
 try{
   const body=resolveContent(key,c[key]);
   const obj=JSON.parse(body);
   return isGrid(obj.terminal_grid) ? obj.terminal_grid : null;
 }catch(e){ return null; }
}
function _predGrid(s){  // model's prediction for the action ACTUALLY committed this step (not the
                        // planner's hypothetical next step). The agent records this in reasoning.
                        // predicted_frame; if absent, return null and the chip is skipped.
                        // (wm_pred.plan[0].grid is the PLANNER's first step from this state — used
                        // for the plan-trajectory overlay, NOT as the per-step prediction, because
                        // the planner's plan[0] may differ from the actor's actually-committed action.)
 const r=s.reasoning||{};
 if(isGrid(r.predicted_frame)) return r.predicted_frame;
 return null;
}
function _gridDiffCount(a,b){
 if(!isGrid(a)||!isGrid(b)||a.length!==b.length) return null;
 for(let r=0;r<a.length;r++) if(a[r].length!==b[r].length) return null;
 let n=0; for(let r=0;r<a.length;r++){const ar=a[r],br=b[r]; for(let c=0;c<ar.length;c++) if(ar[c]!==UNKNOWN && ar[c]!==br[c]) n++;} return n;
}
function _gridUnknownCount(a){
 if(!isGrid(a)) return null;
 let n=0; for(let r=0;r<a.length;r++){const ar=a[r]; for(let c=0;c<ar.length;c++) if(ar[c]===UNKNOWN) n++;} return n;
}
function renderWmPanel(s, i){
 const el=document.getElementById('wmpanel'); if(!el) return;
 const r=s.reasoning||{};
 // (1) verify accuracy chip (best-effort: agent stashes it when it ran verify)
 let acc=null, accLbl='';
 for(const [k,lbl] of [['simulation_accuracy','simulation_acc'],['verify','verify']])
   if(typeof r[k]==='number'){acc=r[k]; accLbl=lbl; break;}
 if(acc===null && r.verify && typeof r.verify.simulation_accuracy==='number'){acc=r.verify.simulation_accuracy; accLbl='simulation_acc';}
 // (2) prediction-vs-reality: model's predicted result of this step's committed action vs the
 // observed grid stored on this same step. Mismatched cells = where the world model is wrong.
 const pred=_predGrid(s); const obs=s.grid;
 const diff=_gridDiffCount(pred,obs);
 const unk=_gridUnknownCount(pred);
 const outcome=r.outcome||r.goal;   // outcome channel; r.goal is old-run display compatibility.
 const hasPlan=s.wm_pred&&s.wm_pred.plan&&s.wm_pred.plan.length;
 if(acc===null && diff===null && !outcome && !hasPlan){ el.innerHTML=''; return; }   // no model signal this step
 let h='<div class="wmhdr"><span class="lbl" style="text-transform:none">world-model impact</span>';
 if(acc!==null) h+=`<span class="wmacc" title="model verify ${esc(accLbl)} at this step; simulation_acc accepts bounded observation_variants when present">${esc(accLbl)} ${acc<=1?acc.toFixed(3):acc.toFixed(0)}</span>`;
 if(r.verify && r.verify.variant_used){
   const strict=r.verify.strict_simulation_accuracy;
   h+=`<span class="wmacc" title="observation_variants accepted ${r.verify.variant_used} transition(s); strict canonical render accuracy=${strict}">variants ${r.verify.variant_used}</span>`;
 }
 if(r.verify && (r.verify.unknown_used || (r.verify.prediction_coverage!=null && r.verify.prediction_coverage<0.999))){
   const cov=r.verify.prediction_coverage;
   const known=r.verify.known_cell_accuracy;
   const cstat=r.verify.prediction_coverage_status || (cov==null?'unobserved':(+cov<=0?'vacuous':(+cov<0.999?'partial':'complete')));
   h+=`<span class="wmacc" title="prediction_coverage=${cov}; status=${cstat}; render() used UNKNOWN=-1 on ${r.verify.unknown_used||0} transition(s); claimed-cell accuracy=${known}; vacuous means render() claimed no cells">coverage ${cov==null?'n/a':(+cov).toFixed(2)} ${esc(cstat)}</span>`;
 }
 // OUTCOME channel chip — terminal status is a separate inference from dynamics.
 if(outcome && (outcome.outcome_observable || outcome.goal_observable || outcome.n_terminal)){
   const ok=outcome.ok;
   if(outcome.level_complete_on_terminal!==undefined || outcome.game_over_on_death!==undefined){
     const lct=outcome.level_complete_on_terminal, lcnt=outcome.level_complete_on_nonterminal;
     const got=outcome.game_over_on_death, gont=outcome.game_over_on_nonterminal;
     let why='';
     if(lct!=null && lct<1) why='missed level_complete';
     else if(lcnt) why='false level_complete';
     else if(got!=null && got<1) why='missed game_over';
     else if(gont) why='false game_over';
     else if(outcome.terminal_render_ok===false) why='terminal render';
     const tcov=outcome.terminal_render_prediction_coverage;
     const tcstat=outcome.terminal_render_coverage_status || (tcov==null?'unobserved':(+tcov<=0?'vacuous':(+tcov<0.999?'partial':'complete')));
     const tvar=outcome.terminal_render_variant_used||0, tunk=outcome.terminal_render_unknown_used||0;
     const renderNote=(tvar||tunk)?` · terminal render ${tcstat}`:'';
     h+=`<span class="${ok?'wmok':'wmbad'}" title="outcome() returns ongoing, level_complete, or game_over. level_complete_on_terminal=${lct}, level_complete_on_nonterminal=${lcnt}, game_over_on_death=${got}, game_over_on_nonterminal=${gont}; terminal_render_coverage=${tcov}; terminal_render_coverage_status=${tcstat}; terminal variants=${tvar}; terminal UNKNOWN accepts=${tunk}">`
       +`${ok?('✓ outcome verified'+renderNote):('⚠ outcome '+(why||'wrong'))}</span>`;
   }else{
     const ft=outcome.fires_on_terminal, fnt=outcome.fires_on_nonterminal;
     h+=`<span class="${ok?'wmok':'wmbad'}" title="legacy goal() channel: fires_on_terminal=${ft}, fires_on_nonterminal=${fnt}">`
       +`${ok?'✓ goal verified':('⚠ goal '+((ft||0)<1?'too tight':'too loose'))}</span>`;
   }
 }
 // PLAN chip — the planner found a path to level_complete from here (col-1 planning visibility)
 if(hasPlan) h+=`<span class="wmacc" title="the planner found a path to outcome() == 'level_complete' from this state; the predicted trajectory is overlaid on the grid">▸ plan: ${s.wm_pred.plan.length} step(s)</span>`;
 if(diff!==null){
   const cls = diff===0?'wmok':'wmbad';
   const unknownNote=unk?` (${unk} unknown)`:'';
   h+=`<span class="${cls}" title="claimed cells where the model's predicted grid differs from the observed result of this action; UNKNOWN=-1 cells are abstentions and are shown with red diagonal crosses">`
     +`${diff===0?('✓ prediction matched'+unknownNote):('⚠ '+diff+' claimed cell'+(diff>1?'s':'')+' mispredicted'+unknownNote)}</span>`;
 }
 h+='</div>';
 // when there's a mispredict or an abstention, offer the side-by-side on the prediction canvas.
 if((diff || unk) && pred && obs){
   h+=`<div class="k" style="font-size:11px;margin-top:5px">predicted next grid (this turn's action) vs actually observed — `
     +`<a href="#" onclick="window._wmAudit=!window._wmAudit;draw();return false">${window._wmAudit?'hide':'show'} diff ▦</a></div>`;
 }
 el.innerHTML=h;
 // render the audit overlay onto the prediction canvas (reuse #cvp): predicted, red = where it was wrong
 if(window._wmAudit && (diff || unk) && pred && obs){
   document.getElementById('predwrap').style.display='block';
   drawGrid(document.getElementById('cvp'), pred, obs);   // diffAgainst=observed → red = mispredicted
   document.getElementById('predwrap').lastElementChild.textContent=
     'world model: PREDICTED next grid (red outline = claimed cell differs from observed; red diagonal cross = UNKNOWN/unclaimed)';
 }
}

function draw(){
 const D=DATA();
 if(!D.length){document.getElementById('col-chat').innerHTML=
   '<div class="panel k">No viewable steps in this record (run launched without --viz, so no frames or per-step trace were captured). RHAE/levels are in the run list; re-run with --viz for full replay.</div>';
  document.getElementById('bar').textContent=G+'  (no steps)';return;}
 const s=D[i];
 window._lines=(LAYERS.find(x=>x.id==='gridlines')||{}).on;
 const cv=document.getElementById('cv'); const CELL=drawGrid(cv,s.grid,null);
 const ctx=cv.getContext('2d');
 for(const L of LAYERS) if(L.on&&L.id!=='gridlines') L.fn(ctx,s,CELL);
 // side-by-side second grid in the prediction canvas. Two cases (in priority order):
 //   (a) the agent attached a predicted_frame (verifier mismatch) — show predicted vs observed;
 //   (b) GAME_OVER / automatic RESET — show the terminal frame explicitly so reset is not mistaken
 //       for ordinary dynamics;
 //   (c) LEVEL BOUNDARY (this step is turn_in_level 0 of level>0) — show the PREVIOUS level's
 //       observed terminal frame next to this new level's first acted-from frame when terminal
 //       evidence is available. The trace frame on the level-completing action is frame[-1],
 //       i.e. already the next playable level, so it must not be labelled as the old level FINAL.
 const pf=s.reasoning&&s.reasoning.predicted_frame;
 const r=s.reasoning||{};
 const atBoundary = r.turn_in_level===0 && (r.level||0)>0 && i>0;
 const completedTerm = s.just_completed ? terminalGridForCompletedStep(s) : null;
 const prevStep = i>0 ? DATA()[i-1] : null;
 const gameOverPrev = s.game_over && prevStep && isGrid(prevStep.grid) ? prevStep.grid : null;
 const resetGameOver = s.auto_reset && prevStep && prevStep.game_over && isGrid(prevStep.grid) ? prevStep.grid : null;
 const pw=document.getElementById('predwrap');
 if(isGrid(pf)){
   pw.style.display='block'; drawGrid(document.getElementById('cvp'),pf,s.grid);
   pw.lastElementChild.textContent='world model: PREDICTED next grid (red outline = claimed differs from observed; red diagonal = UNKNOWN/unclaimed)';
 } else if(isGrid(gameOverPrev)){
   pw.style.display='block'; drawGrid(document.getElementById('cvp'),gameOverPrev,null);
   pw.lastElementChild.textContent='◀ previous playable frame — vs observed GAME_OVER terminal on the left';
 } else if(isGrid(resetGameOver)){
   pw.style.display='block'; drawGrid(document.getElementById('cvp'),resetGameOver,null);
   pw.lastElementChild.textContent='◀ observed GAME_OVER terminal frame — vs automatic RESET fresh frame on the left';
 } else if(isGrid(completedTerm)){
   pw.style.display='block'; drawGrid(document.getElementById('cvp'),completedTerm,null);
   pw.lastElementChild.textContent=`◀ completed level (L${s.level}) observed TERMINAL frame — vs engine's next playable frame on the left`;
 } else if(atBoundary){
   const term=terminalGridForBoundary(s);
   const prev=term || DATA()[i-1].grid;
   pw.style.display='block'; drawGrid(document.getElementById('cvp'),prev,null);
   const prevLevel=(r.level||0)-1;
   pw.lastElementChild.textContent=term
     ? `◀ previous level (L${prevLevel}) observed TERMINAL frame — vs this level's first acted-from frame on the left`
     : `◀ previous trace frame (terminal evidence not captured in this legacy record) — vs this level's first acted-from frame on the left`;
 } else {
   pw.style.display='none';
 }
 renderWmPred(s);   // world-model plan/per-action prediction overlay (--wm-predict)
 renderWmPanel(s, i);   // Verification accuracy and current executable-model diagnostics.
 const _liveG=(MANIFEST.find(m=>m.id===G)||{}).live;
 document.getElementById('bar').innerHTML=`${esc(G)}  step ${i+1}/${D.length}  turn ${s.turn}`
   +(_liveG?`  <span class="livebadge">● IN PROGRESS — partial (last completed level); refresh for newer</span>`:'');
 const mx=Math.max(...D.map(x=>x.levels));
 document.getElementById('lvlbar').innerHTML=`<span class="v">LEVEL ${s.level+1}</span> <span class="k">(${mx} done this run)</span>`;
 const hasWorkspace = !!(s.reasoning && s.reasoning.workspace);
 // Center column: status plus the Tycho transcript when workspace evidence is available.
 const chat=document.getElementById('col-chat');
 let mid = PANELS[0](s)+PANELS[1](s)+PANELS[2](s)+PANELS[3](s);
 mid += hasWorkspace ? renderTranscript(s) : PANELS.slice(4).map(p=>p(s)).join('');
 chat.innerHTML=mid;
 chat.scrollTop=0;
 // COL 3 (tree) + COL 4 (file content) — Tycho only
 document.getElementById('col-tree').innerHTML = hasWorkspace ? renderTree(s)
   : '<div class="k" style="font-size:12px">file tree (Tycho only)</div>';
 document.getElementById('col-file').innerHTML = hasWorkspace ? renderFile(s) : '';
}
runsel();gsel();layerToggles();ensureLoaded(G,draw);  // run picker + game picker, then lazy-load + render
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')step(1);if(e.key==='ArrowLeft')step(-1);});
</script></body></html>"""


def _env_token_cost(env: dict, model: str):
    """Sum token buckets across every captured LLM call in an env's trace and convert to a USD
    list-price cost, pricing each cache tier separately (base input / output / cache-read /
    cache-write). Returns (tokens_in_total, tokens_out, cost_usd_or_None, cost_cacheon_or_None):
      cost          = the ACTUAL list price for how the run billed. If caching was OFF (legacy
                      context, no cache buckets) this is the NO-CACHE upper bound — every turn re-sends
                      the growing history at full base-input price.
      cost_cacheon  = an ESTIMATE of what the SAME run would cost with prompt caching ON, so it is
                      comparable to leaderboard figures from cache-using runs. Model: per turn you pay
                      base price only on the NEW prompt tokens since the previous call (the delta) and
                      cache-read price (~0.1× base) on the repeated prefix; output is unchanged. This
                      is a near-LOWER-bound (assumes near-perfect prefix reuse). When the run ALREADY
                      used caching (real cache_read/write present), cost_cacheon == cost (no estimate
                      needed). None when the model has no known price.
    tokens_in_total = fresh + cache_read + cache_write (the total prompt size), for display."""
    fresh = out = cr = cw = 0
    ins_per_call = []  # per-call tokens_in, in trace order, for the cache-on delta model
    for t in env.get("trace", []):
        for c in (t.get("reasoning") or {}).get("llm_calls") or []:
            ti = int(c.get("tokens_in") or 0)
            fresh += ti; ins_per_call.append(ti)
            out += int(c.get("tokens_out") or 0)
            cr += int(c.get("cache_read") or 0)
            cw += int(c.get("cache_write") or 0)
    price = _price_for(model)
    if price is None:
        return fresh + cr + cw, out, None, None
    base_in, p_out, p_cr, p_cw = price
    cost = (fresh / 1e6) * base_in + (out / 1e6) * p_out + (cr / 1e6) * p_cr + (cw / 1e6) * p_cw
    out_cost = (out / 1e6) * p_out
    if cr or cw:
        # the run already used caching → the actual cost IS the cache-aware cost.
        cost_cacheon = cost
    elif ins_per_call:
        # no-cache run → estimate cache-on: pay base on the per-turn NEW tokens (delta), cache-read on
        # the repeated prefix. First call is all-new (cache miss). Negative deltas (a level reset /
        # context eviction shrank the prompt) are treated as all-new for that call (conservative).
        in_cacheon = ins_per_call[0] * base_in
        for i in range(1, len(ins_per_call)):
            new = ins_per_call[i] - ins_per_call[i - 1]
            if new < 0:
                new = ins_per_call[i]            # prompt shrank → treat whole call as fresh (no reuse)
            repeated = ins_per_call[i] - new
            in_cacheon += new * base_in + max(0, repeated) * p_cr
        cost_cacheon = in_cacheon / 1e6 + out_cost
    else:
        cost_cacheon = cost
    return fresh + cr + cw, out, cost, cost_cacheon


def _env_records(record: Path):
    """Yield per-env record dicts from either input shape:
      - a run_parallel OUT-DIR of game_*.json (one EnvRunRecord each), or
      - a legacy single run JSON with an "envs" list."""
    if record.is_dir():
        for p in sorted(record.glob("game_*.json")):
            # SKIP a corrupt/half-written game record instead of crashing the whole listing. A run
            # still mid-write (or a truncated file from a crashed run) leaves malformed JSON; without
            # this guard, one bad file can make every viewer request fail or stall.
            try:
                yield json.loads(p.read_text())
            except (json.JSONDecodeError, OSError) as e:
                import sys as _sys
                print(f"[viewer] skipping unreadable record {p}: {e}", file=_sys.stderr)
                continue
    else:
        yield from json.loads(record.read_text()).get("envs", [])


def _record_mode(record: Path) -> str | None:
    """Best-effort run mode from a run_parallel directory manifest or a legacy JSON record."""
    try:
        if record.is_dir():
            mp = record / "manifest.json"
            if mp.exists():
                return json.loads(mp.read_text()).get("mode")
        else:
            return json.loads(record.read_text()).get("mode")
    except Exception:  # noqa: BLE001
        return None
    return None


def _compute_wm_predictions(record: Path, tmp_root: str | None, game_filter: str | None = None) -> dict:
    """For each game in a run, locate its workspace and recompute world-model predictions
    (plan-to-level_complete + per-action). Returns {short_game_id: {predictions, verify, ...}}. Best-effort:
    games with stub/unloadable models or missing temp dirs are simply absent from the map."""
    from tycho.viewer.wm_predict import find_workspaces, predict_for_workspace
    if _record_mode(record) == "no_world_model":
        print("    wm-predict: skipped for no_world_model run")
        return {}
    games = [env["game_id"].split("-")[0] for env in _env_records(record)]
    if game_filter:
        games = [g for g in games if g == game_filter]
    ws = find_workspaces(games, str(record) if record.is_dir() else tmp_root)
    if tmp_root:
        ws = {**find_workspaces(games, tmp_root), **ws}
    out = {}
    for g, d in ws.items():
        res = predict_for_workspace(d)
        npred = len(res.get("predictions") or {})
        has_quality = bool(res.get("verify") or res.get("outcome") or res.get("goal"))
        if npred or has_quality:
            out[g] = res
            print(f"    wm-predict {g}: {npred} predicted frames "
                  f"(verify simulation-acc={(res.get('verify') or {}).get('simulation_accuracy')})")
        else:
            print(f"    wm-predict {g}: none ({res.get('error', 'stub/no plan')})")
    return out


def _build_run(record: Path, out_dir: Path, game_filter: str | None, wm_predictions: dict | None = None):
    """Build one run into <out_dir>/<runId>/game_*.js. Returns (runId, manifest) or None
    if the record has no viz-able frames. Each game's data assigns
    window.GAMES[runId][gameId] so ids don't collide across runs."""
    run_id = record.stem if not record.is_dir() else record.name
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # model id (for per-token pricing) from manifest.json if present, else first recorded call.
    _model = ""
    _mp = record / "manifest.json" if record.is_dir() else None
    if _mp and _mp.exists():
        try:
            _model = json.loads(_mp.read_text()).get("model") or ""
        except Exception:  # noqa: BLE001
            pass
    if not _model:
        for env in _env_records(record):
            for t in env.get("trace", []):
                lc = (t.get("reasoning") or {}).get("llm_calls") or []
                if lc:
                    _model = lc[0].get("model") or ""
                    break
            if _model:
                break
    manifest, n_steps = [], 0
    for env in _env_records(record):
        short = env["game_id"].split("-")[0]
        if game_filter and short != game_filter:
            continue
        steps = build_steps(env)
        if not steps:
            continue
        # WORLD-MODEL PREDICTIONS (defensive, opt-in via --wm-predict): if the game produced a
        # loadable coded world_model.py, recompute its plan-to-level_complete / per-action predictions and
        # bake them onto the matching steps so the viewer can overlay them on the grid pane. This
        # Silently does nothing when the model is a stub, is unloadable, or its workspace is gone,
        # so optional overlays never break the base replay build.
        if wm_predictions and short in wm_predictions:
            pmap = wm_predictions[short].get("predictions") or {}
            vinfo = wm_predictions[short].get("verify") or {}
            oinfo = wm_predictions[short].get("outcome") or wm_predictions[short].get("goal") or {}
            for st in steps:
                r = st.get("reasoning")
                if not isinstance(r, dict):
                    r = {}
                    st["reasoning"] = r
                til = r.get("turn_in_level")
                key = f"{st['level']}_{til}" if til is not None else f"{st['level']}_{st['turn']}"
                if key in pmap:
                    st["wm_pred"] = pmap[key]
                if vinfo and not vinfo.get("error"):
                    r["verify"] = vinfo
                    if vinfo.get("simulation_accuracy") is not None:
                        r["simulation_accuracy"] = vinfo["simulation_accuracy"]
                if oinfo and not oinfo.get("error") and (oinfo.get("outcome_observable") or oinfo.get("goal_observable")):
                    if "level_complete_on_terminal" in oinfo or "game_over_on_death" in oinfo:
                        r["outcome"] = oinfo
                    else:
                        r["goal"] = oinfo
            if pmap or vinfo or oinfo:  # stamp run-level verify quality onto the manifest entry below
                env["_wm_verify"] = vinfo
        (run_dir / f"game_{short}.js").write_text(
            f"(window.GAMES[{json.dumps(run_id)}]=window.GAMES[{json.dumps(run_id)}]||{{}})"
            f"[{json.dumps(short)}]=" + json.dumps(steps) + ";")
        ti, to, cost, cost_co = _env_token_cost(env, _model)
        cr = cw = 0
        for t in env.get("trace", []):
            for c in (t.get("reasoning") or {}).get("llm_calls") or []:
                cr += int(c.get("cache_read") or 0); cw += int(c.get("cache_write") or 0)
        manifest.append({"id": short, "steps": len(steps),
                         "levels": max(s["levels"] for s in steps),
                         # Run-level stats: RHAE (env_score) and wall-clock per game.
                         "rhae": round(env.get("env_score", 0.0), 2),
                         "wall_s": round(env.get("wall_clock_s", 0.0), 1),
                         # token spend + USD list-price cost (None if model has no known price);
                         # cost_cacheon = cache-on equivalent estimate (== cost if caching was on)
                         "tok_in": ti, "tok_out": to,
                         "cache_read": cr, "cache_write": cw,
                         "cost": (round(cost, 2) if cost is not None else None),
                         "cost_cacheon": (round(cost_co, 2) if cost_co is not None else None)})
        # NOTE: the live server (serve.py:_run_manifest) also attaches a per-game "wm" (WM/plan
        # coverage) here, read from run_parallel's precomputed manifest. This LEGACY static builder
        # has no precomputed wm_activity and doesn't bake per-step verify.per_level, so it omits "wm".
        # The live server is the path in use; this exporter is kept for offline/static snapshots.
        n_steps += len(steps)
    if not manifest:
        return None
    # Run-level metadata for the viewer header. Source: the
    # run's manifest.json (run_parallel writes model/effort) + the dir mtime as the run time.
    meta = {}
    mpath = record / "manifest.json" if record.is_dir() else None
    if mpath and mpath.exists():
        try:
            mj = json.loads(mpath.read_text())
            meta["model"] = mj.get("model"); meta["effort"] = mj.get("effort")
            meta["git_version"] = mj.get("git_version")
            meta["hardware"] = mj.get("hardware")
            meta["workers"] = mj.get("workers")
            meta["wall_clock_s"] = mj.get("wall_clock_s")  # TRUE elapsed (new runs only)
        except Exception:  # noqa: BLE001
            pass
    # fall back to a recorded call's model/effort if the manifest lacked them
    if not meta.get("model"):
        for env in _env_records(record):
            for t in env.get("trace", []):
                lc = (t.get("reasoning") or {}).get("llm_calls") or []
                if lc:
                    meta["model"] = lc[0].get("model"); meta.setdefault("effort", lc[0].get("effort"))
                    break
            if meta.get("model"):
                break
    import datetime as _dt
    src = (record / "manifest.json") if (record.is_dir() and (record / "manifest.json").exists()) else record
    try:
        meta["run_time"] = _dt.datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        meta["run_time"] = ""
    print(f"  run {run_id}: {len(manifest)} game(s), {n_steps} steps")
    return run_id, {"games": manifest, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record", nargs="+",
                    help="one or more run_parallel out-dirs (game_*.json) / legacy results/*.json. "
                         "Multiple runs become a run-switcher dropdown in the viewer.")
    ap.add_argument("--game", default=None, help="only this short game id (default: all)")
    ap.add_argument("--out-dir", default=None,
                    help="viewer dir (default: results/viz_<first-record-name>/)")
    ap.add_argument("--wm-predict", action="store_true",
                    help="recompute world-model plan/per-action predictions from each game's "
                         "workspace world_model.py and overlay them on the grid pane. No-op for "
                         "games whose model is a stub / unloadable / whose temp dir is gone.")
    ap.add_argument("--wm-tmp", default=None,
                    help="extra root to search for arcws_* workspaces (default: $TMPDIR, /tmp).")
    args = ap.parse_args()

    records = [Path(r) for r in args.record]
    first = records[0]
    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO / "results" / f"viz_{first.name if first.is_dir() else first.stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for rec in records:
        wm_pred = None
        if args.wm_predict:
            wm_pred = _compute_wm_predictions(rec, args.wm_tmp, args.game)
        built = _build_run(rec, out_dir, args.game, wm_predictions=wm_pred)
        if built:
            runs[built[0]] = built[1]
    if not runs:
        raise SystemExit("no viz-able frames found in any record. Re-run the eval with --viz.")

    title = first.name if first.is_dir() else first.stem
    html = (HTML.replace("__RUNS__", json.dumps(runs))
                .replace("__PAL__", json.dumps(ARC16))
                .replace("__TITLE__", title))
    (out_dir / "index.html").write_text(html)
    print(f"viewer: {len(runs)} run(s) -> {out_dir}/index.html")
    print(f"open: file://{(out_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
