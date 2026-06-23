#!/usr/bin/env bash
# Launch all 15 ARC-AGI-3 baseline experiments in 5 batches (3 parallel per batch).
# Each batch = one baseline method across all 3 games.
# Batches run sequentially to avoid OpenRouter concurrent-request limits.
#
# Results layout:
#   runs/arc_zero_shot/<model>/<timestamp>/<game>_L*/   zero-shot
#   runs/arc_few_shot/<model>/<timestamp>/<game>_L*/    few-shot ICL
#   runs/arc_rag/<model>/<timestamp>/<game>_L*/         RAG (online buffer)
#   runs/arc_withrule/<model>/<timestamp>/<game>_L*/    human-written rules
#   runs/arc_agi3/<model>/<timestamp>/<game>_L*/        Schema (our method)
#
# Logs:  runs/logs/<method>_<game>.log
# Summary: .venv/bin/python scripts/eval_arc_summary.py

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
MODEL="google/gemini-2.5-pro"
MAX_STEPS=200
RUNS_DIR="runs"
LOGDIR="runs/logs"

GAMES=("sk48-d8078629" "cd82-fb555c5d" "tu93-0768757b")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_batch() {
    local MODE=$1
    log "━━━ Batch: $MODE (3 games in parallel) ━━━"
    for GAME in "${GAMES[@]}"; do
        GNAME="${GAME%%-*}"
        LOG="$LOGDIR/${MODE}_${GNAME}.log"
        log "  START $MODE $GAME  →  $LOG"
        if [ "$MODE" = "schema" ]; then
            $PY -m socialclaw.run_arc_agi3 \
                --game-id "$GAME" --auto-yes \
                --model "$MODEL" --max-steps "$MAX_STEPS" --runs-dir "$RUNS_DIR" \
                > "$LOG" 2>&1 &
        else
            $PY scripts/run_arc_baselines.py \
                --mode "$MODE" --game-id "$GAME" \
                --model "$MODEL" --max-steps "$MAX_STEPS" --runs-dir "$RUNS_DIR" \
                > "$LOG" 2>&1 &
        fi
    done
    log "  Waiting for $MODE batch to finish ..."
    wait
    log "  $MODE batch done."
    echo ""
}

log "=== ARC-AGI-3 Full Baseline Run ==="
log "Model: $MODEL | Max steps/level: $MAX_STEPS"
log "Games: ${GAMES[*]}"
log "Methods: zero_shot  few_shot  rag  withrule  schema"
log "Logs → $LOGDIR/"
log ""

run_batch "zero_shot"
run_batch "few_shot"
run_batch "rag"
run_batch "withrule"
run_batch "schema"

log "=== All 15 experiments finished ==="
log ""
log "Results summary:"
$PY scripts/eval_arc_summary.py
