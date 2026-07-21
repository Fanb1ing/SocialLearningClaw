#!/usr/bin/env bash
# Run the eight ARC-AGI-3 baselines plus the schema method on three games.
# Methods are sequential; games within one method run in parallel.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=".venv/bin/python"
MODEL="${MODEL:-google/gemini-2.5-pro}"
MAX_STEPS="${MAX_STEPS:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
LOG_DIR="$OUTPUT_ROOT/logs/arc_agi3"

GAMES=("sk48-d8078629" "cd82-fb555c5d" "tu93-0768757b")
METHODS=("naive" "icl" "rag" "withrule" "reflexion" "expel" "amem" "tgm" "schema")

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_batch() {
    local method="$1"
    log "Batch: $method"
    for game in "${GAMES[@]}"; do
        local short_name="${game%%-*}"
        local log_path="$LOG_DIR/${method}_${short_name}.log"
        "$PY" -m socialclaw.run_arc \
            --method "$method" \
            --game-id "$game" \
            --model "$MODEL" \
            --max-steps "$MAX_STEPS" \
            --max-attempts 1 \
            --output-root "$OUTPUT_ROOT" \
            >"$log_path" 2>&1 &
    done
    wait
}

log "ARC-AGI-3 experiments: model=$MODEL max_steps=$MAX_STEPS"
for method in "${METHODS[@]}"; do
    run_batch "$method"
done

"$PY" scripts/eval_arc_summary.py --runs-dir "$OUTPUT_ROOT" --model "$MODEL"
