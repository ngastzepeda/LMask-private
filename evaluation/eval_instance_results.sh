#!/bin/bash

# Wrapper for eval_instance_results.py: source the venv, then run the per-instance
# eval on the workstation's GPU. Works from the project root or from evaluation/.
# stdout/stderr are mirrored into a timestamped log file, in addition to the
# per-checkpoint csvs eval_instance_results.py writes under
# evaluation/instance_results/{size}/ and its own rotating
# evaluation/logs/eval_instance_results.log.
#
#   bash evaluation/eval_instance_results.sh              # eval all best ckpts on cuda
#   bash evaluation/eval_instance_results.sh --modes best last  # forward extra args as-is

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "Error: Virtual environment not found in $ROOT_DIR/.venv"
    exit 1
fi

LOG_DIR="$ROOT_DIR/evaluation/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/instance_results_$(date +%Y%m%d_%H%M%S).log"

python "$ROOT_DIR/evaluation/eval_instance_results.py" --device cuda "$@" 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
