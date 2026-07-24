#!/bin/bash

# Wrapper for eval_benchmarks.py: source the venv, then evaluate the trained
# TSPTW checkpoints on the classic OOD benchmark libraries under
# data/benchmarks/tsptw/ (Dumas, GendreauDumasExtended, OhlmannThomas, Langevin).
# Works from the project root or from evaluation/benchmarks/. stdout/stderr are
# mirrored into a timestamped log file, in addition to the per-checkpoint csvs
# eval_benchmarks.py writes under evaluation/benchmarks/results/{set}/ and its own
# rotating evaluation/benchmarks/logs/eval_benchmarks.log.
#
#   bash evaluation/benchmarks/eval_benchmarks.sh                    # best ckpts, size-matched, cuda
#   bash evaluation/benchmarks/eval_benchmarks.sh --modes best last  # forward extra args as-is

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
else
    echo "Error: Virtual environment not found in $ROOT_DIR/.venv"
    exit 1
fi

LOG_DIR="$ROOT_DIR/evaluation/benchmarks/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/benchmarks_$(date +%Y%m%d_%H%M%S).log"

python "$ROOT_DIR/evaluation/benchmarks/eval_benchmarks.py" --device cuda "$@" 2>&1 | tee "$LOG_FILE"
exit "${PIPESTATUS[0]}"
