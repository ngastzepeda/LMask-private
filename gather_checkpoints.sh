#!/bin/bash

# Wrapper for gather_checkpoints.py (same convention as resume_jobs.sh):
# load the Python module, source the venv, then run the gather script.
#
#   bash gather_checkpoints.sh                # gather into checkpoints/ and git add
#   bash gather_checkpoints.sh --no-git-add   # just copy, don't touch git

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed for torch to read epoch/score from ckpts)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

python "gather_checkpoints.py" $*
