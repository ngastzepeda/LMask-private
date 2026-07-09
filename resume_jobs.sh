#!/bin/bash

# Wrapper for resume_jobs.py (same convention as AMAI2025's submit_jobs_skillvrp.sh):
# load the Python module, source the venv, then run the discovery script.
#
#   bash resume_jobs.sh                    # scan logs/runs and sbatch the resumes
#   bash resume_jobs.sh --dry_run          # only print what would be submitted
#   bash resume_jobs.sh --include-finished # also resume runs that reached max_epochs

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed e.g. for torch in the epoch check)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

python "resume_jobs.py" $*
