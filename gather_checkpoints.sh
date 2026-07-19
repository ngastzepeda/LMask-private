#!/bin/bash
#SBATCH --job-name=lmask-gather-ckpts
#SBATCH --output=out/gather_ckpt/%j.out
#SBATCH --error=out/gather_ckpt/%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Thread control (matching start_job.sh)
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Create output directory
mkdir -p out/gather_ckpt

# Wrapper for gather_checkpoints.py (same convention as resume_jobs.sh):
# load the Python module, source the venv, then run the gather script.
#
#   sbatch gather_checkpoints.sh                # copy into checkpoints/ (no git)
#   sbatch gather_checkpoints.sh --git-add      # ...and git add the folder

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment (needed for torch to read epoch/score from ckpts)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Error: Virtual environment not found in $(pwd)/.venv"
    exit 1
fi

python "gather_checkpoints.py" $*
