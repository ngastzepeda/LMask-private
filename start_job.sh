#!/bin/bash

#SBATCH --output=out/%x_%j.out
#SBATCH --error=out/%x_%j.err
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "lmask"
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-winf4gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nayeli.gast.zepeda@univie.ac.at

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# wandb-core sometimes needs >30s (default) to start on busy nodes; a timeout
# kills the job at wandb.init with ServicePollForTokenError. Wait up to 5 min.
export WANDB__SERVICE_WAIT=300

# Use the directory from which the job was submitted
current_folder="$SLURM_SUBMIT_DIR"
cd "$current_folder" || { echo "Error: Could not change to $current_folder"; exit 1; }

module load lang/Python/3.12.3-GCCcore-13.2.0

# Source the virtual environment
if [ -f "$current_folder/.venv/bin/activate" ]; then
    source "$current_folder/.venv/bin/activate"
else
    echo "Error: Virtual environment not found in $current_folder/.venv"
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: sbatch start_job.sh <experiment_name>"
    echo "  e.g. sbatch start_job.sh feasible/lmask/n20_amai"
    exit 1
fi

experiment="$1"
shift

# Build the wandb run name from the experiment path:
#   feasible/<group>/<cfg>  ->  <group>_<cfg with "amai" stripped>
#   e.g. feasible/amai/n100_amai_mw -> amai_n100_mw ; feasible/lmask/n20_amai -> lmask_n20
group="$(basename "$(dirname "$experiment")")"
cfg="$(basename "$experiment")"
cleaned="${cfg//amai/}"      # drop the redundant "amai" token
cleaned="${cleaned//__/_}"   # collapse any resulting double underscore
cleaned="${cleaned#_}"       # trim leading underscore
cleaned="${cleaned%_}"       # trim trailing underscore
run_name="${group}_${cleaned}"

mkdir -p "$current_folder/out"

# Start the cluster job.
# RichProgressBar crashes when stdout is not a TTY (Slurm redirects to a file)
python "$current_folder/run.py" \
    experiment="$experiment" \
    logger.wandb.name="$run_name" \
    '~callbacks.rich_progress_bar' \
    ++trainer.enable_progress_bar=False \
    "$@"
