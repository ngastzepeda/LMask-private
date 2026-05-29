#!/bin/bash

# Submits one Slurm job per experiment config under configs/experiment/feasible/.
# Each job is launched via start_job.sh with the experiment path passed as the
# Hydra `experiment=` argument (no .yaml suffix, relative to configs/experiment/).

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_root="$script_dir/configs/experiment"
feasible_dir="$config_root/feasible"

if [ ! -d "$feasible_dir" ]; then
    echo "Error: $feasible_dir not found"
    exit 1
fi

shopt -s nullglob
configs=("$feasible_dir"/*/*.yaml)
shopt -u nullglob

if [ ${#configs[@]} -eq 0 ]; then
    echo "No .yaml configs found under $feasible_dir"
    exit 1
fi

for cfg in "${configs[@]}"; do
    # Strip the configs/experiment/ prefix and the .yaml suffix
    rel="${cfg#$config_root/}"
    experiment="${rel%.yaml}"

    # Use the experiment slug as the Slurm job name (drives %x in output files)
    job_name="${experiment//\//_}"

    echo "Submitting: $experiment (job-name=$job_name)"
    sbatch --job-name="$job_name" "$script_dir/start_job.sh" "$experiment"
done
