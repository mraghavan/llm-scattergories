#!/bin/bash
#SBATCH --job-name=count_params
#SBATCH --output=log_count_parameters_%j.out
#SBATCH --error=log_count_parameters_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-rtx8000

export HF_TOKEN=$(cat ~/.HF_TOKEN)

# Usage examples:
#   sbatch run_count_parameters.slurm
#   sbatch run_count_parameters.slurm --models cogview4 flux1.dev
#   sbatch run_count_parameters.slurm --enable-cpu-offload

source ../venv/bin/activate
time python3 count_parameters.py "$@"

