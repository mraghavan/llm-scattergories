#!/bin/bash
#SBATCH --job-name=evaluate_logos
#SBATCH --output=log_evaluate_logos_%j.out
#SBATCH --error=log_evaluate_logos_%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

# Optional: HF token not strictly needed here unless models require auth
export HF_TOKEN=$(cat ~/.HF_TOKEN)


# Usage examples:
#   sbatch run_evaluate_images.slurm
#   sbatch run_evaluate_images.slurm --device cuda
#   sbatch run_evaluate_images.slurm --device cpu --no-dino
#   sbatch run_evaluate_images.slurm --output-csv results_clip_lpips_dino.csv

time python3 evaluate_images.py "$@"


