#!/bin/bash
#SBATCH --job-name=sample_images
#SBATCH --output=log_sample_images_%j.out
#SBATCH --error=log_sample_images_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

export HF_TOKEN=$(cat ~/.HF_TOKEN)

# Change to the logos directory to run the script
cd logos

# Pass all arguments to the Python script
# Usage examples:
#   sbatch run_sample_images.slurm
#   sbatch run_sample_images.slurm --models sd15 sdxl --num-images 5
#   sbatch run_sample_images.slurm --models flux --num-images 3 --no-quantize
time python3 sample_images.py "$@"

