#!/bin/bash
#SBATCH --job-name=verify_samples
#SBATCH --output=log_verify_samples_%j.out
#SBATCH --error=log_verify_samples_%j.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

# Print out current environment details
echo "GPU information:"
nvidia-smi

export HF_TOKEN=$(cat ~/.HF_TOKEN)

echo "Running script"
# Run the Python script
time python3 verify_samples.py -m nemotron,phi3.5,smollm -n 10 -v qwen2.5
