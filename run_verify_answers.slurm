#!/bin/bash
#SBATCH --job-name=verify_answers
#SBATCH --output=log_verify_answers_%j.out
#SBATCH --error=log_verify_answers_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

export HF_TOKEN=$(cat ~/.HF_TOKEN)

time python3 verify_samples.py -c models -v qwen2.5 -i out
