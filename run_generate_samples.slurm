#!/bin/bash
#SBATCH --job-name=generate_samples
#SBATCH --output=log_generate_samples_%A_%a.out
#SBATCH --error=log_generate_samples_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100
#SBATCH --array=0-5%3

export HF_TOKEN=$(cat ~/.HF_TOKEN)

time python3 generate_samples.py -m gemma2,llama3.1,llama3.2,mistral,nemotron,phi3.5 -n 25 -s 2000 -b 6 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
