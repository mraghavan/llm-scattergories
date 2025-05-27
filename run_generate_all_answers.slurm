#!/bin/bash
#SBATCH --job-name=generate_answers
#SBATCH --output=log_generate_answers_%A_%a.out
#SBATCH --error=log_generate_answers_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100
#SBATCH --array=0-10%4

export HF_TOKEN=$(cat ~/.HF_TOKEN)

time python3 generate_samples.py -c models -n 10 -s 50 -b 6 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
