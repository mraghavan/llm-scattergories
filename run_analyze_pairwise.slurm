#!/bin/bash
#SBATCH --job-name=analyze_pairwise
#SBATCH --output=log_analyze_pairwise_%A_%a.out
#SBATCH --error=log_analyze_pairwise_%A_%a.err
#SBATCH --time=0-10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --array=0-80

# Run the Python script
time python3 analyze_pairwise.py -m gemma2,llama3.1,llama3.2,mistral,nemotron,phi3.5 -v qwen2.5 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
