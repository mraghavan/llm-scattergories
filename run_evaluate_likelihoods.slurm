#!/bin/bash
#SBATCH --job-name=eval_likelihoods
#SBATCH --output=log_eval_likelihoods_%A_%a.out
#SBATCH --error=log_eval_likelihoods_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --array=0-10%4

# Check if min_count argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide a min-count value"
    echo "Usage: sbatch $0 <min-count>"
    exit 1
fi

export HF_TOKEN=$(cat ~/.HF_TOKEN)

time python3 evaluate_answer_likelihoods.py --job-num $SLURM_ARRAY_TASK_ID --num-jobs $SLURM_ARRAY_TASK_COUNT --min-count $1 -n 5
