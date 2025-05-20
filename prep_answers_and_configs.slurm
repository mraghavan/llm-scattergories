#!/bin/bash
#SBATCH --job-name=prep_answers_configs
#SBATCH --output=log_prep_answers_configs_%A_%a.out
#SBATCH --error=log_prep_answers_configs_%A_%a.err
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu

export HF_TOKEN=$(cat ~/.HF_TOKEN)

# Run make_model_configs first
time python3 make_model_configs.py

# Then run candidate_answers
time python3 candidate_answers.py --min-count 100