#!/bin/bash
#SBATCH --job-name=get_answer_space_size
#SBATCH --output=log_get_answer_space_size_%j.out
#SBATCH --error=log_get_answer_space_size_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

export HF_TOKEN=$(cat ~/.HF_TOKEN)

python3 get_answer_space_size.py -m gemma2,llama3.1,llama3.2,mistral,nemotron,phi3.5

