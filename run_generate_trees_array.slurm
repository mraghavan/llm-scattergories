#!/bin/bash
#SBATCH --job-name=generate_trees
#SBATCH --output=log_generate_trees_%A_%a.out
#SBATCH --error=log_generate_trees_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100
#SBATCH --array=0-6%3

# Print out current environment details
echo "GPU information:"
nvidia-smi

export HF_TOKEN=$(cat ~/.HF_TOKEN)

echo "Running script"
# Run the Python script
python3 generate_trees.py -f -m all -n 50 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
