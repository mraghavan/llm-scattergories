#!/bin/bash
#SBATCH --job-name=test_generate_trees
#SBATCH --output=log_test_generate_trees_%j.out
#SBATCH --error=log_test_generate_trees_%j.err
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
python3 generate_trees.py -m llama3 -n 10 -d 5 -o test_trees
