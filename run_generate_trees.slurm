#!/bin/bash
#SBATCH --job-name=generate_trees
#SBATCH --output=log_generate_trees_%j.out
#SBATCH --error=log_generate_trees_%j.err
#SBATCH --time=03:00:00
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
python3 generate_trees.py -f -m nemotron,minitron -v llama3.1 -n 20
