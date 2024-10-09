#!/bin/bash
#SBATCH --job-name=hf_model_test
#SBATCH --output=log_generate_trees_%j.out
#SBATCH --error=log_generate_trees_%j.err
#SBATCH --time=00:10:00
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
python3 generate_trees.py -f -t 0.5 -m smollm -n 1
