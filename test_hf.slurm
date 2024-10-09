#!/bin/bash
#SBATCH --job-name=hf_model_test
#SBATCH --output=hf_completion_test.out
#SBATCH --error=hf_completion_test.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100

# Print out current environment details
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"
echo "GPU information:"
nvidia-smi

export HF_TOKEN=$(cat ~/.HF_TOKEN)

echo "Running script"

# Run the Python script
python3 completion_hf.py
