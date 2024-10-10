#!/bin/bash
#SBATCH --job-name=hf_cache_test
#SBATCH --output=log_cache_test.out
#SBATCH --error=log_cache_test.err
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
python3 completion_hf.py
python3 completion_cached_hf.py
