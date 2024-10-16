#!/bin/bash
#SBATCH --job-name=generate_samples
#SBATCH --output=log_generate_samples_%j.out
#SBATCH --error=log_generate_samples_%j.err
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
for b in 2 4 8 16; do
    time python3 generate_samples.py -m nemotron -n 5 -s 10000 -b $b
done
