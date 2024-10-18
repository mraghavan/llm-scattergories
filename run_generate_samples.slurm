#!/bin/bash
#SBATCH --job-name=generate_samples
#SBATCH --output=log_generate_samples_%A_%a.out
#SBATCH --error=log_generate_samples_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-v100
#SBATCH --array=0-5%3

# Print out current environment details
echo "GPU information:"
nvidia-smi

export HF_TOKEN=$(cat ~/.HF_TOKEN)

echo "Running script"
# Run the Python script
time python3 generate_samples.py -m nemotron,phi3.5,llama3.2,mistral,smollm,llama3 -n 20 -s 2000 -b 6 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
#for b in 6 16; do
    #time python3 generate_samples.py -m nemotron -n 10 -s 500 -b $b
#done
