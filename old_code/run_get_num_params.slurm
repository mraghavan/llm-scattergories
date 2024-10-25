#!/bin/bash
#SBATCH --job-name=num_params
#SBATCH --output=log_num_params.out
#SBATCH --error=log_num_params.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu

echo "Running script"
# Run the Python script
python3 get_num_params.py
