#!/bin/bash
#SBATCH --job-name=analyze_games
#SBATCH --output=log_analyze_games_%j.out
#SBATCH --error=log_analyze_games_%j.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu
##SBATCH --array=0-5

echo "Running script"
# Run the Python script
python3 analyze_games.py -f -m llama3,smollm,llama3.1,llama3.2 -v llama3.1 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
