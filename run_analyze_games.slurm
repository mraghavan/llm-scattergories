#!/bin/bash
#SBATCH --job-name=analyze_games
#SBATCH --output=log_analyze_games_%A_%a.out
#SBATCH --error=log_analyze_games_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu
#SBATCH --array=0-29

echo "Running script"
# Run the Python script
echo $SLURM_ARRAY_TASK_ID
echo $SLURM_ARRAY_TASK_COUNT
python3 analyze_games.py -m all -v llama3.1 -j $SLURM_ARRAY_TASK_ID -t $SLURM_ARRAY_TASK_COUNT
