#!/bin/bash
#SBATCH --job-name=analyze_games
#SBATCH --output=log_analyze_games_%j.out
#SBATCH --error=log_analyze_games_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --partition=cpu

echo "Running script"
# Run the Python script
python3 analyze_games.py -f -m qwen2,llama3 -v llama3.1 -j 0 -t 2
