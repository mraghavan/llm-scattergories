#!/bin/bash
#SBATCH --job-name=eval_pairwise
#SBATCH --output=log_eval_pairwise_%j.out
#SBATCH --error=log_eval_pairwise_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-rtx8000

# Optional: HF token not strictly needed here unless models require auth
export HF_TOKEN=$(cat ~/.HF_TOKEN)

echo "Current working directory: $(pwd)"

# Activate virtual environment if it exists
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

# Run the pairwise evaluation script
# Pass all arguments through (e.g., --base-name, --no-dreamsim, --models, etc.)
# If --base-name is not provided, it will process all base_names
time python3 evaluate_pairwise.py "$@"

