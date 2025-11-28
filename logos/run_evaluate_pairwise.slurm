#!/bin/bash
#SBATCH --job-name=eval_pairwise
#SBATCH --output=log_eval_pairwise_%A_%a.out
#SBATCH --error=log_eval_pairwise_%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --array=0-24%3

# Optional: HF token not strictly needed here unless models require auth
export HF_TOKEN=$(cat ~/.HF_TOKEN)

# Change to the logos directory (where this script is located)
# This allows the script to be run from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Script directory: $SCRIPT_DIR"
echo "Current working directory: $(pwd)"
echo "Checking for generated_images directory..."
if [ ! -d "generated_images" ]; then
    echo "ERROR: generated_images directory not found!"
    echo "Looking in: $(pwd)"
    ls -la . | head -10
    exit 1
fi

echo "Found generated_images directory"
echo "Looking for *_original.png files..."

# Find all base names from original images
# Pattern: {base_name}_original.png
BASE_NAMES=()
shopt -s nullglob  # Make globs return empty if no matches
for orig_file in generated_images/*_original.png; do
    if [ -f "$orig_file" ]; then
        # Extract base name: remove path, remove _original.png, remove extension
        base_name=$(basename "$orig_file" _original.png)
        BASE_NAMES+=("$base_name")
        echo "  Found: $orig_file -> base_name: $base_name"
    fi
done
shopt -u nullglob  # Restore default glob behavior

echo "Total base names found: ${#BASE_NAMES[@]}"

# Sort base names for consistent ordering
IFS=$'\n' BASE_NAMES=($(sort <<<"${BASE_NAMES[*]}"))
unset IFS

NUM_BASE_NAMES=${#BASE_NAMES[@]}

if [ $NUM_BASE_NAMES -eq 0 ]; then
    echo "Error: No base names found in generated_images/"
    exit 1
fi

# Check if SLURM_ARRAY_TASK_ID is within valid range
if [ $SLURM_ARRAY_TASK_ID -ge $NUM_BASE_NAMES ]; then
    echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is >= number of base names ($NUM_BASE_NAMES)"
    echo "This array task will do nothing."
    exit 0
fi

# Get the base name for this array task
BASE_NAME="${BASE_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "Processing base_name: $BASE_NAME (task $SLURM_ARRAY_TASK_ID of $NUM_BASE_NAMES)"

# Activate virtual environment if it exists
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

# Run the pairwise evaluation script
# Pass through any additional arguments (e.g., --no-lpips, --models, etc.)
time python3 evaluate_pairwise.py --base-name "$BASE_NAME" "$@"

