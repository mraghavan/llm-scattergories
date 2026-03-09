#!/bin/bash
# Helper script to automatically calculate the number of icons and submit
# the SLURM array job with the correct array size.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Count the number of base names (icons)
NUM_ICONS=$(ls generated_images/*_original.png 2>/dev/null | wc -l | tr -d ' ')

if [ "$NUM_ICONS" -eq 0 ]; then
    echo "Error: No icons found in generated_images/"
    echo "Expected files matching pattern: *_original.png"
    exit 1
fi

# Array indices are 0-based, so we need 0 to (NUM_ICONS-1)
ARRAY_MAX=$((NUM_ICONS - 1))

echo "Found $NUM_ICONS icon(s)"
echo "Submitting SLURM array job with array size: 0-$ARRAY_MAX"

# Create a temporary SLURM script with the correct array size
TEMP_SLURM=$(mktemp)
sed "s/^#SBATCH --array=.*/#SBATCH --array=0-${ARRAY_MAX}%20/" run_evaluate_pairwise.slurm > "$TEMP_SLURM"

# Submit the job
sbatch "$TEMP_SLURM" "$@"

# Clean up
rm "$TEMP_SLURM"

echo "Job submitted. Use 'squeue -u $USER' to check status."

