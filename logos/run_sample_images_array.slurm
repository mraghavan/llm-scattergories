#!/bin/bash
#SBATCH --job-name=sample_images_array
#SBATCH --output=log_sample_images_array_%A_%a.out
#SBATCH --error=log_sample_images_array_%A_%a.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --array=0-24%5
# Note: Array range 0-24 for 25 icons (one per task)

# Optional: HF token for models that require authentication
export HF_TOKEN=$(cat ~/.HF_TOKEN)

# Parse arguments: --model MODEL --max-samples N --samples-per-task N
# Default values
MODEL="sd15"
MAX_SAMPLES=25
SAMPLES_PER_TASK=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --samples-per-task)
            SAMPLES_PER_TASK="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments
            break
            ;;
    esac
done

# Calculate which samples this array task should process
START_IDX=$((SLURM_ARRAY_TASK_ID * SAMPLES_PER_TASK))
END_IDX=$((START_IDX + SAMPLES_PER_TASK))

# Make sure we don't exceed max_samples
if [ $START_IDX -ge $MAX_SAMPLES ]; then
    echo "SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) would process samples starting at $START_IDX, which exceeds max_samples ($MAX_SAMPLES)"
    echo "This array task will do nothing."
    exit 0
fi

# Cap end_idx at max_samples
if [ $END_IDX -gt $MAX_SAMPLES ]; then
    END_IDX=$MAX_SAMPLES
fi

echo "=========================================="
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Processing samples: $START_IDX to $((END_IDX - 1)) (indices $START_IDX-$((END_IDX - 1)))"
echo "Total samples to process: $MAX_SAMPLES"
echo "Samples per task: $SAMPLES_PER_TASK"
echo "=========================================="

# Activate virtual environment if it exists
if [ -f "../venv/bin/activate" ]; then
    source ../venv/bin/activate
fi

# Run the sample_images script with the specific model and sample range
# Pass through any remaining arguments (e.g., --num-images, --enable-cpu-offload, etc.)
time python3 sample_images.py \
    --models "$MODEL" \
    --max-samples "$MAX_SAMPLES" \
    --start-index "$START_IDX" \
    --end-index "$END_IDX" \
    "$@"

echo "Completed processing samples $START_IDX to $((END_IDX - 1)) for model: $MODEL"
