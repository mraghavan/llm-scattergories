#!/bin/bash
# Script to compute equilibria, plot them, and plot diversity for dreamsim and lpips metrics
# For models: cogview4, flux1.dev, pixart-sigma, sd3.5
#
# Usage:
#   ./run_equilibria_analysis.sh              # Run full pipeline
#   ./run_equilibria_analysis.sh --skip-compute  # Skip equilibrium computation, only plot

set -e  # Exit on error

# Parse arguments
SKIP_COMPUTE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-compute)
            SKIP_COMPUTE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--skip-compute]"
            exit 1
            ;;
    esac
done

# Models to use
MODELS=("cogview4" "flux1.dev" "pixart-sigma" "sd3.5")

# Metrics to process
METRICS=("dreamsim" "lpips")

echo "=========================================="
echo "Running equilibria analysis pipeline"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Metrics: ${METRICS[*]}"
if [ "$SKIP_COMPUTE" = true ]; then
    echo "Skipping equilibrium computation"
fi
echo ""

# Step 1: Compute equilibria for each metric
if [ "$SKIP_COMPUTE" = false ]; then
    echo "Step 1: Computing equilibria..."
    for metric in "${METRICS[@]}"; do
        echo ""
        echo "Computing equilibria for metric: $metric"
        python3 compute_equilibria.py \
            --distance-metric "$metric" \
            --models "${MODELS[@]}" \
            --max-players 20
    done
else
    echo "Step 1: Skipping equilibrium computation (--skip-compute flag set)"
fi

# Step 2: Plot equilibria for each metric
echo ""
echo "Step 2: Plotting equilibria..."
for metric in "${METRICS[@]}"; do
    echo ""
    echo "Plotting equilibria for metric: $metric"
    python3 plot_equilibria.py \
        --distance-metric "$metric"
done

# Step 3: Plot diversity for all metrics at once
echo ""
echo "Step 3: Plotting diversity..."
python3 plot_diversity.py \
    --metrics "${METRICS[@]}"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="

