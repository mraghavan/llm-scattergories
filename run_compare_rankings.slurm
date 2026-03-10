#!/bin/bash
#SBATCH --job-name=compare_rankings
#SBATCH --output=log_compare_rankings_%j.out
#SBATCH --error=log_compare_rankings_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail

MIN_COUNT="${1:-75}"
NUM_INSTANCES="${2:-10}"
OUTPUT_PKL="${3:-info/compare_rankings_n${NUM_INSTANCES}_min${MIN_COUNT}.pkl}"

time python3 compare_rankings.py \
  --no-plots \
  --min-count "$MIN_COUNT" \
  -n "$NUM_INSTANCES" \
  --output-pkl "$OUTPUT_PKL"
