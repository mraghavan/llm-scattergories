#!/bin/bash
#SBATCH --job-name=candidate_answers
#SBATCH --output=log_candidate_answers_%j.out
#SBATCH --error=log_candidate_answers_%j.err
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu

set -euo pipefail

MIN_COUNT="${1:-10}"
NUM_INSTANCES="${2:-25}"

python3 candidate_answers.py --min-count "$MIN_COUNT" -n "$NUM_INSTANCES" --input-dir out
