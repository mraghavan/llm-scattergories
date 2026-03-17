#!/bin/bash
set -euo pipefail

MIN_COUNT="${1:-10}"

find models -maxdepth 1 -name '*.json' -delete
python3 make_model_configs.py

gen_id=$(sbatch --parsable run_generate_all_answers.slurm)
echo "Submitted generation job: $gen_id"

ver_id=$(sbatch --parsable --dependency=afterok:"$gen_id" run_verify_answers.slurm)
echo "Submitted verification job: $ver_id"

cands_id=$(sbatch --parsable --dependency=afterok:"$ver_id" run_candidate_answers.slurm "$MIN_COUNT" 25)
echo "Submitted candidate-answers job: $cands_id"

eval_id=$(sbatch --parsable --dependency=afterok:"$cands_id" run_evaluate_likelihoods.slurm "$MIN_COUNT")
echo "Submitted likelihood-eval job: $eval_id"

compare_id=$(sbatch --parsable --dependency=afterok:"$eval_id" run_compare_rankings.slurm "$MIN_COUNT" 25 "info/compare_rankings_n25_min${MIN_COUNT}.pkl")
echo "Submitted compare-rankings job: $compare_id"

echo "Done. Submitted canonical models pipeline for n=25."
