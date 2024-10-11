#!/bin/bash

gen_id=$(sbatch --parsable run_generate_trees_array.slurm)
echo "Submitted generate_trees job with ID $gen_id"
v_id=$(sbatch --parsable --dependency=afterok:$gen_id run_generate_trees.slurm)
echo "Submitted verify_trees job with ID $v_id"
sbatch --dependency=afterok:$v_id run_analyze_games.slurm
