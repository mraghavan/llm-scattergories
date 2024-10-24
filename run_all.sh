#!/bin/bash

gen_id=$(sbatch --parsable run_generate_samples.slurm)
echo "Submitted generate_trees job with ID $gen_id"
v_id=$(sbatch --parsable --dependency=afterok:$gen_id run_verify_samples.slurm)
echo "Submitted verify_trees job with ID $v_id"
