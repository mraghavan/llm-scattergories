#!/bin/bash

NUM_INSTANCES=25
NUM_SAMPLES=2000


# Should run the following on a cluster, and might want to split the them over array jobs
# Generate samples (can reduce -s 2000 to something like 100 for faster testing)
python3 generate_samples.py -m nemotron,phi3.5,llama3.2,mistral,smollm,llama3.1 -n $NUM_INSTANCES -s $NUM_SAMPLES -b 6

# Verify samples
# Current slurm script isn't set up to run as an array job. Can modify to do so, but would also need to modify verify_samples.py
python3 verify_samples.py -m nemotron,phi3.5,llama3.2,mistral,smollm,llama3.1 -n $NUM_INSTANCES -v qwen2.5

# Could use ./run_all.sh instead of the above two commands if running on a cluster.
# Need to appropriately modify run_generate_samples.slurm and run_verify_samples.slurm

# Analyze equilibria (can be done locally)
python3 analyze_samples.py -m nemotron,phi3.5,llama3.2,mistral,smollm,llama3.1 -v qwen2.5

# Probably should run this on a cluster (see run_analyze_pairwise.slurm)
python3 analyze_pairwise.py -m nemotron,phi3.5,llama3.2,mistral,smollm,llama3.1 -v qwen2.5

# plots
# Main temp and utility plots (creates 5 figures)
python3 make_plots.py -m smollm,mistral,nemotron,llama3.2,phi3.5,llama3.1

# Utility as a function of temp for different values of n
# Creates 6 figures
python3 plot_opt_over_temp.py -m phi3.5,llama3.1,llama3.2,smollm,mistral,nemotron -v qwen2.5

# Inversions: Weighted inversions and prob mass captured (2 figures)
python3 plot_inversions.py -m phi3.5,llama3.1,llama3.2,smollm,mistral,nemotron

# Pairwise
# Separately run 2 for main figures
python3 plot_pairwise.py -m llama3.1,llama3.2
python3 plot_pairwise.py -m llama3.1,phi3.5
# One big plot with all pairwise comparisons
python3 make_big_pairwise_plot.py -m llama3.1,llama3.2,phi3.5,smollm,nemotron,mistral
