MODELS=$1
# plots
# Main temp and utility plots (creates 5 figures)
python3 make_plots.py -m $MODELS

# Utility as a function of temp for different values of n
# Creates 6 figures
python3 plot_opt_over_temp.py -m $MODELS -v qwen2.5

# Inversions: Weighted inversions and prob mass captured (2 figures)
python3 plot_inversions.py -m $MODELS
# Inversions on d
python3 plot_prs.py -m $MODELS -v qwen2.5

# Pairwise
# Separately run 2 for main figures
python3 plot_pairwise.py -m llama3.1,llama3.2
python3 plot_pairwise.py -m llama3.1,phi3.5
# One big plot with all pairwise comparisons
python3 make_big_pairwise_plot.py -m $MODELS
