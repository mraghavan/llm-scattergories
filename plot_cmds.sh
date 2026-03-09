MODELS=$1
# plots

# Score functions (Figure 1)
python3 plot_s.py

# Theory for competition going to infinity (Figure 2)
python3 plot_inf.py

# Main temp and utility plots (Figures 4, 5, 10, 11, 12)
python3 make_plots.py -m $MODELS

# Utility as a function of temp for different values of n (Figures 3, 9)
python3 plot_opt_over_temp.py -m $MODELS -v qwen2.5

# Inversions: Weighted inversions and prob mass captured (Figure 14)
python3 plot_inversions.py -m $MODELS

# Inversions on d (not in paper)
python3 plot_prs.py -m $MODELS -v qwen2.5

# Pairwise (Figure 6)
# Separately run 2 for main figures
python3 plot_pairwise.py -m llama3.1,llama3.2
python3 plot_pairwise.py -m llama3.1,phi3.5

# One big plot with all pairwise comparisons (Figure 13)
python3 make_big_pairwise_plot.py -m $MODELS

# Spectral analysis (Figures 7, 8, 15)
# Note: Requires additional data generation steps. See README.md.
# python3 compare_rankings.py -n 10 --min-count 10
