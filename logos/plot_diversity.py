"""
Plot expected pairwise similarity (diversity) for equilibrium and socially optimal strategies.

For each similarity metric, computes the expected pairwise similarity between n players
given their strategy profiles (equilibrium or socially optimal), averaged across all icons.
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import comb


def parse_args():
    parser = argparse.ArgumentParser(description="Plot expected pairwise similarity (diversity)")
    parser.add_argument(
        "--pairwise-csv",
        type=str,
        default="pairwise_similarity_results.csv",
        help="Path to pairwise similarity results CSV (relative to logos directory)",
    )
    parser.add_argument(
        "--equilibria-dir",
        type=str,
        default="plots",
        help="Directory containing equilibria JSON files (relative to logos directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots (relative to logos directory)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["dino_cosine", "clip_cosine", "lpips"],
        help="Similarity metrics to plot",
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock equilibria files (for testing with limited pairwise data)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for similarity computation",
    )
    return parser.parse_args()


def compute_k_per_model_per_icon(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """
    Compute k (number of images) for each model-icon combination.
    
    For each model-icon pair, counts the unique seeds that appear in either
    model_name1/seed1 or model_name2/seed2 columns.
    
    Returns:
        Dict mapping base_name -> Dict mapping model_name -> k
    """
    k_dict = {}
    for base_name in df["base_name"].unique():
        k_dict[base_name] = {}
        icon_df = df[df["base_name"] == base_name]
        
        # Get all models that appear for this icon
        all_models = set(icon_df["model_name1"].dropna().unique()) | set(icon_df["model_name2"].dropna().unique())
        
        for model in all_models:
            # Collect all seeds where this model appears (either as model1 or model2)
            seeds = set()
            
            # Seeds where model is model_name1
            model1_rows = icon_df[icon_df["model_name1"] == model]
            seeds.update(model1_rows["seed1"].dropna().tolist())
            
            # Seeds where model is model_name2
            model2_rows = icon_df[icon_df["model_name2"] == model]
            seeds.update(model2_rows["seed2"].dropna().tolist())
            
            k_dict[base_name][model] = len(seeds)
    
    return k_dict


def compute_expected_pairwise_similarity(
    df: pd.DataFrame,
    assignment: List[str],
    metric: str,
    k_dict: Dict[str, Dict[str, int]],
    debug: bool = False
) -> float:
    """
    Compute expected pairwise similarity for a given strategy profile.
    
    Args:
        df: DataFrame with pairwise similarity data
        assignment: List of model names (one per player)
        metric: Similarity metric name (e.g., "dino_cosine")
        k_dict: Dict mapping base_name -> model_name -> k (number of images)
        debug: If True, print debugging information
    
    Returns:
        Average expected pairwise similarity across all icons
    """
    n = len(assignment)
    model_counts = Counter(assignment)
    
    # Filter dataframe to only include rows with valid metric values
    df_with_metric = df.dropna(subset=[metric])
    
    if debug:
        print(f"    DEBUG: Assignment = {assignment}")
        print(f"    DEBUG: Model counts = {dict(model_counts)}")
        print(f"    DEBUG: Total rows with {metric}: {len(df_with_metric)} (out of {len(df)} total)")
    
    # Get all icons that have data for this metric
    icons = df_with_metric["base_name"].unique()
    
    if debug:
        print(f"    DEBUG: Icons with {metric} data: {len(icons)}")
    
    total_similarity = 0.0
    valid_icons = 0
    
    for icon in icons:
        icon_df = df_with_metric[df_with_metric["base_name"] == icon]
        
        if debug:
            print(f"    DEBUG: Processing icon {icon}, {len(icon_df)} rows with {metric} data")
        
        # Check if we have data for all models in assignment
        icon_models = set(icon_df["model_name1"].dropna().unique()) | set(icon_df["model_name2"].dropna().unique())
        missing_models = set(model_counts.keys()) - icon_models
        if missing_models:
            if debug:
                print(f"    DEBUG:   Skipping {icon}: missing models {missing_models}")
            continue
        
        # Get k for each model for this icon
        icon_k = k_dict.get(icon, {})
        missing_k = [m for m in model_counts.keys() if m not in icon_k]
        if missing_k:
            if debug:
                print(f"    DEBUG:   Skipping {icon}: missing k for models {missing_k}")
            continue
        
        if debug:
            print(f"    DEBUG:   Icon {icon} passed checks, computing similarity...")
        
        # Compute expected pairwise similarity for this icon
        icon_similarity = 0.0
        contributions = 0
        
        # Iterate over all pairs of models in the assignment
        for model1, count1 in model_counts.items():
            for model2, count2 in model_counts.items():
                k1 = icon_k[model1]
                k2 = icon_k[model2]
                
                # Get all pairs between these two models
                if model1 == model2:
                    # Same model: pairs within model
                    pairs_df = icon_df[
                        (icon_df["model_name1"] == model1) & 
                        (icon_df["model_name2"] == model2)
                    ]
                    
                    if len(pairs_df) == 0:
                        if debug:
                            print(f"    DEBUG:     No pairs found for {model1} vs {model1}")
                        continue
                    
                    # Weight: C(count1, 2) / (C(n, 2) * C(k1, 2))
                    if count1 < 2 or k1 < 2:
                        if debug:
                            print(f"    DEBUG:     Skipping {model1} vs {model1}: count1={count1}, k1={k1}")
                        continue
                    
                    weight = comb(count1, 2, exact=True) / (comb(n, 2, exact=True) * comb(k1, 2, exact=True))
                    
                    # Average similarity over all pairs
                    # For LPIPS, use 1-score so higher values mean more similar
                    raw_avg = pairs_df[metric].mean()
                    avg_sim = 1 - raw_avg if metric == "lpips" else raw_avg
                    contribution = weight * avg_sim
                    icon_similarity += contribution
                    contributions += 1
                    
                    if debug:
                        print(f"    DEBUG:     {model1} vs {model1}: {len(pairs_df)} pairs, avg={raw_avg:.6f}, weight={weight:.6f}, contribution={contribution:.6f}")
                else:
                    # Different models: pairs between models
                    pairs_df = icon_df[
                        ((icon_df["model_name1"] == model1) & (icon_df["model_name2"] == model2)) |
                        ((icon_df["model_name1"] == model2) & (icon_df["model_name2"] == model1))
                    ]
                    
                    if len(pairs_df) == 0:
                        if debug:
                            print(f"    DEBUG:     No pairs found for {model1} vs {model2}")
                        continue
                    
                    # Weight: C(count1, 1) * C(count2, 1) / (C(n, 2) * k1 * k2)
                    if count1 < 1 or count2 < 1 or k1 < 1 or k2 < 1:
                        if debug:
                            print(f"    DEBUG:     Skipping {model1} vs {model2}: count1={count1}, count2={count2}, k1={k1}, k2={k2}")
                        continue
                    
                    weight = (comb(count1, 1, exact=True) * comb(count2, 1, exact=True)) / (comb(n, 2, exact=True) * k1 * k2)
                    
                    # Average similarity over all pairs
                    # For LPIPS, use 1-score so higher values mean more similar
                    raw_avg = pairs_df[metric].mean()
                    avg_sim = 1 - raw_avg if metric == "lpips" else raw_avg
                    contribution = weight * avg_sim
                    icon_similarity += contribution
                    contributions += 1
                    
                    if debug:
                        print(f"    DEBUG:     {model1} vs {model2}: {len(pairs_df)} pairs, avg={raw_avg:.6f}, weight={weight:.6f}, contribution={contribution:.6f}")
        
        if debug:
            print(f"    DEBUG:   Icon {icon} similarity: {icon_similarity:.6f} (from {contributions} contributions)")
        
        if icon_similarity > 0 or contributions > 0:
            total_similarity += icon_similarity
            valid_icons += 1
    
    if debug:
        print(f"    DEBUG: Total similarity: {total_similarity:.6f}, Valid icons: {valid_icons}")
    
    if valid_icons == 0:
        if debug:
            print(f"    DEBUG: No valid icons found, returning 0.0")
        return 0.0
    
    result = total_similarity / valid_icons
    if debug:
        print(f"    DEBUG: Final result: {result:.6f}")
    return result


def load_equilibria_data(json_path: Path) -> Dict:
    """Load equilibria JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()
    logos_dir = Path(__file__).parent
    
    pairwise_csv_path = logos_dir / args.pairwise_csv
    equilibria_dir = logos_dir / args.equilibria_dir
    output_dir = logos_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    if not pairwise_csv_path.exists():
        raise FileNotFoundError(f"Pairwise CSV not found at {pairwise_csv_path}")
    
    print(f"Loading pairwise similarity data from {pairwise_csv_path}...")
    df = pd.read_csv(pairwise_csv_path)
    print(f"Loaded {len(df)} pairwise comparisons")
    
    # Compute k (number of images per model per icon)
    print("Computing k (number of images per model per icon)...")
    k_dict = compute_k_per_model_per_icon(df)
    print(f"Computed k for {len(k_dict)} icons")
    
    # Process each metric
    results = {}  # metric -> List[Dict with n, type, similarity]
    
    for metric in args.metrics:
        print(f"\n{'='*60}")
        print(f"Processing metric: {metric}")
        print(f"{'='*60}")
        
        # Check if metric exists in CSV
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in CSV. Skipping.")
            continue
        
        # Load equilibria JSON
        json_filename = f"equilibria_{metric}_mock.json" if args.use_mock else f"equilibria_{metric}.json"
        json_path = equilibria_dir / json_filename
        if not json_path.exists():
            print(f"Warning: Equilibria JSON not found at {json_path}. Skipping.")
            continue
        
        print(f"Loading equilibria data from {json_path}...")
        equilibria_data = load_equilibria_data(json_path)
        
        metric_results = []
        
        # Process equilibria
        print("Processing equilibria...")
        equilibria_by_n = {}
        for eq in equilibria_data.get("equilibria", []):
            n = eq["n"]
            assignment = eq["assignment"]
            if n not in equilibria_by_n:
                equilibria_by_n[n] = []
            equilibria_by_n[n].append(assignment)
        
        for n, assignments in equilibria_by_n.items():
            # Average across all equilibria for this n
            similarities = []
            for assignment in assignments:
                sim = compute_expected_pairwise_similarity(df, assignment, metric, k_dict, debug=args.debug)
                similarities.append(sim)
            
            avg_sim = np.mean(similarities)
            metric_results.append({
                "n": n,
                "type": "Equilibrium",
                "similarity": avg_sim
            })
            print(f"  n={n}: avg pairwise similarity = {avg_sim:.6f} (across {len(assignments)} equilibria)")
        
        # Process socially optimal
        print("Processing socially optimal strategies...")
        for opt in equilibria_data.get("socially_optimal", []):
            n = opt["n"]
            assignment = opt["assignment"]
            sim = compute_expected_pairwise_similarity(df, assignment, metric, k_dict, debug=args.debug)
            metric_results.append({
                "n": n,
                "type": "Socially Optimal",
                "similarity": sim
            })
            print(f"  n={n}: pairwise similarity = {sim:.6f}")
        
        results[metric] = metric_results
    
    # Plot results
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    
    sns.set_theme(style="whitegrid")
    
    for metric in results.keys():
        df_plot = pd.DataFrame(results[metric])
        
        if df_plot.empty:
            print(f"No data to plot for {metric}")
            continue
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_plot, x="n", y="similarity", hue="type", marker="o")
        plt.title(f"Expected Pairwise Similarity vs Number of Players ({metric})")
        plt.xlabel("Number of Players (n)")
        plt.ylabel("Expected Pairwise Similarity")
        plt.legend(title="Strategy Type")
        plt.tight_layout()
        
        output_path = output_dir / f"diversity_{metric}.png"
        plt.savefig(output_path, dpi=600)
        plt.close()
        print(f"Saved plot to {output_path}")
    
    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()

