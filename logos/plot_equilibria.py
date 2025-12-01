import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Plot equilibria and socially optimal strategies from JSON files")
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="lpips",
        choices=["lpips", "dreamsim", "all"],
        help="Distance metric to use. Use 'all' to run for all metrics.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="plots",
        help="Directory containing JSON files (equilibria_*.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots.",
    )
    return parser.parse_args()


def load_equilibria_data(json_path: Path) -> Dict:
    """Load equilibria data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_market_share_from_equilibria(equilibria_data: List[Dict], n: int, available_models: List[str]) -> Dict[str, float]:
    """Compute average market share from equilibria for a given n."""
    # Filter equilibria for this n
    eq_for_n = [eq for eq in equilibria_data if eq["n"] == n]
    if not eq_for_n:
        return {model: 0.0 for model in available_models}
    
    total_model_counts = Counter()
    for eq in eq_for_n:
        total_model_counts.update(eq["assignment"])
    
    num_equilibria = len(eq_for_n)
    market_share = {}
    for model in available_models:
        avg_count = total_model_counts[model] / num_equilibria
        share = avg_count / n  # Percentage of players using this model
        market_share[model] = share
    
    return market_share


def compute_performance_scores(equilibria_data: List[Dict], optimal_data: List[Dict], 
                                metric: str) -> List[Dict]:
    """Compute performance scores from equilibria and optimal data stored in JSON."""
    performance_data = []
    
    # Get all n values
    all_n_values = sorted(set(eq["n"] for eq in equilibria_data))
    
    for n in all_n_values:
        # Compute Nash equilibrium performance from JSON data
        eq_for_n = [eq for eq in equilibria_data if eq["n"] == n]
        if eq_for_n:
            # Average scores across all equilibria for this n
            total_score = 0.0
            count_with_scores = 0
            for eq in eq_for_n:
                if "score" in eq:
                    total_score += eq["score"]
                    count_with_scores += 1
            
            if count_with_scores > 0:
                avg_score = total_score / count_with_scores
                # For distance metrics, transform to 1 - score so higher is better for visualization
                plot_score = 1 - avg_score
                performance_data.append({
                    "n": n,
                    "score": plot_score,
                    "type": "Strategic (Nash)"
                })
        
        # Compute socially optimal performance from JSON data
        opt_for_n = [opt for opt in optimal_data if opt["n"] == n]
        if opt_for_n:
            # Should only be one optimal per n
            opt = opt_for_n[0]
            optimal_score = opt["score"]
            # For distance metrics, transform to 1 - score so higher is better for visualization
            plot_score = 1 - optimal_score
            performance_data.append({
                "n": n,
                "score": plot_score,
                "type": "Socially Optimal"
            })
    
    return performance_data


def plot_equilibria(metric, input_dir, output_dir):
    print(f"\n{'='*60}")
    print(f"Plotting equilibria for metric: {metric}")
    print(f"{'='*60}")
    
    # Load JSON data
    json_path = input_dir / f"equilibria_{metric}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Equilibria JSON file not found at {json_path}")
    
    data = load_equilibria_data(json_path)
    available_models = data["available_models"]
    equilibria_data = data["equilibria"]
    optimal_data = data["socially_optimal"]
    
    print(f"Loaded {len(equilibria_data)} equilibria records and {len(optimal_data)} optimal strategy records")
    
    # Get all n values
    all_n_values = sorted(set(eq["n"] for eq in equilibria_data))
    
    # Compute market share data
    market_share_data = []
    for n in all_n_values:
        market_share = compute_market_share_from_equilibria(equilibria_data, n, available_models)
        for model in available_models:
            market_share_data.append({
                "n": n,
                "model": model,
                "share": market_share[model]
            })
    
    # Compute optimal market share data
    optimal_market_share_data = []
    for opt in optimal_data:
        n = opt["n"]
        optimal_model_counts = Counter(opt["assignment"])
        for model in available_models:
            count = optimal_model_counts[model]
            share = count / n  # Percentage of players using this model
            optimal_market_share_data.append({
                "n": n,
                "model": model,
                "share": share
            })
    
    # Compute performance data (only from JSON, no CSV needed)
    performance_data = compute_performance_scores(
        equilibria_data, optimal_data, metric
    )
    
    # Plotting
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Market Share (Stacked Bar Chart) - Nash Equilibria
    print("\nGenerating Market Share Plot (Nash Equilibria)...")
    df_share = pd.DataFrame(market_share_data)
    
    if not df_share.empty:
        # Pivot for stacked bar
        df_pivot = df_share.pivot(index="n", columns="model", values="share").fillna(0)
        # Sort columns in reverse alphabetical order (Z to A) so alphabetically first appears at top of stack
        df_pivot = df_pivot.sort_index(axis=1, ascending=False)
        
        ax = df_pivot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        plt.title(f"Equilibrium Model Selection vs Number of Players ({metric})")
        plt.xlabel("Number of Players (n)")
        plt.ylabel("Share of Players")
        
        # Set legend order to alphabetical (A to Z) from top to bottom
        handles, labels = ax.get_legend_handles_labels()
        # Create a mapping of label to handle for correct pairing
        label_to_handle = dict(zip(labels, handles))
        # Get all unique labels sorted alphabetically (A to Z)
        sorted_labels = sorted(set(labels))
        # Create handles list in the same alphabetical order
        sorted_handles = [label_to_handle[label] for label in sorted_labels]
        ax.legend(sorted_handles, sorted_labels, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"market_share_{metric}.png", dpi=600)
        plt.close()
    else:
        print("No market share data to plot.")
    
    # Plot 1b: Market Share (Stacked Bar Chart) - Socially Optimal
    print("Generating Market Share Plot (Socially Optimal)...")
    df_optimal_share = pd.DataFrame(optimal_market_share_data)
    
    if not df_optimal_share.empty:
        # Pivot for stacked bar
        df_pivot = df_optimal_share.pivot(index="n", columns="model", values="share").fillna(0)
        # Sort columns in reverse alphabetical order (Z to A) so alphabetically first appears at top of stack
        df_pivot = df_pivot.sort_index(axis=1, ascending=False)
        
        ax = df_pivot.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
        plt.title(f"Socially Optimal Model Selection vs Number of Players ({metric})")
        plt.xlabel("Number of Players (n)")
        plt.ylabel("Share of Players")
        
        # Set legend order to alphabetical (A to Z) from top to bottom
        handles, labels = ax.get_legend_handles_labels()
        # Create a mapping of label to handle for correct pairing
        label_to_handle = dict(zip(labels, handles))
        # Get all unique labels sorted alphabetically (A to Z)
        sorted_labels = sorted(set(labels))
        # Create handles list in the same alphabetical order
        sorted_handles = [label_to_handle[label] for label in sorted_labels]
        ax.legend(sorted_handles, sorted_labels, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"market_share_optimal_{metric}.png", dpi=600)
        plt.close()
    else:
        print("No socially optimal market share data to plot.")
    
    # Plot 2: System Performance
    print("Generating System Performance Plot...")
    df_perf = pd.DataFrame(performance_data)
    
    if not df_perf.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_perf, x="n", y="score", hue="type", marker="o")
        plt.title(f"System Performance vs Number of Players ({metric})")
        plt.xlabel("Number of Players (n)")
        plt.ylabel("Expected Min Distance Score (transformed)")
        plt.tight_layout()
        plt.savefig(output_dir / f"system_performance_{metric}.png", dpi=600)
        plt.close()
    else:
        print("No performance data to plot.")
    
    print(f"\nPlots saved to {output_dir}")


def main():
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_dir = logos_dir / args.input_dir
    output_dir = logos_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    metrics_to_run = []
    if args.distance_metric == "all":
        metrics_to_run = ["lpips", "dreamsim"]
    else:
        metrics_to_run = [args.distance_metric]
        
    for metric in metrics_to_run:
        plot_equilibria(
            metric=metric,
            input_dir=input_dir,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()

