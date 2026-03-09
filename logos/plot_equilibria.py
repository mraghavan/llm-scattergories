import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import game_simulation

metric_to_text = {
    'dreamsim': 'DreamSim',
    'lpips': 'LPIPS'
}


def format_model_name_for_display(model_name: str) -> str:
    """Format model name for display in plots. Changes flux1.dev to FLUX.1-dev."""
    if model_name == "flux1.dev":
        return "FLUX.1-dev"
    return model_name


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
    parser.add_argument(
        "--input-csv",
        type=str,
        default="image_similarity_results.csv",
        help="Input CSV filename (should be in the logos directory).",
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
    
    # Get all n values from optimal_data (which should contain all computed n values)
    all_n_values = sorted(set(opt["n"] for opt in optimal_data))
    # Also include any n values from equilibria (defensive)
    n_from_equilibria = set(eq["n"] for eq in equilibria_data)
    all_n_values = sorted(set(all_n_values) | n_from_equilibria)
    
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
                # Use raw distance score (lower is better for distance metrics)
                plot_score = avg_score
                performance_data.append({
                    "n": n,
                    "score": plot_score,
                    "type": "Equilibrium"
                })
        
        # Compute socially optimal performance from JSON data
        opt_for_n = [opt for opt in optimal_data if opt["n"] == n]
        if opt_for_n:
            # Should only be one optimal per n
            opt = opt_for_n[0]
            optimal_score = opt["score"]
            # Use raw distance score (lower is better for distance metrics)
            plot_score = optimal_score
            performance_data.append({
                "n": n,
                "score": plot_score,
                "type": "Socially optimal"
            })
    
    return performance_data


def compute_average_distance_per_model(df: pd.DataFrame, metric: str, 
                                       available_models: List[str]) -> Dict[str, float]:
    """Compute average distance for each model from raw CSV data.
    
    This computes the average distance across all icons and generated images for each model,
    regardless of whether the model appears in equilibria or optimal assignments.
    
    Returns a dictionary mapping model names to their average distances.
    """
    # Use game_simulation's function to compute averages from raw data
    model_averages = game_simulation.compute_model_averages(df, metric)
    
    # Ensure all available models are included (even if they have no data)
    result = {}
    for model in available_models:
        if model in model_averages:
            result[model] = model_averages[model]
        else:
            # Model has no data in CSV
            result[model] = 0.0
    
    return result


def format_latex_table(df: pd.DataFrame, metric: str, available_models: List[str]) -> str:
    """Format model averages as a LaTeX table using pandas.
    
    Args:
        df: DataFrame with distance measurements
        metric: The distance metric name (for table caption)
        available_models: List of all available models
    
    Returns:
        Formatted LaTeX table as a string
    """
    # Compute mean for each model
    stats_data = []
    print(f"\n  Computing statistics over {metric} values:")
    for model in sorted(available_models):
        model_data = df[df["model_name"] == model][metric].dropna()
        num_values = len(model_data)
        if num_values > 0:
            mean_val = model_data.mean()
            # Format as mean only
            stats_data.append(f"{mean_val:.3f}")
            print(f"    {model}: {num_values} values")
        else:
            stats_data.append("N/A")
            print(f"    {model}: 0 values (no data)")
    
    # Create DataFrame with formatted values
    table_df = pd.DataFrame({
        'Model': sorted(available_models),
        'Distance': stats_data  # Shorter name to avoid line wrapping
    })
    table_df = table_df.set_index('Model')
    
    # Format as LaTeX table using pandas
    latex_table = table_df.to_latex(
        caption=f"Average Distance per Model ({metric})",
        label=f"tab:model_performance_{metric}",
        index=True,
        column_format="lc",
        escape=False
    )
    
    # Add \centering after \begin{table}
    latex_table = latex_table.replace("\\begin{table}", "\\begin{table}\n\\centering")
    
    # Fix header to be on one line - replace any multirow or wrapped headers
    lines = latex_table.split('\n')
    found_toprule = False
    for i, line in enumerate(lines):
        if '\\toprule' in line:
            found_toprule = True
        elif found_toprule and '&' in line:
            # This should be the header line (first line with & after \toprule)
            # Replace with single-line header
            lines[i] = 'Model & Average Distance \\\\'
            break
        elif '\\midrule' in line:
            # If we hit midrule without finding header, stop looking
            break
    
    return '\n'.join(lines)


def plot_equilibria(metric, input_dir, output_dir, csv_path):
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
    
    # Load CSV data for computing model averages
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {csv_path}")
    df = game_simulation.load_distance_data(csv_path, metric)
    
    print(f"Loaded {len(equilibria_data)} equilibria records and {len(optimal_data)} optimal strategy records")
    
    # Get all n values from both equilibria and optimal data
    # Use optimal_data as the source of truth for all n values since it should contain
    # all n values that were computed (even if some have no equilibria)
    all_n_values = sorted(set(opt["n"] for opt in optimal_data))
    
    # Also include any n values from equilibria that might be missing from optimal (defensive)
    n_from_equilibria = set(eq["n"] for eq in equilibria_data)
    all_n_values = sorted(set(all_n_values) | n_from_equilibria)
    
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
    # Set larger default font sizes
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.titlesize': 16})
    plt.rcParams.update({'axes.labelsize': 14})
    plt.rcParams.update({'xtick.labelsize': 12})
    plt.rcParams.update({'ytick.labelsize': 12})
    plt.rcParams.update({'legend.fontsize': 12})
    
    # Plot 1: Market Share (Stacked Bar Chart) - Combined Equilibrium and Optimal
    print("\nGenerating Market Share Plot (Combined)...")
    df_share = pd.DataFrame(market_share_data)
    df_optimal_share = pd.DataFrame(optimal_market_share_data)
    
    if not df_share.empty or not df_optimal_share.empty:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        metric_name = metric_to_text[metric]
        
        # Left panel: Equilibrium market shares
        if not df_share.empty:
            df_pivot_eq = df_share.pivot(index="n", columns="model", values="share").fillna(0)
            # Sort columns in reverse alphabetical order (Z to A) so alphabetically first appears at top of stack
            df_pivot_eq = df_pivot_eq.sort_index(axis=1, ascending=False)
            
            df_pivot_eq.plot(kind="bar", stacked=True, ax=ax1, colormap="viridis", legend=False)
            ax1.set_title(f"Equilibrium market shares ({metric_name})", fontsize=20)
            ax1.set_xlabel("$n$", fontsize=18)
            ax1.set_ylabel("Market share", fontsize=18)
            ax1.tick_params(axis='x', labelsize=16)
            ax1.tick_params(axis='y', labelsize=16)
        else:
            ax1.text(0.5, 0.5, "No equilibrium data", ha='center', va='center', transform=ax1.transAxes, fontsize=16)
            ax1.set_title(f"Equilibrium market shares ({metric_name})", fontsize=20)
            ax1.set_xlabel("$n$", fontsize=18)
            ax1.set_ylabel("Market share", fontsize=18)
            ax1.tick_params(axis='x', labelsize=16)
            ax1.tick_params(axis='y', labelsize=16)
        
        # Right panel: Socially optimal market shares
        if not df_optimal_share.empty:
            df_pivot_opt = df_optimal_share.pivot(index="n", columns="model", values="share").fillna(0)
            # Sort columns in reverse alphabetical order (Z to A) so alphabetically first appears at top of stack
            df_pivot_opt = df_pivot_opt.sort_index(axis=1, ascending=False)
            
            df_pivot_opt.plot(kind="bar", stacked=True, ax=ax2, colormap="viridis")
            ax2.set_title(f"Socially optimal market shares ({metric_name})", fontsize=20)
            ax2.set_xlabel("$n$", fontsize=18)
            # Remove y-axis label and tick labels since they're the same as the left plot
            ax2.set_ylabel("")
            ax2.set_yticklabels([])
            ax2.tick_params(axis='x', labelsize=16)
            
            # Set legend order to alphabetical (A to Z) from top to bottom
            handles2, labels2 = ax2.get_legend_handles_labels()
            label_to_handle2 = dict(zip(labels2, handles2))
            sorted_labels2 = sorted(set(labels2))
            formatted_labels2 = [format_model_name_for_display(label) for label in sorted_labels2]
            sorted_handles2 = [label_to_handle2[label] for label in sorted_labels2]
            ax2.legend(sorted_handles2, formatted_labels2, title="Model", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16, title_fontsize=18)
        else:
            ax2.text(0.5, 0.5, "No optimal data", ha='center', va='center', transform=ax2.transAxes, fontsize=16)
            ax2.set_title(f"Socially optimal market shares ({metric_name})", fontsize=20)
            ax2.set_xlabel("$n$", fontsize=18)
            # Remove y-axis label and tick labels since they're the same as the left plot
            ax2.set_ylabel("")
            ax2.set_yticklabels([])
            ax2.tick_params(axis='x', labelsize=16)
        
        # Add spacing between the two panels
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.15)  # Add spacing after tight_layout to override it
        plt.savefig(output_dir / f"market_share_{metric}.png", dpi=600)
        plt.close()
    else:
        print("No market share data to plot.")
    
    # Plot 2: System Performance
    print("Generating System Performance Plot...")
    df_perf = pd.DataFrame(performance_data)
    
    if not df_perf.empty:
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df_perf, x="n", y="score", hue="type", marker="o")
        metric_name = metric_to_text[metric]
        plt.title(f"Distance between chosen image and target ({metric_name})", fontsize=16)
        plt.xlabel("$n$", fontsize=14)
        plt.ylabel("Average distance to target", fontsize=14)
        # Remove the "type" title from the legend
        ax.legend(title="", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / f"system_performance_{metric}.png", dpi=600)
        plt.close()
    else:
        print("No performance data to plot.")
    
    # Plot 3: Average Distance per Model
    print("Generating Average Distance per Model Plot...")
    model_averages = compute_average_distance_per_model(df, metric, available_models)
    
    if model_averages:
        # Include all models, even those that never appeared (they will have 0.0)
        sorted_models = sorted(model_averages.keys())  # Alphabetical for display
        distances = [model_averages[model] for model in sorted_models]
        
        # Check for models with no data (have 0.0)
        missing_models = [model for model in sorted_models if model_averages[model] == 0.0]
        if missing_models:
            print(f"  Note: Models with 0.0 have no data in CSV: {missing_models}")
        
        # Get viridis colormap colors matching the market share plots
        # Market share plots sort in reverse alphabetical (Z to A) for color assignment
        # So we need to assign colors in reverse alphabetical order to match
        try:
            viridis = plt.colormaps['viridis']
        except (AttributeError, KeyError):
            # Fallback for older matplotlib versions
            viridis = plt.cm.get_cmap('viridis')
        num_models = len(sorted_models)
        
        # Create color mapping: assign colors in reverse alphabetical order (matching market share plot)
        reverse_sorted_models = sorted(model_averages.keys(), reverse=True)
        model_to_color = {}
        for i, model in enumerate(reverse_sorted_models):
            model_to_color[model] = viridis(i / max(num_models - 1, 1))
        
        # Get colors in alphabetical order (for display order)
        colors = [model_to_color[model] for model in sorted_models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sorted_models, distances, color=colors)
        plt.title(f"Average Distance per Model ({metric})", fontsize=16)
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Average Distance", fontsize=14)
        # Format model names for display on x-axis
        formatted_model_names = [format_model_name_for_display(model) for model in sorted_models]
        plt.xticks(range(len(sorted_models)), formatted_model_names, rotation=45, ha='right', fontsize=12)
        
        # Set y-axis limits to make differences more visible (don't start at 0)
        if distances:
            min_dist = min(d for d in distances if d > 0)  # Minimum non-zero distance
            max_dist = max(distances)
            # Set y-axis to start slightly below minimum, with some padding
            y_min = max(0, min_dist - (max_dist - min_dist) * 0.1)
            y_max = max_dist + (max_dist - min_dist) * 0.05
            plt.ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"model_performance_{metric}.png", dpi=600)
        plt.close()
        print(f"  Average distances: {dict(zip(sorted_models, distances))}")
        
        # Print LaTeX table
        print("\nLaTeX Table:")
        print("-" * 60)
        latex_table = format_latex_table(df, metric, available_models)
        print(latex_table)
        print("-" * 60)
    else:
        print("No model performance data to plot.")
    
    print(f"\nPlots saved to {output_dir}")


def main():
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_dir = logos_dir / args.input_dir
    output_dir = logos_dir / args.output_dir
    csv_path = logos_dir / args.input_csv
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
            output_dir=output_dir,
            csv_path=csv_path
        )


if __name__ == "__main__":
    main()

