import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import game_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Compute equilibria and socially optimal strategies")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="image_similarity_results.csv",
        help="Input CSV filename (should be in the logos directory).",
    )
    parser.add_argument(
        "--distance-metric",
        type=str,
        default="lpips",
        choices=["lpips", "dreamsim", "all"],
        help="Distance metric to use. Use 'all' to run for all metrics.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Maximum number of players to simulate (default: limited by available samples).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save JSON files.",
    )
    parser.add_argument(
        "--max-icons",
        type=int,
        default=None,
        help="Limit analysis to the first k icons.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Restrict analysis to specific models. If not provided, all available models are used.",
    )
    return parser.parse_args()


def compute_equilibria(metric, input_path, output_dir, max_players, max_icons, restricted_models):
    print(f"\n{'='*60}")
    print(f"Computing equilibria for metric: {metric}")
    print(f"{'='*60}")
    
    # Load data
    df = game_simulation.load_distance_data(input_path, metric)
    print(f"Loaded {len(df)} distance measurements")
    
    grouped_data = game_simulation.group_by_icon_and_model(df, metric)
    
    if max_icons is not None:
        print(f"Limiting to first {max_icons} icons.")
        # Slice the dictionary
        grouped_data = dict(list(grouped_data.items())[:max_icons])
        
    available_models = game_simulation.get_available_models(df)
    print(f"Available models: {available_models}")
    
    # Filter models if restricted_models is provided
    if restricted_models is not None:
        # Validate that all requested models are available
        invalid_models = [m for m in restricted_models if m not in available_models]
        if invalid_models:
            raise ValueError(
                f"Invalid models specified: {invalid_models}. "
                f"Available models: {available_models}"
            )
        available_models = [m for m in available_models if m in restricted_models]
        print(f"Restricted to models: {available_models}")
    
    # Validate sample counts
    min_samples = float('inf')
    for icon, models in grouped_data.items():
        for model, samples in models.items():
            if model in available_models:
                min_samples = min(min_samples, len(samples))
    
    print(f"Minimum samples per icon: {min_samples}")
    
    if max_players is None:
        max_n = min_samples + 1
        print(f"Setting max_players to {max_n} based on available data (min_samples + 1).")
    else:
        max_n = max_players
        if max_n > min_samples + 1:
            print(f"Warning: Requested max_players={max_n} but data only has {min_samples} samples per icon.")
            print(f"Limiting max_players to {min_samples + 1} (allowing 1 extra for symmetric play).")
            max_n = min_samples + 1
    
    # Data collection
    all_equilibria_data = []  # List of dicts: {n: int, assignment: List[str], utilities: Dict[int, float]}
    all_optimal_data = []  # List of dicts: {n: int, assignment: List[str], score: float, utilities: Dict[int, float]}
    
    # Loop over n
    for n in range(2, max_n + 1):
        print(f"\nSimulating n={n}...")
        
        # 1. Compute utilities for all assignments
        utilities = game_simulation.compute_all_utilities(
            grouped_data, n, available_models, metric
        )
        print(f"  Computed utilities for {len(utilities)} assignments")
        
        # 2. Find Nash Equilibria
        equilibria = game_simulation.find_nash_equilibria(utilities, available_models)
        print(f"  Found {len(equilibria)} equilibria")
        for i, (assignment, util) in enumerate(equilibria):
            print(f"    Eq {i+1}: {assignment}")
            # Compute expected max score for this equilibrium
            eq_score = game_simulation.get_expected_max_score(
                grouped_data, list(assignment), metric
            )
            # Save equilibrium data
            all_equilibria_data.append({
                "n": n,
                "assignment": list(assignment),  # Convert tuple to list for JSON
                "score": float(eq_score),  # Add score for performance plotting
                "utilities": {str(k): float(v) for k, v in util.items()}  # Convert int keys to str for JSON
            })
        
        if not equilibria:
            print(f"  Warning: No equilibria found for n={n}")
        
        # Socially Optimal: Find assignment that minimizes expected min score
        print(f"  Finding socially optimal assignment...")
        optimal_score = float('inf')
        optimal_assignment = None
        
        for assignment in utilities.keys():
            score = game_simulation.get_expected_max_score(
                grouped_data, list(assignment), metric
            )
            # For distance metrics, lower is better
            if score < optimal_score:
                optimal_score = score
                optimal_assignment = assignment
        
        if optimal_assignment is not None:
            print(f"    Optimal assignment: {optimal_assignment} (score: {optimal_score:.4f})")
            # Save optimal assignment data
            optimal_utilities = utilities.get(optimal_assignment, {})
            all_optimal_data.append({
                "n": n,
                "assignment": list(optimal_assignment),  # Convert tuple to list for JSON
                "score": float(optimal_score),
                "utilities": {str(k): float(v) for k, v in optimal_utilities.items()}  # Convert int keys to str for JSON
            })
        else:
            print(f"    Warning: Could not find optimal assignment")
    
    # Save equilibria and optimal strategies to JSON file
    equilibria_file = output_dir / f"equilibria_{metric}.json"
    print(f"\nSaving equilibria and optimal strategies to {equilibria_file}...")
    with open(equilibria_file, 'w') as f:
        json.dump({
            "metric": metric,
            "available_models": available_models,
            "equilibria": all_equilibria_data,
            "socially_optimal": all_optimal_data
        }, f, indent=2)
    print(f"Saved {len(all_equilibria_data)} equilibria records and {len(all_optimal_data)} optimal strategy records to {equilibria_file}")
    
    print(f"\nResults saved to {output_dir}")


def main():
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_path = logos_dir / args.input_csv
    output_dir = logos_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")
    
    metrics_to_run = []
    if args.distance_metric == "all":
        metrics_to_run = ["lpips", "dreamsim"]
    else:
        metrics_to_run = [args.distance_metric]
        
    for metric in metrics_to_run:
        compute_equilibria(
            metric=metric,
            input_path=input_path,
            output_dir=output_dir,
            max_players=args.max_players,
            max_icons=args.max_icons,
            restricted_models=args.models
        )


if __name__ == "__main__":
    main()

