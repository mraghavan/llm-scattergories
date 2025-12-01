"""
Game simulation for model selection game.

Players choose models, and we compute expected utility over:
- Random selection of original icon (base_name)
- Sampling distances from each model (without replacement if multiple players use same model)
- Winner is player with lowest distance (using lpips)
"""

import argparse
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"{desc}...")
        return iterable


def load_distance_data(csv_path: Path, distance_metric: str = "lpips") -> pd.DataFrame:
    """Load distance results CSV and return DataFrame."""
    df = pd.read_csv(csv_path)
    
    if distance_metric not in df.columns:
        raise ValueError(f"Distance metric '{distance_metric}' not found in CSV. Available: {df.columns.tolist()}")
    
    # Filter out rows with missing distance values
    df = df.dropna(subset=[distance_metric])
    
    return df


def get_available_models(df: pd.DataFrame) -> List[str]:
    """Get list of unique model names in the data that have samples for all icons."""
    all_icons = set(df["base_name"].unique())
    valid_models = []
    
    for model in df["model_name"].unique():
        model_icons = set(df[df["model_name"] == model]["base_name"].unique())
        if all_icons.issubset(model_icons):
            valid_models.append(model)
            
    return sorted(valid_models)


def get_available_icons(df: pd.DataFrame) -> List[str]:
    """Get list of unique base_name (icon) identifiers."""
    return sorted(df["base_name"].unique().tolist())


def compute_model_averages(df: pd.DataFrame, distance_metric: str = "lpips") -> Dict[str, float]:
    """
    Compute average distance for each model across all icons and generated images.
    
    Args:
        df: DataFrame with distance measurements
        distance_metric: Name of distance metric (e.g., "lpips", "dreamsim")
    
    Returns:
        Dictionary mapping model_name -> average distance
        Note: For LPIPS and DreamSim, lower is better.
    """
    model_averages = {}
    
    for model in df["model_name"].unique():
        model_data = df[df["model_name"] == model][distance_metric].dropna()
        if len(model_data) > 0:
            avg = model_data.mean()
            model_averages[model] = avg
    
    return model_averages


def get_best_single_shot_model(df: pd.DataFrame, distance_metric: str = "lpips") -> Tuple[Optional[str], Optional[float]]:
    """
    Find the best model in the single-shot case.
    
    For distance metrics: best = lowest average distance (lower is better)
    
    Args:
        df: DataFrame with distance measurements
        distance_metric: Name of distance metric
    
    Returns:
        Tuple of (best_model_name, average_distance)
    """
    model_averages = compute_model_averages(df, distance_metric)
    if not model_averages:
        return None, None
    
    # For distance metrics, lower is better (minimum distance)
    best_model = min(model_averages.items(), key=lambda x: x[1])
    
    return best_model


def is_lower_better(metric: str) -> bool:
    """Check if lower values are better for this metric."""
    # All distance metrics have lower is better
    return True


def group_by_icon_and_model(df: pd.DataFrame, distance_metric: str = "lpips") -> Dict[str, Dict[str, List[float]]]:
    """
    Group distance scores by icon (base_name) and model.
    
    Returns: {icon_name: {model_name: [distance_scores]}}
    """
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for _, row in df.iterrows():
        icon = row["base_name"]
        model = row["model_name"]
        distance = row[distance_metric]
        grouped[icon][model].append(distance)
    
    # Convert defaultdicts to regular dicts and sort scores
    # For distance metrics (lower is better), sort ascending.
    result = {}
    for icon, models in grouped.items():
        result[icon] = {
            model: sorted(scores)
            for model, scores in models.items()
        }
    
    return result


def nCr(n: int, r: int) -> float:
    """Compute combinations nCr."""
    if r < 0 or r > n:
        return 0.0
    if r == 0 or r == n:
        return 1.0
    if r > n // 2:
        r = n - r
    
    import math
    return float(math.comb(n, r))


def get_win_probabilities_efficient(
    grouped_data: Dict[str, Dict[str, List[float]]],
    player_models: List[str],
    distance_metric: str = "lpips"
) -> Tuple[float, Dict[int, float]]:
    """
    Compute win probabilities efficiently without generating combinations.
    
    Args:
        grouped_data: {icon_name: {model_name: [distance_scores]}}
        player_models: List of model names, one per player
        distance_metric: Name of distance metric
    
    Returns:
        (overall_expected_utility, {player_index: expected_utility})
    """
    n_players = len(player_models)
    model_counts = Counter(player_models)
    lower_is_better = is_lower_better(distance_metric)
    
    # Track total utility per player
    player_total_utility = np.zeros(n_players)
    n_icons = len(grouped_data)
    
    if n_icons == 0:
        return 0.0, {i: 0.0 for i in range(n_players)}
    
    import bisect
    
    for icon, icon_data in grouped_data.items():
        # Collect all samples from all models to iterate over
        all_samples = []
        for model, samples in icon_data.items():
            if model in model_counts:
                for s in samples:
                    all_samples.append((s, model))
        
        # Sort samples to easily count "worse" ones
        all_samples.sort(key=lambda x: x[0])
        
        # Pre-calculate sorted samples for each model
        model_samples_sorted = {
            m: sorted(icon_data.get(m, [])) for m in model_counts.keys()
        }
        
        # Check for insufficient samples
        # Check for insufficient samples
        insufficient_samples = False
        for model, count in model_counts.items():
            if len(model_samples_sorted[model]) < count:
                if count == n_players:
                    # Special case: All players chose same model, but we ran out of samples.
                    # By symmetry, each player wins with prob 1/n.
                    for i in range(n_players):
                        player_total_utility[i] += 1.0 / n_players
                    insufficient_samples = True
                    break
                else:
                    raise ValueError(
                        f"Model '{model}' has only {len(model_samples_sorted[model])} samples "
                        f"for icon '{icon}', but {count} players selected it. "
                        "Cannot sample without replacement."
                    )
        
        if insufficient_samples:
            continue
        
        # Calculate win prob for each model
        model_win_probs = defaultdict(float)
        
        for s_val, s_model in all_samples:
            # Probability that this specific sample 's_val' is chosen and is the winner
            
            vals = model_samples_sorted[s_model]
            N = len(vals)
            k = model_counts[s_model]
            
            if N < k:
                continue
            
            # Count "worse" samples for s_model
            if lower_is_better:
                # Worse = strictly greater
                idx = bisect.bisect_right(vals, s_val)
                count_worse = len(vals) - idx
            else:
                # Worse = strictly smaller
                idx = bisect.bisect_left(vals, s_val)
                count_worse = idx
            
            # Prob for s_model group:
            # We need to pick s_val (1 way) AND k-1 others from the 'worse' set.
            # Total ways to pick k from N is nCr(N, k).
            # Ways to pick s_val + (k-1) worse is 1 * nCr(count_worse, k-1).
            prob_model = nCr(count_worse, k - 1) / nCr(N, k)
            
            if prob_model == 0:
                continue
                
            # Prob for other models
            prob_others = 1.0
            for other_model, other_count in model_counts.items():
                if other_model == s_model:
                    continue
                
                other_vals = model_samples_sorted[other_model]
                other_N = len(other_vals)
                
                if other_N < other_count:
                    prob_others = 0.0
                    break
                
                if lower_is_better:
                    idx = bisect.bisect_right(other_vals, s_val)
                    other_count_worse = len(other_vals) - idx
                else:
                    idx = bisect.bisect_left(other_vals, s_val)
                    other_count_worse = idx
                
                term = nCr(other_count_worse, other_count) / nCr(other_N, other_count)
                prob_others *= term
            
            total_prob = prob_model * prob_others
            model_win_probs[s_model] += total_prob

        # Distribute model win probs to players
        for i, model in enumerate(player_models):
            count = model_counts[model]
            if count > 0:
                player_total_utility[i] += model_win_probs[model] / count

    # Average over icons
    expected_utilities = {i: player_total_utility[i] / n_icons for i in range(n_players)}
    overall_expected = sum(expected_utilities.values()) / n_players if n_players > 0 else 0.0
    
    return overall_expected, expected_utilities


def get_expected_max_score(
    grouped_data: Dict[str, Dict[str, List[float]]],
    player_models: List[str],
    distance_metric: str = "lpips"
) -> float:
    """
    Compute the expected value of the minimum distance score (the winning score).
    
    Args:
        grouped_data: {icon_name: {model_name: [distance_scores]}}
        player_models: List of model names, one per player
        distance_metric: Name of distance metric
    
    Returns:
        Expected minimum score (averaged over icons)
    """
    n_players = len(player_models)
    model_counts = Counter(player_models)
    lower_is_better = is_lower_better(distance_metric)
    
    total_expected_max = 0.0
    n_icons = len(grouped_data)
    
    if n_icons == 0:
        return 0.0
    
    import bisect
    
    for icon, icon_data in grouped_data.items():
        # Collect all samples from all models to iterate over
        all_samples = []
        for model, samples in icon_data.items():
            if model in model_counts:
                for s in samples:
                    all_samples.append((s, model))
        
        # Sort samples
        all_samples.sort(key=lambda x: x[0])
        
        # Pre-calculate sorted samples for each model
        model_samples_sorted = {
            m: sorted(icon_data.get(m, [])) for m in model_counts.keys()
        }
        
        # Check for insufficient samples
        valid_icon = True
        use_all_samples = False
        for model, count in model_counts.items():
            if len(model_samples_sorted[model]) < count:
                if count == n_players:
                    use_all_samples = True
                else:
                    valid_icon = False
                    break
        if not valid_icon:
            continue

        if use_all_samples:
            # If we use all samples (and more players than samples), the max score 
            # is simply the max of the available samples for this model.
            # We assume the model is the same for all players (checked above).
            # Since we use the entire population of samples, there is no randomness 
            # in the set of samples chosen.
            # Note: This assumes n_players > len(samples) AND all players chose same model.
            # We find the max of the samples for that model.
            # Since model_counts has only one key (checked by count == n_players),
            # we can get the model name easily.
            model = list(model_counts.keys())[0]
            if model_samples_sorted[model]:
                icon_expected_max = model_samples_sorted[model][-1] if not lower_is_better else model_samples_sorted[model][0]
                total_expected_max += icon_expected_max
            continue

        icon_expected_max = 0.0
        
        for s_val, s_model in all_samples:
            # Probability that this specific sample 's_val' is chosen and is the winner
            
            vals = model_samples_sorted[s_model]
            N = len(vals)
            k = model_counts[s_model]
            
            if N < k:
                continue
            
            # Count "worse" samples for s_model
            if lower_is_better:
                # Worse = strictly greater
                idx = bisect.bisect_right(vals, s_val)
                count_worse = len(vals) - idx
            else:
                # Worse = strictly smaller
                idx = bisect.bisect_left(vals, s_val)
                count_worse = idx
            
            # Prob for s_model group:
            prob_model = nCr(count_worse, k - 1) / nCr(N, k)
            
            if prob_model == 0:
                continue
                
            # Prob for other models
            prob_others = 1.0
            for other_model, other_count in model_counts.items():
                if other_model == s_model:
                    continue
                
                other_vals = model_samples_sorted[other_model]
                other_N = len(other_vals)
                
                if lower_is_better:
                    idx = bisect.bisect_right(other_vals, s_val)
                    other_count_worse = len(other_vals) - idx
                else:
                    idx = bisect.bisect_left(other_vals, s_val)
                    other_count_worse = idx
                
                term = nCr(other_count_worse, other_count) / nCr(other_N, other_count)
                prob_others *= term
            
            total_prob = prob_model * prob_others
            icon_expected_max += total_prob * s_val

        total_expected_max += icon_expected_max

    return total_expected_max / n_icons


def compute_expected_utility(
    grouped_data: Dict[str, Dict[str, List[float]]],
    player_models: List[str],
    distance_metric: str = "lpips"
) -> Tuple[float, Dict[int, float]]:
    """
    Compute expected utility for each player given their model choices.
    
    Uses efficient probability calculation.
    """
    return get_win_probabilities_efficient(grouped_data, player_models, distance_metric)


def canonicalize_profile(profile: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Convert a strategy profile to its canonical (lexicographically sorted) form.
    
    Args:
        profile: Strategy profile (model_0, model_1, ..., model_{n-1})
    
    Returns:
        Canonical form (sorted lexicographically)
    """
    return tuple(sorted(profile))


def get_profile_mapping(original: Tuple[str, ...], canonical: Tuple[str, ...]) -> Dict[int, int]:
    """
    Get mapping from original player indices to canonical player indices.
    
    Args:
        original: Original strategy profile
        canonical: Canonical (sorted) form of the profile
    
    Returns:
        Dictionary mapping original_player_idx -> canonical_player_idx
    """
    # Create mapping by matching strategies
    mapping = {}
    canonical_used = [False] * len(canonical)
    
    for orig_idx, strategy in enumerate(original):
        # Find first unused position in canonical with same strategy
        for canon_idx, canon_strategy in enumerate(canonical):
            if canon_strategy == strategy and not canonical_used[canon_idx]:
                mapping[orig_idx] = canon_idx
                canonical_used[canon_idx] = True
                break
    
    return mapping


def generate_all_assignments(n_players: int, available_models: List[str]) -> List[Tuple[str, ...]]:
    """
    Generate all canonical (lexicographically sorted) model assignments for n players.
    
    Only generates unique strategy profiles up to permutation.
    
    Args:
        n_players: Number of players
        available_models: List of available model names
    
    Returns:
        List of canonical tuples, each tuple is sorted lexicographically
    """
    def _generate_sorted_tuples(n: int, models: List[str], start_idx: int = 0) -> List[Tuple[str, ...]]:
        """
        Recursively generate all sorted tuples of length n from models[start_idx:].
        This directly generates only canonical forms, avoiding duplicate generation.
        """
        if n == 0:
            return [()]
        
        if start_idx >= len(models):
            return []
        
        result = []
        # For each model from start_idx onwards, we can use it 0 to n times
        for count in range(n + 1):
            # Use current model 'count' times
            prefix = tuple([models[start_idx]] * count)
            # Recursively generate remaining n - count elements from remaining models
            for suffix in _generate_sorted_tuples(n - count, models, start_idx + 1):
                result.append(prefix + suffix)
        
        return result
    
    if n_players == 0:
        return [()]
    
    if not available_models:
        return []
    
    # Sort models to ensure lexicographic ordering
    sorted_models = sorted(available_models)
    return _generate_sorted_tuples(n_players, sorted_models)


def compute_all_utilities(
    grouped_data: Dict[str, Dict[str, List[float]]],
    n_players: int,
    available_models: List[str],
    distance_metric: str = "lpips"
) -> Dict[Tuple[str, ...], Dict[int, float]]:
    """
    Compute expected utility for all canonical model assignments.
    
    Args:
        grouped_data: {icon_name: {model_name: [distance_scores]}}
        n_players: Number of players
        available_models: List of available model names
        distance_metric: Name of distance metric
    
    Returns:
        Dictionary mapping canonical_profile -> {canonical_player_index: expected_utility}
        Utilities are indexed by position in the canonical (sorted) profile.
    """
    all_assignments = generate_all_assignments(n_players, available_models)
    utilities = {}
    
    for canonical_assignment in tqdm(all_assignments, desc="Computing utilities", total=len(all_assignments)):
        _, player_utilities = compute_expected_utility(
            grouped_data, list(canonical_assignment), distance_metric
        )
        utilities[canonical_assignment] = player_utilities
    
    return utilities


def is_nash_equilibrium(
    canonical_assignment: Tuple[str, ...],
    utilities: Dict[Tuple[str, ...], Dict[int, float]],
    available_models: List[str]
) -> bool:
    """
    Check if a canonical assignment is a Nash equilibrium.
    
    A Nash equilibrium is where no player can unilaterally deviate
    to get strictly higher utility.
    
    Args:
        canonical_assignment: Canonical (sorted) model assignment
        utilities: Dictionary mapping canonical assignments to player utilities
        available_models: List of available model names
    
    Returns:
        True if assignment is a Nash equilibrium, False otherwise
    """
    n_players = len(canonical_assignment)
    current_utilities = utilities[canonical_assignment]
    
    # Group players by strategy (since players using same strategy are symmetric)
    strategy_to_players: Dict[str, List[int]] = defaultdict(list)
    for player_idx, strategy in enumerate(canonical_assignment):
        strategy_to_players[strategy].append(player_idx)
    
    # Check each unique strategy group
    for strategy, player_indices in strategy_to_players.items():
        # All players using this strategy have the same utility (by symmetry)
        current_utility = current_utilities[player_indices[0]]
        
        # Try all possible deviations for players using this strategy
        for deviated_model in available_models:
            if deviated_model == strategy:
                continue  # Not a deviation
            
            # Create deviated assignment (one player deviates)
            deviated_assignment = list(canonical_assignment)
            # Replace first occurrence of strategy with deviated_model
            for i, s in enumerate(deviated_assignment):
                if s == strategy:
                    deviated_assignment[i] = deviated_model
                    break
            
            # Canonicalize the deviated assignment
            deviated_canonical = canonicalize_profile(tuple(deviated_assignment))
            
            # Check if deviation gives strictly higher utility
            if deviated_canonical in utilities:
                # Find which player in deviated_canonical corresponds to the deviating player
                # The deviating player uses deviated_model in the canonical form
                deviated_utilities = utilities[deviated_canonical]
                
                # Find utility for a player using deviated_model in the deviated profile
                for canon_idx, canon_strategy in enumerate(deviated_canonical):
                    if canon_strategy == deviated_model:
                        deviated_utility = deviated_utilities[canon_idx]
                        if deviated_utility > current_utility:
                            return False  # Player can deviate to get higher utility
                        break
    
    return True  # No player can deviate to get strictly higher utility


def find_nash_equilibria(
    utilities: Dict[Tuple[str, ...], Dict[int, float]],
    available_models: List[str]
) -> List[Tuple[Tuple[str, ...], Dict[int, float]]]:
    """
    Find all Nash equilibria.
    
    Args:
        utilities: Dictionary mapping assignments to player utilities
        available_models: List of available model names
    
    Returns:
        List of (assignment, utilities) tuples for all Nash equilibria
    """
    equilibria = []
    
    assignments_list = list(utilities.keys())
    for assignment in tqdm(assignments_list, desc="Checking for Nash equilibria", total=len(assignments_list)):
        if is_nash_equilibrium(assignment, utilities, available_models):
            equilibria.append((assignment, utilities[assignment]))
    
    return equilibria


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute expected utility for model selection game"
    )
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
        choices=["lpips", "dreamsim"],
        help="Distance metric to use for determining winners.",
    )
    parser.add_argument(
        "--player-models",
        nargs="+",
        default=None,
        help="Model names for each player (e.g., --player-models sd3 flux sd3). Required unless --find-equilibria is used.",
    )
    parser.add_argument(
        "--find-equilibria",
        action="store_true",
        help="Find all Nash equilibria for n-player game.",
    )
    parser.add_argument(
        "--n-players",
        type=int,
        default=None,
        help="Number of players (required when --find-equilibria is used).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--list-icons",
        action="store_true",
        help="List available icons and exit.",
    )
    parser.add_argument(
        "--show-all-assignments",
        action="store_true",
        help="Show utilities for all assignments when finding equilibria.",
    )
    parser.add_argument(
        "--debug-equilibrium",
        type=str,
        default=None,
        help="Debug why a specific assignment is or isn't an equilibrium (e.g., 'sd3,flux,sd3').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_path = logos_dir / args.input_csv
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")
    
    # Load data
    df = load_distance_data(input_path, args.distance_metric)
    print(f"Loaded {len(df)} distance measurements from {input_path}")
    
    # List models/icons if requested
    if args.list_models:
        models = get_available_models(df)
        print(f"\nAvailable models ({len(models)}):")
        for model in models:
            count = len(df[df["model_name"] == model])
            print(f"  {model}: {count} samples")
        return
    
    if args.list_icons:
        icons = get_available_icons(df)
        print(f"\nAvailable icons ({len(icons)}):")
        for icon in icons:
            count = len(df[df["base_name"] == icon])
            print(f"  {icon}: {count} samples")
        return
    
    # Group data by icon and model
    grouped_data = group_by_icon_and_model(df, args.distance_metric)
    print(f"Grouped data for {len(grouped_data)} icons")
    
    available_models = get_available_models(df)
    
    # Find equilibria mode
    if args.find_equilibria:
        if args.n_players is None:
            raise ValueError("--n-players is required when --find-equilibria is used")
        
        n_players = args.n_players
        print(f"\nFinding Nash equilibria for {n_players} players")
        print(f"Available models: {available_models}")
        
        # Compute and print best single-shot model
        best_model, best_avg = get_best_single_shot_model(df, args.distance_metric)
        if best_model:
            print(f"\nBest single-shot model (average {args.distance_metric}): {best_model} ({best_avg:.4f})")
            model_averages = compute_model_averages(df, args.distance_metric)
            print(f"All model averages:")
            for model in sorted(model_averages.keys()):
                avg = model_averages[model]
                marker = " (best)" if model == best_model else ""
                print(f"  {model}: {avg:.4f}{marker}")
        
        # Compute utilities for all assignments
        all_utilities = compute_all_utilities(
            grouped_data, n_players, available_models, args.distance_metric
        )
        
        # Debug specific assignment if requested
        if args.debug_equilibrium:
            debug_models = [m.strip() for m in args.debug_equilibrium.split(',')]
            if len(debug_models) != n_players:
                raise ValueError(f"Debug assignment must have {n_players} models, got {len(debug_models)}")
            debug_assignment = tuple(debug_models)
            debug_canonical = canonicalize_profile(debug_assignment)
            
            if debug_canonical not in all_utilities:
                print(f"Assignment {debug_assignment} (canonical: {debug_canonical}) not found in computed utilities.")
                return
            
            print(f"\n{'='*80}")
            print(f"Debugging assignment: {debug_assignment}")
            print(f"Canonical form: {debug_canonical}")
            print(f"{'='*80}")
            current_utilities = all_utilities[debug_canonical]
            print(f"Current utilities (canonical positions):")
            for player_idx, utility in current_utilities.items():
                print(f"  Position {player_idx} ({debug_canonical[player_idx]}): {utility:.4f}")
            
            # Group players by strategy
            strategy_to_players: Dict[str, List[int]] = defaultdict(list)
            for orig_idx, strategy in enumerate(debug_assignment):
                strategy_to_players[strategy].append(orig_idx)
            
            print(f"\nChecking deviations:")
            is_eq = True
            for strategy, player_indices in strategy_to_players.items():
                # Map to canonical position
                canon_idx = None
                for i, s in enumerate(debug_canonical):
                    if s == strategy:
                        canon_idx = i
                        break
                if canon_idx is None:
                    continue
                
                current_utility = current_utilities[canon_idx]
                print(f"\n  Players {player_indices} using {strategy} (utility {current_utility:.4f}):")
                for deviated_model in available_models:
                    if deviated_model == strategy:
                        continue
                    deviated_assignment = list(debug_canonical)
                    # Replace first occurrence of strategy with deviated_model
                    for i, s in enumerate(deviated_assignment):
                        if s == strategy:
                            deviated_assignment[i] = deviated_model
                            break
                    deviated_canonical = canonicalize_profile(tuple(deviated_assignment))
                    
                    if deviated_canonical in all_utilities:
                        # Find utility for player using deviated_model
                        deviated_utility = None
                        for i, s in enumerate(deviated_canonical):
                            if s == deviated_model:
                                deviated_utility = all_utilities[deviated_canonical][i]
                                break
                        
                        if deviated_utility is not None:
                            if deviated_utility > current_utility:
                                print(f"    -> Deviate to {deviated_model}: utility {deviated_utility:.4f} (BETTER - not equilibrium)")
                                is_eq = False
                            elif deviated_utility == current_utility:
                                print(f"    -> Deviate to {deviated_model}: utility {deviated_utility:.4f} (SAME)")
                            else:
                                print(f"    -> Deviate to {deviated_model}: utility {deviated_utility:.4f} (WORSE)")
            
            print(f"\n{'='*80}")
            print(f"Is Nash equilibrium: {is_eq}")
            print(f"{'='*80}")
            return
        
        # Find equilibria
        equilibria = find_nash_equilibria(all_utilities, available_models)
        
        print(f"\n{'='*80}")
        print(f"Found {len(equilibria)} Nash equilibrium/equilibria:")
        print(f"{'='*80}")
        
        for i, (canonical_assignment, utilities) in enumerate(equilibria, 1):
            print(f"\nEquilibrium {i}:")
            print(f"  Canonical assignment: {canonical_assignment}")
            print(f"  Utilities (by strategy):")
            # Group by strategy for cleaner output
            strategy_to_utility: Dict[str, float] = {}
            for player_idx, utility in utilities.items():
                strategy = canonical_assignment[player_idx]
                if strategy not in strategy_to_utility:
                    strategy_to_utility[strategy] = utility
            for strategy, utility in sorted(strategy_to_utility.items()):
                count = canonical_assignment.count(strategy)
                print(f"    {strategy}: {utility:.4f} ({count} player{'s' if count > 1 else ''})")
            print(f"  Average utility: {sum(utilities.values()) / len(utilities):.4f}")
        
        if len(equilibria) == 0:
            print("\nNo Nash equilibria found.")
        
        return
    
    # Single assignment mode
    if args.player_models is None:
        raise ValueError("--player-models is required unless --find-equilibria is used")
    
    player_models = args.player_models
    n_players = len(player_models)
    
    print(f"\nPlayer model assignments ({n_players} players):")
    for i, model in enumerate(player_models):
        print(f"  Player {i}: {model}")
    
    # Validate models
    invalid_models = [m for m in player_models if m not in available_models]
    if invalid_models:
        raise ValueError(f"Invalid models: {invalid_models}. Available: {available_models}")
    
    # Compute expected utility
    overall_expected, player_utilities = compute_expected_utility(
        grouped_data, player_models, args.distance_metric
    )
    
    print(f"\nResults (using {args.distance_metric}):")
    print(f"  Overall expected utility: {overall_expected:.4f}")
    print(f"  Per-player expected utility:")
    for i, utility in player_utilities.items():
        print(f"    Player {i} ({player_models[i]}): {utility:.4f}")


if __name__ == "__main__":
    main()

