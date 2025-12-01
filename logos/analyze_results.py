import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze image distance evaluation results and compute statistics "
            "about which models perform best according to each metric."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="image_similarity_results.csv",
        help="Input CSV filename (should be in the logos directory).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional: Output CSV filename for model statistics summary.",
    )
    return parser.parse_args()


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def get_available_metrics(df: pd.DataFrame) -> list[str]:
    """Get list of metrics that exist in the dataframe and have at least one non-null value."""
    metrics = ["lpips", "dreamsim"]
    return [m for m in metrics if m in df.columns and df[m].notna().any()]


def compute_model_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics per model for each metric."""
    available_metrics = get_available_metrics(df)

    stats_list = []
    for model in df["model_name"].dropna().unique():
        model_df = df[df["model_name"] == model]
        stat_row = {"model_name": model, "n_samples": len(model_df)}

        for metric in available_metrics:
            values = model_df[metric].dropna()
            if len(values) > 0:
                stat_row[f"{metric}_mean"] = values.mean()
                stat_row[f"{metric}_median"] = values.median()
                stat_row[f"{metric}_std"] = values.std()
                stat_row[f"{metric}_min"] = values.min()
                stat_row[f"{metric}_max"] = values.max()
                stat_row[f"{metric}_count"] = len(values)
            else:
                stat_row[f"{metric}_mean"] = None
                stat_row[f"{metric}_median"] = None
                stat_row[f"{metric}_std"] = None
                stat_row[f"{metric}_min"] = None
                stat_row[f"{metric}_max"] = None
                stat_row[f"{metric}_count"] = 0

        stats_list.append(stat_row)

    return pd.DataFrame(stats_list)


def rank_models_by_metric(stats_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models by a metric (handles both lower-is-better and higher-is-better)."""
    mean_col = f"{metric}_mean"
    if mean_col not in stats_df.columns:
        return pd.DataFrame()

    # Filter to models with data for this metric
    metric_df = stats_df[stats_df[mean_col].notna()].copy()

    if len(metric_df) == 0:
        return pd.DataFrame()

    # All distance metrics are lower-is-better
    metric_df = metric_df.sort_values(mean_col, ascending=True)
    metric_df["rank"] = range(1, len(metric_df) + 1)

    return metric_df[["model_name", "rank", mean_col, f"{metric}_std", f"{metric}_count"]]


def print_model_rankings(stats_df: pd.DataFrame, available_metrics: list[str]) -> None:
    """Print rankings for each metric."""
    for metric in available_metrics:
        ranking = rank_models_by_metric(stats_df, metric)
        if len(ranking) == 0:
            continue

        print_section(f"Model Rankings by {metric.upper().replace('_', ' ')}")
        print("(Lower is better)")

        print(f"\n{'Rank':<6} {'Model':<20} {'Mean':<12} {'Std':<12} {'Count':<8}")
        print("-" * 60)
        for _, row in ranking.iterrows():
            mean_val = row[f"{metric}_mean"]
            std_val = row[f"{metric}_std"]
            count_val = int(row[f"{metric}_count"])
            print(
                f"{int(row['rank']):<6} {row['model_name']:<20} "
                f"{mean_val:<12.6f} {std_val:<12.6f} {count_val:<8}"
            )


def print_detailed_statistics(stats_df: pd.DataFrame) -> None:
    """Print detailed statistics table."""
    print_section("Detailed Model Statistics")
    print(stats_df.to_string(index=False))


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print overall summary statistics."""
    print_section("Overall Summary")
    print(f"Total comparisons: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Unique base names: {df['base_name'].nunique()}")

    available_metrics = get_available_metrics(df)

    print("\nMetric availability:")
    for metric in available_metrics:
        count = df[metric].notna().sum()
        print(f"  {metric}: {count} samples ({count/len(df)*100:.1f}%)")


def print_per_base_statistics(df: pd.DataFrame) -> None:
    """Print statistics broken down by base_name."""
    print_section("Statistics by Base Name")

    available_metrics = get_available_metrics(df)

    for base_name in sorted(df["base_name"].unique()):
        base_df = df[df["base_name"] == base_name]
        print(f"\n{base_name}:")
        print(f"  Total samples: {len(base_df)}")
        print(f"  Models: {', '.join(sorted(base_df['model_name'].dropna().unique()))}")

        for metric in available_metrics:
            values = base_df[metric].dropna()
            if len(values) > 0:
                # All distance metrics: lower is better
                best_idx = values.idxmin()
                best_model = base_df.loc[best_idx, "model_name"]
                best_val = values.min()

                print(
                    f"  {metric}: mean={values.mean():.6f}, "
                    f"std={values.std():.6f}, "
                    f"best_single_sample={best_model} ({best_val:.6f})"
                )


def main() -> None:
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_path = logos_dir / args.input_csv

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_path}")

    print(f"Reading results from: {input_path}")
    df = pd.read_csv(input_path)

    if len(df) == 0:
        print("No data found in CSV.")
        return

    # Get available metrics (those with actual data)
    available_metrics = get_available_metrics(df)

    # Print overall summary
    print_summary_statistics(df)

    # Compute and print model statistics
    stats_df = compute_model_statistics(df)
    print_detailed_statistics(stats_df)

    # Print rankings
    print_model_rankings(stats_df, available_metrics)

    # Print per-base statistics
    print_per_base_statistics(df)

    # Optionally save statistics to CSV
    if args.output_csv:
        output_path = logos_dir / args.output_csv
        stats_df.to_csv(output_path, index=False)
        print(f"\nSaved model statistics to: {output_path}")

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

