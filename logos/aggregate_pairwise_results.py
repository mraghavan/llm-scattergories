"""
Aggregate individual pairwise similarity CSV files (one per base_name) 
into a single combined CSV file.

This script reads all CSV files from the pairwise_results directory and combines
them into pairwise_similarity_results.csv, which is consumed by plot_diversity.py.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate individual pairwise similarity CSV files (one per base_name) "
            "into a single combined CSV file."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="pairwise_results",
        help="Directory containing individual CSV files (relative to logos directory).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="pairwise_similarity_results.csv",
        help="Output CSV filename for aggregated results (will be written in the logos directory).",
    )
    return parser.parse_args()


def normalize_optional_str(value: Optional[object]) -> Optional[str]:
    """Normalize a value to an optional string, handling None and NaN."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None


def normalize_optional_int(value: Optional[object]) -> Optional[int]:
    """Normalize a value to an optional int, handling None, NaN, and string conversions."""
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    if isinstance(value, int):
        return value
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return int(value_str)
    except ValueError:
        try:
            return int(float(value_str))
        except ValueError:
            return None


def make_result_key_pairwise(
    base_name: str,
    model_name1: Optional[object],
    seed1: Optional[object],
    model_name2: Optional[object],
    seed2: Optional[object],
) -> Tuple[str, Optional[str], Optional[int], Optional[str], Optional[int]]:
    """
    Create a comparable key for identifying a pairwise generated image comparison.
    
    This function ensures consistent ordering: model1 <= model2, and if equal, seed1 <= seed2.
    This matches the logic in evaluate_pairwise.py for deduplication.
    """
    base = str(base_name).strip()
    model1 = normalize_optional_str(model_name1)
    model2 = normalize_optional_str(model_name2)
    norm_seed1 = normalize_optional_int(seed1)
    norm_seed2 = normalize_optional_int(seed2)
    
    # Ensure consistent ordering: model1 <= model2, and if equal, seed1 <= seed2
    if model1 is not None and model2 is not None:
        if model1 > model2:
            model1, model2 = model2, model1
            norm_seed1, norm_seed2 = norm_seed2, norm_seed1
        elif model1 == model2 and norm_seed1 is not None and norm_seed2 is not None:
            if norm_seed1 > norm_seed2:
                norm_seed1, norm_seed2 = norm_seed2, norm_seed1
    
    return base, model1, norm_seed1, model2, norm_seed2


def main() -> None:
    args = parse_args()
    logos_dir = Path(__file__).parent
    input_dir = logos_dir / args.input_dir
    output_path = logos_dir / args.output_csv

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found at {input_dir}")

    # Find all CSV files matching the pattern pairwise_similarity_results_*.csv
    csv_files = sorted(input_dir.glob("pairwise_similarity_results_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        print("Expected files matching pattern: pairwise_similarity_results_*.csv")
        return

    print(f"Found {len(csv_files)} CSV file(s) to aggregate")

    # Read all CSV files and combine, deduplicating by key
    all_records: List[Dict] = []
    seen_keys = set()
    duplicates = 0
    total_rows = 0

    for csv_file in csv_files:
        print(f"  Reading {csv_file.name}...", end=" ", flush=True)
        try:
            df = pd.read_csv(csv_file)
            records = df.to_dict("records")
            file_rows = len(records)
            total_rows += file_rows
            
            file_duplicates = 0
            for record in records:
                # Skip records without base_name (shouldn't happen, but be defensive)
                if record.get("base_name") is None:
                    continue
                
                key = make_result_key_pairwise(
                    record.get("base_name"),
                    record.get("model_name1"),
                    record.get("seed1"),
                    record.get("model_name2"),
                    record.get("seed2"),
                )
                
                if key not in seen_keys:
                    all_records.append(record)
                    seen_keys.add(key)
                else:
                    file_duplicates += 1
                    duplicates += 1
            
            print(f"{file_rows} rows ({file_duplicates} duplicates skipped)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

    if not all_records:
        print("No records found to aggregate.")
        return

    # Write aggregated results
    df_output = pd.DataFrame(all_records)
    df_output.to_csv(output_path, index=False)
    
    print(f"\nAggregation complete:")
    print(f"  Files processed: {len(csv_files)}")
    print(f"  Total rows read: {total_rows}")
    print(f"  Unique records: {len(all_records)}")
    if duplicates > 0:
        print(f"  Duplicates skipped: {duplicates}")
    print(f"  Output CSV: {output_path}")
    
    # Print summary statistics
    if len(all_records) > 0:
        print(f"\nSummary statistics:")
        print(f"  Unique base_names: {df_output['base_name'].nunique()}")
        print(f"  Unique model combinations: {df_output[['model_name1', 'model_name2']].drop_duplicates().shape[0]}")
        # Check which metrics are present
        metrics = ['lpips', 'dreamsim']
        for metric in metrics:
            if metric in df_output.columns:
                non_null = df_output[metric].notna().sum()
                print(f"  {metric}: {non_null} non-null values")


if __name__ == "__main__":
    main()
