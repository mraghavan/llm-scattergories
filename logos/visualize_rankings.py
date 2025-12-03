import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from PIL import Image

# Define a color palette for models
MODEL_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize top ranked generated images by distance to original."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="logos/image_similarity_results.csv",
        help="Path to the CSV file containing similarity results.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="logos/generated_images",
        help="Directory containing the original and generated images.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="lpips",
        choices=["lpips", "dreamsim", "all"],
        help="Metric to use for ranking (default: lpips). Use 'all' to run for all metrics.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for the visualization plot. If not provided, defaults to 'logos/similarity_rankings_<metric>.png'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top similar images to show per icon (default: 10).",
    )
    parser.add_argument(
        "--farthest",
        action="store_true",
        help="If set, show the farthest (least similar) images instead of the closest.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model names to include in the visualization. If not provided, all models are included.",
    )
    return parser.parse_args()


def load_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def generate_plot(
    df: pd.DataFrame,
    images_dir: Path,
    metric: str,
    top_k: int,
    farthest: bool,
    output_path: Path
) -> None:
    # Filter out rows where the metric is missing
    if metric not in df.columns:
        print(f"Warning: Metric '{metric}' not found in CSV columns. Skipping.")
        return
    
    df_metric = df.dropna(subset=[metric])

    # Determine sorting order
    # Distance metrics: lower is better (ascending). If farthest, we want descending.
    ascending = True
    if farthest:
        ascending = False
    
    # Group by base_name (icon)
    grouped = df_metric.groupby("base_name")
    unique_icons = sorted(grouped.groups.keys())
    
    if not unique_icons:
        print(f"No icons found for metric {metric}.")
        return

    # Get unique models for color mapping
    unique_models = sorted(df_metric["model_name"].unique())
    model_color_map = {
        model: MODEL_COLORS[i % len(MODEL_COLORS)] 
        for i, model in enumerate(unique_models)
    }

    num_icons = len(unique_icons)
    # 1 for original + 1 gap + k for top ranked
    cols = top_k + 2 
    
    # Calculate figure size
    # Assuming each subplot is roughly 2x2 inches
    fig_width = cols * 2.0
    fig_height = num_icons * 2.0 + 1.0 # Extra height for legend
    
    print(f"Creating plot for {metric} ({'farthest' if farthest else 'closest'}) - {num_icons} icons...")
    fig, axes = plt.subplots(num_icons, cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9, bottom=0.05)

    for i, base_name in enumerate(unique_icons):
        # 1. Plot Original Image (Column 0)
        orig_filename = f"{base_name}_original.png"
        orig_path = images_dir / orig_filename
        orig_img = load_image(orig_path)
        
        ax_orig = axes[i, 0]
        if orig_img:
            ax_orig.imshow(orig_img)
            if i == 0:
                ax_orig.set_title("Original", fontsize=12, fontweight="bold")
        else:
            ax_orig.text(0.5, 0.5, "Missing", ha="center", va="center")
        ax_orig.axis("off")
        
        # 2. Gap Column (Column 1)
        ax_gap = axes[i, 1]
        ax_gap.axis("off")
        
        # 3. Plot Top k Generated Images (Columns 2 to k+1)
        group_df = grouped.get_group(base_name)
        sorted_df = group_df.sort_values(by=metric, ascending=ascending).head(top_k)
        
        for j, (_, row) in enumerate(sorted_df.iterrows()):
            model_name = row["model_name"]
            seed = row["seed"]
            
            gen_filename = f"{base_name}_{model_name}_{seed}.png"
            gen_path = images_dir / gen_filename
            gen_img = load_image(gen_path)
            
            ax_gen = axes[i, j + 2] # Start after gap
            if gen_img:
                ax_gen.imshow(gen_img)
                
                # Add color-coded frame
                color = model_color_map.get(model_name, "black")
                
                # Turn axis on to show spines (frame)
                ax_gen.axis("on")
                # Remove ticks and labels
                ax_gen.set_xticks([])
                ax_gen.set_yticks([])
                
                # Add padding (whitespace) between image and frame
                w, h = gen_img.size
                pad_w = w * 0.05  # 5% padding
                pad_h = h * 0.05
                ax_gen.set_xlim(-pad_w, w + pad_w)
                ax_gen.set_ylim(h + pad_h, -pad_h) # Inverted Y for images
                
                # Set spine color and width
                for spine in ax_gen.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(8) # Thicker frame
            else:
                ax_gen.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax_gen.axis("off")
            
        # Hide any unused columns if we have fewer than top_k results
        for j in range(len(sorted_df), top_k):
             axes[i, j + 2].axis("off")

    # Add Legend
    legend_patches = [
        mpatches.Patch(color=color, label=model)
        for model, color in model_color_map.items()
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98), # Position at top
        ncol=min(len(unique_models), 5), # Max 5 columns
        fontsize=12,
        frameon=False
    )
    
    # Add extra margin at top for legend
    plt.subplots_adjust(top=0.92)

    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig) # Close figure to free memory
    print("Done!")


def main() -> None:
    args = parse_args()
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    logos_dir = Path(__file__).parent
    
    # If input_csv is a relative path, look in logos directory
    input_csv_arg = Path(args.input_csv)
    if input_csv_arg.is_absolute():
        input_csv_path = input_csv_arg
    else:
        # Check if it exists in logos directory first, otherwise try repo root
        logos_csv_path = logos_dir / args.input_csv
        if logos_csv_path.exists():
            input_csv_path = logos_csv_path
        else:
            input_csv_path = repo_root / args.input_csv
    
    images_dir = repo_root / args.images_dir
    
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {input_csv_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found at {images_dir}")

    # Load results
    print(f"Loading results from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    # Filter by models if specified
    if args.models:
        available_models = df["model_name"].unique()
        requested_models = set(args.models)
        available_models_set = set(available_models)
        
        # Check for invalid models
        invalid_models = requested_models - available_models_set
        if invalid_models:
            print(f"Warning: The following models were not found in the data: {sorted(invalid_models)}")
            print(f"Available models: {sorted(available_models)}")
        
        # Filter to only requested models that exist
        valid_models = requested_models & available_models_set
        if valid_models:
            df = df[df["model_name"].isin(valid_models)]
            print(f"Filtering to models: {sorted(valid_models)}")
        else:
            raise ValueError(f"No valid models found. Available models: {sorted(available_models)}")
    
    metrics_to_run = []
    if args.metric == "all":
        metrics_to_run = ["lpips", "dreamsim"]
    else:
        metrics_to_run = [args.metric]
        
    for metric in metrics_to_run:
        # Determine output path
        if args.output and args.metric != "all":
            # If explicit output provided and single metric, use it
            output_path = repo_root / args.output
        else:
            # Otherwise construct default path
            suffix = "_farthest" if args.farthest else ""
            output_path = repo_root / f"logos/similarity_rankings_{metric}{suffix}.png"
            
        generate_plot(
            df=df,
            images_dir=images_dir,
            metric=metric,
            top_k=args.top_k,
            farthest=args.farthest,
            output_path=output_path
        )

if __name__ == "__main__":
    main()
