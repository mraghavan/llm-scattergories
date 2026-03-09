from file_manager import FileManager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import argparse
from scat_utils import get_model_list, MAX_TEMPS
from pathlib import Path
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, default='')
parser.add_argument('--from-config', '-c', type=str, default='',
                    help='Load model configuration from a config file')
parser.add_argument('--verifier', '-v', type=str, required=True,
                    help='Verifier model name to use for correctness')
parser.add_argument('--no_save', '-n', action='store_true', default=False)
parser.add_argument('--output', '-o', type=str, default='correctness_vs_entropy.png',
                    help='Output filename for the plot')
parser.add_argument('--output-conditional', type=str, default='',
                    help='Output filename for the conditional entropy plot (default: derived from --output)')
parser.add_argument('--output-combined', type=str, default='',
                    help='Output filename for the combined plot (default: derived from --output)')
parser.add_argument('--empirical-entropy', '-e', action='store_true', default=False,
                    help='Use empirical entropy instead of Miller-Madow corrected')
SAVE = True

def get_entropy(samples: dict[str, int]) -> float:
    """Calculate entropy from sample counts"""
    total = sum(samples.values())
    if total == 0:
        return 0.0
    entropy = 0
    for count in samples.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * np.log(p)
    return entropy

def get_entropy3(samples: dict[str, int]) -> float:
    """Miller-Madow corrected entropy"""
    total = sum(samples.values())
    if total == 0:
        return 0.0
    entropy = get_entropy(samples)
    entropy += (len(samples) - 1) / (2 * total)
    return entropy

def calculate_pr_correct(samples: dict[str, int], verified_yes: set[str]) -> float:
    """Calculate Pr[answer is correct] from sample distribution and verified answers"""
    total = sum(samples.values())
    if total == 0:
        return 0.0
    correct_count = sum(count for answer, count in samples.items() if answer in verified_yes)
    return correct_count / total

def get_conditional_entropy(samples: dict[str, int], verified_yes: set[str], use_empirical: bool = False) -> float:
    """Calculate entropy of the distribution conditional on correctness (only correct answers)"""
    # Filter samples to only include correct answers
    correct_samples = {answer: count for answer, count in samples.items() if answer in verified_yes}
    
    if len(correct_samples) == 0:
        return 0.0
    
    # Calculate entropy on the filtered distribution
    entropy_fn = get_entropy if use_empirical else get_entropy3
    return entropy_fn(correct_samples)

def make_plot(models_data: list[tuple[str, list[tuple[float, float, float]]]], fm: FileManager, output_fname: str, use_empirical: bool = False):
    """
    Plot Pr[answer is correct] vs entropy for each (model, temperature) pair.
    
    Args:
        models_data: List of (model_name, [(temp, pr_correct, entropy), ...]) tuples
        fm: FileManager instance
        output_fname: Output filename for the plot
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    """
    plt.figure(figsize=(10, 8))
    
    # Get color cycle from matplotlib's default prop cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle([item['color'] for item in prop_cycle])
    
    for model_name, points in models_data:
        # Sort points by temperature to connect them properly
        points_sorted = sorted(points, key=lambda x: x[0])
        temps = [p[0] for p in points_sorted]
        pr_correct = [p[1] for p in points_sorted]
        entropies = [p[2] for p in points_sorted]
        
        # Get base color for this model
        color = next(colors)
        
        # Normalize temperatures to [0, 1] for this model
        min_temp = min(temps)
        max_temp = max(temps)
        if max_temp > min_temp:
            temp_normalized = [(t - min_temp) / (max_temp - min_temp) for t in temps]
        else:
            temp_normalized = [0.5] * len(temps)  # All same temp, use middle darkness
        
        # Plot line with base color (entropy on x-axis, accuracy on y-axis)
        plt.plot(entropies, pr_correct, '-', label=model_name, color=color, linewidth=1.5)
        
        # Plot markers with darker edge colors for higher temperatures (open circles)
        # Higher temperature = darker edge color (brightness range 0.1 to 1.0)
        for i, (x, y, temp_norm) in enumerate(zip(entropies, pr_correct, temp_normalized)):
            brightness = 1.0 - 0.9 * temp_norm
            marker_color = tuple(c * brightness for c in mcolors.to_rgb(color))
            plt.plot(x, y, 'o', color=marker_color, markerfacecolor='none',
                    markeredgewidth=1.7, markersize=7, zorder=3)
    
    entropy_label = 'Average entropy' if use_empirical else 'Average entropy (Miller-Madow corrected)'
    plt.xlabel(entropy_label, fontsize=16)
    plt.ylabel('Average accuracy', fontsize=16)
    plt.title('Average accuracy vs entropy', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if SAVE:
        fname = fm.locations.plots_dir / output_fname
        print('Saving to', fname)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def make_plot_combined(models_data: list[tuple[str, list[tuple[float, float, float]]]], 
                       models_data_conditional: list[tuple[str, list[tuple[float, float, float]]]], 
                       fm: FileManager, output_fname: str, use_empirical: bool = False):
    """
    Plot Pr[answer is correct] vs both regular entropy and conditional entropy 
    on the same axes for each (model, temperature) pair.
    
    Args:
        models_data: List of (model_name, [(temp, pr_correct, entropy), ...]) tuples for regular entropy
        models_data_conditional: List of (model_name, [(temp, pr_correct, conditional_entropy), ...]) tuples
        fm: FileManager instance
        output_fname: Output filename for the plot
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    """
    plt.figure(figsize=(10, 8))
    
    # Get color cycle from matplotlib's default prop cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle([item['color'] for item in prop_cycle])
    
    # Create a mapping from model name to color to ensure same color for both curves
    model_colors = {}
    for model_name, _ in models_data:
        if model_name not in model_colors:
            model_colors[model_name] = next(colors)
    
    # Plot regular entropy curves (solid lines)
    for model_name, points in models_data:
        points_sorted = sorted(points, key=lambda x: x[0])
        pr_correct = [p[1] for p in points_sorted]
        entropies = [p[2] for p in points_sorted]
        
        color = model_colors[model_name]
        
        # Plot line with solid style (no label - we'll create custom legend)
        plt.plot(entropies, pr_correct, '-', color=color, linewidth=1.5)
    
    # Plot conditional entropy curves (dashed lines)
    for model_name, points in models_data_conditional:
        points_sorted = sorted(points, key=lambda x: x[0])
        pr_correct = [p[1] for p in points_sorted]
        conditional_entropies = [p[2] for p in points_sorted]
        
        color = model_colors.get(model_name, next(colors))
        
        # Plot line with dashed style (no label - we'll create custom legend)
        plt.plot(conditional_entropies, pr_correct, '--', color=color, linewidth=1.5)
    
    # Create custom legend: 2 columns (Entropy | Conditional) for each model
    # Similar to make_legend in make_plots.py
    legend_handles = []
    
    # Create handles for entropy (solid lines) with model names
    for model_name, color in model_colors.items():
        handle = Line2D([0], [0], color=color, linestyle='-', linewidth=1.5, label=model_name)
        legend_handles.append(handle)
    
    # Create handles for conditional entropy (dashed lines) with model names
    for model_name, color in model_colors.items():
        handle = Line2D([0], [0], color=color, linestyle='--', linewidth=1.5, label=model_name)
        legend_handles.append(handle)
    
    # Set labels to empty for first half (entropy handles) - model names will appear in conditional column
    num_models = len(model_colors)
    for i in range(num_models):
        legend_handles[i].set_label('')
    
    # Create header patches - use Line2D with very short lines for better alignment
    entropy_header = Line2D([0], [0], color='None', linestyle='None', linewidth=0, label='Entropy')
    conditional_header = Line2D([0], [0], color='None', linestyle='None', linewidth=0, label='Conditional')
    
    # Arrange: [Entropy header] + [entropy handles (empty labels)] + [Conditional header] + [conditional handles (with names)]
    new_handles = [entropy_header] + legend_handles[:num_models] + [conditional_header] + legend_handles[num_models:]
    
    entropy_label = 'Average entropy' if use_empirical else 'Average entropy (Miller-Madow corrected)'
    plt.xlabel(entropy_label, fontsize=16)
    plt.ylabel('Average accuracy', fontsize=16)
    plt.title('Average accuracy vs entropy (combined)', fontsize=18)
    
    legend = plt.legend(handles=new_handles, loc='upper right', ncol=2, 
                        columnspacing=-1.5, handletextpad=0.0, handlelength=1.5)
    
    # Adjust text positioning: move headers up and align properly
    texts = legend.get_texts()
    for i, text in enumerate(texts):
        if i == 0:  # Entropy header (first column)
            text.set_position((-70, -10))  # Move the text up
        elif i == num_models + 1:  # Conditional header (second column)
            text.set_position((-70, -10))  # Move the text up, same x offset for alignment
        else:
            text.set_position((10, 0))  # Add some padding to other labels
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if SAVE:
        fname = fm.locations.plots_dir / output_fname
        print('Saving to', fname)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def make_plot_conditional(models_data: list[tuple[str, list[tuple[float, float, float]]]], fm: FileManager, output_fname: str, use_empirical: bool = False):
    """
    Plot Pr[answer is correct] vs conditional entropy (entropy of correct answers only) 
    for each (model, temperature) pair.
    
    Args:
        models_data: List of (model_name, [(temp, pr_correct, conditional_entropy), ...]) tuples
        fm: FileManager instance
        output_fname: Output filename for the plot
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    """
    plt.figure(figsize=(10, 8))
    
    # Get color cycle from matplotlib's default prop cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle([item['color'] for item in prop_cycle])
    
    for model_name, points in models_data:
        # Sort points by temperature to connect them properly
        points_sorted = sorted(points, key=lambda x: x[0])
        temps = [p[0] for p in points_sorted]
        pr_correct = [p[1] for p in points_sorted]
        conditional_entropies = [p[2] for p in points_sorted]
        
        # Get base color for this model
        color = next(colors)
        
        # Normalize temperatures to [0, 1] for this model
        min_temp = min(temps)
        max_temp = max(temps)
        if max_temp > min_temp:
            temp_normalized = [(t - min_temp) / (max_temp - min_temp) for t in temps]
        else:
            temp_normalized = [0.5] * len(temps)  # All same temp, use middle darkness
        
        # Plot line with base color (conditional entropy on x-axis, accuracy on y-axis)
        plt.plot(conditional_entropies, pr_correct, '-', label=model_name, color=color, linewidth=1.5)
        
        # Plot markers with darker edge colors for higher temperatures (open circles)
        # Higher temperature = darker edge color (brightness range 0.1 to 1.0)
        for i, (x, y, temp_norm) in enumerate(zip(conditional_entropies, pr_correct, temp_normalized)):
            brightness = 1.0 - 0.9 * temp_norm
            marker_color = tuple(c * brightness for c in mcolors.to_rgb(color))
            plt.plot(x, y, 'o', color=marker_color, markerfacecolor='none',
                    markeredgewidth=1.7, markersize=7, zorder=3)
    
    entropy_label = 'Average entropy conditional on correctness' if use_empirical else 'Average entropy conditional on correctness, (Miller-Madow corrected)'
    plt.xlabel(entropy_label, fontsize=16)
    plt.ylabel('Average accuracy', fontsize=16)
    plt.title('Average accuracy vs entropy (conditional on correctness)', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if SAVE:
        fname = fm.locations.plots_dir / output_fname
        print('Saving to', fname)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def make_plot_two_panel(models_data: list[tuple[str, list[tuple[float, float, float]]]], 
                       models_data_conditional: list[tuple[str, list[tuple[float, float, float]]]], 
                       fm: FileManager, output_fname: str, use_empirical: bool = False):
    """
    Plot Pr[answer is correct] vs entropy in a two-panel plot.
    Left panel: unconditional entropy
    Right panel: conditional entropy
    
    Args:
        models_data: List of (model_name, [(temp, pr_correct, entropy), ...]) tuples for unconditional entropy
        models_data_conditional: List of (model_name, [(temp, pr_correct, conditional_entropy), ...]) tuples
        fm: FileManager instance
        output_fname: Output filename for the plot
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    
    # Get color cycle from matplotlib's default prop cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = cycle([item['color'] for item in prop_cycle])
    
    # Create a mapping from model name to color to ensure same color across both panels
    model_colors = {}
    for model_name, _ in models_data:
        if model_name not in model_colors:
            model_colors[model_name] = next(colors)
    
    # Plot unconditional entropy on left panel
    for model_name, points in models_data:
        points_sorted = sorted(points, key=lambda x: x[0])
        temps = [p[0] for p in points_sorted]
        pr_correct = [p[1] for p in points_sorted]
        entropies = [p[2] for p in points_sorted]
        
        color = model_colors[model_name]
        
        # Normalize temperatures to [0, 1] for this model
        min_temp = min(temps)
        max_temp = max(temps)
        if max_temp > min_temp:
            temp_normalized = [(t - min_temp) / (max_temp - min_temp) for t in temps]
        else:
            temp_normalized = [0.5] * len(temps)
        
        # Plot line (no label - legend will be on right panel)
        ax1.plot(entropies, pr_correct, '-', color=color, linewidth=1.5)
        
        # Plot markers
        for i, (x, y, temp_norm) in enumerate(zip(entropies, pr_correct, temp_normalized)):
            brightness = 1.0 - 0.9 * temp_norm
            marker_color = tuple(c * brightness for c in mcolors.to_rgb(color))
            ax1.plot(x, y, 'o', color=marker_color, markerfacecolor='none',
                    markeredgewidth=1.7, markersize=7, zorder=3)
    
    # Plot conditional entropy on right panel
    for model_name, points in models_data_conditional:
        points_sorted = sorted(points, key=lambda x: x[0])
        temps = [p[0] for p in points_sorted]
        pr_correct = [p[1] for p in points_sorted]
        conditional_entropies = [p[2] for p in points_sorted]
        
        color = model_colors.get(model_name, next(colors))
        
        # Normalize temperatures to [0, 1] for this model
        min_temp = min(temps)
        max_temp = max(temps)
        if max_temp > min_temp:
            temp_normalized = [(t - min_temp) / (max_temp - min_temp) for t in temps]
        else:
            temp_normalized = [0.5] * len(temps)
        
        # Plot line with label for legend
        ax2.plot(conditional_entropies, pr_correct, '-', label=model_name, color=color, linewidth=1.5)
        
        # Plot markers
        for i, (x, y, temp_norm) in enumerate(zip(conditional_entropies, pr_correct, temp_normalized)):
            brightness = 1.0 - 0.9 * temp_norm
            marker_color = tuple(c * brightness for c in mcolors.to_rgb(color))
            ax2.plot(x, y, 'o', color=marker_color, markerfacecolor='none',
                    markeredgewidth=1.7, markersize=7, zorder=3)
    
    # Set labels and titles
    xlabel = 'Average entropy' if use_empirical else 'Average entropy (Miller-Madow corrected)'
    
    # Remove individual x-axis labels and add shared label
    ax1.set_xlabel('', fontsize=16)
    ax1.set_ylabel('Average accuracy', fontsize=16)
    ax1.set_title('Unconditional', fontsize=18)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('', fontsize=16)
    ax2.set_title('Conditional on correctness', fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add shared x-axis label
    fig.supxlabel(xlabel, fontsize=16)
    
    plt.tight_layout()
    
    if SAVE:
        fname = fm.locations.plots_dir / output_fname
        print('Saving to', fname)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def compute_metrics_for_model_temp(fm: FileManager, model_id: str, temp: float, 
                                   verified_map: dict[tuple[str, str], set[str]], 
                                   use_empirical: bool = False) -> tuple[float, float]:
    """
    Compute average Pr[correct] and entropy for a (model, temperature) pair across all instances.
    
    Args:
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    
    Returns:
        (pr_correct, entropy) averaged across all instances
    """
    samples_df = fm.get_all_samples(model=model_id, max_temp=temp)
    samples_df = samples_df[samples_df['temperature'] == temp]
    
    if len(samples_df) == 0:
        return 0.0, 0.0
    
    total_pr_correct = 0.0
    total_entropy = 0.0
    num_instances = 0
    
    # Choose entropy function
    entropy_fn = get_entropy if use_empirical else get_entropy3
    
    for _, row in samples_df.iterrows():
        letter = row['letter']
        category = row['category']
        fname = Path(row['fname'])
        
        # Load samples
        info = fm.load_from_path(fname)
        samples = info['dist']
        
        # Get verified answers for this instance
        verified_yes = verified_map.get((letter, category), set())
        
        # Calculate metrics
        pr_correct = calculate_pr_correct(samples, verified_yes)
        entropy = entropy_fn(samples)
        
        total_pr_correct += pr_correct
        total_entropy += entropy
        num_instances += 1
    
    if num_instances == 0:
        return 0.0, 0.0
    
    return total_pr_correct / num_instances, total_entropy / num_instances

def compute_conditional_metrics_for_model_temp(fm: FileManager, model_id: str, temp: float, 
                                               verified_map: dict[tuple[str, str], set[str]], 
                                               use_empirical: bool = False) -> tuple[float, float]:
    """
    Compute average Pr[correct] and conditional entropy (entropy of correct answers only) 
    for a (model, temperature) pair across all instances.
    
    Args:
        use_empirical: If True, use empirical entropy; if False, use Miller-Madow corrected
    
    Returns:
        (pr_correct, conditional_entropy) averaged across all instances
    """
    samples_df = fm.get_all_samples(model=model_id, max_temp=temp)
    samples_df = samples_df[samples_df['temperature'] == temp]
    
    if len(samples_df) == 0:
        return 0.0, 0.0
    
    total_pr_correct = 0.0
    total_conditional_entropy = 0.0
    num_instances = 0
    
    for _, row in samples_df.iterrows():
        letter = row['letter']
        category = row['category']
        fname = Path(row['fname'])
        
        # Load samples
        info = fm.load_from_path(fname)
        samples = info['dist']
        
        # Get verified answers for this instance
        verified_yes = verified_map.get((letter, category), set())
        
        # Calculate metrics
        pr_correct = calculate_pr_correct(samples, verified_yes)
        conditional_entropy = get_conditional_entropy(samples, verified_yes, use_empirical)
        
        total_pr_correct += pr_correct
        total_conditional_entropy += conditional_entropy
        num_instances += 1
    
    if num_instances == 0:
        return 0.0, 0.0
    
    return total_pr_correct / num_instances, total_conditional_entropy / num_instances

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Add validation for required arguments
    if bool(args.models) == bool(args.from_config):
        parser.error("Either --models or --from-config must be specified, but not both")
    
    SAVE = not args.no_save
    fm = FileManager.from_base('.')
    
    # Load verification data
    print(f'Loading verification data for verifier: {args.verifier}')
    all_verified = fm.get_all_verified(verifier=args.verifier)
    verified_map = {}
    for _, row in all_verified.iterrows():
        letter = row['letter']
        category = row['category']
        verified_data = fm.load_verified(letter, category, args.verifier)
        verified_map[(letter, category)] = verified_data['yes']
    
    print(f'Loaded verification data for {len(verified_map)} instances')
    
    # Get model configurations
    if args.from_config:
        from completion_hf import MODELS
        configs_df = fm.get_all_model_configs()
        if configs_df.empty:
            raise ValueError(f"No configs found in {fm.locations.models_dir}")
        
        # Group by model nickname and collect all temperatures
        model_configs = {}
        for _, row in configs_df.iterrows():
            config = row.to_dict()
            nickname = config['model']
            model_id = config['id']
            temp = config['temperature']
            
            if nickname not in model_configs:
                model_configs[nickname] = []
            model_configs[nickname].append((model_id, temp))
        
        models_data = []
        for nickname, configs in model_configs.items():
            points = []
            for model_id, temp in configs:
                pr_correct, entropy = compute_metrics_for_model_temp(fm, model_id, temp, verified_map, args.empirical_entropy)
                points.append((temp, pr_correct, entropy))
                print(f'{nickname} (id: {model_id}) at temp {temp}: Pr[correct]={pr_correct:.4f}, entropy={entropy:.4f}')
            if points:
                models_data.append((nickname, points))
    else:
        from completion_hf import MODELS
        models = get_model_list(args.models, set(MODELS.keys()))
        max_temps = {model: MAX_TEMPS[MODELS[model]] for model in models}
        
        models_data = []
        for model in models:
            max_temp = max_temps[model]
            samples_df = fm.get_all_samples(model=model, max_temp=max_temp)
            temps = sorted(samples_df['temperature'].unique())
            
            points = []
            for temp in temps:
                pr_correct, entropy = compute_metrics_for_model_temp(fm, model, temp, verified_map, args.empirical_entropy)
                points.append((temp, pr_correct, entropy))
                print(f'{model} at temp {temp}: Pr[correct]={pr_correct:.4f}, entropy={entropy:.4f}')
            
            if points:
                models_data.append((model, points))
    
    plt.rcParams.update({'font.size': 14})
    
    # Compute conditional metrics
    if args.from_config:
        models_data_conditional = []
        for nickname, configs in model_configs.items():
            points = []
            for model_id, temp in configs:
                pr_correct, conditional_entropy = compute_conditional_metrics_for_model_temp(fm, model_id, temp, verified_map, args.empirical_entropy)
                points.append((temp, pr_correct, conditional_entropy))
                print(f'{nickname} (id: {model_id}) at temp {temp}: Pr[correct]={pr_correct:.4f}, conditional entropy={conditional_entropy:.4f}')
            if points:
                models_data_conditional.append((nickname, points))
    else:
        models_data_conditional = []
        for model in models:
            max_temp = max_temps[model]
            samples_df = fm.get_all_samples(model=model, max_temp=max_temp)
            temps = sorted(samples_df['temperature'].unique())
            
            points = []
            for temp in temps:
                pr_correct, conditional_entropy = compute_conditional_metrics_for_model_temp(fm, model, temp, verified_map, args.empirical_entropy)
                points.append((temp, pr_correct, conditional_entropy))
                print(f'{model} at temp {temp}: Pr[correct]={pr_correct:.4f}, conditional entropy={conditional_entropy:.4f}')
            
            if points:
                models_data_conditional.append((model, points))
    
    # Generate two-panel plot (unconditional and conditional)
    make_plot_two_panel(models_data, models_data_conditional, fm, args.output, args.empirical_entropy)
    
    # Generate combined plot
    # Determine output filename for combined plot
    if args.output_combined:
        combined_output = args.output_combined
    else:
        # Derive from original output filename
        base_output = Path(args.output)
        combined_output = base_output.stem + '_combined' + base_output.suffix
    
    make_plot_combined(models_data, models_data_conditional, fm, combined_output, args.empirical_entropy)

