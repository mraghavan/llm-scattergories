from scat_utils import get_model_list, MAX_TEMPS
import numpy as np
from analyze_samples import compute_scores
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from file_manager import FileManager
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--scores_dir', '-s', type=str, default='./info')
parser.add_argument('--no_save', '-n', action='store_true', default=False)
SAVE = True

def plot_opt_over_temp(
        model: str,
        v_model: str,
        fm: FileManager,
        max_n: int,
        max_temp: float,
        gamma: float=1.0,
        ):
    samples = fm.get_all_samples(model=model)
    temps = sorted(samples['temperature'].unique().tolist())
    temps = [temp for temp in temps if temp <= max_temp]
    verified_map = {}
    # cmap = cm.get_cmap('cool')
    cmap = mpl.colormaps['copper']
    norm = mcolors.Normalize(vmin=1, vmax=max_n)
    custom_cmap =cmap
    # mcolors.LinearSegmentedColormap.from_list(
        # 'custom_blues', cmap(np.linspace(0.3, 0.9, max_n))
    # )
    _, ax = plt.subplots()
    for n in range(1, max_n+1):
        utilities = np.zeros(len(temps))
        for i, temp in enumerate(temps):
            sample_at_temp = samples[(samples['temperature'] == temp)]
            num_instances = len(sample_at_temp)
            for letter, category in sample_at_temp[['letter', 'category']].values:
                loaded = fm.load_samples(letter, category, model, temp)
                sample_dist = loaded['dist']
                if (letter, category) not in verified_map:
                    verified_map[(letter, category)] = fm.load_verified(letter, category, v_model)['yes']
                score, _ = compute_scores(
                        verified_map[(letter, category)],
                        sample_dist,
                        sample_dist,
                        n,
                        gamma=gamma,
                        )
                utilities[i] += score / num_instances
        ax.plot(temps, utilities, label=f'n={n}', lw=0.7, color=custom_cmap(norm(n-1)))
    sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])  # You need to set an array for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("$n$")
    plt.xlabel('Temperature')
    plt.ylabel('Utility per player')
    plt.title(f'$n$-player utility as a function of temperature ({model})')
    if SAVE:
        fname = fm.locations.plots_dir / f'{model}_opt_over_temp.png'
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    models = get_model_list(args.models, set(MODELS.keys()))
    if args.no_save:
        SAVE = False
    # increase font size for plt
    plt.rcParams.update({'font.size': 12})
    fm = FileManager.from_base('.')
    for model in models:
        max_temp = MAX_TEMPS[MODELS[model]]
        plot_opt_over_temp(model, args.verifier, fm, 15, max_temp)
