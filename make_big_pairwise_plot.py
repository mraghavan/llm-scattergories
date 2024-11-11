import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from file_manager import FileManager
from scat_utils import get_model_list
import argparse
from plot_pairwise import load_eqs

parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--scores_dir', '-s', type=str, default='./info')
parser.add_argument('--no_save', '-n', action='store_true', default=False)
parser.add_argument('--gamma', '-g', type=float, default=1.0)
SAVE = True
LIM = 14.5
SCALE = 1.7

def plot_results(results, ax: plt.Axes):
    ns = sorted(results.keys())
    counts = {}
    gamma = -1
    models = sorted(list(results[ns[0]][0]['scores'].keys()))
    for n in ns:
        for result in results[n]:
            if not result['converged']:
                continue
            if gamma == -1:
                gamma = result['gamma']
            if n not in counts:
                counts[n] = {model: [] for model in models}
            for model in models:
                counts[n][model].append(result['counts'][model])
    if gamma == -1:
        gamma = 1.0
    cmap = mpl.colormaps['copper']
    norm = mcolors.Normalize(vmin=min(ns), vmax=max(ns))
    for n in ns:
        if n not in counts:
            print('No results for n =', n)
            continue
        ls = [counts[n][model] for model in models]
        ax.scatter(ls[0], ls[1], color=cmap(norm(n)), s=10)
        ax.plot([min(ls[0]), max(ls[0])], [max(ls[1]), min(ls[1])], ls='--', lw=0.3, color=cmap(norm(n)))
    xlim = LIM
    ylim = LIM

    min_val = -0.5
    max_val = max(xlim, ylim)

    # Set both axes to the same limits
    ax.axis('square')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    if args.no_save:
        SAVE = False
    plt.rcParams.update({'font.size': 11})
    models = get_model_list(args.models, set(MODELS.keys()))
    dimension = len(models) - 1
    fm = FileManager.from_base('.')
    rows = models[:-1]
    columns = models[1:]
    fig, axs = plt.subplots(dimension, dimension, figsize=(SCALE*dimension, SCALE*dimension))
    gamma = args.gamma
    for i, model1 in enumerate(rows):
        for j, model2 in enumerate(columns[i:]):
            print(model1, model2)
            results = load_eqs(model1, model2, gamma, fm)
            ax = axs[i+j,i]
            if i == 0:
                ax.set_ylabel(model2)
            if i+j == dimension - 1:
                ax.set_xlabel(model1)
            plot_results(results, ax)
    for i in range(dimension):
        for j in range(dimension):
            if j > i:
                axs[i, j].axis('off')
            if j != 0:
                axs[i, j].set_yticklabels([])
            if i != dimension - 1:
                axs[i, j].set_xticklabels([])
    
    ax = plt.gca()
    cmap = mpl.colormaps['copper']
    ns = range(1, 16)
    norm = mcolors.Normalize(vmin=min(ns), vmax=max(ns))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set an array for ScalarMappable
    cbar = fig.colorbar(sm, ax=axs, fraction=0.0175)
    cbar.set_label("$n$")
    fig.suptitle('Market share at equilibrium')
    plt.tight_layout()
    if SAVE:
        fname = fm.locations.plots_dir / 'all_market_share_pairwise.png'
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
    else:
        plt.show()
