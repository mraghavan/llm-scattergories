import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from file_manager import FileManager
from scat_utils import get_model_list
import argparse
from itertools import combinations
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--scores_dir', '-s', type=str, default='./info')
parser.add_argument('--no_save', '-n', action='store_true', default=False)
parser.add_argument('--gamma', '-g', type=float, default=1.0)
SAVE = True

def save_if_needed(fname: str, to_save: bool):
    if to_save:
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_results(results, fm: FileManager):
    ns = sorted(results.keys())
    scores = {}
    counts = {}
    temps = {}
    gamma = -1
    # first plot: scatter eq counts
    models = sorted(list(results[ns[0]][0]['scores'].keys()))
    for n in ns:
        for result in results[n]:
            if not result['converged']:
                continue
            if gamma == -1:
                gamma = result['gamma']
            if n not in counts:
                counts[n] = {model: [] for model in models}
            if n not in scores:
                scores[n] = {model: [] for model in models}
            if n not in temps:
                temps[n] = {model: [] for model in models}
            for model in models:
                counts[n][model].append(result['counts'][model])
                scores[n][model].append(result['scores'][model])
                temps[n][model].append(result['temps'][model])
    if gamma == -1:
        gamma = 1.0
    cmap = mpl.colormaps['copper']
    norm = mcolors.Normalize(vmin=min(ns), vmax=max(ns))
    for n in ns:
        if n not in counts:
            print('No results for n =', n)
            continue
        ls = [counts[n][model] for model in models]
        plt.scatter(ls[0], ls[1], color=cmap(norm(n)))
        plt.plot([min(ls[0]), max(ls[0])], [max(ls[1]), min(ls[1])], ls='--', lw=0.3, color=cmap(norm(n)))
    plt.xlabel(models[0])
    plt.ylabel(models[1])
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Find the min and max of both axes
    # min_val = min(xlim[0], ylim[0])
    min_val = -0.5
    max_val = max(xlim[1], ylim[1])
    plt.title('Market share at equilibrium')

    # Set both axes to the same limits
    plt.axis('square')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    ax = plt.gca()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set an array for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("$n$")
    # equalize axes
    fname_base = fm.locations.plots_dir / ('_'.join(models) + f'_gamma_{gamma}' + '_pairwise_{tag}.png')
    fname = str(fname_base).format(tag='counts_scatter')
    save_if_needed(fname, SAVE)
    markers = {
            models[0]: '^',
            models[1]: 'v',
            }
    colors = {
            models[0]: 'r',
            models[1]: 'b',
            }
    flags = {
            models[0]: False,
            models[1]: False,
            }
    for n in ns:
        for model in models:
            if n not in temps:
                continue
            for i, t in enumerate(temps[n][model]):
                if counts[n][model][i] == 0:
                    continue
                if not flags[model]:
                    plt.scatter(n, t, marker=markers[model], color=colors[model], label=model)
                    flags[model] = True
                else:
                    plt.scatter(n, t, marker=markers[model], color=colors[model])
    plt.xlabel('$n$')
    plt.ylabel('Temperature')
    plt.legend()
    fname = str(fname_base).format(tag='temp')
    save_if_needed(fname, SAVE)

    # do the same thing for scores
    flags = {
            models[0]: False,
            models[1]: False,
            }
    for n in ns:
        for model in models:
            if n not in scores:
                continue
            for i, s in enumerate(scores[n][model]):
                if counts[n][model][i] == 0:
                    continue
                if not flags[model]:
                    plt.scatter(n, s, marker=markers[model], color=colors[model], label=model)
                    flags[model] = True
                else:
                    plt.scatter(n, s, marker=markers[model], color=colors[model])
    plt.xlabel('$n$')
    plt.ylabel('Utility')
    plt.legend()
    fname = str(fname_base).format(tag='utility')
    save_if_needed(fname, SAVE)

def load_eqs(
        model1: str,
        model2: str,
        gamma: float,
        fm: FileManager,
        ):
    results = {}
    df = fm.get_all_pairwise_info(model1=model1, model2=model2, gamma=gamma)
    for _, row in df.iterrows():
        n = row['n']
        results[n] = fm.load_from_path(row['fname']) # type: ignore
    return results

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
    fm = FileManager.from_base('.')
    # get all combinations
    gamma = args.gamma
    for model1, model2 in combinations(models, 2):
        results = load_eqs(model1, model2, gamma, fm)
        plot_results(results, fm)
