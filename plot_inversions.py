from file_manager import FileManager
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from scat_utils import get_model_list
import argparse
from l1_isotonic import isotonic_regression_l1_total_order
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--no_save', '-n', action='store_true', default=False)
SAVE = True

class LinearInterpolater:
    def __init__(self, temps, vals):
        self.temps = temps
        self.vals = vals

    def __repr__(self):
        return f'LinearInterpolater({self.temps}, {self.vals})'

    def __call__(self, temp):
        if temp in self.temps:
            return self.vals[self.temps.index(temp)]
        elif temp < self.temps[0]:
            return np.nan
        elif temp > self.temps[-1]:
            return np.nan
        else:
            i = 0
            while self.temps[i] < temp:
                i += 1
            # linear interpolation
            return (self.vals[i] - self.vals[i-1]) * (temp - self.temps[i-1]) / (self.temps[i] - self.temps[i-1]) + self.vals[i-1]

def get_ranking(dist: dict[str, float]):
    key_vals = list(dist.items())
    key_vals.sort(key=lambda x: x[1], reverse=True)
    # separate keys and vals
    keys, vals = zip(*key_vals)
    return keys, vals

def reorder_keys(
        ranked_keys: list[str],
        ranked_vals: list[float],
        other_ranked_keys: list[str]):
    # reorder ranked_keys with the same vals according to other_ranked_keys
    equivalence_classes = find_equivalance_classes(ranked_keys, ranked_vals)
    rk_copy = ranked_keys.copy()
    for eq_class in equivalence_classes:
        if len(eq_class) == 1:
            continue
        keys_to_reorder = [rk_copy[i] for i in eq_class]
        keys_to_reorder.sort(key=lambda x: other_ranked_keys.index(x))
        rk_copy[eq_class[0]:eq_class[-1]+1] = keys_to_reorder
    return rk_copy

def find_equivalance_classes(ranked_keys: list[str], ranked_vals: list[float]):
    # find the equivalence classes
    classes = []
    current_class = []
    for i in range(len(ranked_keys) - 1):
        current_class.append(i)
        if ranked_vals[i] == ranked_vals[i+1]:
            continue
        else:
            classes.append(current_class)
            current_class = []
    current_class.append(len(ranked_keys) - 1)
    classes.append(current_class)
    return classes

def total_variation_distance(l1: list[float] | np.ndarray, l2: list[float] | np.ndarray):
    return sum(abs(l1[i] - l2[i]) for i in range(len(l1)))/2

def get_inversions(interpolators: dict[str, LinearInterpolater], temps: list[float]):
    weighted_inversions = []
    for t1, t2 in zip(temps[:-1], temps[1:]):
        dist1 = {key: li(t1) for key, li in interpolators.items() if not np.isnan(li(t1))}
        dist2 = {key: li(t2) for key, li in interpolators.items() if not np.isnan(li(t2))}
        intersection = set(dist1.keys()) & set(dist2.keys())
        dist1 = {key: dist1[key] for key in intersection}
        dist2 = {key: dist2[key] for key in intersection}
        keys1, vals1 = get_ranking(dist1)
        keys2, _ = get_ranking(dist2)
        keys1 = list(keys1)
        keys1 = reorder_keys(keys1, vals1, keys2)
        # sort vals2 according to keys1
        new_vals2 = []
        for key in keys1:
            new_vals2.append(dist2[key])
        for key in dist2:
            if key not in keys1:
                print("Shound't happen")
                new_vals2.append(dist2[key])
        if total_variation_distance(vals1, new_vals2) == 0:
            weighted_inversions.append(0.0)
        else:
            ordered = np.abs(isotonic_regression_l1_total_order(-np.array(new_vals2), np.ones(len(new_vals2))))
            weighted_inversions.append(total_variation_distance(new_vals2, ordered) / total_variation_distance(vals1, new_vals2))
            assert weighted_inversions[-1] <= 1
    return weighted_inversions

def get_num_inversions(l: list[float]):
    n = len(l)
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if l[i] < l[j]:
                inversions += 1
    return inversions

def make_plots(models: list[str], fm: FileManager, max_temps: dict[str, float]):
    all_mass_captured = {}
    for model in models:
        max_temp = max_temps[model]
        samples = fm.get_all_samples(model=model, max_temp=max_temp)
        grouped = samples.groupby(['letter', 'category'])
        temps = sorted(samples['temperature'].unique())
        weighted_inversions = np.zeros(len(temps) - 1)
        all_mass_captured[model] = np.zeros(len(temps))
        for (letter, category), group in grouped:
            temps_and_paths = group[['temperature', 'fname']].values
            sorted_temps_and_paths = sorted(temps_and_paths, key=lambda x: x[0])
            _, paths = zip(*sorted_temps_and_paths)
            all_samples = []
            for i, path in enumerate(paths):
                info = fm.load_from_path(path)
                all_samples.append(info['probs'])
                all_mass_captured[model][i] += sum(info['probs'].values()) / len(grouped)
            responses = Counter()
            for sample in all_samples:
                for key in sample:
                    responses[key] += 1
            more_than_once = {key for key, value in responses.items() if value > 1}
            interpolators = {}
            for key in more_than_once:
                temps_to_plot = []
                vals_to_plot = []
                for temp, sample in zip(temps, all_samples):
                    if key in sample:
                        temps_to_plot.append(temp)
                        vals_to_plot.append(sample[key])
                li = LinearInterpolater(temps_to_plot, vals_to_plot)
                interpolators[key] = li
            weighted_inversions += np.array(get_inversions(interpolators, temps))
        weighted_inversions /= len(grouped)
        plt.plot(temps[:-1], weighted_inversions, label=model)
    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel(r'$\widehat{WI}(\hat{\mathbf{p}}^{(\tau)} \| \hat{\mathbf{p}}^{(\tau + 0.05)})$')
    if SAVE:
        fname = fm.locations.plots_dir / 'weighted_inversions.png'
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()
    for model in models:
        max_temp = MAX_TEMPS[MODELS[model]]
        samples = fm.get_all_samples(model=model, max_temp=max_temp)
        temps = sorted(samples['temperature'].unique())
        plt.plot(temps, all_mass_captured[model], label=model)
    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel('Probability mass captured')
    if SAVE:
        fname = fm.locations.plots_dir / 'mass_captured.png'
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

if __name__ == '__main__':
    fm = FileManager.from_base('.')
    args = parser.parse_args()
    plt.rcParams.update({'font.size': 11})
    if args.no_save:
        SAVE = False
    from completion_hf import MODELS
    models = get_model_list(args.models, set(MODELS.keys()))
    from scat_utils import MAX_TEMPS
    max_temps = {model: MAX_TEMPS[MODELS[model]] for model in models}
    make_plots(models, fm, max_temps)
