from file_manager import FileManager
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scat_utils import get_model_list

parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--no_save', '-n', action='store_true', default=False)
parser.add_argument('--verifier', '-v', type=str, default='qwen2.5')
SAVE = True

def make_plots(models: list[str], verifier: str, fm: FileManager, max_temps: dict[str, float]):
    for model in models:
        max_temp = max_temps[model]
        samples = fm.get_all_samples(model=model, max_temp=max_temp)
        grouped = samples.groupby(['letter', 'category'])
        temps = sorted(samples['temperature'].unique())
        pr_correct = np.zeros(len(temps))
        for (letter, category), group in grouped:
            verified = fm.load_verified(letter, category, verifier)
            verified_yes = verified['yes']
            temps_and_paths = group[['temperature', 'fname']].values
            sorted_temps_and_paths = sorted(temps_and_paths, key=lambda x: x[0])
            _, paths = zip(*sorted_temps_and_paths)
            for i, path in enumerate(paths):
                info = fm.load_from_path(path)
                samples = info['dist']
                pr_correct[i] += get_pr_correct(samples, verified_yes)
        pr_correct /= len(grouped)
        plt.plot(temps, pr_correct, label=model)
    plt.xlabel('Temperature')
    plt.ylabel('Probability an answer is valid')
    plt.legend()
    if SAVE:
        fname = fm.locations.plots_dir / 'pr_correct.png'
        print('Saving to', fname)
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def get_pr_correct(samples: dict[str, int], verified_yes: set[str]) -> float:
    correct = 0
    total = 0
    for answer, count in samples.items():
        total += count
        if answer in verified_yes:
            correct += count
    return correct / total

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
    make_plots(models, args.verifier, fm, max_temps)
