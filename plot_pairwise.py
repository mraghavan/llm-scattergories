import matplotlib.pyplot as plt
import os
import pickle
from analyze_pairwise import get_pairwise_fname

def plot_results(results):
    ns = sorted(results.keys())
    scores = {}
    counts = {}
    temps = {}
    all_models = set()
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
    for n in ns:
        if n not in counts:
            print('No results for n =', n)
            continue
        ls = [counts[n][model] for model in models]
        plt.scatter(ls[0], ls[1], color='k')
    plt.xlabel(models[0])
    plt.ylabel(models[1])
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Find the min and max of both axes
    min_val = min(xlim[0], ylim[0])
    max_val = max(xlim[1], ylim[1])
    plt.title('Counts for each model at equilibrium')

    # Set both axes to the same limits
    plt.axis('square')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # equalize axes
    fname_base = 'img/' + '_'.join(models) + f'_gamma_{gamma}' + '_pairwise_{tag}.png'
    fname = fname_base.format(tag='counts_scatter')
    print('Saving to', fname)
    # plt.savefig(fname)
    plt.show()
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
    fname = fname_base.format(tag='temp')
    print('Saving to', fname)
    # plt.savefig(fname)
    plt.show()

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
    fname = fname_base.format(tag='utility')
    print('Saving to', fname)
    # plt.savefig(fname)
    plt.show()
    # for n in ns:
        # if not results[n]['converged']:
            # for model in results[n]['scores']:
                # all_models.add(model)
                # if model not in scores:
                    # scores[model] = []
                # scores[model].append(None)
                # if model not in counts:
                    # counts[model] = []
                # counts[model].append(None)
                # if model not in temps:
                    # temps[model] = []
                # temps[model].append(None)
            # continue
        # for result in results[n]:
            # for model in result['scores']:
                # all_models.add(model)
                # if model not in counts:
                    # counts[model] = []
                # counts[model].append(result['counts'][model]/n)
                # if model not in scores:
                    # scores[model] = []
                # if counts[model][-1] > 0:
                    # scores[model].append(result['scores'][model])
                # else:
                    # scores[model].append(None)
                # if model not in temps:
                    # temps[model] = []
                # if counts[model][-1] > 0:
                    # temps[model].append(result['temps'][model])
                # else:
                    # temps[model].append(None)
    # for model in scores:
        # plt.plot(ns, scores[model], label=model, marker='o')
    # plt.legend()
    # plt.xlabel('n')
    # plt.ylabel('Utility')
    # fname_base = 'img/' + '_'.join(sorted(list(all_models))) + '_pairwise_{tag}.png'
    # print(all_models)
    # fname = fname_base.format(tag='utility')
    # print('Saving to', fname)
    # plt.savefig(fname, dpi=300)
    # plt.clf()
    # for model in counts:
        # plt.plot(ns, counts[model], label=model, marker='o')
    # plt.legend()
    # plt.xlabel('n')
    # plt.ylabel('share of equilibria')
    # fname = fname_base.format(tag='eq_share')
    # print('Saving to', fname)
    # plt.savefig(fname, dpi=300)
    # plt.clf()
    # for model in temps:
        # plt.plot(ns, temps[model], label=model, marker='o')
    # plt.legend()
    # plt.xlabel('n')
    # plt.ylabel('temp')
    # fname = fname_base.format(tag='temp')
    # print('Saving to', fname)
    # plt.savefig(fname, dpi=300)

def load_eqs(
        folder: str,
        model1: str,
        model2: str,
        ns: list[int],
        gamma: float,
        ):
    results = {}
    for n in ns:
        fname = get_pairwise_fname(folder, model1, model2, n, gamma)
        if not os.path.exists(fname):
            print(f'File {fname} does not exist')
            continue
        with open(fname, 'rb') as f:
            results[n] = pickle.load(f)
    return results

if __name__ == '__main__':
    ns = range(1, 21)
    # models_to_compare = ['llama3', 'phi3.5']
    # results = load_eqs('./info', 'llama3', 'phi3.5', ns, 1.0)
    model1 = 'llama3.1'
    model2 = 'nemotron'
    results = load_eqs('./info', model1, model2, ns, 1.0)

    # results = find_eqs_pairwise(models_to_compare, ns, gamma=0.5)
    # print(results)
    plot_results(results)
