# from analyze_games import get_score_fname, generate_score_data
import matplotlib.patches as mpatches
from generate_trees import get_model_list
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--scores_dir', '-s', type=str, default='./info')
parser.add_argument('--no_save', '-n', action='store_true', default=False)
SAVE = True

EQ_MARKER = '2'
OPT_MARKER = '1'

def load_scores(scores_dir: str, models: list[str]):
    scores = []
    for fname in os.listdir(scores_dir):
        if fname.endswith('_info.pkl'):
            with open(f'{scores_dir}/{fname}', 'rb') as f:
                new_scores = pickle.load(f)
            model = new_scores['model']
            if model not in models:
                continue
            scores.append(new_scores)
    return scores

def load_scores_into_df(scores_dir: str, models: list[str]):
    scores = load_scores(scores_dir, models)
    df = pd.DataFrame(scores)
    return df

def plot_symmetric_utility(
        scores: pd.DataFrame,
        model: str,
        ):
    ns = sorted(list(set(scores['n'].unique())))
    gammas = sorted(list(set(scores['gamma'].unique())))
    for gamma in gammas:
        utilities = []
        for n in ns:
            df = scores[(scores['n'] == n) & (scores['gamma'] == gamma)]
            if len(df) == 0:
                continue
            temperatures = df['temperatures'].iloc[0]
            # if tag not in scores:
                # print('Warning: tag not found:', tag)
                # continue
            score_array = df['scores'].iloc[0]
            if n == 1:
                utilities = score_array
            else:
                utilities = [score_array[i,i] for i in range(score_array.shape[0])]
            plt.plot(temperatures, utilities, label=f'$n={n}$')
            max_index = np.argmax(utilities)
            plt.scatter(temperatures[max_index], utilities[max_index], color=plt.gca().lines[-1].get_color(), marker='^')
        plt.xlabel('Temperature')
        plt.ylabel('Utility for symmetric play')
        plt.legend()
        plt.title(fr'{model} symmetric utility; $\gamma = {gamma}$')
        plt.savefig(f'img/{model}_symmetric_utility_gamma{gamma}.png', dpi=300)
        plt.clf()

        
def plot_opt_and_eq_over_gamma(
        scores: pd.DataFrame,
        model: str,
        ):
    ns = sorted(list(set(scores['n'].unique())))
    gammas = sorted(list(set(scores['gamma'].unique())))
    for n in ns:
        eqs = []
        opts = []
        for gamma in gammas:
            tag = (model, n, gamma)
            if tag not in scores:
                print('Warning: tag not found:', tag)
                continue
            eqs.append(scores[tag]['nash_eq'])
            opts.append(scores[tag]['opt'])
        plt.plot(gammas, eqs, label=fr'eq; $n={n}$')
        plt.plot(gammas, opts, label=fr'opt; $n={n}$', linestyle='--', color=plt.gca().lines[-1].get_color())
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Temperature')
    plt.legend()
    # plt.legend(bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.title(f'{model} optimal and equilibrium temperatures')
    plt.savefig(f'img/{model}_opt_and_eq_over_gamma.png', dpi=300)
    plt.clf()

def plot_opt_and_eq_over_n(
        scores: dict[tuple[str, int, float], dict],
        model: str,
        ):
    ns = sorted(list(set(tag[1] for tag in scores.keys())))
    gammas = sorted(list(set(tag[2] for tag in scores.keys())))
    for gamma in gammas:
        eqs = []
        opts = []
        for n in ns:
            tag = (model, n, gamma)
            if tag not in scores:
                print('Warning: tag not found:', tag)
                continue
            eqs.append(scores[tag]['nash_eq'])
            opts.append(scores[tag]['opt'])
        plt.plot(ns, eqs, label=fr'eq; $\gamma = {gamma}$')
        plt.plot(ns, opts, label=fr'opt; $\gamma = {gamma}$', linestyle='--', color=plt.gca().lines[-1].get_color())
    plt.xlabel('n')
    plt.ylabel('Temperature')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title(f'{model} optimal and equilibrium temperatures')
    plt.tight_layout()
    plt.savefig(f'img/{model}_opt_and_eq.png', dpi=300)
    plt.clf()

# def plot_sw_and_nash_welfare_over_axis_helper(all_scores, metric, axis='n', ls='-'):
    # for model in sorted(all_scores['model'].unique()):
        # model_scores = all_scores[(all_scores['model'] == model) & (all_scores['gamma'] == gamma)]
        # ns = sorted(model_scores['n'].unique())
        # print(model, ns)
        # opts = []
        # for n in ns:
            # if n > 10:
                # opts.append(None)
                # continue
            # score = model_scores[model_scores['n'] == n]
            # # print(model, n, gamma)
            # # print(model_scores)
            # if score[metric].iloc[0] == max(score['temperatures'].iloc[0]):
                # opts.append(None)
            # else:
                # opts.append(score[metric + '_util'].iloc[0])
        # plt.plot(ns, opts, label=model)
    # plt.xlabel(r'$n$')
    # plt.ylabel(f'Social welfare at {metric}')
    # plt.title(rf'$\gamma = {gamma}$')
    # plt.legend()
    # fname = f'img/sw_over_n_{metric}.png'
    # print(f'Saving to {fname}')
    # plt.savefig(fname, dpi=300)
    # plt.clf()

# def plot_temp_over_axis(all_scores: pd.DataFrame, metric='opt', axis='n'):
    # plot_temp_over_axis_helper(all_scores, metric, axis)
    # plt.xlabel(axis)
    # plt.ylabel('Temperature')
    # plt.title(f'{metric} over {axis}, {other[0]}={other[1]}')
    # plt.legend()
    # fname = f'img/{metric}_temp_over_{axis}_{other[0]}{other[1]}.png'
    # print(f'Saving to {fname}')
    # plt.savefig(fname, dpi=300)
    # plt.clf()

def plot_temp_over_axis_helper(all_scores: pd.DataFrame, metric='opt', axis='n', marker='.'):
    handles = []
    if axis == 'n':
        other = ('gamma', 1.0)
    else:
        other = ('n', 3)
    for model in sorted(all_scores['model'].unique()):
        maxed_out = []
        model_scores = all_scores[all_scores['model'] == model]
        axis_vals = sorted(model_scores[axis].unique())
        # if axis == 'n':
            # axis_vals = [val for val in axis_vals if val <= 10]
        temps = []
        for val in axis_vals:
            df = model_scores[(model_scores[axis] == val) & (model_scores[other[0]] == other[1])]
            if len(df) == 0:
                temps.append(None)
                continue
            t = df[metric].iloc[0]
            if np.isnan(t):
                temps.append(None)
                print(f'Nan received for {model} {metric} {val}')
                continue
            if metric.endswith('_util'):
                check = df[metric.replace('_util', '')].iloc[0]
            else:
                check = t
            if check == max(df['temperatures'].iloc[0]):
                temps.append(None)
                maxed_out.append((val, t))
            else:
                temps.append(df[metric].iloc[0])
        handles.append(plt.plot(axis_vals, temps, label=model, marker=marker, ls=':', lw=.2)[0])
        if len(maxed_out) > 0:
            print(f'Model {model} maxed out: {maxed_out}')
            plt.scatter([x[0] for x in maxed_out], [x[1] for x in maxed_out], color=plt.gca().lines[-1].get_color(), marker='X')
    return handles

def make_legend(handles: list[plt.Line2D]):
    new_handles = []
    for i, handle in enumerate(handles):
        new_handle = plt.Line2D([0], [0], color=handle.get_color(), label=handle.get_label(), marker=handle.get_marker(), linestyle='None')
        if i < len(handles) // 2:
            new_handle.set_label('')
        new_handles.append(new_handle)
    len_handles = len(new_handles)
    opt = mpatches.Patch(color='None', label='OPT', linestyle='None')
    eq = mpatches.Patch(color='None', label='EQ', linestyle='None')
    # opt = plt.Line2D([0], [0], color='black', label='OPT', linestyle='None')
    # eq = plt.Line2D([0], [0], color='black', label='EQ', linestyle='None')
    new_handles = [eq] + new_handles[:len(new_handles) // 2] + [opt] + new_handles[len(new_handles) // 2:]
    legend = plt.legend(handles=new_handles, ncol=2, columnspacing=-1.5, handletextpad=0.0)
    for i, text in enumerate(legend.get_texts()):
        if i == 0 or i == len_handles//2 + 1:  # Title patches
            text.set_position((-54, -10))  # Move the text up
        else:
            text.set_position((10, 0))  # Add some padding to other labels

def plot_sw_and_nash_temp_over_axis(
        all_scores: pd.DataFrame,
        axis='n',
        ):
    handles = plot_temp_over_axis_helper(all_scores, metric='nash_eq', axis=axis, marker=EQ_MARKER)
    # reset color cycle
    plt.gca().set_prop_cycle(None)
    handles += plot_temp_over_axis_helper(all_scores, metric='opt', axis=axis, marker=OPT_MARKER)
    make_legend(handles)
    # plt.legend(handles=new_handles, ncol=2, columnspacing=-1.5)
    plt.xlabel(axis)
    plt.ylabel('Temperature')
    plt.title(f'Optimal and Nash equilibrium temperatures over {axis}')
    fname = f'img/opt_and_eq_temp_over_{axis}.png'
    if SAVE:
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_sw_and_nash_welfare_over_axis(
        all_scores: pd.DataFrame,
        axis='n',
        ):
    handles = plot_temp_over_axis_helper(all_scores, metric='nash_eq_util', axis=axis, marker=EQ_MARKER)
    # reset color cycle
    plt.gca().set_prop_cycle(None)
    handles += plot_temp_over_axis_helper(all_scores, metric='opt_util', axis=axis, marker=OPT_MARKER)
    make_legend(handles)
    if axis == 'n':
        plt.xlabel('$n$')
    else:
        plt.xlabel(axis)
    plt.ylabel('Utility')
    plt.title(f'Optimal and Nash equilibrium utility over {axis}')
    fname = f'img/opt_and_eq_sw_over_{axis}.png'
    if SAVE:
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()
    plt.clf()

def plot_surface(temps: np.ndarray, scores: np.ndarray, nash_eq: float, nash_eq_util: float):
    temp_grid1, temp_grid2 = np.meshgrid(temps, temps)
    max_indices = np.argmax(scores, axis=0)

    # plt.plot(temps, scores[:, 0])
    # plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(temp_grid1, temp_grid2, scores)
    ax.set_xlabel('Temperature 2')
    ax.set_ylabel('Temperature 1')
    ax.set_zlabel('Utility')
    if nash_eq is not np.NaN:
        ax.scatter(nash_eq, nash_eq, nash_eq_util, color='red', marker='^')
    for i, temp in enumerate(temps):
        ax.scatter(temp, temps[max_indices[i]], scores[max_indices[i], i], color='green', marker='o')
    for i, temp in enumerate(temps):
        ax.scatter(temp, temp, scores[i, i], color='blue', marker='x')
    plt.title('Best response surface')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    models = get_model_list(args.models, set(MODELS.keys()))

    all_scores = load_scores_into_df(args.scores_dir, models)
    if args.no_save:
        SAVE = False
    print(all_scores.columns)
    plot_sw_and_nash_temp_over_axis(all_scores, 'n')
    plot_sw_and_nash_welfare_over_axis(all_scores, 'n')
    plot_sw_and_nash_temp_over_axis(all_scores, 'gamma')
    plot_sw_and_nash_welfare_over_axis(all_scores, 'gamma')
    # TODO
    # - plot quality into the tail:
    #  - take the ranking as temp goes to 0
    #  - plot response quality as a function of fractional rank
    #  - smooth over all games
    # - something about the good answers only

    # phi = all_scores[all_scores['model'] == 'phi3.5']
    # print(phi['opt'])
    # print(phi['nash_eq'])

    # llama3_2 = all_scores[all_scores['model'] == 'llama3.2']
    # for i, row in llama3_2.iterrows():
        # if row['n'] == 1:
            # continue
        # if np.isnan(row['nash_eq']):
            # print('No Nash equilibrium')
        # else:
            # print('Nash eq:', row['nash_eq'])
        # plot_surface(row['temperatures'], row['scores'], row['nash_eq'], row['nash_eq_util'])



    # qwen = all_scores[all_scores['model'] == 'qwen2']
    # print(qwen)
    # ns = sorted(qwen['n'].unique())
    # print(ns)
    # for n in ns:
        # if n == 1:
            # continue
        # row = qwen[qwen['n'] == n].iloc[0]
        # # print(row['nash_eq_util'])
        # # print(row['opt'])
        # # print(row['opt_util'])
        # i = np.argmax([row['scores'][i][i] for i in range(len(row['temperatures']))])
        # print(i)
        # print(row['scores'][i][i])
        # plt.plot(row['temperatures'], [row['scores'][i][i] for i in range(len(row['temperatures']))], label=f'n={n}')
    # plt.legend()
    # plt.show()
    # llama3_2 = all_scores[all_scores['model'] == 'smollm']
    # print(llama3_2['nash_eq'])
    # i = 2
    # row = llama3_2.iloc[i]
    # plot_surface(row['temperatures'], row['scores'], row['nash_eq'], row['nash_eq_util'])

    # NOTES
    # qwen doesn't always have a nash equilibrium


    # plot_temp_over_axis(all_scores, 'opt', 'n')
    # plot_temp_over_axis(all_scores, 'nash_eq', 'n')
    # plot_temp_over_axis(all_scores, 'opt', 'gamma')
    # plot_temp_over_axis(all_scores, 'nash_eq', 'gamma')
    # print(all_scores.keys())
    # all_models = set(tag[0] for tag in all_scores.keys())
    # ns = [2, 5, 10, 20, 50]

    # for model in models:
        # print(model)
        # model_scores = {tag: score for tag, score in all_scores.items() if tag[0] == model}
        # plot_opt_and_eq_over_n(all_scores[all_scores['model'] == model], model)
        # plot_opt_and_eq_over_gamma(all_scores[all_scores['model'] == model], model)
        # plot_symmetric_utility(all_scores[all_scores['model'] == model], model)
