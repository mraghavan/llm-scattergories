import matplotlib.patches as mpatches
from file_manager import FileManager
from pathlib import Path
from scat_utils import get_model_list
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
PLOT_OPTS = {
        'lw': .2,
        'ls': ':',
        }

def load_scores(fm: FileManager, models: list[str]):
    df = fm.get_all_info(models=models)
    scores = []
    for i, row in df.iterrows():
        scores.append(fm.load_from_path(row['fname']))
    return scores

def load_scores_into_df(fm: FileManager, models: list[str]):
    scores = load_scores(fm, models)
    df = pd.DataFrame(scores)
    return df


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
            text.set_position((-70, -10))  # Move the text up
        else:
            text.set_position((10, 0))  # Add some padding to other labels

def plot_util_over_gamma(
        all_scores: pd.DataFrame,
        fm: FileManager,
        n=3,
        ):
    relevant_df = all_scores[all_scores['n'] == n]
    plot_helper(relevant_df, 'gamma', 'nash_eq_util', 'opt_util')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Utility per player')
    plt.title(rf'OPT and EQ utility per player over $\gamma$ for $n={n}$')
    if SAVE:
        fname = fm.locations.plots_dir / 'opt_and_eq_util_over_gamma.png'
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_temp_over_gamma(
        all_scores: pd.DataFrame,
        fm: FileManager,
        n=3,
        ):
    relevant_df = all_scores[all_scores['n'] == n]
    plot_helper(relevant_df, 'gamma', 'nash_eq', 'opt')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Temperature')
    plt.title(rf'OPT and EQ temperature over $\gamma$ for $n={n}$')
    if SAVE:
        fname = fm.locations.plots_dir / 'opt_and_eq_temp_over_gamma.png'
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_sw_over_gamma(
        all_scores: pd.DataFrame,
        fm: FileManager,
        n=3,
        ):
    relevant_df = all_scores[all_scores['n'] == n]
    plot_helper(relevant_df, 'gamma', 'nash_eq_sw', 'opt_sw')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('Social welfare per player')
    plt.title(rf'OPT and EQ social welfare per player over $\gamma$ for $n={n}$')
    if SAVE:
        fname = fm.locations.plots_dir / 'opt_and_eq_sw_over_gamma.png'
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_util_over_n(
        all_scores: pd.DataFrame,
        fm: FileManager,
        gamma=1.0,
        ):
    relevant_df = all_scores[all_scores['gamma'] == gamma]
    plot_helper(relevant_df, 'n', 'nash_eq_util', 'opt_util')
    plt.xlabel(r'$n$')
    plt.ylabel('Utility per player')
    plt.title(rf'OPT and EQ utility per player over $n$ for $\gamma={gamma}$')
    if SAVE:
        fname = fm.locations.plots_dir / 'opt_and_eq_util_over_n.png'
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_temp_over_n(
        all_scores: pd.DataFrame,
        fm: FileManager,
        gamma=1.0,
        ):
    relevant_df = all_scores[all_scores['gamma'] == gamma]
    plot_helper(relevant_df, 'n', 'nash_eq', 'opt')
    plt.xlabel(r'$n$')
    plt.ylabel('Temperature')
    plt.title(rf'OPT and EQ temperature over $n$ for $\gamma={gamma}$')
    if SAVE:
        fname = fm.locations.plots_dir / 'opt_and_eq_temp_over_n.png'
        print(f'Saving to {fname}')
        plt.savefig(fname, dpi=300)
        plt.clf()
    else:
        plt.show()

def plot_helper(
        df: pd.DataFrame,
        axis: str,
        eq_metric: str,
        opt_metric: str,
        ):
    models = sorted(df['model'].unique())
    relevant_df = df.sort_values(by=axis)
    handles = []
    for model in models:
        model_df = relevant_df[relevant_df['model'] == model]
        handles.append(plt.plot(
            model_df[axis],
            model_df[eq_metric],
            label=model,
            marker=EQ_MARKER,
            **PLOT_OPTS)[0])
    plt.gca().set_prop_cycle(None)
    for model in models:
        model_df = relevant_df[relevant_df['model'] == model]
        handles.append(plt.plot(
            model_df[axis],
            model_df[opt_metric],
            label=model,
            marker=OPT_MARKER,
            **PLOT_OPTS)[0])
    make_legend(handles)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    models = get_model_list(args.models, set(MODELS.keys()))

    fm = FileManager.from_args(info_dir=args.scores_dir)
    all_scores = load_scores_into_df(fm, models)
    if args.no_save:
        SAVE = False
    plt.rcParams.update({'font.size': 11})
    plot_util_over_gamma(all_scores, fm)
    plot_temp_over_gamma(all_scores, fm)
    plot_sw_over_gamma(all_scores, fm)
    plot_util_over_n(all_scores, fm)
    plot_temp_over_n(all_scores, fm)
