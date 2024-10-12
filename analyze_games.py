import os
import time
from functools import lru_cache
import numpy as np
import pickle
from generate_trees import parse_pickle_filename, get_v_filename, MAX_TEMPS, get_model_list
from collections.abc import Callable
from completion_base import CompletionNode
from math import comb
from itertools import product
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./scores')
parser.add_argument('--tree_dir', '-r', type=str, default='./trees')
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--total_jobs', '-t', type=int, default=1)

GRID_EPS = 0.02

def get_score_fname(output_dir: str, model: str, n: int, gamma: float):
    return f'{output_dir}/{model}_n{n}_gamma{gamma}.pkl'

def get_score(tree: CompletionNode,
              verified_y: set,
              temperature: float,
              n: int,
              gamma: float=1.0,
              ):
    dist = tree.get_dist(temperature)
    verified_dist = {k: v for k, v in dist.items() if k in verified_y}
    if len(verified_dist) == 0:
        return 0
    if n == 1:
        return sum(verified_dist.values())
    score = 0.0
    # slower but more numerically stable
    for k in verified_dist:
        if verified_dist[k] > 0:
            score += verified_dist[k] * binomial_sum(n-1, verified_dist[k], lambda i: 1/(1 + i)**gamma)
    return score
    # return 1/n * (len(verified_dist) - sum((1-p)**n for p in verified_dist.values()))

def binomial_sum_gamma_one(n, p):
    # only works for lambda i: 1/(1+i)
    candidate1 = (1 - (1-p)**(n+1)) / np.float64((n+1)*p)
    candidate2 = 1 - n*p/2
    return max(candidate1, candidate2)


def binomial_sum(n: int, p: float, f: Callable[[int], float]):
    s = 0
    for i in range(n+1):
        s += comb(n, i) * p**i * (1-p)**(n-i) * f(i)
    return s

def binomial_sum_vec(n: int, p: float, fs: np.ndarray):
    s = 0
    for i in range(n+1):
        s += comb(n, i) * p**i * (1-p)**(n-i) * fs[i]
    return s

@lru_cache
def get_comb_array(n: int):
    return np.array([comb(n, i) for i in range(n+1)])

def faster_binomial_sum_vector(n: int, p: float, fs: np.ndarray):
    s = np.sum(get_comb_array(n) * p**np.arange(n+1) * (1-p)**np.arange(n, -1, -1) * fs)
    return s

@lru_cache
def get_f_vector(n: int, gamma: float):
    return np.array([1/(1+i)**gamma for i in range(n)])

def get_score_two_temps_from_dists(dist1: dict,
                                   dist2: dict,
                                   verified_y: set,
                                   n: int,
                                   gamma: float=1.0,
                                   ):
    assert n > 1
    verified_dist1 = {k: v for k, v in dist1.items() if k in verified_y}
    verified_dist2 = {k: v for k, v in dist2.items() if k in verified_y}
    if len(verified_dist1) == 0:
        return 0
    if len(verified_dist2) == 0:
        return sum(verified_dist1.values())
    total = 0.0
    for k in verified_dist1:
        if k not in verified_dist2 or verified_dist2[k] == 0:
            total += verified_dist1[k]
        else:
            if gamma == 1.0:
                total += verified_dist1[k] * binomial_sum_gamma_one(n-1, verified_dist2[k])
            elif n > 10:
                total += verified_dist1[k] * faster_binomial_sum_vector(n-1, verified_dist2[k], get_f_vector(n, gamma)) # seems to be better for n > 10
            else:
                total += verified_dist1[k] * binomial_sum_vec(n-1, verified_dist2[k], get_f_vector(n, gamma))
                # total += verified_dist1[k] * binomial_sum(n-1, verified_dist2[k], lambda i: 1/(1 + i)**gamma)

            # Use a bound from the Taylor expansion to deal with numeric instability
            # candidate1 = (1 - (1-verified_dist2[k])**n) / np.float64(n*verified_dist2[k])
            # candidate2 = 1 - (n-1)*verified_dist2[k]/2
            # total += verified_dist1[k] * max(candidate1, candidate2)
    return total

def get_all_jobs(models: list[str], ns: list[int], gammas: list[float]):
    return list(product(models, ns, gammas))

def generate_score_data(
        trees: list[CompletionNode],
        verified_ys: list[set[str]],
        n: int,
        gamma: float=1.0,
        info: dict = {},
        ):
    max_temperature = min(tree.max_temperature for tree in trees)
    temps = np.arange(0, max_temperature, GRID_EPS)
    num_games = len(trees)
    if n == 1:
        scores = np.zeros_like(temps)
        for tree, verified_y in zip(trees, verified_ys):
            for i, temp in enumerate(temps):
                scores[i] += 1/num_games * get_score(tree, verified_y, temp, n, gamma)
        info['nash_eq'] = temps[np.argmax(scores)]
        info['nash_eq_util'] = np.max(scores)
        info['opt'] = temps[np.argmax(scores)]
        info['opt_util'] = np.max(scores)
    else:
        temp_grid1, temp_grid2 = np.meshgrid(temps, temps)
        scores = np.zeros_like(temp_grid1)
        for tree, verified_y in zip(trees, verified_ys):
            dists = {}
            for temp in temps:
                dists[temp] = tree.get_dist(temp)
            for i, temp1 in enumerate(temps):
                for j, temp2 in enumerate(temps):
                    # i is this player, j is the n-1 other players
                    scores[i, j] += 1/num_games * get_score_two_temps_from_dists(dists[temp1], dists[temp2], verified_y, n, gamma)
        # for j in range(len(temps)):
            # if not check_is_quasiconcave(scores[:, j]):
                # print('Warning: quasiconcavity not satisfied at temperature', temps[j])
        eq_inds = []
        # best response to each other temperature
        max_indices = np.argmax(scores, axis=0)
        for i, ind in enumerate(max_indices):
            if i == ind:
                print(f'n={n}', 'Exact Nash equilibrium: ', temps[i], 'Avg. welfare', scores[i, ind])
                eq_inds.append(i)
        if len(eq_inds) > 0:
            eq_start = eq_inds[0]
            eq_end = eq_inds[-1]
            info['nash_eq_exact'] = True
            if eq_end - eq_start + 1 != len(eq_inds):
                info['nash_eq_contiguous'] = False
            else:
                info['nash_eq_contiguous'] = True
        else:
            # moving on to aproximate equilibria
            for i, ind in enumerate(max_indices):
                # print(f'Temperature {temps[i]:.2f} best response to {temps[ind]:.2f}')
                if abs(i - ind) <= 1:
                    print(f'n={n}', 'Nash equilibrium: ', temps[i], 'Avg. welfare', scores[i, ind])
                    eq_inds.append(i)
            # make sure the eq inds are contiguous
            if len(eq_inds) > 0:
                info['nash_eq_exact'] = False
                eq_start = eq_inds[0]
                eq_end = eq_inds[-1]
                if eq_end - eq_start + 1 != len(eq_inds):
                    print('Warning: Nash equilibria not contiguous')
                    info['nash_eq_contiguous'] = False
                else:
                    info['nash_eq_contiguous'] = True
        if len(eq_inds) == 0:
            info['nash_eq'] = np.NaN
            info['nash_eq_util'] = np.NaN
        else:
            info['nash_eq'] = np.mean([temps[i] for i in eq_inds])
            info['nash_eq_util'] = np.mean([scores[i, i] for i in eq_inds])
        symmetric = [scores[i, i] for i in range(len(temps))]
        info['opt'] = temps[np.argmax(symmetric)]
        info['opt_util'] = np.max(symmetric)
    print(info)

    info['scores'] = scores
    info['temperatures'] = temps

def load_games_for_model(model: str, tree_dir: str, temp: float):
    candidates = os.listdir(tree_dir)
    game_map = {}
    for c in candidates:
        parsed = parse_pickle_filename(c)
        if parsed is None:
            continue
        letter, category, max_temperature, model_name = parsed
        if model_name != model:
            continue
        if abs(max_temperature - temp) > 1e-6:
            continue
        with open(tree_dir + '/' + c, 'rb') as f:
            game_map[(letter, category)] = pickle.load(f)
    return game_map


if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    print('[LOG ARGS]', args)
    models = get_model_list(args.models, set(MODELS.keys()))

    output_dir = args.output_dir
    tree_dir = args.tree_dir

    gammas = [0.2, 0.5, 1.0, 2.0, 5.0]
    ns = [1, 2, 3, 5, 10, 20, 35]
    all_jobs = get_all_jobs(models, ns, gammas)
    print('All jobs:', all_jobs)
    filtered_jobs = [(model, n, gamma) for model, n, gamma in all_jobs if not os.path.exists(get_score_fname(output_dir, model, n, gamma))]
    print(filtered_jobs)
    print(len(filtered_jobs))
    job_num = args.job_num
    total_jobs = args.total_jobs
    my_jobs = filtered_jobs[job_num::total_jobs]
    print(f'Job {job_num+1} of {total_jobs}: {my_jobs}')
    print(len(my_jobs))
    loaded_models = {}
    loaded_verified = {}
    for model, n, gamma in my_jobs:
        fname = get_score_fname(output_dir, model, n, gamma)
        if os.path.exists(fname):
            # This shouldn't happen
            print(f'{fname} already exists')
            continue
        max_temp = MAX_TEMPS[MODELS[model]]
        if model not in loaded_models:
            loaded_models[model] = load_games_for_model(model, tree_dir, max_temp)
        for (letter, category) in loaded_models[model]:
            if (letter, category) not in loaded_verified:
                with open(get_v_filename(tree_dir, letter, category, args.verifier), 'rb') as f:
                    loaded_verified[(letter, category)] = pickle.load(f)
        letter_category_pairs = sorted(loaded_models[model].keys())
        info = {'model': model, 'n': n, 'gamma': gamma, 'games': letter_category_pairs, 'max_temperature': max_temp}
        game_trees = [loaded_models[model][(letter, category)] for letter, category in letter_category_pairs]
        game_verified = [loaded_verified[(letter, category)]['yes'] for letter, category in letter_category_pairs]
        start = time.time()
        generate_score_data(game_trees, game_verified, n, gamma, info)
        elapsed = time.time() - start
        print(f'[LOG TIME]: Elapsed time: {elapsed:.2f}s')
        with open(fname, 'wb') as f:
            print(f'Saving scores to {fname}')
            pickle.dump(info, f)
