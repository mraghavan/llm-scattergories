import os
import random
import time
import numpy as np
import pickle
from generate_trees import MAX_TEMPS, get_model_list
from collections import Counter
from itertools import product
import argparse
from math import comb
from scat_utils import get_random_instances
from generate_samples import get_temps, get_sample_fname
from verify_samples import get_v_fname
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--input_dir', '-i', type=str, default='./samples')
parser.add_argument('--ouput_dir', '-o', type=str, default='./info')
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--total_jobs', '-t', type=int, default=1)

def get_info_fname(output_dir: str, model: str, n: int, gamma: float) -> str:
    return os.path.join(output_dir, f'{model}_n{n}_gamma{gamma:.2f}_info.pkl')

def get_eq(scores: np.ndarray, temps: list[float]):
    # scores[0, :] += 0.0000001
    print('Scores: ', scores)
    info = {}
    eq_inds = []
    max_indices = np.argmax(scores, axis=0)
    print('Max indices: ', max_indices)
    for i, ind in enumerate(max_indices):
        if i == ind:
            print('Exact Nash equilibrium: ', temps[i], 'Avg. welfare', scores[i, ind])
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
                print('Nash equilibrium: ', temps[i], 'Avg. welfare', scores[i, ind])
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
    print('eq_inds', eq_inds)
    if len(eq_inds) == 0:
        info['nash_eq'] = np.NaN
        info['nash_eq_util'] = np.NaN
    else:
        fake_scores = scores.copy()
        fake_scores[eq_inds[0], eq_inds[0]] = 0
        print('Margin:', scores[eq_inds[0], eq_inds[0]] - np.max(fake_scores[:, eq_inds[0]]))
        info['nash_eq'] = np.mean([temps[i] for i in eq_inds])
        info['nash_eq_util'] = np.mean([scores[i, i] for i in eq_inds])
    symmetric = [scores[i, i] for i in range(len(temps))]
    info['opt'] = temps[np.argmax(symmetric)]
    info['opt_util'] = np.max(symmetric)
    hit_max_temp = scores[-1,:] - np.max(scores, axis=0)
    a_temps = np.array(temps)
    if any(hit_max_temp >= 0):
        print(hit_max_temp >= 0)
        print('Hit max temp in best response:', a_temps[hit_max_temp>=0])
    else:
        print('Did not hit max temp in best response. Margin:', -np.max(hit_max_temp))
    print(info)
    info['scores'] = scores
    info['temperatures'] = a_temps
    return info

def get_score(verified_yes: dict, answer1: str, answers2: list[str], gamma: float=1.0) -> float:
    # Use the faster version instead
    if answer1 not in verified_yes:
        return 0.0
    others_same = sum(1 for a2 in answers2 if answer1 == a2)
    return (1 + others_same) **(-gamma)

def get_score_faster(verified_yes: dict, answer1: str, answers2: Counter, n: int, same: bool=False, gamma: float=1.0) -> float:
    # Pr[exactly k red balls sampling without replacement]
    # comb(num_red, k) * comb(N - num_red, n-1-k) / comb(N, n-1)
    # sum k from 0 to n-1
    if answer1 not in verified_yes:
        return 0.0
    if answer1 not in answers2:
        return 1.0
    num_same = answers2[answer1]
    num_diff = sum(answers2.values()) - num_same
    if same:
        assert num_same > 0
        num_same -= 1
    s = 0.0
    # sum k from 0 to n-1
    for k in range(n):
        s += comb(num_same, k) * comb(num_diff, n-1-k) / comb(num_same + num_diff, n-1) * (1 + k) ** (-gamma)
    return s

def compute_scores(verified_yes: dict, s1: Counter, s2: Counter, n: int, gamma: float=1.0) -> tuple[float, int]:
    s = 0
    if n == 1:
        for k, v in s1.items():
            if k in verified_yes:
                s += v
        return s/sum(s1.values()), sum(s1.values())
    # use symmetric

    if s1 is s2:
        effective_sample_size = sum(s1.values()) // n
    else:
        effective_sample_size = min(sum(s2.values()) // (n-1), sum(s1.values()))
    for answer1, count1 in s1.items():
        s += count1 * get_score_faster(verified_yes, answer1, s2, n, same=s1 is s2, gamma=gamma)
    return s/sum(s1.values()), effective_sample_size

    # num_iterations = 50
    # for _ in range(num_iterations):
        # if s1 is s2:
            # # symmetric case
            # max_size = sum(s1.values()) // n
            # all_samples = samples_from_counter(s1, max_size * n)
            # samples = [all_samples[i::max_size] for i in range(max_size)]
            # samples1 = [samp[0] for samp in samples]
            # samples2 = [samp[1:] for samp in samples]
        # else:
            # max_size = sum(s2.values()) // (n-1)
            # samples1 = samples_from_counter(s1, max_size)
            # all_samples2 = samples_from_counter(s2, max_size * (n-1))
            # # list of lists, inner lists are samples of size max_size
            # samples2 = [all_samples2[i::max_size] for i in range(max_size)]
        # for sample1, sample2 in zip(samples1, samples2):
            # s += get_score(verified_yes, sample1, sample2, gamma)
        # l = len(samples1)
    # return s/l/num_iterations, l

def samples_from_counter(d: Counter, num_samples: int) -> list:
    # sample without replacement
    keys = list(d.keys())
    values = list(d.values())
    samples = []
    for _ in range(num_samples):
        idx = random.choices(range(len(keys)), values)[0]
        samples.append(keys[idx])
        values[idx] -= 1
    return samples

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    # get models
    models = get_model_list(args.models, set(MODELS.keys()))
    random.seed(0)
    ns = [1, 2, 3, 5, 10, 20]
    gammas = [1.0]
    instances = get_random_instances(args.num_instances)
    for model in models:
        #  iterate over all temps
        temps = get_temps(MAX_TEMPS[MODELS[model]])
        temps = [round(temp, 3) for temp in temps]
        sample_map = {}
        verifier_map = {}
        for letter, category in instances:
            verifier_fname = get_v_fname(args.input_dir, letter, category, args.verifier)
            with open(verifier_fname, 'rb') as f:
                verified_yes = pickle.load(f)['yes']
            verifier_map[(letter, category)] = verified_yes
            sample_map[(letter, category)] = {}
            for temp in temps:
                temp = round(temp, 3)
                sample_fname = get_sample_fname(args.input_dir, letter, category, model, temp)
                with open(sample_fname, 'rb') as f:
                    samples = pickle.load(f)
                dist = samples['dist']
                sample_map[(letter, category)][temp] = dist
        # compute scores
        all_scores = {}
        sample_sizes = {}
        for n, gamma in product(ns, gammas):
            # if n == 1, don't need to do all this
            info_fname = get_info_fname(args.ouput_dir, model, n, gamma)
            if os.path.exists(info_fname):
                continue
            scores = np.zeros((len(temps), len(temps)))
            all_scores[n] = scores
            ss = np.zeros((len(temps), len(temps)), dtype=int)
            sample_sizes[n] = ss
            for (i, t1), (j, t2) in product(enumerate(temps), repeat=2):
                for letter, category in instances:
                    score, sample_size = compute_scores(
                            verifier_map[(letter, category)],
                            sample_map[(letter, category)][t1],
                            sample_map[(letter, category)][t2],
                            n,
                            gamma=gamma,
                            )
                    scores[i, j] +=  score / len(instances)
                    ss[i, j] += sample_size
            print(f'{model} {n} {gamma}')
            # print(scores)
            # print(ss)
            info = {'model': model, 'n': n, 'gamma': gamma, 'games': instances, 'max_temperature': max(temps)}
            info.update(get_eq(scores, temps))
            with open(info_fname, 'wb') as f:
                print('Writing to', info_fname)
                pickle.dump(info, f)
            print('CI bound:', np.max(1.96 * np.sqrt(scores * (1 - scores) / ss)))
