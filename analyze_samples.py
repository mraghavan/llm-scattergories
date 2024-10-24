import os
from file_manager import FileManager
import random
import numpy as np
from generate_trees import MAX_TEMPS, get_model_list
from collections import Counter
from itertools import product
import argparse
from math import comb
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--input_dir', '-i', type=str, default='./samples')
parser.add_argument('--output_dir', '-o', type=str, default='./info')

def get_info_fname(output_dir: str, model: str, n: int, gamma: float) -> str:
    return os.path.join(output_dir, f'{model}_n{n}_gamma{gamma:.2f}_info.pkl')

def get_eq(scores: np.ndarray, temps: list[float]):
    info = {}
    eq_inds = []
    max_indices = np.argmax(scores, axis=0)
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
        print('Hit max temp in best response:', a_temps[hit_max_temp>=0])
    else:
        print('Did not hit max temp in best response. Margin:', -np.max(hit_max_temp))
    print(info)
    info['scores'] = scores
    info['temperatures'] = a_temps
    return info

def get_score(verified_yes: dict, answer1: str, answers2: Counter, n: int, same: bool=False, gamma: float=1.0) -> float:
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

    if s1 is s2:
        effective_sample_size = sum(s1.values()) // n
    else:
        effective_sample_size = min(sum(s2.values()) // (n-1), sum(s1.values()))
    for answer1, count1 in s1.items():
        s += count1 * get_score(verified_yes, answer1, s2, n, same=s1 is s2, gamma=gamma)
    return s/sum(s1.values()), effective_sample_size

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
    fm = FileManager.from_args(samples_dir=args.input_dir, info_dir=args.output_dir)
    models = get_model_list(args.models, set(MODELS.keys()))
    ns = [1, 2, 3, 5, 10, 15, 20]
    gammas = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    all_verified = fm.get_all_verified(verifier=args.verifier)
    verified_instances = set(all_verified[['letter', 'category']].drop_duplicates().itertuples(index=False, name=None))
    for model in models:
        all_samples = fm.get_all_samples(model=model, max_temp=MAX_TEMPS[MODELS[model]])
        model_instances = list(all_samples[['letter', 'category']].drop_duplicates().itertuples(index=False, name=None))
        sample_map = {}
        verifier_map = {}
        all_temps = set()
        for letter, category in model_instances:
            assert (letter, category) in verified_instances
            verified_yes = fm.load_verified(letter, category, args.verifier)['yes']
            verifier_map[(letter, category)] = verified_yes
            sample_map[(letter, category)] = {}
            temp_df = all_samples[(all_samples['letter'] == letter) & (all_samples['category'] == category)]
            for temp in temp_df['temperature']:
                all_temps.add(temp)
                samples = fm.load_samples(letter, category, model, temp)
                sample_map[(letter, category)][temp] = samples['dist']
        print(f'Number of instances for model {model}: {len(model_instances)}')
        # compute scores
        all_scores = {}
        sample_sizes = {}
        # assume temps is the same for all (letter, category pairs)
        temps = sorted(list(all_temps))
        for n, gamma in product(ns, gammas):
            # if n == 1, don't need to do all this
            info_fname = fm.get_info_fname(model, n, gamma)
            if os.path.exists(info_fname):
                continue
            scores = np.zeros((len(temps), len(temps)))
            all_scores[n] = scores
            ss = np.zeros((len(temps), len(temps)), dtype=int)
            sample_sizes[n] = ss
            for (i, t1), (j, t2) in product(enumerate(temps), repeat=2):
                for letter, category in model_instances:
                    score, sample_size = compute_scores(
                            verifier_map[(letter, category)],
                            sample_map[(letter, category)][t1],
                            sample_map[(letter, category)][t2],
                            n,
                            gamma=gamma,
                            )
                    scores[i, j] +=  score / len(model_instances)
                    ss[i, j] += sample_size
            print(f'{model} {n} {gamma}')
            info = {'model': model, 'n': n, 'gamma': gamma, 'games': model_instances, 'max_temperature': max(temps)}
            info.update(get_eq(scores, temps))
            fm.write_info(model, n, gamma, info)
            print('CI bound:', np.max(1.96 * np.sqrt(scores * (1 - scores) / ss)))
