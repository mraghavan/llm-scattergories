from completion_utils import MODELS
import os
import numpy as np
import random
from analyze_games import load_all_games
from collections.abc import Callable
import pickle
import multiprocessing as mp

TEMP_DELTA = 0.02
TEMP_ALPHA = 0.8
TEMP_NUM = 30

def get_game_intersection(tree_map):
    # print(tree_map.keys())
    letter_category_sets = {}
    for model_name, games in tree_map.items():
        letter_category_sets[model_name] = set()
        for letter, category, _ in games:
            letter_category_sets[model_name].add((letter, category))
    # print(letter_category_sets)
    # get the intersection of all sets
    intersection = set.intersection(*letter_category_sets.values())
    return intersection

def remove_temp_from_keys(tree_map):
    new_tree_map = {}
    for model in tree_map:
        new_tree_map[model] = {}
        for (letter, category, _), games in tree_map[model].items():
            new_tree_map[model][(letter, category)] = games
    return new_tree_map

# def get_score_multiple_extra_player(
        # extra_model: str,
        # extra_temp: float,
        # tree_map: dict,
        # verified_map: dict,
        # model_counts: dict[str, int],
        # temps: dict[str, float],
        # eta: float,
        # ):
    # games = list(tree_map[next(iter(tree_map))].keys())
    # scores = {model: 0.0 for model in tree_map}
    # scores['extra'] = 0.0
    # num_games = len(games)
    # temps['extra'] = extra_temp
    # for game in games:
        # dists = {}
        # for model in tree_map:
            # dists[model] = tree_map[model][game].get_dist(temps[model])
            # dists[model] = {k: v for k, v in dists[model].items() if k in verified_map[game]['yes']}
        # dists['extra'] = dists[extra_model]
        # utils = dists_to_util(dists, model_counts, eta)

def get_score_multiple(
        tree_map: dict[str, dict],
        verified_map: dict,
        model_counts: dict[str, int],
        temps: dict[str, float],
        eta: float,
        ) -> dict[str, float]:
    games = list(tree_map[next(iter(tree_map))].keys())
    scores = {model: 0.0 for model in tree_map}
    num_games = len(games)
    for game in games:
        dists = {}
        for model in tree_map:
            dists[model] = tree_map[model][game].get_dist(temps[model])
            dists[model] = {k: v for k, v in dists[model].items() if k in verified_map[game]['yes']}
        utils = dists_to_util(dists, model_counts, eta)
        # print(game)
        # print(utils)
        for model in utils:
            scores[model] += utils[model] / num_games
    return scores

def dists_to_util(
        dists: dict[str, dict[str, float]],
        model_counts: dict[str, int],
        eta: float,
        ):
    f = lambda x: (1+x)**(-eta)
    utils = {model: 0.0 for model in dists}
    for model in dists:
        if model_counts[model] == 0:
            continue
        self_model_counts = model_counts.copy()
        self_model_counts[model] -= 1
        for k in dists[model]:
            utils[model] += dists[model][k] * multiple_binom_sum(self_model_counts, {model: dists[model][k] if k in dists[model] else 0 for model in dists}, f)
    return utils

def multiple_binom_sum(
        model_counts: dict[str, int],
        ps: dict[str, float],
        f: Callable[[int], float],
        ):
    total = 0
    array_dists = {model: get_discrete_dist_array(model_counts[model], ps[model]) for model in model_counts if model_counts[model] > 0}
    # iterate through dists and convolve them
    running_dist = np.array([1.0])
    for model in array_dists:
        running_dist = np.convolve(running_dist, array_dists[model])
    for i, prob in enumerate(running_dist):
        total += f(i) * prob
    return total

def get_discrete_dist_array(n: int, p: float):
    if p == 0:
        return np.array([1.0])
    elif p == 1:
        return np.array(([0.0] * n) + [1.0])
    dist = np.zeros(n+1)
    dist[0] = (1-p)**n
    for i in range(1, n+1):
        dist[i] = dist[i-1] * (p/(1-p)) * ((n-i+1)/i)
    return dist

def get_possible_moves(model_counts: dict[str, int]) -> list[tuple[str, str]]:
    moves = []
    for model in model_counts:
        if model_counts[model] > 0:
            for other_model in model_counts:
                moves.append((model, other_model))
    random.shuffle(moves)
    return moves

def find_multiple_eq(
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        n: int,
        max_temp: float,
        max_iter: int = 1000,
        eta: float = 1.0,
        counts_guess: dict[str, int] = {},
        ):
    if not counts_guess:
        model_counts = {model: 0 for model in tree_map}
        model_counts[next(iter(tree_map))] = n
    else:
        model_counts = counts_guess.copy()
    # start with an arbitrary model
    # model = next(iter(tree_map))
    # model_counts[model] = n
    temps = {model: 0.0 for model in tree_map}
    old_scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
    moves = get_possible_moves(model_counts)
    temp_cache = {}
    alternating_flag = False
    temps_history = {model: [] for model in tree_map}
    for i in range(max_iter):
        if not moves:
            break
        print()
        print('Iteration', i)
        print('Scores:', old_scores)
        print('counts', model_counts)
        print('temps', temps)
        print()
        key = tuple(model_counts.items())
        if key not in temp_cache:
            temp_cache[key] = {model: set() for model in tree_map}
        flag = True
        for model in tree_map:
            if temps[model] not in temp_cache[key][model]:
                flag = False
            temp_cache[key][model].add(temps[model])
        alternating_flag = alternating_flag or flag
        # pick a random move and remove it from the list
        # assuming moves is already shuffled
        random_move = moves.pop()
        source_model, dest_model = random_move
        # source_model = np.random.choice(list(nonzero_counts.keys()))
        # dest_model = np.random.choice(list(model_counts.keys()))
        print('Trying to move player from', source_model, 'to', dest_model)
        print(len(moves), 'moves left')
        new_counts = model_counts.copy()
        new_counts[source_model] -= 1
        other_model_counts = new_counts.copy()
        optimized_temp, new_score = find_optimal_temp_extra(dest_model, tree_map, verified_map, other_model_counts, temps, max_temp, eta)
        for model in tree_map:
            temps_history[model].append(temps[model])
        new_counts[dest_model] += 1
        print('Optimized temperature for', dest_model, 'is', optimized_temp, 'with score', new_score)
        if new_score > old_scores[source_model]:
            print('Improved from', old_scores[source_model], 'to', new_score)
            model_counts = new_counts
            thresh = max_iter // 4
            remaining = max_iter - thresh
            if dest_model == source_model and alternating_flag:
                print('Alternating detected')
                # alpha = (remaining - (i - thresh)/2) / remaining
                # temps[dest_model] = optimized_temp * alpha + temps[dest_model] * (1-alpha)
                temps[dest_model] = np.mean(temps_history[dest_model][-1:] + [optimized_temp])
            else:
                temps[dest_model] = optimized_temp
            old_scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
            moves = get_possible_moves(model_counts)
    # optimized_temp = optimize_temps(tree_map, verified_map, model_counts, temps, max_temp, eta, max_iter=10)
    print()
    print('Final scores:', old_scores)
    print('Final counts:', model_counts)
    print('Final temps:', temps)
    results = {'counts': model_counts, 'temps': temps, 'scores': old_scores}
    if not moves:
        results['converged'] = True
    else:
        results['converged'] = False
    return results

def optimize_temps(
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        model_counts: dict[str, int],
        starting_temps: dict[str, float],
        max_temp: float,
        eta: float,
        max_iter: int = 10,
        ):
    temps = starting_temps.copy()
    old_scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
    for i in range(max_iter):
        print('Iteration', i)
        print('Scores:', old_scores)
        print('Temps:', temps)
        print()
        for model in tree_map:
            temps_down = temps.copy()
            temps_up = temps.copy()
            temps_down[model] = max(0.0, temps[model] - TEMP_DELTA)
            temps_up[model] = min(max_temp, temps[model] + TEMP_DELTA)
            scores_down = get_score_multiple(tree_map, verified_map, model_counts, temps_down, eta)
            scores_up = get_score_multiple(tree_map, verified_map, model_counts, temps_up, eta)
            if scores_down[model] > old_scores[model]:
                print('Improved by decreasing temperature for', model)
                temps = temps_down
                old_scores = scores_down
            elif scores_up[model] > old_scores[model]:
                print('Improved by increasing temperature for', model)
                temps = temps_up
                old_scores = scores_up
    return temps

def find_optimal_temp_extra(
        model: str,
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        other_model_counts: dict[str, int],
        temps: dict[str, float],
        max_temp: float,
        eta: float = 1.0,
        ):
    candidate_temps = np.linspace(0.0, max_temp, TEMP_NUM)
    best_temp = -1.0
    best_score = -1.0
    tree_map = tree_map.copy()
    tree_map['extra'] = tree_map[model]
    other_model_counts = other_model_counts.copy()
    other_model_counts['extra'] = 1
    for temp in candidate_temps:
        new_temps = temps.copy()
        new_temps['extra'] = temp
        scores = get_score_multiple(tree_map, verified_map, other_model_counts, new_temps, eta)
        if scores['extra'] > best_score:
            best_score = scores['extra']
            best_temp = temp
    return best_temp, best_score


def get_max_temp(tree_map):
    # TODO np.inf
    max_temp = 1000000
    for model in tree_map:
        for game in tree_map[model]:
            max_temp = min(max_temp, game[2])
    return max_temp

def find_eq(
        models_to_compare: list[str],
        n: int,
        counts_guess: dict[str, int] = {},
        ):
    fname = get_fname_single(models_to_compare, n)
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading from', fname)
            return pickle.load(f)
    print('Generating data for models', models_to_compare, 'n', n)
    tree_map, verified_map = load_all_games()
    reduced_tree_map = {m: tree_map[m] for m in models_to_compare}
    without_temp = remove_temp_from_keys(reduced_tree_map)
    max_temp = get_max_temp(reduced_tree_map)
    if not counts_guess:
        results = find_multiple_eq(without_temp, verified_map, n, max_temp, max_iter=100)
    else:
        assert sum(counts_guess.values()) == n
        results = find_multiple_eq(without_temp, verified_map, n, max_temp, max_iter=100, counts_guess=counts_guess)
    print(results)
    with open(fname, 'wb') as f:
        print('Saving to', fname)
        pickle.dump(results, f)
    return results

def find_eqs(
        models_to_compare: list[str],
        ns: list[int],
        ):
    all_results = {}
    prev_counts = {}
    for n in ns:
        if prev_counts:
            counts_guess = get_counts_guess(n, prev_counts)
            print('Guessing counts', counts_guess)
            all_results[n] = find_eq(models_to_compare, n, counts_guess)
        else:
            all_results[n] = find_eq(models_to_compare, n)
        prev_counts = all_results[n]['counts']
    return all_results

def get_counts_guess(n: int, model_counts: dict[str, int]):
    prev_n = sum(model_counts.values())
    fracs = {model: model_counts[model] / prev_n for model in model_counts}
    scaled = {model: int(fracs[model] * n) for model in model_counts}
    # make sure they sum to n
    diff = n - sum(scaled.values())
    if diff > 0:
        model = np.random.choice(list(model_counts.keys()))
        scaled[model] += diff
    return scaled

def get_fname_single(models_to_compare: list[str], n: int, eta: float):
    return 'eqs/results_' + '_'.join(models_to_compare) + f'_n{n}' + f'_eta{eta}' + '.pkl'
            
# New implemenation here

def find_eq_temp(
        model: str,
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        model_counts: dict[str, int],
        temps: dict[str, float],
        max_temp: float,
        eta: float,
        ):
    if model_counts[model] == 0:
        return 0.0
    seen_so_far = []
    true_model_counts = model_counts.copy()
    model_counts = model_counts.copy()
    model_counts[model] -= 1
    while True:
        print('Next iteration', model, 'with counts', model_counts)
        print('Current temps', temps)
        prev_score = get_score_multiple(tree_map, verified_map, true_model_counts, temps, eta)[model]
        print('Previous score was', prev_score, 'at temp', temps[model])
        best_temp, best_score = find_optimal_temp_extra(
                model, tree_map, verified_map, model_counts, temps, max_temp, eta
                )
        print(model, 'best response temp', best_temp, 'best response score', best_score)
        if best_score <= prev_score:
            print('Found equilibrium', temps[model], prev_score)
            return temps[model]
        prev_score = best_score
        seen_so_far.append(best_temp)
        if best_temp in seen_so_far[:-1]:
            print('Alternating detected')
            print('Seen so far', seen_so_far)
            ind = seen_so_far.index(best_temp)
            best_temp = float(np.mean(seen_so_far[ind+1:]))
        temps[model] = best_temp
        if len(seen_so_far) > 25:
            print('Too many iterations. Returning best so far', best_temp)
            return best_temp

def find_multiple_eq_temps_only(
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        model_counts: dict[str, int],
        max_temp: float,
        eta: float = 1.0,
        temp_guess: dict[str, float] = {},
        ):
    candidate_temps = np.linspace(0.0, max_temp, TEMP_NUM)
    if not temp_guess:
        temps = {model: 0.0 for model in tree_map}
    else:
        temps = temp_guess.copy()
    old_temps = temps
    while True:
        print('Next outer loop with temps', temps)
        old_temps = temps.copy()
        for model in tree_map:
            print('Optimizing temp for', model)
            temps[model] = find_eq_temp(model, tree_map, verified_map, model_counts, temps, max_temp, eta)
        if max([abs(temps[model] - old_temps[model]) for model in tree_map]) < TEMP_DELTA:
            print('Converged', old_temps, temps)
            break
    for model, temp in temps.items():
        if temp == max(candidate_temps):
            print('Warning: max temp reached for', model)
            return {}
    return temps

def verify_eq(
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        model_counts: dict[str, int],
        temps: dict[str, float],
        max_temp: float,
        eta: float,
        ):
    scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
    moves = get_possible_moves(model_counts)
    for move in moves:
        pass

def try_move(
        move: tuple[str, str],
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        model_counts: dict[str, int],
        temps: dict[str, float],
        max_temp: float,
        eta: float,
        ):
    source_model, dest_model = move
    new_counts = model_counts.copy()
    new_counts[source_model] -= 1
    other_model_counts = new_counts.copy()
    optimized_temp, new_score = find_optimal_temp_extra(dest_model, tree_map, verified_map, other_model_counts, temps, max_temp, eta)
    new_counts[dest_model] += 1
    return new_counts, optimized_temp, new_score

def find_eq_for_counts(
        i: int,
        n: int,
        model_list: list[str],
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        max_temp: float,
        eta: float,
        ):
        model_counts = {
                model_list[0]: i,
                model_list[1]: n-i,
                }
        temps = find_multiple_eq_temps_only(
                tree_map, verified_map, model_counts, max_temp, eta,
                )
        if not temps:
            # no equilibrium found, probably because max_temp reached
            return None
        moves = get_possible_moves(model_counts)
        print(moves)
        is_eq = True
        print('Verifying')
        scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
        print('Scores', scores)
        for move in moves:
            _, _, new_score = try_move(
                    move, tree_map, verified_map, model_counts, temps, max_temp, eta
                    )
            print(move)
            print('Score from moving from', move[0], 'to', move[1], new_score)
            if new_score > scores[move[0]]:
                # not an equilibrium
                is_eq = False
                print('Not an equilibrium')
                break
        if is_eq:
            print('Found equilibrium')
            print(model_counts, temps)
            print()
            return (model_counts, temps, scores)

def find_multiple_eq_iterative(
        tree_map: dict[str, dict],
        verified_map: dict[str, dict],
        n: int,
        max_temp: float,
        eta: float = 1.0,
        ) -> list[dict]:
    assert len(tree_map) == 2
    model_list = list(tree_map.keys())
    temps = {model: 0.0 for model in tree_map}
    all_eqs = []
    prev_temps = {}
    with mp.Pool(8) as pool:
        results = pool.starmap(
                find_eq_for_counts,
                [(i, n, model_list, tree_map, verified_map, max_temp, eta) for i in range(n+1)]
                )
    all_eqs = [result for result in results if result is not None]
    # for i in range(n+1):
        # # model_counts = {
                # # model_list[0]: i,
                # # model_list[1]: n-i,
                # # }
        # result = find_eq_for_counts(i, n, model_list, tree_map, verified_map, max_temp, eta)
        # if result is None:
            # continue
        # model_counts, temps, scores = result
        # all_eqs.append((model_counts, temps, scores))
        # print()
        # print('Trying counts', model_counts)
        # if prev_temps:
            # temps = find_multiple_eq_temps_only(
                    # tree_map, verified_map, model_counts, max_temp, eta, prev_temps
                    # )
        # else:
            # temps = find_multiple_eq_temps_only(
                    # tree_map, verified_map, model_counts, max_temp, eta, temps
                    # )
        # if not temps:
            # # no equilibrium found, probably because max_temp reached
            # continue
        # moves = get_possible_moves(model_counts)
        # print(moves)
        # is_eq = True
        # print('Verifying')
        # scores = get_score_multiple(tree_map, verified_map, model_counts, temps, eta)
        # print('Scores', scores)
        # for move in moves:
            # _, _, new_score = try_move(
                    # move, tree_map, verified_map, model_counts, temps, max_temp, eta
                    # )
            # print(move)
            # print('Score from moving from', move[0], 'to', move[1], new_score)
            # if new_score > scores[move[0]]:
                # # not an equilibrium
                # is_eq = False
                # print('Not an equilibrium')
        # if is_eq:
            # print('Found equilibrium')
            # print(model_counts, temps)
            # print()
            # all_eqs.append((model_counts, temps, scores))
    if not all_eqs:
        return []
    results = [{'counts': model_counts, 'temps': temps, 'scores': scores, 'converged': True, 'eta': eta} for model_counts, temps, scores in all_eqs]
    return results

def find_eqs_pairwise(
        models_to_compare: list[str],
        ns: list[int],
        eta: float = 1.0,
        ):
    assert len(models_to_compare) == 2
    all_results = {}
    for n in ns:
        results = find_eq_pairwise(models_to_compare, n, eta)
        all_results[n] = results
        print('Equilibria for n', n, results)
        if results is None:
            print('Warning: no equilibrium found for n', n)
    return all_results

def find_eq_pairwise(
        models_to_compare: list[str],
        n: int,
        eta: float = 1.0,
        ):
    fname = get_fname_single(models_to_compare, n, eta)
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            print('Loading from', fname)
            return pickle.load(f)
    print('Generating data for models', models_to_compare, 'n', n)
    tree_map, verified_map = load_all_games()
    reduced_tree_map = {m: tree_map[m] for m in models_to_compare}
    without_temp = remove_temp_from_keys(reduced_tree_map)
    max_temp = get_max_temp(reduced_tree_map)
    results = find_multiple_eq_iterative(without_temp, verified_map, n, max_temp, eta)
    with open(fname, 'wb') as f:
        print('Saving to', fname)
        pickle.dump(results, f)
    return results

#TODO change to max_temp for each model
if __name__ == '__main__':
    ns = [2, 3, 5, 7, 10, 15, 20]
    models_to_compare = MODELS.keys()
    # models_to_compare = ['qwen1.5', 'smollm', 'nemo']
    # models_to_compare = ['llama3', 'smollm']
    models_to_compare = ['smollm', 'nemo']
    # find_eqs(models_to_compare, ns)

    tree_map, verified_map = load_all_games()
    reduced_tree_map = {m: tree_map[m] for m in models_to_compare}
    without_temp = remove_temp_from_keys(reduced_tree_map)
    max_temp = get_max_temp(reduced_tree_map)

    results = find_multiple_eq_iterative(
            without_temp,
            verified_map,
            5,
            max_temp,
            eta=1.0,
            )
    print(results)

    # temps = find_multiple_eq_temps_only(
            # without_temp,
            # verified_map,
            # {'smollm': 5, 'nemo': 5},
            # max_temp,
            # eta=1.0,
            # )
