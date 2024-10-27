import os
from typing import Generator
from pathlib import Path
from functools import lru_cache
import pickle
from scat_utils import MAX_TEMPS, get_model_list
from itertools import product
import argparse
from math import comb
from file_manager import FileManager
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--input_dir', '-i', type=str, default='./samples')
parser.add_argument('--output_dir', '-o', type=str, default='./info')
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--total_jobs', '-t', type=int, default=1)

def load_samples(model: str, max_temp: float, fm: FileManager):
    all_samples = fm.get_all_samples(model=model)
    samples = {}
    for _, row in all_samples.iterrows():
        if row['temperature'] > max_temp:
            continue
        t = row['temperature']
        if t not in samples:
            samples[t] = {}
        path = Path(row['fname']) # type: ignore
        with open(path, 'rb') as f:
            samples[t][(row['letter'], row['category'])] = pickle.load(f)['dist']
    return samples

def load_verified_map(verifier: str, fm: FileManager):
    all_verified = fm.get_all_verified(verifier=verifier)
    verified_map = {}
    for _, row in all_verified.iterrows():
        path = Path(row['fname']) # type: ignore
        with open(path, 'rb') as f:
            verified_map[(row['letter'], row['category'])] = pickle.load(f)['yes']
    return verified_map

class PairwiseEquilibria:
    def __init__(self, model1: str, model2: str, verifier: str, fm: FileManager, max_temps: dict[str, float]):
        temp1 = max_temps[model1]
        temp2 = max_temps[model2]
        samples1 = load_samples(model1, temp1, fm)
        samples2 = load_samples(model2, temp2, fm)
        verified_map = load_verified_map(verifier, fm)
        self.samples1 = samples1
        self.samples2 = samples2
        self.verified_map = verified_map
        self.temp1 = temp1
        self.temp2 = temp2
        self.instances = sorted(list(samples1[temp1].keys()))

    def why_not_eq(
            self,
            n1: int,
            n2: int,
            t1: float,
            t2: float,
            gamma: float,
            eps: float=1.0,
            ) -> tuple[int, int, float, float] | None:
        utility1 = self.get_utility(1, n1, n2, t1, t2, gamma)
        utility2 = self.get_utility(2, n1, n2, t1, t2, gamma)
        for player_start, player_end, new_temp in self.get_valid_deviations(n1, n2, t1, t2):
            dev_utility = self.get_deviating_utility(player_start, player_end, new_temp, n1, n2, t1, t2, gamma)
            assert 0 <= dev_utility <= 1
            if player_start == 1:
                if dev_utility > utility1 * (1 + eps):
                    return (player_start, player_end, new_temp, dev_utility)
            else:
                if dev_utility > utility2 * (1 + eps):
                    return (player_start, player_end, new_temp, dev_utility)
        return None

    @lru_cache
    def get_utility(self, player: int, n1: int, n2: int, temp1: float, temp2: float, gamma: float) -> float:
        """Utility for n1, n2, temp1, temp2 for player 1 or 2."""
        assert player in [1, 2]
        if player == 1 and n1 == 0:
            return 0
        elif player == 2 and n2 == 0:
            return 0
        score = 0
        dists1 = self.samples1[temp1]
        dists2 = self.samples2[temp2]
        if player == 2:
            dists2, dists1 = dists1, dists2
            n2, n1 = n1, n2
        n1 -= 1
        for letter, category in self.instances:
            verified_yes = self.verified_map[(letter, category)]
            score += get_score_two_dists(dists1[(letter, category)], dists2[(letter, category)], n1, n2, verified_yes, gamma)
        score /= len(self.instances)
        return score

    def get_deviating_utility(
            self,
            player_start: int,
            player_end: int,
            new_temp: float,
            n1: int,
            n2: int,
            temp1: float,
            temp2: float,
            gamma: float
            ) -> float:
        assert player_start in [1, 2]
        assert player_end in [1, 2]
        if player_start == 1:
            assert n1 > 0
        else:
            assert n2 > 0
        score = 0
        if player_end == 1 and new_temp == temp1:
            if player_start == 1:
                # no change
                return self.get_utility(1, n1, n2, temp1, temp2, gamma)
            else:
                return self.get_utility(1, n1+1, n2-1, temp1, temp2, gamma)
        if player_end == 2 and new_temp == temp2:
            if player_start == 2:
                # no change
                return self.get_utility(2, n1, n2, temp1, temp2, gamma)
            else:
                return self.get_utility(2, n1-1, n2+1, temp1, temp2, gamma)
        # assume new_temp != temp[player_end]
        score = 0
        new_dists = self.samples1[new_temp] if player_end == 1 else self.samples2[new_temp]
        dists1 = self.samples1[temp1]
        dists2 = self.samples2[temp2]
        if player_start == 1:
            n1 -= 1
        else:
            n2 -= 1
        for letter, category in self.instances:
            verified_yes = self.verified_map[(letter, category)]
            score += get_score_three_dists(
                    new_dists[(letter, category)],
                    dists1[(letter, category)],
                    dists2[(letter, category)],
                    n1,
                    n2,
                    verified_yes,
                    gamma,
                    )
        score /= len(self.instances)
        return score

    def get_valid_deviations(
            self,
            n1: int,
            n2: int,
            temp1: float,
            temp2: float,
            ) -> Generator[tuple[int, int, float], None, None]:
        temps1 = sorted(list(self.samples1.keys()))
        temps2 = sorted(list(self.samples2.keys()))
        # start with deviations without changing temp
        if n1 > 0:
            yield (1, 2, temp2)
        if n2 > 0:
            yield (2, 1, temp1)
        # now deviations with changing temp
        if n1 > 0:
            for t1 in temps1:
                if t1 == temp1:
                    continue
                yield (1, 1, t1)
            for t2 in temps2:
                yield (1, 2, t2)
        if n2 > 0:
            for t2 in temps2:
                if t2 == temp2:
                    continue
                yield (2, 2, t2)
            for t1 in temps1:
                yield (2, 1, t1)

    def get_all_possible_eqs(self, n: int) -> Generator[tuple[int, float, int, float], None, None]:
        for n1, t1, t2 in product(range(n+1), self.samples1.keys(), self.samples2.keys()):
            if n1 == 0 and t1 > 0:
                continue
            n2 = n - n1
            if n2 == 0 and t2 > 0:
                continue
            yield (n1, t1, n2, t2)
            
    def find_eqs(self, n: int, gamma: float, eps: float=0.0) -> list[dict]:
        all_eqs = []
        for n1, t1, n2, t2 in self.get_all_possible_eqs(n):
            # maybe_eq = True
            why_not = self.why_not_eq(n1, n2, t1, t2, gamma, eps)
            if not why_not:
                utility1 = self.get_utility(1, n1, n2, t1, t2, gamma)
                utility2 = self.get_utility(2, n1, n2, t1, t2, gamma)
                counts = {model1: n1, model2: n2}
                temps = {model1: t1, model2: t2}
                scores = {model1: utility1, model2: utility2}
                all_eqs.append({
                    'counts': counts,
                    'temps': temps,
                    'scores': scores,
                    'converged': True,
                    'gamma': gamma,
                    'n': n,
                    'eps': eps,
                })
        return all_eqs

def get_score_two_dists(dist1: dict[str, int], dist2: dict[str, int], n1: int, n2: int, verified_yes: set[str], gamma: float=1.0) -> float:
    assert dist1 is not dist2
    s = 0
    num_total_1 = sum(dist1.values()) - 1
    num_total_2 = sum(dist2.values())
    for answer, count in dist1.items():
        if answer not in verified_yes:
            continue
        if n1 == 0 and n2 == 0:
            s += count
            continue
        num_same_1 = dist1[answer] - 1
        num_same_2 = dist2[answer]
        for k1, k2 in product(range(n1+1), range(n2+1)):
            if k1 > num_same_1:
                continue
            if k2 > num_same_2:
                continue
            s += count\
                    * prob_of_k_of_m_red(k1, n1, num_same_1, num_total_1)\
                    * prob_of_k_of_m_red(k2, n2, num_same_2, num_total_2)\
                    * (1 + k1 + k2)**(-gamma)
    return s / sum(dist1.values())

def get_score_three_dists(
        new_dist: dict[str, int],
        dist1: dict[str, int],
        dist2: dict[str, int],
        n1: int,
        n2: int,
        verified_yes: set[str],
        gamma: float=1.0
        ) -> float:
    assert new_dist is not dist1
    assert new_dist is not dist2
    assert dist1 is not dist2
    s = 0
    num_total_1 = sum(dist1.values())
    num_total_2 = sum(dist2.values())
    for answer, count in new_dist.items():
        if answer not in verified_yes:
            continue
        num_same_1 = dist1[answer]
        num_same_2 = dist2[answer]
        for k1, k2 in product(range(n1+1), range(n2+1)):
            if k1 > num_same_1:
                continue
            if k2 > num_same_2:
                continue
            s += count\
                    * prob_of_k_of_m_red(k1, n1, num_same_1, num_total_1)\
                    * prob_of_k_of_m_red(k2, n2, num_same_2, num_total_2)\
                    * (1 + k1 + k2)**(-gamma)
    s = s / sum(new_dist.values())
    assert 0 <= s <= 1
    return s

def prob_of_k_of_m_red(k: int, m: int, n_red: int, n_total: int) -> float:
    return comb(n_red, k) * comb(n_total - n_red, m - k) / comb(n_total, m)

def get_pairwise_fname(output_dir: str, model1: str, model2: str, n: int, gamma: float) -> str:
    return os.path.join(output_dir, f'{model1}_{model2}_n{n}_gamma{gamma:.2f}_pairwise.pkl')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    fm = FileManager.from_args(samples_dir=args.input_dir, info_dir=args.output_dir)
    # get models
    models = get_model_list(args.models, set(MODELS.keys()))
    assert len(models) == 2
    model1 = models[0]
    model2 = models[1]
    print('Models:', model1, model2)
    verifier = args.verifier
    ns = range(1, 16)
    gamma = 1.0
    nicknames_to_max_temps = {nickname: MAX_TEMPS[real_name] for nickname, real_name in MODELS.items() if real_name in MAX_TEMPS}
    pairwise_eq_finder = PairwiseEquilibria(model1, model2, verifier, fm, nicknames_to_max_temps)

    for n in ns:
        fname = fm.get_pairwise_fname(model1, model2, n, gamma)
        if os.path.exists(fname):
            print('Skipping', fname)
            continue
        actual_eqs = pairwise_eq_finder.find_eqs(n, gamma, eps=0.01)
        fm.write_pairwise(model1, model2, n, gamma, actual_eqs)
