import time
import numpy as np
import sys
import gc
import torch
import os
import pickle
import random
from collections import Counter
from completion_base import CompletionEngine, softmax_temperature, softmax_temperature_2d
from scat_utils import get_scat_prompt, get_deterministic_instances, standardize_str
from file_manager import FileManager
import argparse
from scat_utils import get_model_list, MAX_TEMPS, get_scat_prompt
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='')
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./samples')
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--num_jobs', '-t', type=int, default=1)
parser.add_argument('--num_samples', '-s', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=4)

EPS_GRID = 0.05
LARGE_TEMP = 3.0
LARGE_P = 0.99
CACHE_MIN = 0.01

class SortedLogitsAndTokens():
    # Necessary to optimize cache. Otherwise, we sort every time
    def __init__(self, logits: np.ndarray):
        # logits is 1D
        # guarantee that logits are sorted in increasing order
        probs_at_large_temp = softmax_temperature(logits, LARGE_TEMP)
        sorted_indices = np.argsort(probs_at_large_temp)
        sorted_probs = probs_at_large_temp[sorted_indices]
        # sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        mask = np.array(cumulative_probs > 1 - LARGE_P, copy=False)
        self.tokens = np.array(sorted_indices[mask], copy=True)
        self.logits = np.array(logits[self.tokens], copy=True)
        # print('Size reduction:', logits.size, '->', self.tokens.size, '(', self.tokens.size / logits.size, ')')


def get_new_sample_prefix(temperature: float, top_p: float, cache: dict[tuple, SortedLogitsAndTokens]) -> tuple[list[int], float]:
    sample = ()
    lp = 0.0
    while sample in cache:
        logits_and_tokens = cache[sample]
        # both length 1
        tok, log_prob = sample_from_sorted_logits(logits_and_tokens, temperature, top_p)
        sample += (tok,)
        lp += log_prob
    return list(sample), lp

def sample_from_sorted_logits(sorted_logits: SortedLogitsAndTokens, temp: float, top_p: float) -> tuple[int, float]:
    logits = sorted_logits.logits
    tokens = sorted_logits.tokens
    probs = softmax_temperature(logits, temp)
    cumulative_probs = np.cumsum(probs)
    mask = np.array(cumulative_probs > 1 - top_p)
    masked_probs = probs[mask]
    masked_probs = masked_probs / np.sum(masked_probs)
    sampled_index = random.choices(np.arange(len(masked_probs)), weights=masked_probs, k=1)[0]
    sampled_prob = masked_probs[sampled_index]
    sampled_token = tokens[mask][sampled_index]
    return int(sampled_token), np.log(sampled_prob)

def sample_from_logits(logits: np.ndarray, temperature: float, top_p: float) -> tuple[list[int], list[float]]:
    if len(logits.shape) == 1:
        logits = logits[None]
    probs = softmax_temperature_2d(logits, temperature)
    sorted_indices = np.argsort(probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
    # sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs, axis=1)
    mask = np.array(cumulative_probs > 1 - top_p, copy=False)
    tokens = []
    log_probs = []
    for i in range(logits.shape[0]):
        masked_probs = sorted_probs[i, mask[i]]
        new_prob_sum = np.sum(masked_probs)
        masked_probs = masked_probs / new_prob_sum
        masked_indices = sorted_indices[i, mask[i]]
        r = np.arange(len(masked_probs))
        index_of_index = random.choices(r, weights=masked_probs, k=1)[0]
        prob = masked_probs[index_of_index]
        token_index = masked_indices[index_of_index]
        sampled_token = int(token_index)
        tokens.append(sampled_token)
        log_probs.append(np.log(prob))
    return tokens, log_probs

def generate_samples(
        engine: CompletionEngine,
        letter: str,
        category: str,
        temperature: float,
        num_samples: int,
        cache: dict[tuple, SortedLogitsAndTokens] | None=None,
        existing_info: dict | None=None,
        max_tokens: int=6,
        batch_size: int=8,
        top_p: float = 0.95,
        ) -> dict:
    allowed_tokens, allowed_starting_tokens = engine.get_allowed_tokens(letter)
    prompt = get_scat_prompt(letter, category, engine.tokenizer)
    tokenized_prompt = engine.encode_prompt(prompt)

    if cache is None:
        cache = {}

    if existing_info is not None:
        c = existing_info['dist']
        prob_dict = existing_info['probs']
        unfinished = existing_info['unfinished']
    else:
        c = Counter()
        prob_dict = {}
        unfinished = 0

    queue = []
    log_probs = []
    prob_dict = {}
    print('Batch size:', batch_size)

    def sample_complete(sample: list[int]) -> bool:
        is_invalid = (len(sample) > 0 and sample[-1] not in allowed_tokens) or (len(sample) == 1 and sample[-1] not in allowed_starting_tokens)
        too_long = len(sample) >= max_tokens
        finished = len(sample) > 0 and sample[-1] == engine.tokenizer.eos_token_id
        return finished or too_long or is_invalid

    def process_response(sample: list[int], lp: float):
        is_invalid = sample[-1] not in allowed_tokens or (len(sample) == 1 and sample[-1] not in allowed_starting_tokens)
        too_long = len(sample) >= max_tokens
        finished = sample[-1] == engine.tokenizer.eos_token_id
        if finished or too_long or is_invalid:
            generated_text = engine.tokenizer.decode(sample)
            seen_before = generated_text in prob_dict
            if not seen_before:
                prob_dict[generated_text] = np.exp(lp)
            if too_long:
                nonlocal unfinished
                unfinished += 1
            if is_invalid:
                generated_text = engine.tokenizer.decode(sample[:-1])
            generated_text = standardize_str(generated_text, engine.tokenizer.eos_token)
            c[generated_text] += 1

    while len(queue) < batch_size and sum(c.values()) + len(queue) < num_samples:
        new_sample, new_lp = get_new_sample_prefix(temperature, top_p, cache)
        if sample_complete(new_sample):
            process_response(new_sample, new_lp)
        else:
            queue.append(new_sample)
            log_probs.append(new_lp)
    while len(c) < num_samples and len(queue) > 0:
        next_logits = engine.get_logits_raw_batch([tokenized_prompt + t for t in queue])
        assert next_logits.shape[0] == len(queue)
        sorted_logits_list = [SortedLogitsAndTokens(logits) for logits in next_logits]
        new_tokens = []
        new_log_probs = []
        for i, t in enumerate(queue):
            tup = tuple(t)
            # only cache if there's a reasonably large chance of sampling it again
            if tup not in cache and np.exp(log_probs[i]) > CACHE_MIN:
                cache[tup] = sorted_logits_list[i]
            new_token, new_log_prob = sample_from_sorted_logits(sorted_logits_list[i], temperature, top_p)
            new_tokens.append(new_token)
            new_log_probs.append(new_log_prob)
        assert len(new_tokens) == len(queue)
        for i, (tok, lp) in enumerate(zip(new_tokens, new_log_probs)):
            queue[i].append(tok)
            log_probs[i] += lp
        # if any sample is finished, add it to c and replace it with a new sample
        next_queue = []
        next_log_probs = []
        for t, lp in zip(queue, log_probs):
            if sample_complete(t):
                process_response(t, lp)
                while sum(c.values()) + len(queue) <= num_samples:
                    new_sample, new_lp = get_new_sample_prefix(temperature, top_p, cache)
                    if sample_complete(new_sample):
                        process_response(new_sample, new_lp)
                    else:
                        next_queue.append(new_sample)
                        next_log_probs.append(new_lp)
                        break
            else:
                next_queue.append(t)
                next_log_probs.append(lp)
        queue = next_queue
        log_probs = next_log_probs
    print(f'Unfinished samples: {unfinished}')
    # print('Number of disctinct samples:', len(c))
    prob_mass = sum(prob_dict.values())
    print('Mass captured:', prob_mass)
    info = {}
    info['letter'] = letter
    info['category'] = category
    info['num_samples'] = num_samples
    info['unfinished'] = unfinished
    info['prob_mass'] = prob_mass
    info['probs'] = prob_dict
    info['dist'] = c
    return info

def get_temps(max_temp: float) -> np.ndarray:
    return np.arange(0, max_temp + EPS_GRID, EPS_GRID)

def get_temps_clean(max_temp: float) -> list[float]:
    return [round(x, 3) for x in get_temps(max_temp)]

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import CompletionEngineMLX as CE, MODELS
    else:
        from completion_hf import CompletionEngineHF as CE, MODELS
    fm = FileManager.from_args(samples_dir=args.output_dir)
    models = get_model_list(args.models, set(MODELS.keys()))
    if args.job_num >= len(models):
        print(f'Job number {args.job_num} is out of range')
        sys.exit(0)
    instances = get_deterministic_instances(args.num_instances)
    if len(models) > 1 and args.num_jobs > 1:
        models = models[args.job_num::args.num_jobs]
    elif len(models) == 1 and args.num_jobs > 1:
        instances = instances[args.job_num::args.num_jobs]
    print(f'Models for job {args.job_num}: {models}')
    print(f'Instances for job {args.job_num}: {instances}')
    for nickname in models:
        print('Model:', nickname)
        model_name = MODELS[nickname]
        max_temperature = MAX_TEMPS[model_name]
        engine = CE.get_completion_engine(model_name, max_temperature=max_temperature, nickname=nickname, epsilon=0)

        temps = get_temps_clean(max_temperature)
        for letter, category in instances:
            cache_fname = fm.get_cache_fname(letter, category, nickname)
            if os.path.exists(cache_fname):
                with open(cache_fname, 'rb') as f:
                    cache = pickle.load(f)
            else:
                cache = {}
            for temp in temps:
                print('Generating', args.num_samples, 'samples for', letter, category, 'at temperature', temp)
                fname = fm.get_sample_fname(letter, category, nickname, temp)
                if os.path.exists(fname):
                    existing_info = pickle.load(open(fname, 'rb'))
                else:
                    existing_info = None
                if existing_info and existing_info['num_samples'] >= args.num_samples:
                    print('Already have enough samples for', letter, category, 'at temperature', temp)
                    continue
                prompt = get_scat_prompt(letter, category, engine.tokenizer)
                start = time.time()
                info = generate_samples(engine, letter, category, temp, args.num_samples, batch_size = args.batch_size, cache=cache, existing_info=existing_info)
                elapsed = time.time() - start
                print(f'Elapsed time: {elapsed:.2f}')
                fm.write_samples(letter, category, nickname, temp, info)
                fm.write_cache(letter, category, nickname, cache)
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
