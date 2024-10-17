import time
import numpy as np
import sys
import gc
import torch
import os
import re
import pickle
import random
from collections import Counter
from completion_base import build_completion_tree, CompletionEngine, CompletionNode, softmax_temperature, softmax_temperature_2d
from scat_utils import get_scat_prompt, get_random_instances, standardize_str
from verifier import Verifier
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='')
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./samples')
parser.add_argument('--depth', '-d', type=int, default=3)
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--num_jobs', '-t', type=int, default=1)
parser.add_argument('--num_samples', '-s', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=4)

EPS_GRID = 0.1

# TODO move this
from generate_trees import get_scat_prompt, get_model_list, MAX_TEMPS

# def generate_text(engine: CompletionEngine,
                  # prompt_tokens: list,
                  # max_tokens: int,
                  # cache: dict,
                  # allowed_tokens: set,
                  # allowed_starting_tokens: set,
                  # ) -> tuple[str, float, bool]:
    # prompt_tokens = prompt_tokens.copy()
    # sampled_tokens = []
    # log_prob = 0.0
    # while max_tokens > 0:
        # max_tokens -= 1
        # tup_tokens = tuple(sampled_tokens)
        # # TODO set top_p to 0 in the engine and do it here so we cache the full logits
        # if tup_tokens in cache:
            # tokens, logits = cache[tuple(tup_tokens)]
        # else:
            # tokens, logits = engine.get_logits(prompt_tokens)
            # cache[tup_tokens] = (tokens, logits)
        # temperature = engine.max_temperature
        # probs = softmax_temperature(logits, temperature)
        # sampled_index = int(random.choices(range(len(probs)), weights=probs, k=1)[0])
        # sampled_token = int(tokens[sampled_index])
        # token_prob = probs[sampled_index]
        # log_prob += np.log(token_prob)
        # prompt_tokens.append(sampled_token)
        # sampled_tokens.append(sampled_token)
        # if len(sampled_tokens) == 1 and sampled_token not in allowed_starting_tokens:
            # st = engine.tokenizer.decode(sampled_token)
            # # print('Discarding starting token:', st)
            # return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), False
        # elif sampled_token not in allowed_tokens:
            # st = engine.tokenizer.decode(sampled_token)
            # # print('Discarding token:', st, sampled_token)
            # return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), False
        # if sampled_token == int(engine.tokenizer.eos_token_id):
            # break
    # print(engine.tokenizer.decode(sampled_tokens))
    # print(np.exp(log_prob))
    # return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), True

def get_new_sample_prefix(temperature: float, top_p: float, cache: dict) -> tuple[list[int], float]:
    sample = ()
    lp = 0.0
    while sample in cache:
        logits = cache[sample]
        # both length 1
        toks, log_probs = sample_from_logits(logits, temperature, top_p)
        sample += tuple(toks)
        lp += log_probs[0]
    return list(sample), lp

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
        cache: dict | None=None,
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
        num_remaining = num_samples - sum(c.values())
    else:
        c = Counter()
        prob_dict = {}
        unfinished = 0
        num_remaining = num_samples

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
                generated_text = ''
            if finished or too_long:
                generated_text = standardize_str(generated_text, engine.tokenizer.eos_token)
            c[generated_text] += 1

    while len(queue) < batch_size and sum(c.values()) + len(queue) < num_samples:
        new_sample, new_lp = get_new_sample_prefix(temperature, top_p, cache)
        if sample_complete(new_sample):
            process_response(new_sample, new_lp)
        else:
            queue.append(new_sample)
            log_probs.append(new_lp)
    # for _ in range(batch_size):
        # queue.append([])
        # log_probs.append(0.0)
    while len(c) < num_samples and len(queue) > 0:
        next_logits = engine.get_logits_raw_batch([tokenized_prompt + t for t in queue])
        assert next_logits.shape[0] == len(queue)
        for i, t in enumerate(queue):
            tup = tuple(t)
            if tup not in cache:
                cache[tup] = next_logits[i]
        new_tokens, new_log_probs = sample_from_logits(next_logits, temperature, top_p)
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
    # print(prob_dict)
    return info

def get_sample_fname(output_dir: str, letter: str, category: str, model_name: str, temp: float) -> str:
    category = re.sub('[^a-zA-Z0-9 ]+', '', category)
    category = re.sub(' ', '_', category)
    return f'{output_dir}/{letter}_{category}_{model_name}_{temp}_samples.pkl'

def get_cache_fname(output_dir: str, letter: str, category: str, model_name: str) -> str:
    category = re.sub('[^a-zA-Z0-9 ]+', '', category)
    category = re.sub(' ', '_', category)
    return f'{output_dir}/{letter}_{category}_{model_name}_cache.pkl'

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import CompletionEngineMLX as CE, MODELS
    else:
        from completion_hf import CompletionEngineHF as CE, MODELS
    models = get_model_list(args.models, set(MODELS.keys()))
    nickname = models[0]
    print('Model:', nickname)
    model_name = MODELS[nickname]
    max_temperature = MAX_TEMPS[model_name]
    engine = CE.get_completion_engine(model_name, max_temperature=max_temperature, nickname=nickname, epsilon=0)

    random.seed(0)
    random_instances = get_random_instances(args.num_instances)
    temps = np.arange(0, max_temperature + EPS_GRID, EPS_GRID)
    for letter, category in random_instances:
        cache_fname = get_cache_fname(args.output_dir, letter, category, nickname)
        if os.path.exists(cache_fname):
            with open(cache_fname, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = {}
        for temp in temps:
            temp = round(temp, 3)
            print('Generating', args.num_samples, 'samples for', letter, category, 'at temperature', temp)
            fname = get_sample_fname(args.output_dir, letter, category, nickname, temp)
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
            print('Saving to', fname)
            with open(fname, 'wb') as f:
                pickle.dump(info, f)
            print('Saving cache to', cache_fname)
            with open(cache_fname, 'wb') as f:
                pickle.dump(cache, f)
