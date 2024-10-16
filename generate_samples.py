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
from completion_base import build_completion_tree, CompletionEngine, CompletionNode, softmax_temperature
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

# TODO move this
from generate_trees import get_scat_prompt, get_model_list, MAX_TEMPS

def generate_text(engine: CompletionEngine,
                  prompt_tokens: list,
                  max_tokens: int,
                  cache: dict,
                  allowed_tokens: set,
                  allowed_starting_tokens: set,
                  ) -> tuple[str, float, bool]:
    prompt_tokens = prompt_tokens.copy()
    sampled_tokens = []
    log_prob = 0.0
    while max_tokens > 0:
        max_tokens -= 1
        tup_tokens = tuple(sampled_tokens)
        if tup_tokens in cache:
            tokens, logits = cache[tuple(tup_tokens)]
        else:
            tokens, logits = engine.get_logits(prompt_tokens)
            cache[tup_tokens] = (tokens, logits)
        temperature = engine.max_temperature
        probs = softmax_temperature(logits, temperature)
        sampled_index = int(random.choices(range(len(probs)), weights=probs, k=1)[0])
        sampled_token = int(tokens[sampled_index])
        token_prob = probs[sampled_index]
        log_prob += np.log(token_prob)
        prompt_tokens.append(sampled_token)
        sampled_tokens.append(sampled_token)
        if len(sampled_tokens) == 1 and sampled_token not in allowed_starting_tokens:
            st = engine.tokenizer.decode(sampled_token)
            # print('Discarding starting token:', st)
            return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), False
        elif sampled_token not in allowed_tokens:
            st = engine.tokenizer.decode(sampled_token)
            # print('Discarding token:', st, sampled_token)
            return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), False
        if sampled_token == int(engine.tokenizer.eos_token_id):
            break
    print(engine.tokenizer.decode(sampled_tokens))
    print(np.exp(log_prob))
    return engine.tokenizer.decode(sampled_tokens), np.exp(log_prob), True

def generate_samples(engine: CompletionEngine, letter: str, category: str, num_samples: int, max_tokens: int=6) -> dict:
    allowed_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz' + ' ' + '0123456789'
    allowed_characters = set(allowed_characters)
    allowed_starting_tokens = set()
    allowed_tokens = set()
    allowed_tokens.add(engine.tokenizer.eos_token_id)
    vocabulary = list(engine.tokenizer.get_vocab())
    print(len(vocabulary))
    print(vocabulary[:10])
    for i, token in enumerate(vocabulary):
        token_id = engine.tokenizer.convert_tokens_to_ids(token)
        token = engine.tokenizer.decode(token_id)
        # print(token_id, token)
        if all(c in allowed_characters for c in token):
            allowed_tokens.add(token_id)
        if token_id in allowed_tokens and token.lower().lstrip().startswith(letter.lower()):
            allowed_starting_tokens.add(token_id)
    print('Number of allowed tokens:', len(allowed_tokens))
    print('Number of allowed starting tokens:', len(allowed_starting_tokens))
    cache = {}
    prompt = get_scat_prompt(letter, category, engine.tokenizer)
    print(prompt)
    tokenized_prompt = engine.encode_prompt(prompt)
    c = Counter()
    seen = set()
    unfinished = 0
    prob_mass = 0.0
    for i in range(num_samples):
        # print('Sample', i)
        generated_text, prob, is_valid = generate_text(engine, tokenized_prompt, max_tokens, cache, allowed_tokens, allowed_starting_tokens)
        if is_valid and generated_text and not generated_text.endswith(engine.tokenizer.eos_token):
            unfinished += 1
        generated_text = standardize_str(generated_text, engine.tokenizer.eos_token)
        if generated_text not in seen:
            seen.add(generated_text)
            assert prob > 0
            prob_mass += prob
        if not is_valid:
            generated_text = ''
        c[generated_text] += 1
    print(f'Unfinished samples: {unfinished}')
    num_ones = sum(1 for _, v in c.items() if v == 1)
    print('Good-Turing estimate:', num_ones / num_samples)
    print('Number of disctinct samples:', len(c))
    print('Mass captured:', prob_mass)
    info = {}
    info['letter'] = letter
    info['category'] = category
    info['num_samples'] = num_samples
    info['unfinished'] = unfinished
    info['good_turing'] = num_ones / num_samples
    info['prob_mass'] = prob_mass
    info['dist'] = c
    return info

def get_sample_fname(output_dir: str, letter: str, category: str, model_name: str, temp: float, job_num: int, jobs: int):
    category = re.sub('[^a-zA-Z0-9 ]+', '', category)
    category = re.sub(' ', '_', category)
    return f'{output_dir}/{letter}_{category}_{model_name}_{temp}_{job_num}_{jobs}.pkl'

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import CompletionEngineMLX as CE, MODELS
        import mlx.core as mx
    else:
        from completion_hf import CompletionEngineHF as CE, MODELS
    models = get_model_list(args.models, set(MODELS.keys()))
    nickname = models[0]
    print('Model:', nickname)
    model_name = MODELS[nickname]
    max_temperature = MAX_TEMPS[model_name]
    engine = CE.get_completion_engine(model_name, max_temperature=max_temperature, nickname=nickname, epsilon=1e-5)
    random_instances = get_random_instances(args.num_instances)
    for letter, category in random_instances:
        start = time.time()
        c = generate_samples(engine, letter, category, args.num_samples)
        elapsed = time.time() - start
        print(f'Elapsed time: {elapsed:.2f}')
        print(c)
        fname = get_sample_fname(args.output_dir, letter, category, nickname, max_temperature, args.job_num, args.num_jobs)
        print('Saving to', fname)
        with open(fname, 'wb') as f:
            pickle.dump(c, f)
