import time
import sys
import gc
import torch
import os
import re
import pickle
import random
from completion_base import build_completion_tree, CompletionEngine, CompletionNode
from scat_utils import get_scat_prompt, get_random_instances
from verifier import Verifier
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='')
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./trees')
parser.add_argument('--depth', '-d', type=int, default=3)
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--num_jobs', '-t', type=int, default=1)

MAX_TEMPS = {
        'meta-llama/Meta-Llama-3-8B-Instruct': 2.5,
        'meta-llama/Llama-3.1-8B-Instruct': 2.5,
        'meta-llama/Llama-3.2-1B-Instruct': 1.5,
        'HuggingFaceTB/SmolLM-1.7B-Instruct': 1.2,
        'Qwen/Qwen2-7B': 1.5,
        'google/gemma-2-2b-it': 1.5,
        'microsoft/Phi-3.5-mini-instruct': 2.0,
        'mistralai/Mistral-7B-Instruct-v0.3': 1.8,
        'nvidia/Nemotron-Mini-4B-Instruct': 1.8,
        # 'nvidia/Mistral-NeMo-Minitron-8B-Instruct': 1.9, # (too big for gpu)
        'Qwen/Qwen2.5-7B-Instruct': 2.0,
        'mlx-community/Meta-Llama-3.1-8B-Instruct-8bit': 1.5,
        'mlx-community/Meta-Llama-3-8B-Instruct-8bit': 1.5,
        # 'gemma2': 'mlx-community/gemma-2-9b-8bit', (not an instruct model)
        'mlx-community/Qwen1.5-7B-Chat-4bit': 1.5,
        # 'openelm1_1': 'mlx-community/OpenELM-1_1B-Instruct-8bit', (no chat template)
        'mlx-community/SmolLM-1.7B-Instruct-fp16': 1.5,
        'mlx-community/Mistral-Nemo-Instruct-2407-4bit': 1.5,
        # 'phi3': 'mlx-community/Phi-3-small-8k-instruct-AQ4_32', (haven't tried yet)
        # 'mlx-community/Phi-3.5-mini-instruct-bf16': 1.5,
        'mlx-community/Phi-3.5-mini-instruct-bf16': 0.5,
        # "yi1.5": 'mlx-community/Yi-1.5-9B-Chat-4bit', (haven't tried yet)
        # 'mlx-community/Llama-3.2-3B-Instruct-8bit': 1.5,
        'mlx-community/Llama-3.2-3B-Instruct-8bit': 0.5,
        }

def make_str_safe(s: str) -> str:
    return re.sub('/', ' ', s).strip()

def get_pickle_filename(tree_dir: str, letter: str, category: str, max_temperature: float, nickname: str) -> str:
    category = make_str_safe(category)
    if nickname:
        return f'{tree_dir}/{letter}_{category}_{max_temperature}_{nickname}_t.pkl'
    return f'{tree_dir}/{letter}_{category}_{max_temperature}_t.pkl'

def parse_pickle_filename(filename: str) -> tuple[str, str, float, str] | None:
    # TODO: fix this
    m = re.match(r'([A-Z])_(.+)_(\d+(\.\d+)?)_(.+)_t.pkl', filename)
    if m is None:
        return None
    return m.group(1), m.group(2), float(m.group(3)), m.group(5)

def get_v_filename(tree_dir: str, letter: str, category: str, v_model_name: str) -> str:
    category = make_str_safe(category)
    return f'{tree_dir}/{letter}_{category}_{v_model_name}_v.pkl'

def parse_v_filename(filename: str) -> tuple[str, str] | None:
    m = re.match(r'([A-Z])_(.+)_v.pkl', filename)
    if m is None:
        return None
    return m.group(1), m.group(2)

def create_tree_if_necessary(
        tree_dir: str,
        letter: str,
        category: str,
        max_temperature: float,
        engine: CompletionEngine) -> CompletionNode:
    fname = get_pickle_filename(tree_dir, letter, category, max_temperature, engine.nickname)
    # TODO get rid of max_temperature
    if os.path.exists(fname):
        print('Tree already exists')
        with open(fname, 'rb') as f:
            tree = pickle.load(f)
    else:
        prompt = get_scat_prompt(letter, category, engine.tokenizer)
        start = time.time()
        tree = build_completion_tree(prompt, engine, letter, max_depth=args.depth)
        elapsed = time.time() - start
        print(f'[LOG: TIME] Elapsed time for single tree {engine.nickname}: {elapsed:.2f} seconds')
        tree.standardize_tree()
        print(f'[LOG: SIZE] Number of nodes {engine.nickname}:', len(list(tree.iter_leaves())))
        tree.pickle_tree(fname, MODELS[engine.nickname])
    return tree

def verify_trees(trees: list[CompletionNode], verifier: Verifier, output_dir: str, category: str, letter: str):
    v_filename = get_v_filename(output_dir, letter, category, verifier.engine.nickname)
    if os.path.exists(v_filename):
        print('Verifier already exists')
        with open(v_filename, 'rb') as f:
            verified_dict = pickle.load(f)
            verified_y = verified_dict['yes']
            verified_n = verified_dict['no']
    else:
        verified_y = set()
        verified_n = set()
    for tree in trees:
        for node in tree.iter_leaves():
            if node.text in verified_y or node.text in verified_n:
                continue
            if verifier.verify(node.text, category, letter):
                print('Yes:', node.text)
                verified_y.add(node.text)
            else:
                print('No:', node.text)
                verified_n.add(node.text)
    print('Verified yes:', verified_y)
    print('Verified no:', verified_n)
    verified_dict = {'yes': verified_y, 'no': verified_n}
    with open(v_filename, 'wb') as f:
        print(f'Writing to {v_filename}')
        pickle.dump(verified_dict, f)
    return verified_dict

def get_model_list(models: str, allowed_models: set[str]) -> list[str]:
    if models == 'all':
        return sorted(list(allowed_models))
    model_list = models.split(',')
    for model in model_list:
        if model not in allowed_models:
            raise ValueError(f'Invalid model: {model}')
    return sorted(model_list)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import CompletionEngineMLX as CE, MODELS
    else:
        from completion_hf import CompletionEngineHF as CE, MODELS
    print('[LOG ARGS]', args)
    models = get_model_list(args.models, set(MODELS.keys()))
    if args.job_num >= len(models):
        print(f'[LOG] Job number {args.job_num} is out of range')
        sys.exit(0)
    models = models[args.job_num::args.num_jobs]
    print(f'[LOG] Models to be used: {models}')
    max_temperatures = [MAX_TEMPS[MODELS[model]] for model in models]
    random.seed(0)
    instances = get_random_instances(args.num_instances)
    print(instances)
    tree_map = {}
    start = time.time()
    for nickname, max_temperature in zip(models, max_temperatures):
        print(f'[LOG] Loading model: {nickname} at temp {max_temperature}')
        model_name = MODELS[nickname]
        all_file_names = [get_pickle_filename(args.output_dir, letter, category, max_temperature, nickname) for letter, category in instances]
        if all([os.path.exists(fname) for fname in all_file_names]):
            print('[LOG] All trees already exist for model:', nickname)
            for (letter, category) in instances:
                tree = pickle.load(open(get_pickle_filename(args.output_dir, letter, category, max_temperature, nickname), 'rb'))
                if (letter, category) not in tree_map:
                    tree_map[(letter, category)] = [tree]
                else:
                    tree_map[(letter, category)].append(tree)
            continue
        engine = CE.get_completion_engine(model_name, max_temperature=max_temperature, nickname=nickname, epsilon=1e-5)
        for letter, category in instances:
            print(f'Model: {nickname}; Letter: {letter}; Category: {category}')
            tree = create_tree_if_necessary(args.output_dir, letter, category, max_temperature, engine)
            if (letter, category) not in tree_map:
                tree_map[(letter, category)] = [tree]
            else:
                tree_map[(letter, category)].append(tree)
        print('[LOG] Current memory allocated before cleanup:', torch.cuda.memory_allocated() / (1024 ** 2))
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print('[LOG] Cleared cache for model:', nickname)
            print('[LOG] Current memory allocated after cleanup:', torch.cuda.memory_allocated() / (1024 ** 2))
    elapsed = time.time() - start
    print('Finished generating trees')
    print(f'[LOG: TIME] Total elapsed time for all tree generation: {elapsed:.2f} seconds')
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    allocated_memory = torch.cuda.memory_allocated(device)
    print(f"[LOG] Memory allocated: {allocated_memory / (1024 ** 2)} MB")
    reserved_memory = torch.cuda.memory_reserved(device)
    print(f"[LOG] Memory reserved: {reserved_memory / (1024 ** 2)} MB")

    if args.verifier not in MODELS:
        print(f'[LOG] Skipping verification: {args.verifier}')
        sys.exit(0)


    verifier_model_name = MODELS[args.verifier]
    verifier = Verifier(verifier_model_name, CE, nickname=args.verifier)

    allocated_memory = torch.cuda.memory_allocated(device)
    print(f"[LOG] Memory allocated after loading verifier: {allocated_memory / (1024 ** 2)} MB")
    reserved_memory = torch.cuda.memory_reserved(device)
    print(f"[LOG] Memory reserved after loading verifier: {reserved_memory / (1024 ** 2)} MB")
    start = time.time()
    for (letter, category), trees in tree_map.items():
        verify_trees(trees, verifier, args.output_dir, category, letter)
    elapsed = time.time() - start
    print(f'[LOG: TIME] Total elapsed time for verification: {elapsed:.2f} seconds')
