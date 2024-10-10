import time
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
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_hf', '-f', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./trees')

MAX_TEMPS = {
        'meta-llama/Meta-Llama-3-8B-Instruct': 2.5,
        'meta-llama/Llama-3.1-8B-Instruct': 2.5,
        'meta-llama/Llama-3.2-1B-Instruct': 2.0,
        'HuggingFaceTB/SmolLM-1.7B-Instruct': 1.2,
        'Qwen/Qwen2-7B': 1.5,
        'google/gemma-2-2b-it': 1.5,
        'microsoft/Phi-3.5-mini-instruct': 1.8,
        'mistralai/Mistral-7B-Instruct-v0.3': 1.5,
        'mlx-community/Meta-Llama-3.1-8B-Instruct-8bit': 1.5,
        'mlx-community/Meta-Llama-3-8B-Instruct-8bit': 1.5,
        # 'gemma2': 'mlx-community/gemma-2-9b-8bit', (not an instruct mocdel)
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
    if os.path.exists(fname):
        print('Tree already exists')
        with open(fname, 'rb') as f:
            tree = pickle.load(f)
    else:
        prompt = get_scat_prompt(letter, category, engine.tokenizer)
        start = time.time()
        tree = build_completion_tree(prompt, engine, letter, max_depth=3)
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

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_hf:
        from completion_hf import CompletionEngineHF as CE, MODELS
    else:
        from completion_mlx import CompletionEngineMLX as CE, MODELS
    print('[LOG ARGS]', args)
    models = args.models.split(',')
    if len(models) == 1 and models[0] == 'all':
        models = sorted(list(MODELS.keys()))
    for model in models:
        if model not in MODELS:
            raise ValueError(f'Invalid model: {model}')
    max_temperatures = [MAX_TEMPS[MODELS[model]] for model in models]
    random.seed(0)
    instances = get_random_instances(args.num_instances)
    print(instances)
    tree_map = {}
    verifier_model_name = MODELS[args.verifier]
    start = time.time()
    for nickname, max_temperature in zip(models, max_temperatures):
        print(f'Loading model: {nickname}')
        model_name = MODELS[nickname]
        engine = CE.get_completion_engine(model_name, max_temperature=max_temperature, nickname=nickname)
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
