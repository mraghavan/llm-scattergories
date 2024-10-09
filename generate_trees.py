import sys
import time
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
parser.add_argument('--max_temperature', '-t', type=float, default=1.5)
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_hf', '-f', action='store_true', default=False)
parser.add_argument('--output_dir', '-o', type=str, default='./trees')
args = parser.parse_args()
if args.use_hf:
    from completion_hf import CompletionEngineHF as CE, MODELS
else:
    from completion_mlx import CompletionEngineMLX as CE, MODELS

def make_str_safe(s: str) -> str:
    return re.sub('/', ' ', s).strip()

def get_pickle_filename(tree_dir: str, letter: str, category: str, max_temperature: float, nickname: str) -> str:
    category = make_str_safe(category)
    if nickname:
        return f'{tree_dir}/{letter}_{category}_{max_temperature}_{nickname}.pkl'
    return f'{tree_dir}/{letter}_{category}_{max_temperature}.pkl'

def parse_pickle_filename(filename: str) -> tuple[str, str, float, str] | None:
    # TODO: fix this
    m = re.match(r'([A-Z])_(.+)_(\d+(\.\d+)?)_(.+).pkl', filename)
    if m is None:
        return None
    return m.group(1), m.group(2), float(m.group(3)), m.group(5)

def get_v_filename(tree_dir: str, letter: str, category: str) -> str:
    category = make_str_safe(category)
    return f'{tree_dir}/{letter}_{category}_v.pkl'

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
        print(f'Elapsed time: {elapsed:.2f} seconds')
        tree.standardize_tree()
        tree.pickle_tree(fname)
    return tree

def verify_tree(tree: CompletionNode, verifier: Verifier, category: str, letter: str):
    if os.path.exists(get_v_filename(letter, category)):
        print('Verifier already exists')
        with open(get_v_filename(letter, category), 'rb') as f:
            verified_dict = pickle.load(f)
            verified_y = verified_dict['yes']
            verified_n = verified_dict['no']
    else:
        verified_y = set()
        verified_n = set()
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
    return verified_dict

def verify_trees(trees: list[CompletionNode], verifier: Verifier, category: str, letter: str):
    if os.path.exists(get_v_filename(letter, category)):
        print('Verifier already exists')
        with open(get_v_filename(letter, category), 'rb') as f:
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
    return verified_dict

if __name__ == '__main__':
    # print(args)
    # 1/0
    models = args.models.split(',')
    for model in models:
        if model not in MODELS:
            raise ValueError(f'Invalid model: {model}')
    random.seed(0)
    instances = get_random_instances(args.num_instances)
    max_temperature = args.max_temperature
    print(instances)
    tree_map = {}
    VERIFIER_MODEL_NAME = MODELS[args.verifier]
    start = time.time()
    for nickname in models:
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
        del engine
    elapsed = time.time() - start
    print('Finished generating trees')
    print(f'Total elapsed time: {elapsed:.2f} seconds')
    sys.exit(0)
    verifier = Verifier(*load(VERIFIER_MODEL_NAME))
    for (letter, category), trees in tree_map.items():
        verified_dict = verify_trees(trees, verifier, category, letter)
        with open(get_v_filename(letter, category), 'wb') as f:
            print(f'Writing to {get_v_filename(letter, category)}')
            pickle.dump(verified_dict, f)
