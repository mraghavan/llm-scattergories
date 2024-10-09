from mlx_lm.utils import load
import os
import re
import pickle
import random
from completion_utils import build_completion_tree, CompletionEngine, CompletionNode, MODELS
from scat_utils import get_random_letter_and_category, get_scat_prompt
from verifier import Verifier
VERIFIER_MODEL_NAME = 'mlx-community/Meta-Llama-3.1-8B-Instruct-8bit'

def make_str_safe(s: str) -> str:
    return re.sub('/', ' ', s).strip()

def get_random_instances(n: int):
    return [get_random_letter_and_category() for _ in range(n)]

def get_pickle_filename(letter: str, category: str, max_temperature: float, nickname: str):
    category = make_str_safe(category)
    if nickname:
        return f'./pickle_trees/{letter}_{category}_{max_temperature}_{nickname}.pkl'
    return f'./pickle_trees/{letter}_{category}_{max_temperature}.pkl'

def parse_pickle_filename(filename: str):
    # TODO: fix this
    m = re.match(r'([A-Z])_(.+)_(\d+(\.\d+)?)_(.+).pkl', filename)
    if m is None:
        return None
    return m.group(1), m.group(2), float(m.group(3)), m.group(5)

def get_v_filename(letter: str, category: str):
    category = make_str_safe(category)
    return f'./pickle_trees/{letter}_{category}_v.pkl'

def parse_v_filename(filename: str):
    m = re.match(r'([A-Z])_(.+)_v.pkl', filename)
    if m is None:
        return None
    return m.group(1), m.group(2)

def create_tree_if_necessary(letter: str, category: str, max_temperature: float, engine: CompletionEngine):
    fname = get_pickle_filename(letter, category, max_temperature, engine.nickname)
    if os.path.exists(fname):
        print('Tree already exists')
        with open(fname, 'rb') as f:
            tree = pickle.load(f)
    else:
        prompt = get_scat_prompt(letter, category, engine.tokenizer)
        tree = build_completion_tree(prompt, engine, letter, max_depth=3)
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
    NUM_INSTANCES = 20
    random.seed(0)
    instances = get_random_instances(1000)
    instances = instances[:NUM_INSTANCES]
    max_temperature = 1.5
    print(instances)
    tree_map = {}
    for nickname, model_name in MODELS.items():
        model, tokenizer = load(model_name)
        engine = CompletionEngine(model, tokenizer, max_temperature=max_temperature, nickname=nickname)
        for letter, category in instances:
            print(f'Model: {nickname}; Letter: {letter}; Category: {category}')
            tree = create_tree_if_necessary(letter, category, max_temperature, engine)
            if (letter, category) not in tree_map:
                tree_map[(letter, category)] = [tree]
            else:
                tree_map[(letter, category)].append(tree)
        del engine
    verifier = Verifier(*load(VERIFIER_MODEL_NAME))
    for (letter, category), trees in tree_map.items():
        verified_dict = verify_trees(trees, verifier, category, letter)
        with open(get_v_filename(letter, category), 'wb') as f:
            print(f'Writing to {get_v_filename(letter, category)}')
            pickle.dump(verified_dict, f)
