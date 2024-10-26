import string
import re
import random

# GENERIC_PROMPT_TEMPLATE = string.Template("<|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
# ONE_SHOT_PROMPT_TEMPLATE = string.Template("<|start_header_id|>system<|end_header_id|>\n\n${system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")

# SCAT_INSTRUCTIONS = "You are a helpful assistant. Answer in as few words as possible, with no explanations."
# SCAT_PROMPT = ''
# with open('./scattegories_prompt.txt', 'r') as f:
    # SCAT_PROMPT = f.read()
# QUESTION_TEMPLATE = GENERIC_PROMPT_TEMPLATE.safe_substitute(system=SCAT_INSTRUCTIONS).format(prompt=SCAT_PROMPT)

# EOT = "<|eot_id|>"
EOM = "<|eom_id|>"
QUESTION_FILE = './questions.txt'
# EVAL_PROMPT = ''
# with open('./verifier_prompt.txt', 'r') as f:
    # EVAL_PROMPT = f.read()
# EVAL_TEMPLATE = GENERIC_PROMPT_TEMPLATE.safe_substitute(system=SCAT_INSTRUCTIONS).format(prompt=EVAL_PROMPT)

SCAT_INSTRUCTIONS = 'We are playing a game of Scattegories. I will give you a letter and a category, and you will give me a word or short phrase that starts with that letter and matches the category. For example, if I say "Fruit" and "A," you could say "Apple" or "Apricot."'

EVAL_INSTRUCTIONS = "You are judging a Scattergories game. I will give you a category and an answer. You will tell me whether that answer fits the given category. You are strict but fair. You do not accept incomplete answers. Answer with either the word 'yes' or 'no'."

SCAT_EXAMPLES = [
        ("Letter: C\nCategory: Countries", "Canada"),
        ("Letter: X\nCategory: Instruments", "Xylophone"),
        ]

EVAL_EXAMPLES = [
        ("Category: Things that are red\nAnswer: blood", "yes"),
        ("Category: Animals\nAnswer: rock", "no"),
        ]

MAX_TEMPS = {
        'meta-llama/Meta-Llama-3-8B-Instruct': 1.7,
        'meta-llama/Llama-3.1-8B-Instruct': 1.8,
        'meta-llama/Llama-3.2-1B-Instruct': 1.4,
        'HuggingFaceTB/SmolLM-1.7B-Instruct': 1.2,
        'Qwen/Qwen2-7B': 1.5,
        'google/gemma-2-2b-it': 1.5,
        'microsoft/Phi-3.5-mini-instruct': 2.2,
        'mistralai/Mistral-7B-Instruct-v0.3': 1.6,
        'nvidia/Nemotron-Mini-4B-Instruct': 1.7,
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

def get_model_list(models: str, allowed_models: set[str]) -> list[str]:
    if models == 'all':
        return sorted(list(allowed_models))
    model_list = models.split(',')
    for model in model_list:
        if model not in allowed_models:
            raise ValueError(f'Invalid model: {model}')
    return sorted(model_list)

def get_model_list(models: str, allowed_models: set[str]) -> list[str]:
    if models == 'all':
        return sorted(list(allowed_models))
    model_list = models.split(',')
    for model in model_list:
        if model not in allowed_models:
            raise ValueError(f'Invalid model: {model}')
    return sorted(model_list)

def get_random_letter():
    return random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def get_random_category():
    return get_random_categories(1)[0]

def get_random_letter_and_category():
    category = get_random_category()
    letter = get_random_letter()
    return letter, category

def get_random_categories(n: int):
    # taken from https://swellgarfo.com/scattergories/
    with open(QUESTION_FILE, 'r') as f:
        categories = f.readlines()
    for i in range(len(categories)):
        categories[i] = categories[i].strip()
    return random.sample(categories, n)

def get_random_instances(n: int):
    return [get_random_letter_and_category() for _ in range(n)]

def get_deterministic_instances(n: int):
    # save random seed
    seed = random.getstate()
    random.seed(0)
    instances = get_random_instances(n)
    # restore random seed
    random.setstate(seed)
    return instances

def get_eval_prompt(answer: str, category: str, tokenizer):
    messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in as few words as possible, with no explanations."},
            {"role": "user", "content": EVAL_INSTRUCTIONS},
            {"role": "assistant", "content": "I understand."},
            ]
    for ans, response in EVAL_EXAMPLES:
        messages.append({"role": "user", "content": ans})
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": f"Category: {category}\nAnswer: {answer}"})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    return str(text)
    # return EVAL_TEMPLATE.format(answer=answer, category=category)

def get_scat_prompt(letter: str, category: str, tokenizer):
    messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in as few words as possible, with no explanations."},
            {"role": "user", "content": SCAT_INSTRUCTIONS},
            {"role": "assistant", "content": "I understand."},
            ]
    for q, a in SCAT_EXAMPLES:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Letter: {letter}\nCategory: {category}"})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    return str(text)
    # return QUESTION_TEMPLATE.format(letter=letter, category=category)

def is_yes(response: str, EOS: str):
    response = standardize_str(response, EOS)
    return response == 'yes' or response == 'y'

def standardize_str(s: str, EOS_str: str='') -> str:
    s = s.replace(EOS_str, '')
    s = re.sub('[^a-zA-Z ]+', '', s).lower().strip()
    return re.sub('  ', ' ', s)
