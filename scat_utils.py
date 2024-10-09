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

def get_eval_prompt(answer: str, category: str, tokenizer):
    messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in as few words as possible, with no explanations."},
            {"role": "user", "content": EVAL_INSTRUCTIONS},
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

def standardize_str(s: str, EOS_str: str) -> str:
    s = s.replace(EOS_str, '')
    s = re.sub('[^a-zA-Z ]+', '', s).lower().strip()
    return re.sub('  ', ' ', s)

if __name__ == '__main__':
    from completion_utils import MODELS
    import random
    from mlx_lm.utils import load
    model_name = random.choice(list(MODELS.values()))
    print(model_name)
    model, tokenizer = load(model_name)
    scat_prompt = get_scat_prompt('A', 'Food', tokenizer)
    print(scat_prompt)
    eval_prompt = get_eval_prompt('apple', 'Food', tokenizer)
    print(eval_prompt)
