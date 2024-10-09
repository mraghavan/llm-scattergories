from mlx_lm.utils import load
import mlx.nn as nn
from mlx_lm.tokenizer_utils import TokenizerWrapper
import mlx.core as mx
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Union
import pickle
from scat_utils import get_random_letter_and_category, standardize_str
import torch
# import torch
# from transformers import pipeline
# import transformers
# from transformers import TextClassificationPipeline

MODELS = {
        'llama3.1': 'mlx-community/Meta-Llama-3.1-8B-Instruct-8bit',
        'llama3': 'mlx-community/Meta-Llama-3-8B-Instruct-8bit',
        # 'gemma2': 'mlx-community/gemma-2-9b-8bit', (not an instruct mocdel)
        'qwen1.5': 'mlx-community/Qwen1.5-7B-Chat-4bit',
        # 'openelm1_1': 'mlx-community/OpenELM-1_1B-Instruct-8bit', (no chat template)
        'smollm': 'mlx-community/SmolLM-1.7B-Instruct-fp16',
        'nemo': 'mlx-community/Mistral-Nemo-Instruct-2407-4bit',
        # 'phi3': 'mlx-community/Phi-3-small-8k-instruct-AQ4_32', (haven't tried yet)
        'phi3.5': 'mlx-community/Phi-3.5-mini-instruct-bf16',
        # "yi1.5": 'mlx-community/Yi-1.5-9B-Chat-4bit', (haven't tried yet)
        'llama3.2': 'mlx-community/Llama-3.2-3B-Instruct-8bit',
          }

# EOT = "<|eot_id|>"
# EOM = "<|eom_id|>"

class CompletionNode():
    def __init__(
            self,
            tokens: mx.array,
            text: str,
            EOS: int,
            EOS_str: str,
            # tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
            prob: float = 1.0,
            max_temperature: float = 1.0,
            depth: int = 0,
            parent: 'CompletionNode' = None,
        ):
        # For some reason LLama outputs EOM instead of EOT
        # if text.endswith(EOM):
            # text = text.replace(EOM, EOS)
        self.tokens = tokens
        self.text=text
        self.prob = prob
        self.max_temperature = max_temperature
        self.depth = depth
        self.parent = parent
        self.children = []
        self.logits = []
        self.EOS = EOS
        self.EOS_str = EOS_str

    # @property
    # def detokenize(self):
        # if len(self.tokens) == 0:
            # return ''
        # return self.tokenizer.decode(self.tokens.tolist())

    def add_child(self, child: 'CompletionNode', logit: float):
        self.children.append(child)
        self.logits.append(logit)

    def probs(self, temperature):
        assert temperature <= self.max_temperature
        return softmax_temperature(np.array(self.logits, dtype=np.float64), temperature)

    def get_dist(self, temperature):
        # Need to call standardize_tree first
        assert temperature <= self.max_temperature
        if not self.children:
            return {self.text: 1.0}
        dist = {}
        for child, prob in zip(self.children, self.probs(temperature)):
            child_dist = child.get_dist(temperature)
            for k, v in child_dist.items():
                if k in dist:
                    dist[k] += prob * v
                else:
                    dist[k] = prob * v
        return dist

    def __repr__(self):
        return f'Node({self.text}, {self.prob}, {self.depth})'

    def standardize_tree(self):
        self.text = standardize_str(self.text, self.EOS_str)
        for child in self.children:
            child.standardize_tree()

    def pickle_tree(self, filename: str):
        print('Pickling tree to', filename)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def iter_leaves(self):
        if not self.children:
            yield self
        for child in self.children:
            yield from child.iter_leaves()

class CompletionEngine():
    def __init__(
            self,
            model: nn.Module,
            tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
            epsilon: float=1e-4,
            max_temperature: float=1.0,
            top_p: float=0.95,
            nickname: str='',
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.max_temperature = max_temperature
        self.top_p = top_p
        self.nickname = nickname

    def get_logits(self, prompt_tokens: mx.array):
        cache = None
        logits = self.model(prompt_tokens[None], cache=cache)
        logits = logits[:, -1, :]
        logits = logits.squeeze(0)
        if self.max_temperature == 0:
            token = mx.argmax(logits)
            logit = logits[token]
            return np.array(token), np.array(logit)
        probs = mx.softmax(logits / self.max_temperature)
        # sort probs in ascending order
        sorted_indices = mx.argsort(probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = mx.cumsum(sorted_probs)

        mask = np.array(cumulative_probs > 1 - self.top_p, copy=False)
        np_indices = np.array(sorted_indices, copy=False)
        tokens = np_indices[mask]
        np_logits = np.array(logits, copy=False)
        return tokens, np_logits[tokens]

        # select tokens with cumulative probs below threshold
        # top_probs = mx.where(
            # cumulative_probs > 1 - self.top_p,
            # sorted_probs,
            # mx.zeros_like(sorted_probs),
        # )
        # np_indices = np.array(sorted_indices, copy=False)
        # np_probs = np.array(top_probs, copy=False)
        # prec_probs = np.array(np_probs[np_probs > 0], dtype=np.float64)
        # tokens = np_indices[np_probs > 0]
        # renormalized_probs = prec_probs / np.sum(prec_probs)
        # assert np.isclose(np.sum(renormalized_probs), 1)
        # return tokens, np.array(logits, copy=False)[tokens]
        # tokens_and_probs = {self.tokenizer.decode(token): prob for token, prob in zip(tokens, renormalized_probs)}
        # # tokens = mx.where(renormalized_probs > 0, sorted_indices.squeeze(0), -1*mx.ones_like(sorted_indices.squeeze(0)))
        # # tokens = sorted_indices.squeeze(0)[[sorted_token, sorted_token2]]
        # # tokens = np.nonzero(tokens)[0]
        # # print('tokens', tokens)
        # # decoded = tokenizer._decode(tokens)
        # # for token, txt in zip(tokens, decoded):
            # # print(token, f'\t"{txt}"')
        # # logits, cache = model(y[None], cache=cache)
        # return tokens_and_probs

# class CompletionEngineHF(CompletionEngine):
    # def get_logits(self, prompt: str):
        # prompt_tokens = mx.array(self.tokenizer.encode(prompt))
        # y = prompt_tokens
        # cache = None
        # logits = self.model(y[None], cache=cache)
        # logits = logits[:, -1, :]
        # logits = logits.squeeze(0)
        # # return np.array(logits, copy=False)
        # if self.max_temperature == 0:
            # token = mx.argmax(logits)
            # logit = logits[token]
            # return np.array(token), np.array(logit)
        # probs = mx.softmax(logits / self.max_temperature)
        # # sort probs in ascending order
        # sorted_indices = mx.argsort(probs)
        # sorted_probs = probs[sorted_indices]
        # cumulative_probs = mx.cumsum(sorted_probs)

        # mask = np.array(cumulative_probs > 1 - self.top_p, copy=False)
        # np_indices = np.array(sorted_indices, copy=False)
        # tokens = np_indices[mask]
        # np_logits = np.array(logits, copy=False)
        # return tokens, np_logits[tokens]

def build_completion_tree(prompt: str, engine: CompletionEngine, letter: str = '', max_depth: int = 3):
    EOS_str = str(engine.tokenizer.eos_token)
    EOS_id = int(engine.tokenizer.eos_token_id)
    tokenized_prompt = mx.array(engine.tokenizer.encode(prompt))
    root = CompletionNode(
            mx.array([], dtype=tokenized_prompt.dtype),
            '',
            EOS=EOS_id,
            EOS_str=EOS_str,
            prob=1.0,
            max_temperature=engine.max_temperature)
    nodes = [root]
    while nodes:
        node = nodes.pop(0)
        print('Expanding node', node)
        if node.depth == max_depth:
            continue
        if len(node.tokens) > 0 and node.tokens[-1] == EOS_id:
            continue
        # slight optimization to only consider completions starting with the letter
        if len(node.tokens) > 0 and not node.text.strip().lower().startswith(letter.lower()):
            continue
        tokens, logits = engine.get_logits(mx.concatenate([tokenized_prompt, node.tokens]))
        probs = softmax_temperature(logits, engine.max_temperature)
        for token, prob, logit in zip(tokens, probs, logits):
            if prob * node.prob < engine.epsilon:
                continue
            # Consider using engine.tokenizer.eos_token_id somewhere
            token = int(token)
            child_tokens = mx.concatenate([node.tokens, mx.array([token], dtype=node.tokens.dtype)])
            child = CompletionNode(
                tokens=child_tokens,
                text=engine.tokenizer.decode(child_tokens.tolist()),
                EOS=EOS_id,
                EOS_str=EOS_str,
                prob=prob * node.prob,
                max_temperature=engine.max_temperature,
                depth=node.depth + 1,
                parent=node,
            )
            node.add_child(child, logit)
            nodes.append(child)
    return root

def softmax_temperature(logits: np.ndarray, temperature: float):
    # numerically stable softmax
    if temperature == 0:
        probs = np.zeros_like(logits)
        probs[np.argmax(logits)] = 1
        return probs
    logits = logits / temperature
    logits -= np.max(logits)
    exp_logits = np.exp(logits, dtype=np.float64)
    probs = exp_logits / np.sum(exp_logits)
    return probs

def get_completion_dist_from_tree(node: CompletionNode, temperature: float, prob: float=1.0):
    assert temperature <= node.max_temperature
    if not node.children:
        return {node.text: prob}
    probs = node.probs(temperature)
    completions = {}
    for child, p in zip(node.children, probs):
        completions.update(get_completion_dist_from_tree(child, temperature, prob*p))
    return completions

if __name__ == '__main__':
    # model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # model = AutoModelForCausalLM.from_pretrained(model)
    # # pipe = ProbPipeline(model=model, tokenizer=tokenizer)
    # with torch.no_grad():
        # outputs = model(torch.tensor(tokenizer.encode("Hello, my name is"), dtype=torch.long).unsqueeze(0)).logits
    # print(outputs)
    # 1/0
    # # pipe = transformers.pipeline(
        # # "text-generation",
        # # model=model,
        # # torch_dtype=torch.float16,
        # # device_map="auto",
        # # )
    # messages = [
        # {
            # "role": "system",
            # "content": "You are a friendly chatbot who always responds in the style of a pirate",
        # },
        # {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
    # ]
    # assert pipe.tokenizer is not None
    # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # print(outputs[0]["generated_text"])
    
    # tokenizer = AutoTokenizer.from_pretrained('neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16')
    # pipe = pipeline(
    # "text-generation",
    # model="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
    # tokenizer=tokenizer,
    # max_new_tokens=512,
    # temperature=0.7,
    # top_p=0.95,
    # repetition_penalty=1.15,
    # device='cpu'
    # )
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.to(device)
    # text = "On what date was the Declaration of Independence officially signed?"
    # inputs = tokenizer(text, return_tensors="pt").to(device)
    # outputs = model.generate(**inputs, max_new_tokens=10, num_return_sequences=5, temperature=0.8)
    # print(outputs)
    model_name = 'mlx-community/phi-3.5-mini-instruct-bf16'
    model, tokenizer = load(model_name)
    engine = CompletionEngine(model, tokenizer, max_temperature=1.5, nickname='phi3.5')
    from scat_utils import get_scat_prompt
    prompt = get_scat_prompt('S', 'Gifts/Presents', engine.tokenizer)
    build_completion_tree(prompt, engine, 'S')
    # from mlx_lm import generate

    # for _ in range(10):
        # print(generate(model, tokenizer, prompt=get_scat_prompt('S', 'Gifts/Presents', tokenizer) + "Snow", max_tokens=2, temp=1.5))

    # model_name = 'mlx-community/gemma-2-9b-8bit'
    # model, tokenizer = load(model_name)
    # engine = CompletionEngine(model, tokenizer, max_temperature=.8, nickname='qwen1.5')
    # letter, category = get_random_letter_and_category()
    # print(f'Letter: {letter}; Category: {category}')
    # prompt = QUESTION_TEMPLATE.format(letter=letter, category=category)
    # tree = build_completion_tree(prompt, engine, letter)
    # temps = [.1, .5, .8]
    # for temp in temps:
        # completions = get_completion_dist_from_tree(tree, temp)
        # print('Completions at temperature', temp)
        # for comp, prob in completions.items():
            # print(comp, prob)
        # print('Total probability', sum(completions.values()))
        # print()
