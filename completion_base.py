from numpy.typing import ArrayLike
from typing import Union
import re
import numpy as np
import pickle

class CompletionEngine():
    def __init__(
            self,
            model,
            tokenizer,
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

    def get_logits(self, prompt_tokens: list) -> tuple[ArrayLike, ArrayLike]:
        cache = None
        logits = self.get_logits_raw(prompt_tokens)
        if self.max_temperature == 0:
            token = np.argmax(logits)
            logit = logits[token]
            return np.array(token), np.array(logit)
        probs = softmax_temperature(logits, self.max_temperature)
        # sort probs in ascending order
        sorted_indices = np.argsort(probs)
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        mask = np.array(cumulative_probs > 1 - self.top_p, copy=False)
        np_indices = np.array(sorted_indices, copy=False)
        tokens = np_indices[mask]
        np_logits = np.array(logits, copy=False)
        return tokens, np_logits[tokens]

    def get_logits_raw(self, model_input) -> np.ndarray: ...
    # this method should be implemented by the subclass

    def encode_prompt(self, prompt: str) -> Union["torch.Tensor", "mlx.core.array"]: ...
    # this method should be implemented by the subclass

    @staticmethod
    def get_completion_engine(
            model_name: str,
            epsilon: float=1e-4,
            max_temperature: float=1.0,
            top_p: float=0.95,
            nickname: str='',
            ) -> 'CompletionEngine': ...

class CompletionNode():
    def __init__(
            self,
            tokens: list,
            text: str,
            EOS: int,
            EOS_str: str,
            prob: float = 1.0,
            max_temperature: float = 1.0,
            depth: int = 0,
            parent: Union['CompletionNode', None] = None,
        ):
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
        self.model_name = ''

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

    def pickle_tree(self, filename: str, model_name: str = ''):
        print('Pickling tree to', filename)
        self.model_name = model_name
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def iter_leaves(self):
        if not self.children:
            yield self
        for child in self.children:
            yield from child.iter_leaves()

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

def standardize_str(s: str, EOS_str: str) -> str:
    s = s.replace(EOS_str, '')
    s = re.sub('[^a-zA-Z ]+', '', s).lower().strip()
    return re.sub('  ', ' ', s)

def build_completion_tree(prompt: str, engine: CompletionEngine, letter: str = '', max_depth: int = 3):
    EOS_str = str(engine.tokenizer.eos_token)
    EOS_id = int(engine.tokenizer.eos_token_id)
    tokenized_prompt = engine.encode_prompt(prompt)
    root = CompletionNode(
            [],
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
        # tokens, logits = engine.get_logits(mod.concatenate([tokenized_prompt, mod.array(node.tokens, dtype=tokenized_prompt.dtype)]))
        # print(type(tokenized_prompt))
        # print(tokenized_prompt.shape)
        tokens, logits = engine.get_logits(tokenized_prompt + node.tokens)
        probs = softmax_temperature(logits, engine.max_temperature)
        for token, prob, logit in zip(tokens, probs, logits):
            if prob * node.prob < engine.epsilon:
                continue
            # Consider using engine.tokenizer.eos_token_id somewhere
            token = int(token)
            child_tokens = list(node.tokens) + [token]
            child = CompletionNode(
                tokens=child_tokens,
                text=engine.tokenizer.decode(child_tokens),
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
