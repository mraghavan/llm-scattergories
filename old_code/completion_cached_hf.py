from completion_base import CompletionEngine, CompletionNode, softmax_temperature
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
        'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama3.2': 'meta-llama/Llama-3.2-1B-Instruct',
        'smollm': 'HuggingFaceTB/SmolLM-1.7B-Instruct',
          }


class CompletionEngineHFCached(CompletionEngine):
    DEVICE = None
    @staticmethod
    def get_completion_engine(
            model_name: str,
            epsilon: float=1e-4,
            max_temperature: float=1.0,
            top_p: float=0.95,
            nickname: str='',
            ) -> 'CompletionEngine':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
        else:
            # For some reason half precision doesn't work on CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
            )
        return CompletionEngineHFCached(model, tokenizer, epsilon, max_temperature, top_p, nickname)

    def __init__(self, model, tokenizer, epsilon, max_temperature, top_p, nickname):
        super().__init__(model, tokenizer, epsilon, max_temperature, top_p, nickname)
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("GPU is not available. Using CPU.")
            device = torch.device("cpu")
        model.to(device)
        CompletionEngineHFCached.DEVICE = device

    def get_logits_raw(self, model_input: list):
        # TODO speed up by batching
        torch_input = torch.tensor(model_input).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            # TODO use_cache?
            logits = self.model(input_ids = torch_input, use_cache=False).logits
        logits = logits[:, -1, :]
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits) # shape: (vocab_size,)
        return logits

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")['input_ids'].flatten().tolist()

    def set_prefix_prompt(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.DEVICE)
        past_key_values = None
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        self.past_key_values = outputs.past_key_values

    def get_logits_with_prefix(self, prompt_tokens: list):
        logits = self.get_logits_raw_with_prefix(prompt_tokens)
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

    def get_logits_raw_with_prefix(self, model_input: list):
        # TODO speed up by batching
        torch_input = torch.tensor(model_input).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            outputs = self.model(input_ids = torch_input, past_key_values=self.past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits)
        return logits

    def get_logits_with_prefix_batch(self, prompt_tokens_list: list[list]):
        logits = self.get_logits_raw_with_prefix_batch(prompt_tokens_list)
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

    def get_logits_raw_with_prefix_batch(self, model_input: list[list]):
        torch_input = torch.tensor(model_input).to(self.DEVICE)
        print('torch_input shape:', torch_input.shape) # should be (batch_size, seq_len)
        print(torch_input)
        with torch.no_grad():
            outputs = self.model(input_ids = torch_input, past_key_values=self.past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        # make this (batch_size, vocab_size)
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits)
        print('logits shape:', logits.shape)
        return logits

def build_completion_tree(prompt: str, engine: CompletionEngineHFCached, letter: str = '', max_depth: int = 3):
    EOS_str = str(engine.tokenizer.eos_token)
    EOS_id = int(engine.tokenizer.eos_token_id)
    engine.set_prefix_prompt(prompt)
    # tokenized_prompt = engine.encode_prompt(prompt)
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
        tokens, logits = engine.get_logits_with_prefix(node.tokens)
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

def build_completion_tree_batched(prompt: str, engine: CompletionEngineHFCached, letter: str = '', max_depth: int = 3, batch_size: int = 10):
    EOS_str = str(engine.tokenizer.eos_token)
    EOS_id = int(engine.tokenizer.eos_token_id)
    engine.set_prefix_prompt(prompt)
    # tokenized_prompt = engine.encode_prompt(prompt)
    root = CompletionNode(
            [],
            '',
            EOS=EOS_id,
            EOS_str=EOS_str,
            prob=1.0,
            max_temperature=engine.max_temperature)
    nodes = []
    jobs_waiting = [root]
    while nodes or jobs_waiting:
        if len(jobs_waiting) >= batch_size or (not nodes and jobs_waiting) or (nodes and jobs_waiting and nodes[0].depth > jobs_waiting[0].depth):
            token_list = [job.tokens for job in jobs_waiting]
            tokens, logits = engine.get_logits_with_prefix_batch(token_list)
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
        node = nodes.pop(0)
        print('Expanding node', node)
        if node.depth == max_depth:
            continue
        if len(node.tokens) > 0 and node.tokens[-1] == EOS_id:
            continue
        # slight optimization to only consider completions starting with the letter
        if len(node.tokens) > 0 and not node.text.strip().lower().startswith(letter.lower()):
            continue
    return root

if __name__ == '__main__':
    import random
    from scat_utils import get_random_instances, get_scat_prompt
    from completion_base import build_completion_tree
    print('Testing completion with HF and caching')
    model_name = MODELS['smollm']
    engine = CompletionEngineHFCached.get_completion_engine(model_name, max_temperature=1.5, nickname=model_name)
    random.seed(0)
    instances = get_random_instances(3)
    for letter, category in instances:
        print("Letter:", letter)
        print("Category:", category)
        prompt = get_scat_prompt(letter, category, engine.tokenizer)
        start = time.time()
        build_completion_tree_batched(prompt, engine, letter=letter, max_depth=3)
        elapsed = time.time() - start
        print(f"[LOG] Elapsed time: {elapsed:.2f} seconds")
