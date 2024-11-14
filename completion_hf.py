from completion_base import CompletionEngine
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
        'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama3.2': 'meta-llama/Llama-3.2-1B-Instruct',
        'smollm': 'HuggingFaceTB/SmolLM-1.7B-Instruct',
        # 'qwen2': 'Qwen/Qwen2-7B', # (use 2.5 instead)
        'gemma2': 'google/gemma-2-2b-it', # (no system role in chat template)
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
        'phi3.5': 'microsoft/Phi-3.5-mini-instruct',
        'nemotron': 'nvidia/Nemotron-Mini-4B-Instruct',
        # 'minitron': 'nvidia/Mistral-NeMo-Minitron-8B-Instruct', # (too big for gpu)
        'qwen2.5': 'Qwen/Qwen2.5-7B-Instruct', # (similar to mistral, don't need both)
          }

NO_BATCH = {
        'llama3',
        'llama3.1',
        }


class CompletionEngineHF(CompletionEngine):
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
                torch_dtype=torch.float16,
                )
        else:
            # For some reason half precision doesn't work on CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
            )
        return CompletionEngineHF(model, tokenizer, epsilon, max_temperature, top_p, nickname)

    def __init__(self, model, tokenizer, epsilon, max_temperature, top_p, nickname):
        super().__init__(model, tokenizer, epsilon, max_temperature, top_p, nickname)
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"GPU is available. Using GPU: {torch.cuda.get_device_name(0)}")
            device = torch.device("cuda")
        else:
            print("GPU is not available. Using CPU.")
            device = torch.device("cpu")
        print('Current pad token', self.tokenizer.pad_token)
        if not self.tokenizer.pad_token and self.nickname not in NO_BATCH:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        model.to(device)
        CompletionEngineHF.DEVICE = device

    def get_logits_raw(self, model_input: list):
        torch_input = torch.tensor(model_input).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            logits = self.model(input_ids = torch_input, use_cache=False).logits
        logits = logits[:, -1, :]
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits) # shape: (vocab_size,)
        return logits

    def get_logits_raw_batch(self, model_input: list[list]) -> np.ndarray:
        if self.nickname in NO_BATCH:
            all_logits = [self.get_logits_raw(x) for x in model_input]
            return np.array(all_logits)
        max_len = max(len(x) for x in model_input)
        padded_model_inputs = []
        attention_mask = []
        idxs = []
        for x in model_input:
            if self.tokenizer.padding_side == 'right':
                idxs.append(len(x) - 1)
                x = x + [self.tokenizer.pad_token_id] * (max_len - len(x))
                attention_mask.append([1] * len(x) + [0] * (max_len - len(x)))
            elif self.tokenizer.padding_side == 'left':
                x = [self.tokenizer.pad_token_id] * (max_len - len(x)) + x
                attention_mask.append([0] * (max_len - len(x)) + [1] * len(x))
                idxs.append(max_len - 1)
            else:
                raise ValueError('padding_side must be left or right')
            padded_model_inputs.append(x)

        torch_input = torch.tensor(padded_model_inputs).to(self.DEVICE)
        torch_attention_mask = torch.tensor(attention_mask).to(self.DEVICE)
        idxs = np.array(idxs)
        with torch.no_grad():
            outputs = self.model(input_ids = torch_input, use_cache=False, attention_mask=torch_attention_mask)
        logits = np.array(outputs.logits.to('cpu'))
        logits = logits[np.arange(len(model_input)), idxs, :]
        return logits

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")['input_ids'].flatten().tolist()

if __name__ == '__main__':
    import random
    from scat_utils import get_random_instances, get_scat_prompt
    from completion_base import build_completion_tree
    print('Testing completion with HF')
    model_name = MODELS['llama3.1']
    engine = CompletionEngineHF.get_completion_engine(model_name, max_temperature=1.5, nickname=model_name)
    random.seed(0)
    instances = get_random_instances(3)
    for letter, category in instances:
        print("Letter:", letter)
        print("Category:", category)
        prompt = get_scat_prompt(letter, category, engine.tokenizer)
        start = time.time()
        build_completion_tree(prompt, engine, letter=letter, max_depth=3)
        elapsed = time.time() - start
        print(f"[LOG] Elapsed time: {elapsed:.2f} seconds")
