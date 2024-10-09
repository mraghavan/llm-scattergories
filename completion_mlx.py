from completion_base import CompletionEngine, build_completion_tree
import numpy as np
import mlx.core as mx
from mlx_lm.utils import load

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

class CompletionEngineMLX(CompletionEngine):
    @staticmethod
    def get_completion_engine(
            model_name: str,
            epsilon: float=1e-4,
            max_temperature: float=1.0,
            top_p: float=0.95,
            nickname: str='',
            ) -> 'CompletionEngine':
        model, tokenizer = load(model_name)
        return CompletionEngineMLX(model, tokenizer, epsilon, max_temperature, top_p, nickname)

    def get_logits_raw(self, model_input) -> np.ndarray:
        cache = None
        logits = self.model(model_input[None], cache=cache)
        logits = logits[:, -1, :]
        logits = logits.squeeze(0)
        logits = np.array(logits)
        return logits

    def encode_prompt(self, prompt: str) -> mx.array:
        return mx.array(self.tokenizer.encode(prompt))

if __name__ == '__main__':
    import random
    from scat_utils import get_random_letter_and_category, get_scat_prompt
    print('Testing completion with MLX')
    model_name = MODELS['llama3.2']
    # print('Loading model:', model_name)
    # model, tokenizer = load(model_name)
    # print('Model loaded')
    engine = CompletionEngineMLX.get_completion_engine(model_name, max_temperature=0.8, nickname=model_name)
    random.seed(0)
    letter, category = get_random_letter_and_category()
    print("Letter:", letter)
    print("Category:", category)
    prompt = get_scat_prompt(letter, category, engine.tokenizer)
    print("Prompt:")
    print(prompt)
    build_completion_tree(prompt, engine, mx, letter=letter, max_depth=3)
