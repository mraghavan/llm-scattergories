from completion_base import CompletionEngine
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
        'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'llama3.1': 'meta-llama/Llama-3.1-8B-Instruct',
        'llama3.2': 'meta-llama/Llama-3.2-1B-Instruct',
        'smollm': 'HuggingFaceTB/SmolLM-1.7B-Instruct',
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
                torch_dtype=torch.float16
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
        model.to(device)
        CompletionEngineHF.DEVICE = device

    def get_logits_raw(self, model_input: list):
        torch_input = torch.tensor(model_input).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            # TODO use_cache?
            logits = self.model(input_ids = torch_input, use_cache=False).logits
        logits = logits[:, -1, :]
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits)
        return logits

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")['input_ids'].flatten().tolist()

if __name__ == '__main__':
    import random
    from scat_utils import get_random_letter_and_category, get_scat_prompt
    from completion_base import build_completion_tree
    print('Testing completion with HF')
    model_name = MODELS['smollm']
    engine = CompletionEngineHF.get_completion_engine(model_name, max_temperature=0.8, nickname=model_name)
    random.seed(0)
    letter, category = get_random_letter_and_category()
    print("Letter:", letter)
    print("Category:", category)
    prompt = get_scat_prompt(letter, category, engine.tokenizer)
    build_completion_tree(prompt, engine, letter=letter, max_depth=3)
