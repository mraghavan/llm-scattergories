from completion_base import CompletionEngine, build_completion_tree
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
        'llama3.1': 'meta-llama/Meta-Llama-3-8B-Instruct',
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.float16
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

    def get_logits_raw(self, model_input):
        with torch.no_grad():
            logits = self.model(**model_input)
        logits = logits[:, -1, :]
        logits = logits.squeeze(0).to('cpu')
        logits = np.array(logits)
        return logits

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.DEVICE)

if __name__ == '__main__':
    import random
    from scat_utils import get_random_letter_and_category, get_scat_prompt
    print('Testing completion with HF')
    model_name = MODELS['smollm']
    # print('Loading model:', model_name)
    # model, tokenizer = load(model_name)
    # print('Model loaded')
    engine = CompletionEngineHF.get_completion_engine(model_name, max_temperature=0.5, nickname=model_name)
    random.seed(0)
    letter, category = get_random_letter_and_category()
    print("Letter:", letter)
    print("Category:", category)
    prompt = get_scat_prompt(letter, category, engine.tokenizer)
    inputs = engine.encode_prompt(prompt)
    print(type(inputs))
    # TODO do we want to use cache?
    inputs['use_cache'] = False
    print(inputs)
    with torch.no_grad():
        outputs = engine.model(**inputs)
    print(outputs.logits.shape)
    logits = outputs.logits[:, -1, :]
    logits = logits.squeeze(0).to('cpu')
    logits = np.array(logits)
    best_token = np.argmax(logits)
    print(best_token)
    prompt_with_response = torch.cat([
            inputs['input_ids'].flatten(),
            torch.tensor([best_token])
            ])
    print(engine.tokenizer.decode(prompt_with_response))
    # print(outputs.shape)
    # build_completion_tree(prompt, engine, np, letter=letter, max_depth=3)
