from completion_hf import MODELS
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    for nickname in MODELS:
        print(nickname)
        model_name = MODELS[nickname]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(nickname, sum(p.numel() for p in model.parameters()))
