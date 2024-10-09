# from mlx_lm.utils import load
# import mlx.core as mx
from scat_utils import get_eval_prompt, is_yes
# from completion_utils import CompletionEngine

class Verifier():
    def __init__(self, model, tokenizer):
        self.engine = CompletionEngine(model, tokenizer, max_temperature=0)

    def verify(self, answer: str, category: str, letter: str):
        if not answer.startswith(letter.lower()):
            return False
        if answer.startswith(letter.lower() + ' '):
            return False
        prompt = get_eval_prompt(answer, category, self.engine.tokenizer)
        prompt_tokens = mx.array(self.engine.tokenizer.encode(prompt))
        tokens, _ = self.engine.get_logits(prompt_tokens)
        result = self.engine.tokenizer.decode(tokens)
        EOS = str(self.engine.tokenizer.eos_token)
        return is_yes(result, EOS)

if __name__ == '__main__':
    answer = 'Fish'
    category = 'Animals'
    model, tokenizer = load('mlx-community/Meta-Llama-3.1-8B-Instruct-8bit')
    prompt = get_eval_prompt(answer, category, tokenizer)
    print(prompt)
    engine = CompletionEngine(model, tokenizer, max_temperature=0)
    tokens, logits = engine.get_logits(prompt)
    print(tokens)
    print(engine.tokenizer.decode(tokens))
