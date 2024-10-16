from scat_utils import get_eval_prompt, is_yes
from typing import Type
from completion_base import CompletionEngine

class Verifier():
    def __init__(self, model_name: str, CEClass: Type[CompletionEngine], nickname=''):
        self.engine = CEClass.get_completion_engine(model_name, max_temperature=0, nickname=nickname)

    def verify(self, answer: str, category: str, letter: str):
        # TODO batch this
        if not answer.lower().startswith(letter.lower()):
            return False
        if answer.lower().startswith(letter.lower() + ' '):
            return False
        prompt = get_eval_prompt(answer, category, self.engine.tokenizer)
        prompt_tokens = self.engine.encode_prompt(prompt)
        tokens, _ = self.engine.get_logits(prompt_tokens)
        result = self.engine.tokenizer.decode(tokens)
        EOS = str(self.engine.tokenizer.eos_token)
        return is_yes(result, EOS)

if __name__ == '__main__':
    answers = ['rabbit', 'dog', 'rat', 'rhino', 'racooon', 'rasdfas', 'rotten', 'rope']
    category = 'Animals'
    from completion_hf import CompletionEngineHF, MODELS
    nickname = 'llama3.2'
    model_name = MODELS[nickname]
    verifier = Verifier(model_name, CompletionEngineHF, nickname=nickname)
    for answer in answers:
        print(answer, verifier.verify(answer, category, 'R'))
    del verifier
    from completion_mlx import CompletionEngineMLX, MODELS
    # model, tokenizer = load('mlx-community/Meta-Llama-3.1-8B-Instruct-8bit')
    nickname = 'llama3.1'
    model_name = MODELS[nickname]
    verifier = Verifier(model_name, CompletionEngineMLX)
    for answer in answers:
        print(answer, verifier.verify(answer, category, 'R'))
    # prompt = get_eval_prompt(answer, category, verifier.engine.tokenizer)
    # print(prompt)
    # print(verifier.verify(answer, category, 'R'))
    # engine = CompletionEngine(model, tokenizer, max_temperature=0)
    # tokens, logits = engine.get_logits(prompt)
    # print(tokens)
    # print(engine.tokenizer.decode(tokens))
