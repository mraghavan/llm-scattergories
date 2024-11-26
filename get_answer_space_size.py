from completion_hf import MODELS, CompletionEngineHF
import gc
import torch
from scat_utils import get_model_list
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    models = get_model_list(args.models, set(MODELS.keys()))
    for model in models:
        engine = CompletionEngineHF.get_completion_engine(model, max_temperature=0.0, nickname='nemotron', epsilon=0)
        allowed_tokens, allowed_starting_tokens = engine.get_allowed_tokens(letter='A')
        print(model)
        print(len(allowed_tokens))
        print(len(allowed_starting_tokens)**6)
        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
