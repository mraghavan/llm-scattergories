import time
import re
import os
import pickle
import random
from scat_utils import get_random_instances
from verifier import Verifier
import argparse
from scat_utils import MAX_TEMPS, get_model_list
from generate_samples import get_temps, get_sample_fname
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='')
parser.add_argument('--num_instances', '-n', type=int, default=20)
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--input_dir', '-i', type=str, default='./samples')
parser.add_argument('--output_dir', '-o', type=str, default='./samples')
parser.add_argument('--batch_size', '-b', type=int, default=4)

def get_v_fname(output_dir: str, letter: str, category: str, v_name: str) -> str:
    category = re.sub('[^a-zA-Z0-9 ]+', '', category)
    category = re.sub(' ', '_', category)
    return os.path.join(output_dir, f'{letter}_{category}_{v_name}_verified.pkl')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS, CompletionEngineMLX as CEClass
    else:
        from completion_hf import MODELS, CompletionEngineHF as CEClass
    models = get_model_list(args.models, set(MODELS.keys()))
    random.seed(0)
    instances = get_random_instances(args.num_instances)
    verifier_model_name = MODELS[args.verifier]
    verifier = Verifier(verifier_model_name, CEClass, nickname=args.verifier)
    for letter, category in instances:
        print(f'Category: {category}, Letter: {letter}')
        to_be_verified = set()
        # load all responses for all models
        for nickname in models:
            model_name = MODELS[nickname]
            temps = get_temps(MAX_TEMPS[model_name])
            for temp in temps:
                temp = round(temp, 3)
                fname = get_sample_fname(args.input_dir, letter, category, nickname, temp)
                if not os.path.exists(fname):
                    print(f'File {fname} does not exist')
                    continue
                with open(fname, 'rb') as f:
                    samples = pickle.load(f)
                dist = samples['dist']
                to_be_verified.update(dist.keys())
        vfname = get_v_fname(args.output_dir, letter, category, args.verifier)
        if os.path.exists(vfname):
            with open(vfname, 'rb') as f:
                verified = pickle.load(f)
                verified_yes = verified['yes']
                verified_no = verified['no']
        else:
            verified_yes = set()
            verified_no = set()
        verification_batch = set()
        for answer in to_be_verified:
            if answer in verified_yes or answer in verified_no:
                continue
            verification_batch.add(answer)
        print('Number to be verified:', len(verification_batch))
        start = time.time()
        responses = verifier.verify_batch(verification_batch, category, letter, args.batch_size)
        elapsed = time.time() - start
        print('Number of responses:', len(responses))
        print('Time elapsed:', elapsed)
        for answer, is_yes in responses.items():
            if is_yes:
                verified_yes.add(answer)
            else:
                verified_no.add(answer)
        with open(vfname, 'wb') as f:
            print(f'Saving verified samples to {vfname}')
            pickle.dump({'yes': verified_yes, 'no': verified_no}, f)
