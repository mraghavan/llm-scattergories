import time
import re
import os
import pickle
from pathlib import Path
from verifier import Verifier
import argparse
from scat_utils import MAX_TEMPS, get_model_list
from file_manager import FileManager
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str)
parser.add_argument('--from-config', '-c', type=str,
                    help='Load model configuration from a config file')
parser.add_argument('--verifier', '-v', type=str, default='')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--input_dir', '-i', type=str, default='./samples')
parser.add_argument('--batch_size', '-b', type=int, default=4)
parser.add_argument('--job_num', '-j', type=int, default=0)
parser.add_argument('--num_jobs', '-t', type=int, default=1)

def get_v_fname(output_dir: str, letter: str, category: str, v_name: str) -> str:
    category = re.sub('[^a-zA-Z0-9 ]+', '', category)
    category = re.sub(' ', '_', category)
    return os.path.join(output_dir, f'{letter}_{category}_{v_name}_verified.pkl')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Add validation for required arguments
    if bool(args.models) == bool(args.from_config):
        parser.error("Either --models or --from-config must be specified, but not both")
        
    if args.use_mlx:
        from completion_mlx import MODELS, CompletionEngineMLX as CEClass
    else:
        from completion_hf import MODELS, CompletionEngineHF as CEClass
    fm = FileManager.from_args(samples_dir=args.input_dir)
    
    if args.from_config:
        # Load all configs from directory
        configs_df = fm.get_all_model_configs()
        if configs_df.empty:
            raise ValueError(f"No configs found in {fm.locations.models_dir}")
        models = []
        for _, row in configs_df.iterrows():
            config = row.to_dict()
            nickname = config['model']
            model_id = config['id']
            models.append((nickname, model_id))
    else:
        models = get_model_list(args.models, set(MODELS.keys()))
        models = [(m, m) for m in models]
        
    df = fm.get_all_samples(models=models)
    instances = sorted(list(df[['letter', 'category']].drop_duplicates().itertuples(index=False, name=None)))
    if args.num_jobs > 1:
        instances = instances[args.job_num::args.num_jobs]
    verifier_model_name = MODELS[args.verifier]
    verifier = Verifier(verifier_model_name, CEClass, nickname=args.verifier)
    for letter, category in instances:
        print(f'Category: {category}, Letter: {letter}')
        to_be_verified = set()
        # load all responses for all models
        for nickname, model_id in models:
            model_name = MODELS[nickname]
            all_samples = fm.get_all_samples(model=model_id, max_temp=MAX_TEMPS[model_name], letter=letter, category=category)
            for _, row in all_samples.iterrows():
                fname = Path(row['fname']) # type: ignore
                with open(fname, 'rb') as f:
                    samples = pickle.load(f)
                dist = samples['dist']
                to_be_verified.update(dist.keys())
        vfname = fm.get_v_fname(letter, category, args.verifier)
        if os.path.exists(vfname):
            verified = fm.load_verified(letter, category, args.verifier)
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
        fm.write_verified(letter, category, args.verifier, {'yes': verified_yes, 'no': verified_no})
