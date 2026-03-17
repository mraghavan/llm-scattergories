from pathlib import Path
from file_manager import FileManager
import pandas as pd
from scat_utils import get_deterministic_instances, standardize_str
from completion_hf import CompletionEngineHF, MODELS
import numpy as np
from make_model_configs import PROMPT_REGISTRY
import argparse
import random
import gc
import json

def load_model_configs(from_config: str = '') -> pd.DataFrame:
    """
    Load all model configurations from the models directory.
    
    Returns:
        DataFrame containing all model configurations
    """
    if from_config:
        config_path = Path(from_config)
        if not config_path.exists():
            raise ValueError(f"Config path does not exist: {from_config}")
        if config_path.is_file():
            with open(config_path, 'r') as f:
                df = pd.DataFrame([json.load(f)])
                return df[df['model'] != 'smollm'].reset_index(drop=True)
        configs = []
        for fname in sorted(config_path.glob("*.json")):
            with open(fname, 'r') as f:
                configs.append(json.load(f))
        df = pd.DataFrame(configs)
        return df[df['model'] != 'smollm'].reset_index(drop=True)
    fm = FileManager.from_base(Path('./'))
    df = fm.get_all_model_configs()
    return df[df['model'] != 'smollm'].reset_index(drop=True)

def load_candidate_answers(fm: FileManager, letter: str, category: str, min_count: int = 1) -> set:
    """
    Load candidate answers for a given letter and category.
    
    Args:
        fm: FileManager instance
        letter: The starting letter
        category: The category name
        min_count: Minimum count threshold for including an answer
    
    Returns:
        Set of candidate answers
    """
    return fm.load_answer_set(letter, category, min_count)

def load_model(model_name: str, max_temperature: float = 1.0, top_p: float = 0.95) -> CompletionEngineHF:
    """
    Load a model using CompletionEngineHF.
    
    Args:
        model_name: Name of the model (must be a key in MODELS dict)
        max_temperature: Maximum temperature for sampling
        top_p: Top-p sampling parameter
    
    Returns:
        CompletionEngineHF instance
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(MODELS.keys())}")
    
    model_path = MODELS[model_name]
    engine = CompletionEngineHF.get_completion_engine(
        model_name=model_path,
        max_temperature=max_temperature,
        top_p=top_p,
        nickname=model_name
    )
    return engine

def compute_answer_nll(
        prompt: str,
        answer: str,
        allowed_tokens: set,
        allowed_starting_tokens: set,
        completion_engine: CompletionEngineHF,
        temperature: float = 1.0,
        logits_cache: dict = None,
        ) -> float:
    """
    Compute the negative log likelihood of an answer given a prompt.
    
    Args:
        prompt: The input prompt
        answer: The answer to compute likelihood for
        allowed_tokens: Tokens permitted anywhere in the answer
        allowed_starting_tokens: Tokens permitted at the first answer position
        completion_engine: The CompletionEngineHF instance
        temperature: Temperature for probability computation
        logits_cache: Optional dictionary to cache logits between calls
    
    Returns:
        float: Negative log likelihood of the answer
    """
    # Encode the prompt
    prompt_tokens = completion_engine.encode_prompt(prompt)
    
    # Capitalize the first letter of the answer
    answer = answer[0].upper() + answer[1:]
    
    def score_token_path(answer_tokens: list[int]) -> float:
        if not answer_tokens:
            return np.inf
        if answer_tokens[0] not in allowed_starting_tokens:
            return np.inf
        if any(token not in allowed_tokens for token in answer_tokens):
            return np.inf

        total_log_prob = 0.0
        current_input = prompt_tokens.copy()
        for i, token in enumerate(answer_tokens):
            input_key = tuple(current_input)
            if logits_cache is not None and input_key in logits_cache:
                logits = logits_cache[input_key]
            else:
                logits = completion_engine.get_logits_raw(current_input)
                if logits_cache is not None and total_log_prob > np.log(0.01):
                    logits_cache[input_key] = logits

            logits = logits / temperature
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            if i == 0:
                mask = np.zeros_like(probs, dtype=bool)
                mask[list(allowed_starting_tokens)] = True
                restricted_mass = np.sum(probs[mask])
                if not np.isfinite(restricted_mass) or restricted_mass <= 0:
                    return np.inf
                probs = np.where(mask, probs / restricted_mass, 0.0)

            token_prob = probs[token]
            if not np.isfinite(token_prob) or token_prob <= 0:
                return np.inf

            total_log_prob += np.log(token_prob)
            current_input.append(token)

        if not np.isfinite(total_log_prob):
            return np.inf
        return -total_log_prob

    # Score only the answer tokens. For chat-template models, forcing EOS can
    # incorrectly make an otherwise valid answer impossible.
    answer_tokens = completion_engine.tokenizer.encode(answer, add_special_tokens=False)
    if not answer_tokens:
        return np.inf

    canonical_nll = score_token_path(answer_tokens)
    if np.isfinite(canonical_nll):
        return canonical_nll

    # Pragmatic first-token fallback: some models reach the same normalized
    # surface form through a different first tokenization path (e.g. "Q" + ...).
    target_norm = standardize_str(answer)
    best_nll = np.inf
    for first_token in allowed_starting_tokens:
        piece = completion_engine.tokenizer.decode([first_token])
        piece_norm = standardize_str(piece)
        if not piece_norm or not target_norm.startswith(piece_norm):
            continue
        suffix = answer[len(piece):]
        if piece != answer[:len(piece)]:
            continue
        suffix_tokens = completion_engine.tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
        candidate_tokens = [first_token] + suffix_tokens
        decoded_candidate = completion_engine.tokenizer.decode(candidate_tokens)
        if standardize_str(decoded_candidate) != target_norm:
            continue
        candidate_nll = score_token_path(candidate_tokens)
        if candidate_nll < best_nll:
            best_nll = candidate_nll
    if np.isfinite(best_nll):
        return best_nll

    return np.inf

def get_all_jobs(model_configs, instances):
    """
    Generate list of all (letter, category, model_config) triplets to process.
    
    Args:
        model_configs: DataFrame of model configurations
        instances: List of (letter, category) tuples
    
    Returns:
        List of (letter, category, model_config) tuples
    """
    all_jobs = []
    for letter, category in instances:
        for _, model_config in model_configs.iterrows():
            all_jobs.append((letter, category, model_config))
    return all_jobs

def process_job(letter, category, model_config, fm, min_count, prev_engine=None):
    """Process a single job (letter, category, model_config combination)"""
    model_id = model_config['id']
    
    # Get all candidate answers
    candidates = load_candidate_answers(fm, letter, category, min_count=min_count)
    
    print(f"\n{letter} {category}:")
    print(f"Found {len(candidates)} candidate answers")
    
    # Load existing rankings if they exist
    rankings_path = fm.get_ranking_fname(letter, category, model_id, min_count)
    if rankings_path.exists():
        print(f"Loading existing rankings for {letter} {category} {model_id} min{min_count}")
        model_nlls = fm.load_rankings(letter, category, model_id, min_count)
        # Find answers that haven't been processed yet
        new_answers = set(candidates) - set(model_nlls.keys())
        if not new_answers:
            print(f"No new answers to process for {letter} {category} {model_id} min{min_count}")
            return
        print(f"Found {len(new_answers)} new answers to process")
    else:
        model_nlls = {}
        new_answers = candidates
    
    print(model_id)
    temperature = model_config['temperature']
    
    print(f"\nLoading model: {model_id} at temperature {temperature}")
    model_name = model_config['model']
    if prev_engine is not None and prev_engine.nickname == model_name:
        engine = prev_engine
        engine.max_temperature = temperature
    elif prev_engine is not None:
        del prev_engine.model
        del prev_engine.tokenizer
        gc.collect()
        engine = load_model(model_name, max_temperature=temperature)
    else:
        engine = load_model(model_name, max_temperature=temperature)
    
    # Get the prompt function from the registry
    prompt_fn = PROMPT_REGISTRY[model_config['prompt_function']]
    prompt = prompt_fn(letter, category, engine.tokenizer)
    allowed_tokens, allowed_starting_tokens = engine.get_allowed_tokens(letter)
    
    # Initialize cache for this model's answers
    logits_cache = {}
    
    # Compute NLL for each new candidate answer
    for answer in sorted(new_answers):
        nll = compute_answer_nll(
            prompt,
            answer,
            allowed_tokens,
            allowed_starting_tokens,
            engine,
            temperature,
            logits_cache,
        )
        print(f"NLL for {answer}: {nll:.2f}")
        model_nlls[answer] = nll
    
    # Save the updated rankings with min_count parameter
    fm.write_rankings(letter, category, model_id, min_count, model_nlls)
    
    # Print some sample results
    print(f"Sample NLLs for {model_id} (min_count={min_count}):")
    for answer, nll in list(model_nlls.items())[:5]:
        print(f"{answer}: {nll:.2f}")
    return engine

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Evaluate answer likelihoods for scattergories')
    parser.add_argument('--min-count', type=int, default=100,
                      help='Minimum count threshold for including an answer (default: 100)')
    parser.add_argument('--job-num', type=int, default=0,
                      help='Job number (0-based index) (default: 0)')
    parser.add_argument('--num-jobs', type=int, default=1,
                      help='Total number of jobs (default: 1)')
    parser.add_argument('--num-instances', '-n', type=int, default=1,
                      help='Number of instances to evaluate (default: 1)')
    parser.add_argument('--from-config', '-c', type=str, default='',
                      help='Load model configs from a JSON file or directory (default: models dir)')
    args = parser.parse_args()

    # Initialize FileManager
    fm = FileManager.from_base(Path('./'))
    
    # Load all model configurations
    model_configs = load_model_configs(args.from_config)
    print(f"Loaded {len(model_configs)} model configurations")
    
    # Get deterministic instances
    instances = get_deterministic_instances(args.num_instances)
    
    # Get all jobs and select subset for this job
    all_jobs = get_all_jobs(model_configs, instances)
    # apply a deterministic shuffle using Python's random module
    random.seed(0)
    random.shuffle(all_jobs)
    my_jobs = all_jobs[args.job_num::args.num_jobs]
    
    print(f"Processing {len(my_jobs)} jobs out of {len(all_jobs)} total jobs")
    # sort my_jobs by model_id
    my_jobs.sort(key=lambda x: x[2]['id'])
    
    # Process each job
    prev_engine = None
    for letter, category, model_config in my_jobs:
        prev_engine = process_job(letter, category, model_config, fm, args.min_count, prev_engine)

if __name__ == "__main__":
    main() 
