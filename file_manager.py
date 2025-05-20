from pathlib import Path
import pandas as pd
import re
from typing import Dict, List, Union, Optional, Any
from dataclasses import dataclass
import pickle
import json

@dataclass
class FileLocations:
    """Configuration for directory structure"""
    base_dir: Path
    samples_dir: Path
    info_dir: Path
    plots_dir: Path
    answer_sets_dir: Path
    models_dir: Path
    rankings_dir: Path  # New directory for rankings

    @classmethod
    def from_base(cls, base_dir: Union[str, Path]):
        base = Path(base_dir)
        return cls(
            base_dir=base,
            samples_dir=base / "samples",
            info_dir=base / "info",
            plots_dir=base / "img",
            answer_sets_dir=base / "answer_sets",
            models_dir=base / "models",
            rankings_dir=base / "rankings"  # New directory
        )

    def ensure_dirs(self):
        """Create all directories if they don't exist"""
        for dir_path in [self.samples_dir, self.info_dir, self.plots_dir, 
                        self.answer_sets_dir, self.models_dir, self.rankings_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

class FileManager:
    def __init__(self, locations: FileLocations):
        self.locations = locations
        self.locations.ensure_dirs()

    @classmethod
    def from_base(cls, base_dir: Union[str, Path]):
        return cls(FileLocations.from_base(base_dir))

    @classmethod
    def from_args(cls, samples_dir='', info_dir='', plots_dir=''):
        fl = FileLocations.from_base(Path('.'))
        if samples_dir:
            fl.samples_dir = Path(samples_dir)
        if info_dir:
            fl.info_dir = Path(info_dir)
        if plots_dir:
            fl.plots_dir = Path(plots_dir)
        return cls(fl)

    @staticmethod
    def safe_category(category: str) -> str:
        category = re.sub('[^a-zA-Z0-9_ ]+', '', category)
        category = re.sub(' ', '_', category)
        return category

    def load_from_path(self, fname: Path) -> Any:
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def get_sample_fname(self, letter: str, category: str, model_name: str, temp: float) -> Path:
        category = self.safe_category(category)
        return self.locations.samples_dir / f'{letter}_{category}_{model_name}_{temp}_samples.pkl'

    def parse_sample_fname(self, fname: Path) -> dict:
        pattern = re.compile(r'(\w)_(.*)_([a-z0-9.]+)_(\d+\.\d+)_samples.pkl')
        matched = pattern.match(fname.name)
        assert matched is not None
        letter = matched.group(1)
        category = matched.group(2)
        model = matched.group(3)
        temp = matched.group(4)
        return {
            'letter': letter,
            'category': category,
            'model': model,
            'temperature': float(temp)
            }

    def load_samples(self, letter: str, category: str, model_name: str, temp: float) -> dict:
        fname = self.get_sample_fname(letter, category, model_name, temp)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def write_samples(self, letter: str, category: str, model_name: str, temp: float, data: dict):
        fname = self.get_sample_fname(letter, category, model_name, temp)
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(data, f)

    def get_cache_fname(self, letter: str, category: str, model_name: str) -> Path:
        category = self.safe_category(category)
        return self.locations.samples_dir / f'{letter}_{category}_{model_name}_cache.pkl'

    def load_cache(self, letter: str, category: str, model_name: str) -> dict:
        fname = self.get_cache_fname(letter, category, model_name)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def write_cache(self, letter: str, category: str, model_name: str, data: dict):
        fname = self.get_cache_fname(letter, category, model_name)
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(data, f)

    def get_v_fname(self, letter: str, category: str, v_name: str) -> Path:
        category = self.safe_category(category)
        return self.locations.samples_dir / f'{letter}_{category}_{v_name}_verified.pkl'

    def load_verified(self, letter: str, category: str, v_name: str) -> dict:
        fname = self.get_v_fname(letter, category, v_name)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def write_verified(self, letter: str, category: str, v_name: str, data: dict):
        fname = self.get_v_fname(letter, category, v_name)
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(data, f)

    def parse_v_fname(self, fname: Path) -> dict:
        pattern = re.compile(r'(\w)_(.*)_([a-z0-9.]+)_verified.pkl')
        matched = pattern.match(fname.name)
        assert matched is not None
        letter = matched.group(1)
        category = matched.group(2)
        verifier = matched.group(3)
        return {
            'letter': letter,
            'category': category,
            'verifier': verifier
            }

    def get_info_fname(self, model: str, n: int, gamma: float) -> Path:
        return self.locations.info_dir / f'{model}_n{n}_gamma{gamma:.2f}_info.pkl'

    def load_info(self, model: str, n: int, gamma: float) -> dict:
        fname = self.get_info_fname(model, n, gamma)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def write_info(self, model: str, n: int, gamma: float, data: dict):
        fname = self.get_info_fname(model, n, gamma)
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(data, f)

    def parse_info_fname(self, fname: Path) -> dict:
        pattern = re.compile(r'([a-z0-9.]+)_n(\d+)_gamma(\d+\.\d+)_info.pkl')
        matched = pattern.match(fname.name)
        assert matched is not None
        model = matched.group(1)
        n = int(matched.group(2))
        gamma = float(matched.group(3))
        return {
            'model': model,
            'n': n,
            'gamma': gamma
            }

    def get_pairwise_fname(self, model1: str, model2: str, n: int, gamma: float) -> Path:
        return self.locations.info_dir / f'{model1}_{model2}_n{n}_gamma{gamma:.2f}_pairwise.pkl'

    def load_pairwise(self, model1: str, model2: str, n: int, gamma: float) -> dict:
        fname = self.get_pairwise_fname(model1, model2, n, gamma)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def write_pairwise(self, model1: str, model2: str, n: int, gamma: float, data: list[dict]):
        fname = self.get_pairwise_fname(model1, model2, n, gamma)
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(data, f)

    def parse_pairwise_fname(self, fname: Path) -> dict:
        pattern = re.compile(r'([a-z0-9.]+)_([a-z0-9.]+)_n(\d+)_gamma(\d+\.\d+)_pairwise.pkl')
        matched = pattern.match(fname.name)
        assert matched is not None
        model1 = matched.group(1)
        model2 = matched.group(2)
        n = int(matched.group(3))
        gamma = float(matched.group(4))
        return {
            'model1': model1,
            'model2': model2,
            'n': n,
            'gamma': gamma
            }

    def get_all_samples(
            self,
            letter: str='',
            category: str='',
            model: str='',
            models: list = [],
            max_temp: float=0.0) -> pd.DataFrame:
        assert not (model and models)
        category = self.safe_category(category)
        all_samples = list(self.locations.samples_dir.glob(f"*_samples.pkl"))
        data = [self.parse_sample_fname(fname) for fname in all_samples]
        for d, fname in zip(data, all_samples):
            d['fname'] = fname
        df = pd.DataFrame(data)
        if letter:
            df = df[df['letter'] == letter]
        if category:
            df = df[df['category'] == category]
        if model:
            df = df[df['model'] == model]
        if models:
            df = df[df['model'].isin(models)]
        if max_temp > 0:
            df = df[df['temperature'] <= max_temp]
        return df

    def get_all_verified(self, letter: str='', category: str='', verifier: str='') -> pd.DataFrame:
        category = self.safe_category(category)
        all_samples = list(self.locations.samples_dir.glob(f"*_verified.pkl"))
        data = [self.parse_v_fname(fname) for fname in all_samples]
        for d, fname in zip(data, all_samples):
            d['fname'] = fname
        df = pd.DataFrame(data)
        if letter:
            df = df[df['letter'] == letter]
        if category:
            df = df[df['category'] == category]
        if verifier:
            df = df[df['verifier'] == verifier]
        return df

    def get_all_info(
            self,
            model: str='',
            models: list = [],
            n: int=0,
            gamma: float=0.0,
            ) -> pd.DataFrame:
        assert not (model and models)
        all_samples = list(self.locations.info_dir.glob(f"*_info.pkl"))
        data = [self.parse_info_fname(fname) for fname in all_samples]
        for d, fname in zip(data, all_samples):
            d['fname'] = fname
        df = pd.DataFrame(data)
        if model:
            df = df[df['model'] == model]
        if models:
            df = df[df['model'].isin(models)]
        if n > 0:
            df = df[df['n'] == n]
        if gamma > 0:
            df = df[df['gamma'] == gamma]
        return df

    def get_all_pairwise_info(self, model1: str='', model2: str='', n: int=0, gamma: float=0.0) -> pd.DataFrame:
        all_samples = list(self.locations.info_dir.glob(f"*_pairwise.pkl"))
        data = [self.parse_pairwise_fname(fname) for fname in all_samples]
        if len(data) == 0:
            return pd.DataFrame()
        # always in sorted order
        if model1 and model2 and model1 > model2:
            model1, model2 = model2, model1
        for d, fname in zip(data, all_samples):
            d['fname'] = fname
        df = pd.DataFrame(data)
        if model1:
            df = df[df['model1'] == model1]
        if model2:
            df = df[df['model2'] == model2]
        if n > 0:
            df = df[df['n'] == n]
        if gamma > 0:
            df = df[df['gamma'] == gamma]
        return df

    def get_answer_set_fname(self, letter: str, category: str, min_count: int) -> Path:
        category = self.safe_category(category)
        return self.locations.answer_sets_dir / f'{letter}_{category}_min{min_count}_answers.pkl'

    def write_answer_set(self, letter: str, category: str, min_count: int, answers: set):
        fname = self.get_answer_set_fname(letter, category, min_count)
        fname.parent.mkdir(parents=True, exist_ok=True)  # Create answer_sets directory if it doesn't exist
        with open(fname, 'wb') as f:
            print(f'Writing to {fname}')
            pickle.dump(answers, f)

    def load_answer_set(self, letter: str, category: str, min_count: int) -> set:
        fname = self.get_answer_set_fname(letter, category, min_count)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def get_model_config_fname(self, model_id: str) -> Path:
        return self.locations.models_dir / f"{model_id}.json"

    def load_model_config(self, model_id: str) -> dict:
        """Load a single model configuration"""
        fname = self.get_model_config_fname(model_id)
        with open(fname, 'r') as f:
            return json.load(f)

    def write_model_config(self, model_id: str, config: dict):
        """Write a model configuration"""
        fname = self.get_model_config_fname(model_id)
        with open(fname, 'w') as f:
            json.dump(config, f, indent=2)

    def get_all_model_configs(self) -> pd.DataFrame:
        """Load all model configurations into a DataFrame"""
        config_files = list(self.locations.models_dir.glob("*.json"))
        configs = []
        for fname in config_files:
            with open(fname, 'r') as f:
                config = json.load(f)
                configs.append(config)
        return pd.DataFrame(configs)

    def get_ranking_fname(self, letter: str, category: str, model_name: str) -> Path:
        """Get filename for rankings of a specific model, letter, and category"""
        category = self.safe_category(category)
        return self.locations.rankings_dir / f'{letter}_{category}_{model_name}_rankings.pkl'

    def write_rankings(self, letter: str, category: str, model_name: str, rankings: dict):
        """Write rankings (NLLs) to file"""
        fname = self.get_ranking_fname(letter, category, model_name)
        with open(fname, 'wb') as f:
            print(f'Writing rankings to {fname}')
            pickle.dump(rankings, f)

    def load_rankings(self, letter: str, category: str, model_name: str) -> dict:
        """Load rankings (NLLs) from file"""
        fname = self.get_ranking_fname(letter, category, model_name)
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def get_all_rankings(self, letter: str='', category: str='', model: str='') -> pd.DataFrame:
        """Get DataFrame of all ranking files"""
        category = self.safe_category(category) if category else ''
        pattern = f"{letter}_{category}_{model}_rankings.pkl" if all([letter, category, model]) else "*_rankings.pkl"
        all_rankings = list(self.locations.rankings_dir.glob(pattern))
        
        data = []
        for fname in all_rankings:
            parts = fname.stem.split('_')
            if len(parts) >= 3:
                data.append({
                    'letter': parts[0],
                    'category': parts[1],
                    'model': parts[2],
                    'fname': fname
                })
        
        return pd.DataFrame(data)

    def get_all_ranking_files(self, letter: str, category: str) -> list[tuple[str, Path]]:
        """
        Get all ranking files for a given letter and category.
        
        Args:
            letter: The starting letter
            category: The category name
        
        Returns:
            List of tuples containing (model_name, file_path) for each ranking file
        """
        rankings_dir = self.locations.rankings_dir
        category = self.safe_category(category)
        pattern = f"{letter}_{category}_*_rankings.pkl"
        
        results = []
        for ranking_file in rankings_dir.glob(pattern):
            model_name = ranking_file.stem.split('_')[-2]
            results.append((model_name, ranking_file))
        
        return results

if __name__ == '__main__':
    from scat_utils import get_deterministic_instances
    base_dir = Path('./')
    letter, category = get_deterministic_instances(1)[0]
    nickname = 'llama3.2'
    loc = FileLocations.from_base(base_dir)
    fm = FileManager(loc)
    print(fm.locations.samples_dir)
    print(fm.get_sample_fname(letter, category, nickname, 1.0))
    print(fm.get_all_samples(model=nickname))
    print(fm.get_all_verified(verifier='qwen2.5'))
    print(fm.get_all_info(model='llama3.2', gamma=1.0))
    print(fm.get_all_pairwise_info(model2='nemotron'))
