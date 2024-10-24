import re
import pickle
import os

class FileManager():
    def __init__(self):
        self.sample_dir = ''
        self.cache_dir = ''
        self.verified_dir = ''
        self.info_dir = ''

    @staticmethod
    def safe_category(category: str) -> str:
        category = re.sub('[^a-zA-Z0-9 ]+', '', category)
        category = re.sub(' ', '_', category)
        return category

    def get_sample_fname(self, letter: str, category: str, model_name: str, temp: float) -> str:
        assert self.sample_dir != ''
        category = self.safe_category(category)
        return os.path.join(self.sample_dir, f'{letter}_{category}_{model_name}_{temp}_samples.pkl')

    def get_cache_fname(self, letter: str, category: str, model_name: str) -> str:
        assert self.cache_dir != ''
        category = self.safe_category(category)
        return os.path.join(self.cache_dir, f'{letter}_{category}_{model_name}_cache.pkl')

    def get_v_fname(self, letter: str, category: str, v_name: str) -> str:
        assert self.verified_dir != ''
        category = self.safe_category(category)
        return os.path.join(self.verified_dir, f'{letter}_{category}_{v_name}_verified.pkl')

    def get_info_fname(self, model: str, n: int, gamma: float) -> str:
        assert self.info_dir != ''
        return os.path.join(self.info_dir, f'{model}_n{n}_gamma{gamma:.2f}_info.pkl')

    def get_samples_for_model(self, model: str):
        all_samples = {}
        all_files = os.listdir(self.info_dir)
        for fname in all_files:
            if not fname.endswith('_samples.pkl'):
                continue


    def get_info_for_model(self, model: str) -> dict[tuple[int, float], dict] :
        all_info = {}
        all_files = os.listdir(self.info_dir)
        for fname in all_files:
            if not fname.endswith('_info.pkl'):
                continue
            if not fname.startswith(model):
                continue
            with open(fname, 'rb') as f:
                info = pickle.load(f)
                n = info['n']
                gamma = info['gamma']
                all_info[(n, gamma)] = info
        return all_info

FM = FileManager()

