import pickle
import os
from collections import Counter
from scat_utils import standardize_str

if __name__ == '__main__':
    folder = './samples/'
    all_files = os.listdir(folder)
    for fname in all_files:
        if fname.endswith('_samples.pkl'):
            new_dist = Counter()
            print(fname)
            full_path = os.path.join(folder, fname)
            with open(full_path, 'rb') as f:
                info = pickle.load(f)
            dist = info['dist']
            for s, count in dist.items():
                new_dist[standardize_str(s)] += count
            assert sum(new_dist.values()) == sum(dist.values())
            info['dist'] = new_dist
            with open(full_path, 'wb') as f:
                pickle.dump(info, f)
