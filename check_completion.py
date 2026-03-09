from file_manager import FileManager
from scat_utils import MAX_TEMPS
from completion_hf import MODELS
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fm = FileManager.from_base('.')
    all_samples = fm.get_all_samples()
    for model in all_samples['model'].unique():
        all_samples_model = all_samples[all_samples['model'] == model]
        max_temp = MAX_TEMPS[MODELS[model]]
        all_samples_model = all_samples_model[all_samples_model['temperature'] <= max_temp]
        print(model, max(all_samples_model['temperature'].unique()))
        all_unfinished = []
        for _, row in all_samples_model.iterrows():
            info = fm.load_from_path(row['fname'])
            all_unfinished.append(info['unfinished']/info['num_samples'])
    print(f'Average unfinished: {sum(all_unfinished)/len(all_unfinished)}')
    plt.hist(all_unfinished, bins=20)
    plt.show()
