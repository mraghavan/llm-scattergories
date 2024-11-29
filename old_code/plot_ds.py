import matplotlib.pyplot as plt
from collections import Counter
import os
from analyze_games import load_games_for_model
from generate_trees import MAX_TEMPS, get_v_filename, get_model_list
import numpy as np
import pickle
from completion_base import CompletionNode
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--models', '-m', type=str, required=True)
parser.add_argument('--verifier', '-v', type=str, default='llama3.1')
parser.add_argument('--use_mlx', '-x', action='store_true', default=False)
parser.add_argument('--tree_dir', '-r', type=str, default='./trees')


def get_binary_quality(tree: CompletionNode, verified_y: set) -> np.ndarray:
    binary_quality = [node.text in verified_y for node in tree.iter_leaves()]
    binary_quality = np.array(binary_quality)
    return binary_quality

def upsample_smooth_and_average_bqs(bqs: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(bq) for bq in bqs)
    upsampled_bqs = []
    for bq in bqs:
        upsampled = np.interp(np.linspace(0, 1, max_len), np.linspace(0, 1, len(bq)), bq)
        upsampled_bqs.append(upsampled)
    upsampled_bqs = np.array(upsampled_bqs)
    avg_bq = np.mean(upsampled_bqs, axis=0)
    smoothing = max_len/100
    filt = np.blackman(smoothing)
    filt /= np.sum(filt)
    avg_bq = np.convolve(avg_bq, filt, mode='same')
    return avg_bq

def plot_ds(models: list[str], v_model: str, tree_dir: str):
    verified_ys = {}
    correct_counts = Counter()
    total_counts = Counter()
    for model in models:
        print('Model:', model)
        games = load_games_for_model(model, tree_dir, MAX_TEMPS[MODELS[model]])
        bqs = []
        for (letter, category), tree in games.items():
            if (letter, category) not in verified_ys:
                v_filename = get_v_filename(args.tree_dir, letter, category, v_model)
                if os.path.exists(v_filename):
                    with open(v_filename, 'rb') as f:
                        verified_dict = pickle.load(f)
                        verified_y = verified_dict['yes']
                        verified_ys[(letter, category)] = verified_y
                else:
                    print('File not found:', v_filename)
            bq = get_binary_quality(tree, verified_ys[(letter, category)])
            correct_counts[model] += np.sum(bq)
            total_counts[model] += len(bq)
            bqs.append(bq)
        avg_bq = upsample_smooth_and_average_bqs(bqs)
        plt.plot(np.linspace(0, 1, len(avg_bq)), avg_bq, label=model, lw=0.7)
    plt.xlabel('Normalized rank')
    plt.ylabel('Quality')
    plt.legend()
    plt.savefig('./img/quality_by_rank.png', dpi=300)
    plt.clf()

    for model, count in correct_counts.items():
        plt.bar(model, count)
    plt.xlabel('Model')
    plt.ylabel('Correct count')
    plt.savefig('./img/correct_count.png', dpi=300)
    plt.clf()

    for model, count in correct_counts.items():
        denom = total_counts[model]
        plt.bar(model, count/denom)
    plt.xlabel('Model')
    plt.ylabel('Correct rate')
    plt.savefig('./img/correct_rate.png', dpi=300)
    plt.clf()

# def get_inversions(probs: dict[str, float], verified_yes: set[str]) -> float:
    # correct_probs = []
    # incorrect_probs = []
    # for answer, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        # if standardize_str(answer) in verified_yes:
            # correct_probs.append(prob)
        # else:
            # incorrect_probs.append(prob)
    # all_probs = np.array(correct_probs + incorrect_probs)
    # # print('Correct:', correct_probs)
    # # print('Incorrect:', incorrect_probs)
    # isotonic = np.abs(isotonic_regression_l1_total_order(-all_probs, np.ones(len(all_probs))))
    # # print('Isotonic:', isotonic)
    # TVD = total_variation_distance(all_probs, isotonic)
    # return TVD


if __name__ == '__main__':
    args = parser.parse_args()
    if args.use_mlx:
        from completion_mlx import MODELS
    else:
        from completion_hf import MODELS
    models = get_model_list(args.models, set(MODELS.keys()))
    v_model = args.verifier
    plot_ds(models, v_model, args.tree_dir)
    # with open(tree_file, 'rb') as f:
        # tree = pickle.load(f)
    # with open(v_file, 'rb') as f:
        # verified_dict = pickle.load(f)
        # verified_y = verified_dict['yes']
        # verified_n = verified_dict['no']
    # print('Number of verified yes:', len(verified_y))
    # print('Number of verified no:', len(verified_n))
    # print('Number of nodes:', len(list(tree.iter_leaves())))
    # binary_quality = get_binary_quality(tree, verified_y)
