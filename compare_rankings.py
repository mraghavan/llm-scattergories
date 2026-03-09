from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from file_manager import FileManager
import pandas as pd
from scat_utils import get_deterministic_instances
import argparse
import itertools
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
from l1_isotonic import isotonic_regression_l1_total_order
from plot_inversions import total_variation_distance
from scipy.optimize import linear_sum_assignment

def load_all_rankings(fm: FileManager, letter: str, category: str, min_count: int) -> dict:
    """
    Load all ranking files for a given letter and category.
    
    Args:
        fm: FileManager instance
        letter: The starting letter
        category: The category name
        min_count: Minimum count threshold for including an answer
    
    Returns:
        Dictionary mapping model names to their answer rankings (NLLs)
    """
    rankings = {}
    for model_name, _ in fm.get_all_ranking_files(letter, category, min_count):
        model_rankings = fm.load_rankings(letter, category, model_name, min_count)
        rankings[model_name] = model_rankings
    
    return rankings

def normalized_kendall_tau_distance(rankings1: list[int], rankings2: list[int]) -> float:
    """
    Calculate the normalized Kendall Tau distance between two rankings.
    The distance is the number of discordant pairs divided by the maximum possible number of pairs (n choose 2).
    
    Args:
        rankings1: First ranking as a list of integers
        rankings2: Second ranking as a list of integers
    
    Returns:
        Normalized Kendall Tau distance (between 0 and 1)
    """
    distance = 0
    n = len(rankings1)
    
    # For each pair of items
    for i in range(n):
        for j in range(i + 1, n):
            # Check if the relative ordering is different between the two rankings
            if (rankings1[i] < rankings1[j] and rankings2[i] > rankings2[j]) or \
               (rankings1[i] > rankings1[j] and rankings2[i] < rankings2[j]):
                distance += 1
    
    # Normalize by n choose 2 (maximum possible number of discordant pairs)
    max_distance = (n * (n - 1)) // 2
    return distance / max_distance

def create_distance_matrix(rankings: pd.DataFrame) -> pd.DataFrame:
    # for each pair of rankings, calculate the normalized Kendall tau distance
    model_ids = list(rankings.columns)
    distance_matrix = pd.DataFrame(index=model_ids, columns=model_ids)
    
    for model_id1, model_id2 in itertools.product(model_ids, repeat=2):
        distance = normalized_kendall_tau_distance(rankings[model_id1].values, rankings[model_id2].values)
        distance_matrix.loc[model_id1, model_id2] = distance
    
    return distance_matrix

def get_distance_matrix(dicts: dict[str, dict[int, float]]):
    model_ids = sorted(list(dicts.keys()))
    distance_matrix = np.zeros((len(model_ids), len(model_ids)))
    for i, model_id1 in enumerate(model_ids):
        for j, model_id2 in enumerate(model_ids):
            distance_matrix[i, j] = get_weighted_inversions(dicts[model_id1], dicts[model_id2])
            assert np.isfinite(distance_matrix[i, j])
    return distance_matrix

def get_similarity_matrix_gaussian(distance_matrix: np.ndarray):
    sigma = 1.0
    return np.exp(-distance_matrix**2 / (2 * sigma**2))

def cluster_rankings_weighted(distance_matrix: np.ndarray, model_names: list[str], num_clusters: int):
    # Add edges with weights (using similarity instead of distance)
    sigma = 1.0
    similarity_matrix = np.exp(-distance_matrix / (2 * sigma**2))
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2

    # Perform clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=0)
    labels = clustering.fit_predict(similarity_matrix)
    
    # Get 2D embedding
    embedder_2d = SpectralEmbedding(n_components=2,
                             affinity='precomputed',
                             random_state=0)
    node_positions_2d = embedder_2d.fit_transform(similarity_matrix)
    
    # Get 3D embedding
    embedder_3d = SpectralEmbedding(n_components=3,
                             affinity='precomputed',
                             random_state=0)
    node_positions_3d = embedder_3d.fit_transform(similarity_matrix)

    # Get unique model names and assign markers sequentially
    unique_models = sorted(set(name.split('-')[0] for name in model_names))
    marker_styles = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', 'X', 'd', '>', '<', 'P']
    marker_map = {model: marker for model, marker in zip(unique_models, marker_styles)}
    
    # Get markers for each point based on model names
    markers = [marker_map[name.split('-')[0]] for name in model_names]
    
    # Create separate figures instead of subplots
    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(111)

    # 1. Plot similarity matrix heatmap
    im = ax1.imshow(distance_matrix, cmap='viridis_r')
    ax1.set_title('Distance Matrix')
    ax1.set_yticks([])
    
    # Calculate average positions for each model type
    model_positions = {}
    for i, name in enumerate(model_names):
        base_name = name.split('-')[0]
        if base_name not in model_positions:
            model_positions[base_name] = []
        model_positions[base_name].append(i)
    
    # Calculate average position for each model type
    avg_positions = {name: np.mean(positions) for name, positions in model_positions.items()}
    
    # Set x-ticks at average positions
    ax1.set_xticks(list(avg_positions.values()))
    ax1.set_xticklabels(list(avg_positions.keys()), rotation=45, ha='right')
    
    # Set y-ticks at average positions (same as x-ticks)
    ax1.set_yticks(list(avg_positions.values()))
    ax1.set_yticklabels(list(avg_positions.keys()), rotation=45, ha='right')
    
    plt.colorbar(im, ax=ax1)
    plt.tight_layout()
    fig1.savefig('img/distance_matrix.png', dpi=300, bbox_inches='tight')

    # Create second figure for 2D spectral embedding
    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111)

    # 2. Plot 2D spectral embedding using second and third eigenvectors, color by cluster
    # Create a color map for the labels
    cmap = plt.cm.get_cmap('tab10')
    
    # Plot each point with its marker and color
    for i, (x, y) in enumerate(node_positions_2d):
        ax2.plot(x, y, 
                marker=markers[i], 
                color=cmap(labels[i]), 
                linestyle='', 
                markersize=8)
    
    # Add legend for models (markers)
    model_handles = [plt.Line2D([0], [0], marker=marker, color='gray', 
                               markerfacecolor='gray', markersize=8, 
                               label=model)
                    for model, marker in marker_map.items()]
    
    # Only show model legend
    ax2.legend(handles=model_handles, 
              title='Models',
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    ax2.set_title('2D Spectral Embedding')
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.tight_layout()
    fig2.savefig('img/spectral_embedding_2d.png', dpi=300, bbox_inches='tight')
    
    # Create third figure for 3D spectral embedding
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # 3. Plot 3D spectral embedding
    for i, (x, y, z) in enumerate(node_positions_3d):
        ax3.scatter(x, y, z,
                   marker=markers[i],
                   color=cmap(labels[i]),
                   s=100)
    
    # Add legend for models (markers)
    model_handles = [plt.Line2D([0], [0], marker=marker, color='gray',
                               markerfacecolor='gray', markersize=8,
                               label=model)
                    for model, marker in marker_map.items()]
    
    ax3.legend(handles=model_handles,
              title='Models',
              bbox_to_anchor=(1.05, 1),
              loc='upper left')
    
    ax3.set_title('3D Spectral Embedding')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    
    # Set initial view angle
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    fig3.savefig('img/spectral_embedding_3d.png', dpi=300, bbox_inches='tight')

    return labels

def get_weighted_inversions(d1: dict[int, float], d2: dict[int, float]) -> float:
    # get the ranking of d1
    ranking1 = sorted(d1.keys(), key=lambda x: d1[x])
    values2 = [d2[i] for i in ranking1]
    ordered = isotonic_regression_l1_total_order(np.array(values2), np.ones(len(values2)))
    return total_variation_distance(values2, ordered)

def get_weighted_inversions2(d1: dict[int, float], d2: dict[int, float]) -> float:
    # sum over all pairs of items
    # if they're inverted, add p_i * p_j to the total
    total = 0.0
    items = list(d1.keys())
    
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            item_i = items[i]
            item_j = items[j]
            
            # Check if there's an inversion between these items
            if (d1[item_i] < d1[item_j] and d2[item_i] > d2[item_j]) or \
               (d1[item_i] > d1[item_j] and d2[item_i] < d2[item_j]):
                # Add the product of probabilities to the total
                total += max(d1[item_i], d1[item_j])
    
    return total

def evaluate_cluster(labels: list[int], true_labels: list[str], base_ids: list[str]) -> float:
    """
    Calculate the accuracy of the clustering by finding the best possible mapping
    between predicted and true labels, allowing many-to-one mapping.
    
    Args:
        labels: List of predicted cluster labels (integers)
        true_labels: List of true cluster labels (strings)
    
    Returns:
        Accuracy score between 0 and 1
    """
    # Convert string labels to integers
    unique_true = np.unique(true_labels)
    true_to_int = {label: i for i, label in enumerate(unique_true)}
    true_labels_int = np.array([true_to_int[label] for label in true_labels])

    unique_pred = np.unique(labels)
    n_pred = len(unique_pred)
    n_true = len(unique_true)

    # Create confusion matrix
    confusion = np.zeros((n_pred, n_true), dtype=int)
    for pred, true in zip(labels, true_labels_int):
        confusion[pred, true] += 1

    # Sort columns (true labels) alphabetically
    alpha_col_order = np.argsort(unique_true)
    sorted_true_labels = unique_true[alpha_col_order]
    confusion_alpha = confusion[:, alpha_col_order]

    # Run Hungarian algorithm to maximize diagonal
    row_ind, col_ind = linear_sum_assignment(-confusion_alpha)

    # Sort the assignments by col_ind so that row i matches column i (alphabetical order)
    sort_order = np.argsort(col_ind)
    final_confusion = confusion_alpha[row_ind[sort_order], :]
    cluster_names = [f'Cluster {i+1}' for i in range(final_confusion.shape[0])]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(final_confusion, cmap='Blues')
    plt.colorbar(label='Count')
    plt.xticks(range(len(sorted_true_labels)), sorted_true_labels, rotation=45, ha='right')
    plt.yticks(range(len(cluster_names)), cluster_names)
    plt.xlabel('True Model')
    plt.ylabel('Predicted Cluster')
    plt.title('Confusion Matrix')

    # Add text annotations
    for i in range(final_confusion.shape[0]):
        for j in range(final_confusion.shape[1]):
            plt.text(j, i, str(final_confusion[i, j]),
                     ha='center', va='center',
                     color='white' if final_confusion[i, j] > final_confusion.max()/2 else 'black')

    plt.tight_layout()
    plt.savefig('img/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate accuracy
    correct = sum(final_confusion[i, i] for i in range(min(final_confusion.shape)))
    # For each predicted cluster, find the true label with most items
    assignments = []
    for pred_label in unique_pred:
        true_label_idx = np.argmax(confusion[pred_label])
        true_label = unique_true[true_label_idx]
        correct_count = confusion[pred_label, true_label_idx]
        total_count = np.sum(confusion[pred_label])
        assignments.append((pred_label, true_label, total_count, correct_count))
    
    # Create mapping of predicted clusters to their assigned true labels
    pred_to_true = {pred: true for pred, true, _, _ in assignments}
    
    # Print misclassified items
    print("\nMisclassified items:")
    for i, (pred_label, true_label, base_id) in enumerate(zip(labels, true_labels, base_ids)):
        inferred_true_label = pred_to_true[pred_label]
        if inferred_true_label != true_label:
            print(f"Item {base_id}: Predicted {inferred_true_label} -> True {true_label}")
    
    # Calculate accuracy
    correct = sum(correct_count for _, _, _, correct_count in assignments)
    total = len(labels)
    
    return correct / total

def main():
    parser = argparse.ArgumentParser(description='Compare rankings across different models')
    parser.add_argument('--num-instances', '-n', type=int, default=1,
                      help='Number of instances to analyze (default: 1)')
    parser.add_argument('--min-count', type=int, default=100,
                      help='Minimum count threshold for including an answer (default: 100)')
    args = parser.parse_args()

    # Initialize FileManager
    fm = FileManager.from_base(Path('./'))
    # fm.locations.rankings_dir = Path('./rankings2')
    
    # Load rankings for specified letter and category
    instances = get_deterministic_instances(args.num_instances)
    all_valid_model_ids = set()
    
    # First pass: collect all invalid model IDs
    for letter, category in instances:
        valid_model_ids = set()
        rankings = load_all_rankings(fm, letter, category, args.min_count)
        all_answers = next(iter(rankings.values())).keys()
        
        for model_id, model_rankings in rankings.items():
            # Check for missing answers
            missing_answers = all_answers - set(model_rankings.keys())
            if missing_answers:
                # print(f"Missing answers for {model_id} in {letter}/{category}: {missing_answers}")
                pass
            elif not np.any(np.isfinite(list(model_rankings.values()))):
                print(f"Non-finite NLLs for {model_id} in {letter}/{category}")
                pass
            elif model_id.startswith('smollm'):
                pass
            else:
                valid_model_ids.add(model_id)
        if not all_valid_model_ids:
            all_valid_model_ids = valid_model_ids
        else:
            all_valid_model_ids.intersection_update(valid_model_ids)
    sorted_all_valid_model_ids = sorted(all_valid_model_ids)
    
    # Now proceed with the analysis
    distance_matrices = []
    for letter, category in instances:
        rankings = load_all_rankings(fm, letter, category, args.min_count)
        model_ids = sorted(list(rankings.keys()))
        model_names = [model_id.split('-')[0] for model_id in model_ids]
        unique_model_names = list(set(model_names))
        all_answers = next(iter(rankings.values())).keys()
        # verify that all answers are present in all rankings
        # assign a unique integer to each answer
        answer_to_id = {answer: i for i, answer in enumerate(all_answers)}
        id_to_answer = {i: answer for answer, i in answer_to_id.items()}
        # for each model_id, convert NLLs to probability distributions
        rankings_dict = {}
        for model_id in sorted_all_valid_model_ids:
            model_rankings = rankings[model_id]
            # Convert NLLs to log probabilities (multiply by -1)
            if model_id not in all_valid_model_ids:
                continue
            log_probs = np.array([-nll for nll in model_rankings.values()])
            # Apply softmax to get valid probability distribution
            probs = np.exp(log_probs - np.max(log_probs))  # Subtract max for numerical stability
            probs[np.isnan(probs)] = 0
            probs[np.isinf(probs)] = 0
            assert np.all(probs >= 0)
            assert np.sum(probs) > 0
            probs = probs / np.sum(probs)
            assert np.all(np.isfinite(probs))
            # Create dictionary mapping answer IDs to probabilities
            rankings_dict[model_id] = {answer_to_id[answer]: prob 
                                     for answer, prob in zip(model_rankings.keys(), probs)}
            # print out the 2 highest probability answers for each model
            sorted_probs = sorted(probs, reverse=True)
            print(f"{model_id}: {id_to_answer[np.argmax(probs)]} ({sorted_probs[0]:.2f}), {id_to_answer[np.argmax(probs[1])]} ({sorted_probs[1]:.2f})")
        distance_matrix = get_distance_matrix(rankings_dict)
        distance_matrices.append(distance_matrix)
        assert np.all(np.isfinite(distance_matrix))
    print([m.shape for m in distance_matrices])
    avg_distance_matrix = np.mean(distance_matrices, axis=0)
    model_names = [model_id.split('-')[0] for model_id in sorted_all_valid_model_ids]
    # num_cluster_list = list(range(len(unique_model_names), len(unique_model_names)+1))
    # accuracy_list = []
    # for num_clusters in num_cluster_list:
    labels = cluster_rankings_weighted(avg_distance_matrix, model_names, num_clusters=len(set(model_names)))
    assert len(labels) == len(all_valid_model_ids)
    accuracy = evaluate_cluster(labels, model_names, sorted_all_valid_model_ids)
    print(f"Accuracy: {accuracy}")
    # accuracy_list.append(accuracy)
    # plt.plot(num_cluster_list, accuracy_list)
    # plt.show()
        # convert rankings_dict to a DataFrame
        # df = pd.DataFrame(rankings_dict)
        # print(df)
        # cluster_rankings(df, num_clusters=len(unique_model_names))
        
if __name__ == "__main__":
    main() 