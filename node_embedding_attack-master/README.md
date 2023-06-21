## Extension of Adversarial Attacks on Node Embeddings via Graph Poisoning

## example

import numpy as np
from scipy.linalg import eigh
from node_embedding_attack.utils import *
from node_embedding_attack.embedding import *
from node_embedding_attack.perturbation_attack import *

graph = load_dataset('data/cora.npz')
adj_matrix = graph['adj_matrix']
labels = graph['labels']

adj_matrix, labels = standardize(adj_matrix, labels)
num = adj_matrix.shape[0]

n_nodes = 20
budget = 20
dim = 32
window_size = 5

candidates = generate_candidate_nodes_removal(adj_matrix,budget)

b_eig_nodes = baseline_eigencentrality_top_nodes(adj_matrix, candidates, n_nodes, budget)
b_deg_nodes = baseline_degree_top_nodes(adj_matrix, candidates, n_nodes, budget)
b_rnd_nodes = baseline_random_top_nodes(candidates, n_nodes, 0)

our_nodes = perturbation_top_nodes_removal(adj_matrix, candidates, n_nodes*budget, dim, window_size)

for nodes, name in zip([None, b_rnd_nodes, b_deg_nodes, b_eig_nodes, our_nodes],
                             ['cln', 'rnd', 'deg', 'eig', 'our']):
    labels_nodes = labels
    if nodes is not None:
        adj_matrix_nodes, labels_nodes = remove_node(adj_matrix, nodes, labels)
    else:
        adj_matrix_nodes = adj_matrix
        
    embedding = deepwalk_skipgram(adj_matrix_nodes, dim, window_size=window_size)
    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels_nodes)
    print('{}, F1: {:.4f} {:.4f}'.format(name, f1_scores_mean[0], f1_scores_mean[1]))

for nodes, name in zip([None, b_rnd_nodes, b_deg_nodes, b_eig_nodes, our_nodes],['cln', 'rnd', 'deg', 'eig', 'our']):
    labels_nodes=labels
    if nodes is not None:
            adj_matrix_nodes, labels_nodes = remove_node(adj_matrix, nodes, labels)
    else:
            adj_matrix_nodes = adj_matrix
    embedding, _, _, _ = deepwalk_svd(adj_matrix_nodes, window_size, dim)
    f1_scores_mean, _ = evaluate_embedding_node_classification(embedding, labels_nodes)
    print('{}, F1: {:.4f} {:.4f}'.format(name, f1_scores_mean[0], f1_scores_mean[1]))







