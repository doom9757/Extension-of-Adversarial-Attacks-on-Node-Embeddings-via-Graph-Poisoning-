"""
Implementation of the method proposed in the paper:

'Adversarial Attacks on Node Embeddings via Graph Poisoning'
Aleksandar Bojchevski and Stephan Günnemann, ICML 2019
http://proceedings.mlr.press/v97/bojchevski19a.html

Copyright (C) owned by the authors, 2019
"""

import warnings

import numpy as np
import numba
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit


def remove_singleton_nodes(adj_matrix, labels):
    """Remove singleton nodes from the graph.
    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param labels: np.array, shape [n_nodes]
        Node labels
    :return: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph without singleton nodes.
    :return: np.array, shape [n_nodes]
        Updated labels
    """
    # Find the non-singleton nodes (nodes with degree not equal to zero)
    nonsingletons = np.where(adj_matrix.sum(1).A1 != 0)[0]

    # Keep only the rows and columns of the adjacency matrix that correspond to non-singleton nodes
    adj_matrix = adj_matrix[nonsingletons, :][:, nonsingletons]

    # Update labels using numpy indexing
    labels = labels[nonsingletons]

    return adj_matrix, labels

def remove_node(adj_matrix, node_to_remove, labels):
    """
    Remove a node from the adjacency matrix and update the labels accordingly.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param node_to_remove: int
        The node to remove
    :param labels: numpy array
        Node labels
    :return: sp.csr_matrix, shape [n_nodes - 1, n_nodes - 1]
        Adjacency matrix of the graph without the removed node.
    :return: numpy array
        Updated labels
    """

    adj_matrix_removed = adj_matrix.copy().tolil()

    # Remove the node by setting all its edges to 0 (unconnected)
    adj_matrix_removed[node_to_remove, :] = 0
    adj_matrix_removed[:, node_to_remove] = 0

    adj_matrix_removed = adj_matrix_removed.tocsr()

    # Remove the corresponding row and column from the adjacency matrix
    adj_matrix_removed = np.delete(adj_matrix_removed.toarray(), node_to_remove, 0)
    adj_matrix_removed = np.delete(adj_matrix_removed, node_to_remove, 1)

    # Convert it back to sparse matrix format
    adj_matrix_removed = sp.csr_matrix(adj_matrix_removed)

    # Remove the corresponding label
    labels = np.delete(labels, node_to_remove)

    return adj_matrix_removed, labels

def flip_candidates(adj_matrix, candidates, labels):
    """Flip the edges in the candidate set to non-edges and vise-versa.
    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :return: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph with the flipped edges/non-edges.
    """
    adj_matrix_flipped = adj_matrix.copy().tolil()
    
    for candidate in candidates:
        adj_matrix_flipped[candidate[0], candidate[1]] = 1 - adj_matrix_flipped[candidate[0], candidate[1]]
        adj_matrix_flipped[candidate[1], candidate[0]] = 1 - adj_matrix_flipped[candidate[1], candidate[0]]
    
    adj_matrix_flipped = adj_matrix_flipped.tocsr()
    adj_matrix_flipped.eliminate_zeros()

    # Remove singleton nodes
    adj_matrix_flipped, labels = remove_singleton_nodes(adj_matrix_flipped, labels)

    return adj_matrix_flipped, labels


@numba.jit(nopython=True)
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)


def generate_candidates_removal_minimum_spanning_tree(adj_matrix):
    """Generates candidate edge flips for removal (edge -> non-edge),
     disallowing edges that lie on the minimum spanning tree.

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    mst = sp.csgraph.minimum_spanning_tree(adj_matrix)
    mst = mst.maximum(mst.T)
    adj_matrix_sample = adj_matrix - mst
    candidates = np.column_stack(sp.triu(adj_matrix_sample, 1).nonzero())

    return candidates


def generate_candidates_removal(adj_matrix, seed=0):
    """Generates candidate edges for removal (edge -> non-edge),
     targeting edges that, when removed, would create a singleton node.
    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge removals
    """
    n_nodes = adj_matrix.shape[0]

    np.random.seed(seed)

    # Get the degree of each node
    degree = adj_matrix.sum(1).A1

    # Find edges connected to nodes with degree 2
    candidates = np.array(
        [[i, j] for i in range(n_nodes) for j in range(n_nodes) if adj_matrix[i, j] == 1 and (degree[i] == 2 or degree[j] == 2)])

    return candidates
    
def generate_candidate_nodes_removal(adj_matrix, edge_budget, seed=0):
    """
    Generates candidate nodes for removal, where each node has number of edges less than or equal to the edge budget 
    and greater than 0 (i.e., not an isolated node).

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param edge_budget: int
        Maximum edges that a candidate node can have
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?,]
        Candidate set of nodes for removal
    """
    np.random.seed(seed)

    # Calculate degree of each node
    degree = adj_matrix.sum(1).A1

    # Select nodes where the degree is less than or equal to the edge budget and greater than 0
    candidate_nodes = np.where((degree <= edge_budget) & (degree > 0))[0]
    non_positive_definite = np.array([1,11,30,54,84,89,98,104,135,189,197,216,233,234,240,241,247,249,254,255,259,261,264,295,311,323,325,326,347,363,366,377,379,387,420,422,440,464,486,496,508,509,515,516,554,557,568,580,602,610,625,627,629,639,644,646,651,658,660,661,663,683,684,692,693,696,699,701,709,720,731,734,735,737,742,747,755,764,784,801,803,807,808,816,818,822,829,834,835,840,851,861,867,871,875,892,910,911,922,923,927,929,934,936,1009,1013,1015,1041,1049,1059,1060,1063,1067,1099,1112,1127,1141,1144,1146,1154,1159,1166,1199,1203,1223,1244,1246,1252,1254,1257,1265,1268,1270,1299,1306,1307,1312,1327,1328,1385,1389,1398,1399,1419,1422,1439,1450,1454,1455,1458,1467,1468,1480,1498,1517,1522,1526,1532,1538,1539,1542,1545,1565,1575,1578,1582,1621,1636,1637,1665,1669,1673,1680,1682,1695,1702,1710,1724,1725,1730,1731,1736,1738,1740,1744,1760,1773,1787,1789,1792,1795,1802,1803,1805,1808,1828,1841,1852,1857,1859,1860,1865,1866,1875,1911,1914,1916,1919,1923,1925,1927,1929,1933,1939,1955,1957,1962,1968,1974,1983,1987,1989,2000,2014,2036,2049,2065,2066,2068,2071,2080,2094,2095,2103,2116,2120,2122,2130,2140,2142,2152,2180,2189,2197,2216,2230,2250,2259,2260,2261,2269,2274,2283,2290,2309,2320,2329,2337,2343,2362,2379,2390,2398,2401,2402,2407,2441,2448,2463,2465,2470,2481,2483,2493,2504,2506,2509,2511,2522,2540,2568,2582,2590,2598,2604,2618,2627,2631,2633,2641,2643,2644,2649,2663,2672,2685,2694,2698,2700,2701,2704,2706,2708,2738,2742,2747,2751,2753,2758,2763,2769,2787,2794,2796,2802,2803])
    candidate_nodes = np.setdiff1d(candidate_nodes, non_positive_definite)
    
    return candidate_nodes



def generate_candidates_addition(adj_matrix, n_candidates, seed=0):
    """Generates candidate edge flips for addition (non-edge -> edge).

    adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param n_candidates: int
        Number of candidates to generate.
    :param seed: int
        Random seed
    :return: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    """
    np.random.seed(seed)
    num_nodes = adj_matrix.shape[0]

    candidates = np.random.randint(0, num_nodes, [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    candidates = candidates[adj_matrix[candidates[:, 0], candidates[:, 1]].A1 == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates = candidates[:n_candidates]

    assert len(candidates) == n_candidates

    return candidates


def edges_to_sparse(edges, num_nodes, weights=None):
    """Create a sparse adjacency matrix from an array of edge indices and (optionally) values.

    :param edges: array-like, shape [num_edges, 2]
        Array with each row storing indices of an edge as (u, v).
    :param num_nodes: int
        Number of nodes in the resulting graph.
    :param weights: array_like, shape [num_edges], optional, default None
        Weights of the edges. If None, all edges weights are set to 1.
    :return: sp.csr_matrix
        Adjacency matrix in CSR format.
    """
    if weights is None:
        weights = np.ones(edges.shape[0])

    return sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(num_nodes, num_nodes)).tocsr()


def evaluate_embedding_link_prediction(adj_matrix, node_pairs, embedding_matrix, norm=False):
    """Evaluate the node embeddings on the link prediction task.

    :param adj_matrix: sp.csr_matrix, shape [n_nodes, n_nodes]
        Adjacency matrix of the graph
    :param node_pairs:
    :param embedding_matrix: np.ndarray, shape [n_nodes, embedding_dim]
        Embedding matrix
    :param norm: bool
        Whether to normalize the embeddings
    :return: float, float
        Average precision (AP) score and area under ROC curve (AUC) score
    """
    if norm:
        embedding_matrix = normalize(embedding_matrix)

    true = adj_matrix[node_pairs[:, 0], node_pairs[:, 1]].A1
    scores = (embedding_matrix[node_pairs[:, 0]] * embedding_matrix[node_pairs[:, 1]]).sum(1)

    auc_score, ap_score = roc_auc_score(true, scores), average_precision_score(true, scores)

    return auc_score, ap_score


def evaluate_embedding_node_classification(embedding_matrix, labels, train_ratio=0.1, norm=True, seed=0, n_repeats=10):
    """Evaluate the node embeddings on the node classification task..

    :param embedding_matrix: np.ndarray, shape [n_nodes, embedding_dim]
        Embedding matrix
    :param labels: np.ndarray, shape [n_nodes]
        The ground truth labels
    :param train_ratio: float
        The fraction of labels to use for training
    :param norm: bool
        Whether to normalize the embeddings
    :param seed: int
        Random seed
    :param n_repeats: int
        Number of times to repeat the experiment
    :return: [float, float], [float, float]
        The mean and standard deviation of the f1_scores
    """
    if norm:
        embedding_matrix = normalize(embedding_matrix)

    results = []
    for it_seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + it_seed)
        split_train, split_test = next(sss.split(embedding_matrix, labels))

        features_train = embedding_matrix[split_train]
        features_test = embedding_matrix[split_test]
        labels_train = labels[split_train]
        labels_test = labels[split_test]

        lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
        lr.fit(features_train, labels_train)

        lr_z_predict = lr.predict(features_test)
        f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
        f1_macro = f1_score(labels_test, lr_z_predict, average='macro')

        results.append([f1_micro, f1_macro])

    results = np.array(results)

    return results.mean(0), results.std(0)


def construct_line_graph(adj_matrix):
    """Construct a line graph from an undirected original graph.

    Parameters
    ----------
    adj_matrix : sp.spmatrix [n_samples ,n_samples]
        Symmetric binary adjacency matrix.

    Returns
    -------
    L : sp.spmatrix, shape [A.nnz/2, A.nnz/2]
        Symmetric binary adjacency matrix of the line graph.
    """
    N = adj_matrix.shape[0]
    edges = np.column_stack(sp.triu(adj_matrix, 1).nonzero())
    e1, e2 = edges[:, 0], edges[:, 1]

    I = sp.eye(N).tocsr()
    E1 = I[e1]
    E2 = I[e2]

    L = E1.dot(E1.T) + E1.dot(E2.T) + E2.dot(E1.T) + E2.dot(E2.T)

    return L - 2 * sp.eye(L.shape[0])


def load_dataset(file_name):
    """"Load a graph from a Numpy binary file.

    :param file_name: str
        Name of the file to load.

    :return: dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        labels = loader.get('labels')

        graph = {
            'adj_matrix': adj_matrix,
            'labels': labels
        }

        idx_to_node = loader.get('idx_to_node')
        if idx_to_node:
            idx_to_node = idx_to_node.tolist()
            graph['idx_to_node'] = idx_to_node

        idx_to_attr = loader.get('idx_to_attr')
        if idx_to_attr:
            idx_to_attr = idx_to_attr.tolist()
            graph['idx_to_attr'] = idx_to_attr

        idx_to_class = loader.get('idx_to_class')
        if idx_to_class:
            idx_to_class = idx_to_class.tolist()
            graph['idx_to_class'] = idx_to_class

        return graph


def train_val_test_split_adjacency(adj_matrix, p_val=0.10, p_test=0.05, seed=0, neg_mul=1,
                                   every_node=True, connected=False, undirected=False,
                                   use_edge_cover=True, set_ops=True, asserts=False):
    """Split the edges of the adjacency matrix into train, validation and test edges
    and randomly samples equal amount of validation and test non-edges.

    :param adj_matrix: scipy.sparse.spmatrix
        Sparse unweighted adjacency matrix
    :param p_val: float
        Percentage of validation edges. Default p_val=0.10
    :param p_test: float
        Percentage of test edges. Default p_test=0.05
    :param seed: int
        Seed for numpy.random. Default seed=0
    :param neg_mul: int
        What multiplicity of negative samples (non-edges) to have in the test/validation set
        w.r.t the number of edges, i.e. len(non-edges) = L * len(edges). Default neg_mul=1
    :param every_node: bool
        Make sure each node appears at least once in the train set. Default every_node=True
    :param connected: bool
        Make sure the training graph is still connected after the split
    :param undirected: bool
        Whether to make the split undirected, that is if (i, j) is in val/test set then (j, i) is there as well.
        Default undirected=False
    :param use_edge_cover: bool
        Whether to use (approximate) edge_cover to find the minimum set of edges that cover every node.
        Only active when every_node=True. Default use_edge_cover=True
    :param set_ops: bool
        Whether to use set operations to construction the test zeros. Default setwise_zeros=True
        Otherwise use a while loop.
    :param asserts: bool
        Unit test like checks. Default asserts=False
    :return:
        train_ones: array-like, shape [n_train, 2]
            Indices of the train edges
        val_ones: array-like, shape [n_val, 2]
            Indices of the validation edges
        val_zeros: array-like, shape [n_val, 2]
            Indices of the validation non-edges
        test_ones: array-like, shape [n_test, 2]
            Indices of the test edges
        test_zeros: array-like, shape [n_test, 2]
            Indices of the test non-edges
    """
    assert p_val + p_test > 0
    assert adj_matrix.max() == 1  # no weights
    assert adj_matrix.min() == 0  # no negative edges
    assert adj_matrix.diagonal().sum() == 0  # no self-loops
    assert not np.any(adj_matrix.sum(0).A1 + adj_matrix.sum(1).A1 == 0)  # no dangling nodes

    is_undirected = (adj_matrix != adj_matrix.T).nnz == 0

    if undirected:
        assert is_undirected  # make sure is directed
        adj_matrix = sp.tril(adj_matrix).tocsr()  # consider only upper triangular
        adj_matrix.eliminate_zeros()
    else:
        if is_undirected:
            warnings.warn('Graph appears to be undirected. Did you forgot to set undirected=True?')

    np.random.seed(seed)

    n_edges = adj_matrix.nnz
    n_nodes = adj_matrix.shape[0]
    s_train = int(n_edges * (1 - p_val - p_test))

    idx = np.arange(n_nodes)

    # hold some edges so each node appears at least once
    if every_node:
        if connected:
            assert sp.csgraph.connected_components(adj_matrix)[0] == 1  # make sure original graph is connected
            adj_hold = sp.csgraph.minimum_spanning_tree(adj_matrix)
        else:
            adj_matrix.eliminate_zeros()  # makes sure A.tolil().rows contains only indices of non-zero elements
            d = adj_matrix.sum(1).A1

            if use_edge_cover:
                hold_edges = edge_cover(adj_matrix)

                # make sure the training percentage is not smaller than len(edge_cover)/E when every_node is set to True
                min_size = hold_edges.shape[0]
                if min_size > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(min_size / n_edges))
            else:
                # make sure the training percentage is not smaller than N/E when every_node is set to True
                if n_nodes > s_train:
                    raise ValueError('Training percentage too low to guarantee every node. Min train size needed {:.2f}'
                                     .format(n_nodes / n_edges))

                hold_edges_d1 = np.column_stack(
                    (idx[d > 0], np.row_stack(map(np.random.choice, adj_matrix[d > 0].tolil().rows))))

                if np.any(d == 0):
                    hold_edges_d0 = np.column_stack(
                        (np.row_stack(map(np.random.choice, adj_matrix[:, d == 0].T.tolil().rows)),
                         idx[d == 0]))
                    hold_edges = np.row_stack((hold_edges_d0, hold_edges_d1))
                else:
                    hold_edges = hold_edges_d1

            if asserts:
                assert np.all(adj_matrix[hold_edges[:, 0], hold_edges[:, 1]])
                assert len(np.unique(hold_edges.flatten())) == n_nodes

            adj_hold = edges_to_sparse(hold_edges, n_nodes)

        adj_hold[adj_hold > 1] = 1
        adj_hold.eliminate_zeros()
        adj_sample = adj_matrix - adj_hold

        s_train = s_train - adj_hold.nnz
    else:
        adj_sample = adj_matrix

    idx_ones = np.random.permutation(adj_sample.nnz)
    ones = np.column_stack(adj_sample.nonzero())
    train_ones = ones[idx_ones[:s_train]]
    test_ones = ones[idx_ones[s_train:]]

    # return back the held edges
    if every_node:
        train_ones = np.row_stack((train_ones, np.column_stack(adj_hold.nonzero())))

    n_test = len(test_ones) * neg_mul
    if set_ops:
        # generate slightly more completely random non-edge indices than needed and discard any that hit an edge
        # much faster compared a while loop
        # in the future: estimate the multiplicity (currently fixed 1.3/2.3) based on A_obs.nnz
        if undirected:
            random_sample = np.random.randint(0, n_nodes, [int(2.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] > random_sample[:, 1]]
        else:
            random_sample = np.random.randint(0, n_nodes, [int(1.3 * n_test), 2])
            random_sample = random_sample[random_sample[:, 0] != random_sample[:, 1]]

        # discard ones
        random_sample = random_sample[adj_matrix[random_sample[:, 0], random_sample[:, 1]].A1 == 0]
        # discard duplicates
        random_sample = random_sample[
            np.unique(random_sample[:, 0] * n_nodes + random_sample[:, 1], return_index=True)[1]]
        # only take as much as needed
        test_zeros = np.row_stack(random_sample)[:n_test]
        assert test_zeros.shape[0] == n_test
    else:
        test_zeros = []
        while len(test_zeros) < n_test:
            i, j = np.random.randint(0, n_nodes, 2)
            if adj_matrix[i, j] == 0 and (not undirected or i > j) and (i, j) not in test_zeros:
                test_zeros.append((i, j))
        test_zeros = np.array(test_zeros)

    # split the test set into validation and test set
    s_val_ones = int(len(test_ones) * p_val / (p_val + p_test))
    s_val_zeros = int(len(test_zeros) * p_val / (p_val + p_test))

    val_ones = test_ones[:s_val_ones]
    test_ones = test_ones[s_val_ones:]

    val_zeros = test_zeros[:s_val_zeros]
    test_zeros = test_zeros[s_val_zeros:]

    if undirected:
        # put (j, i) edges for every (i, j) edge in the respective sets and form back original A
        symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
        train_ones = symmetrize(train_ones)
        val_ones = symmetrize(val_ones)
        val_zeros = symmetrize(val_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)
        adj_matrix = adj_matrix.maximum(adj_matrix.T)

    if asserts:
        set_of_train_ones = set(map(tuple, train_ones))
        assert train_ones.shape[0] + test_ones.shape[0] + val_ones.shape[0] == adj_matrix.nnz
        assert (edges_to_sparse(np.row_stack((train_ones, test_ones, val_ones)), n_nodes) != adj_matrix).nnz == 0
        assert set_of_train_ones.intersection(set(map(tuple, test_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_ones))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, test_zeros))) == set()
        assert set_of_train_ones.intersection(set(map(tuple, val_zeros))) == set()
        assert len(set(map(tuple, test_zeros))) == len(test_ones) * neg_mul
        assert len(set(map(tuple, val_zeros))) == len(val_ones) * neg_mul
        assert not connected or sp.csgraph.connected_components(adj_hold)[0] == 1
        assert not every_node or ((adj_hold - adj_matrix) > 0).sum() == 0

    return train_ones, val_ones, val_zeros, test_ones, test_zeros


def edge_cover(adj_matrix):
    """Approximately compute minimum edge cover.

    Edge cover of a graph is a set of edges such that every vertex of the graph is incident
    to at least one edge of the set. Minimum edge cover is an  edge cover of minimum size.

    :param adj_matrix: sp.spmatrix
        Sparse adjacency matrix
    :return: array-like, shape [?, 2]
        The edges the form the edge cover
    """
    n_nodes = adj_matrix.shape[0]
    d_in = adj_matrix.sum(0).A1
    d_out = adj_matrix.sum(1).A1

    # make sure to include singleton nodes (nodes with one incoming or one outgoing edge)
    one_in = np.where((d_in == 1) & (d_out == 0))[0]
    one_out = np.where((d_in == 0) & (d_out == 1))[0]

    edges = []
    edges.append(np.column_stack((adj_matrix[:, one_in].argmax(0).A1, one_in)))
    edges.append(np.column_stack((one_out, adj_matrix[one_out].argmax(1).A1)))
    edges = np.row_stack(edges)

    edge_cover_set = set(map(tuple, edges))
    nodes = set(edges.flatten())

    # greedly add other edges such that both end-point are not yet in the edge_cover_set
    cands = np.column_stack(adj_matrix.nonzero())
    for u, v in cands[d_in[cands[:, 1]].argsort()]:
        if u not in nodes and v not in nodes and u != v:
            edge_cover_set.add((u, v))
            nodes.add(u)
            nodes.add(v)
        if len(nodes) == n_nodes:
            break

    # add a single edge for the rest of the nodes not covered so far
    not_covered = np.setdiff1d(np.arange(n_nodes), list(nodes))
    edges = [list(edge_cover_set)]
    not_covered_out = not_covered[d_out[not_covered] > 0]

    if len(not_covered_out) > 0:
        edges.append(np.column_stack((not_covered_out, adj_matrix[not_covered_out].argmax(1).A1)))

    not_covered_in = not_covered[d_out[not_covered] == 0]
    if len(not_covered_in) > 0:
        edges.append(np.column_stack((adj_matrix[:, not_covered_in].argmax(0).A1, not_covered_in)))

    edges = np.row_stack(edges)

    # make sure that we've indeed computed an edge_cover
    assert adj_matrix[edges[:, 0], edges[:, 1]].sum() == len(edges)
    assert len(set(map(tuple, edges))) == len(edges)
    assert len(np.unique(edges)) == n_nodes

    return edges


def standardize(adj_matrix, labels):
    """
    Make the graph undirected and select only the nodes
     belonging to the largest connected component.

    :param adj_matrix: sp.spmatrix
        Sparse adjacency matrix
    :param labels: array-like, shape [n]

    :return:
        standardized_adj_matrix: sp.spmatrix
            Standardized sparse adjacency matrix.
        standardized_labels: array-like, shape [?]
            Labels for the selected nodes.
    """
    # copy the input
    standardized_adj_matrix = adj_matrix.copy()

    # make the graph unweighted
    standardized_adj_matrix[standardized_adj_matrix != 0] = 1

    # make the graph undirected
    standardized_adj_matrix = standardized_adj_matrix.maximum(standardized_adj_matrix.T)

    # select the largest connected component
    _, components = sp.csgraph.connected_components(standardized_adj_matrix)
    c_ids, c_counts = np.unique(components, return_counts=True)
    id_max_component = c_ids[c_counts.argmax()]
    select = components == id_max_component
    standardized_adj_matrix = standardized_adj_matrix[select][:, select]
    standardized_labels = labels[select]

    # remove self-loops
    standardized_adj_matrix = standardized_adj_matrix.tolil()
    standardized_adj_matrix.setdiag(0)
    standardized_adj_matrix = standardized_adj_matrix.tocsr()
    standardized_adj_matrix.eliminate_zeros()

    return standardized_adj_matrix, standardized_labels
