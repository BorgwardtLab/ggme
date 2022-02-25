"""Utility functions used by multiple scripts."""
import gzip
import bz2
import pickle
#from io import BytesIO
from scipy.sparse import csr_matrix
import shutil
import numpy as np
import networkx as nx


def get_open_fn(filepath):
    if filepath.endswith('.gz'):
        open_fn = gzip.open
    elif filepath.endswith('.bz2'):
        open_fn = bz2.open
    else:
        # Assume plain pickle
        open_fn = open
    return open_fn


def load_graphs(filepath):
    """Load graphs from eventually compressed pickle."""
    open_fn = get_open_fn(filepath)

    with open_fn(filepath, 'rb') as f:
        graphs = pickle.load(f)
    if isinstance(graphs[0], nx.Graph):
        pass
    elif isinstance(graphs[0], np.ndarray):
        graphs = adj_to_networkx(graphs)
    elif isinstance(graphs[0], csr_matrix):
        graphs = sparse_to_networkx(graphs)
    else:
        raise ValueError(
            'Unsupported input type.')
    return graphs


def adj_to_networkx(graphs):
    """Convert adj matrices to networkx graphs."""
    return [nx.from_numpy_array(g) for g in graphs]


def sparse_to_networkx(graphs):
    """Convert adj matrices to networkx graphs."""
    return [nx.from_scipy_sparse_matrix(g) for g in graphs]


def networkx_to_adj(graphs):
    """Convert networkx graphs to adjacency matrices."""
    return [nx.to_numpy_array(g) for g in graphs]
