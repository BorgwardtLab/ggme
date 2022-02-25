
"""Utility functions and classes."""

from functools import partial
from itertools import chain

import numpy as np


def pad_to_length(element, length):
    """Pad array to pre-defined length by adding zeros."""
    return np.pad(element, (0, length - len(element)), 'constant')


def ensure_padded(X, Y=None):
    # Warning! This only works because the function
    # nx.degree_histgram(G) returns a vector of length max_degree, so we
    # can simply pad by adding zeros to the end of the vector and know
    # that it works. If you use a different histogram function, this
    # would need to be updated! (This function isn't necessary for the
    # clustering coefficient and normalized laplacian spectrum, since
    # they are already the same size). However, if you intend to add
    # your own histogram function, this function may need to be updated
    # to do padding properly.
    """Ensure that input arrays are padded to the same length."""
    if Y is None:
        max_length = max(map(lambda a: len(a), X))
        return X, None
    else:
        max_length = max(map(lambda a: len(a), chain(X, Y)))

        X_padded = np.asarray([pad_to_length(el, max_length) for el in X])
        Y_padded = np.asarray([pad_to_length(el, max_length) for el in Y])

        return X_padded, Y_padded
