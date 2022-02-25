"""Kernels for comparing graph summary statistics or representations."""

from functools import partial

from sklearn.base import TransformerMixin
from sklearn.metrics import pairwise_kernels
from sklearn.gaussian_process.kernels import Kernel

from scipy.linalg import toeplitz
import pyemd

from metrics.utils import ensure_padded

import numpy as np


def laplacian_total_variation_kernel(x, y, sigma=1.0, **kwargs):
    """Geodesic Laplacian kernel based on total variation distance."""
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-sigma * dist)


def gaussian_kernel(x, y, sigma=1.0, **kwargs):
    """Gaussian (RBF) kernel."""
    return np.exp(-sigma * np.dot(x - y, x - y))


# taken from GRAN code
def gaussian_tv(x, y, sigma=1.0):  
  support_size = max(len(x), len(y))
  # convert histogram values x and y to float, and make them equal len
  x = x.astype(np.float)
  y = y.astype(np.float)
  if len(x) < len(y):
    x = np.hstack((x, [0.0] * (support_size - len(x))))
  elif len(y) < len(x):
    y = np.hstack((y, [0.0] * (support_size - len(y))))

  dist = np.abs(x - y).sum() / 2.0
  return np.exp(-dist * dist / (2 * sigma * sigma))

# gaussian_emd taken directly from graphrnn code
def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0, density=False):
    ''' Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    '''
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return np.exp(-emd * emd / (2 * sigma * sigma))


def linear_kernel(x, y, normalize=False, **kwargs):
    """Linear kernel."""
    if normalize:
        return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.dot(x, y)


class KernelDistributionWrapper:
    """Wrap kernel function to work with distributions.
    The purpose of this class is to wrap a kernel function or other
    class such that it can be directly used with *distributions* of
    samples.
    """

    def __init__(self, kernel, pad=True, **kwargs):
        """Create new wrapped kernel.
        Parameters
        ----------
        kernel : callable
            Kernel function to use for the calculation of a kernel value
            between two samples from a distribution. The wrapper ensures
            that the kernel can be evaluated for distributions, not only
            for samples.
        pad : bool, optional
            If set, ensures that all input sequences will be of the same
            length.
        """
        self.pad = pad
        self.kernel_type = 'kernel_fn'

        # Check whether we have something more complicated than
        # a regular kernel function.
        if isinstance(kernel, TransformerMixin):
            self.kernel_type = 'transformer'
        elif isinstance(kernel, Kernel):
            self.kernel_type = 'passthrough'

        self.original_kernel = partial(kernel, **kwargs)


    def __call__(self, X, Y=None):
        """Call kernel wrapper for two arguments.
        The specifics of this evaluation depend on the kernel that is
        wrapped; the function can call the proper method of a kernel,
        thus ensuring that the output array is always of shape (n, m)
        with n and m being the lengths of the input distributions.
        Parameter
        ---------
        X : array-like
            First input distribution. Needs to be compatible with the
            wrapped kernel function.
        Y : array-like, optional
            Second input distribution. The same caveats as for the first
            one apply.
        Returns
        -------
        Kernel matrix between the samples of X and Y.
        """
        if self.kernel_type == 'transformer':
            return self.original_kernel.transform(X, Y)
        elif self.kernel_type == 'kernel_fn':

            if self.pad:
                X, Y = ensure_padded(X, Y)

            return pairwise_kernels(X, Y, metric=self.original_kernel)

        # By default: just evaluate the kernel!
        return self.original_kernel(X, Y)

    def diag(self, X, Y=None):
        """Return diagonal values of the wrapped kernel."""
        if Y is None:
            Y = X

        if self.kernel_type == 'transformer':
            raise NotImplementedError()
        elif self.kernel_type == 'kernel_fn':

            if self.pad:
                X, Y = ensure_padded(X, Y)

            return [self.original_kernel(x, y) for x, y in zip(X, Y)]

        return self.original_kernel.diag(X, Y)
