"""Maximum mean discrepancy estimation function and utilities."""

import numpy as np


def mmd(X, Y, kernel, estimate_variance=False):
    """Calculate MMD between two sets of samples, using a kernel.
    This function calculates the maximum mean discrepancy between two
    distributions `X` and `Y` by means of a kernel function, which is
    assumed to be *compatible* with the input data.
    This implementation follows Lemma 6 of 'A Kernel Two-Sample Test'
    or Equation 6 of 'A Kernel Method for the Two-Sample Problem'. It
    implements an unbiased empirical estimate of `MMD^2`, the squared
    population MMD.
    Since this is only an *estimate*, negative values may still occur
    in the calculation.
    Parameters
    ----------
    X : `array_like`
        First distribution
    Y : `array_like`
        Second distribution
    kernel : callable
        Kernel function for evaluating the similarity between samples
        from `X` and `Y`. The kernel function must support the *type*
        of `X` and `Y`, respectively. It is supposed to return a real
        value. On the technical side, the kernel must be PSD, i.e. it
        must be a positive semi-definite function.
        Notice that the kernel must support being called with a whole
        distribution instead of individual elements.
    estimate_variance : bool, optional
        If set, estimates the variance and returns it as an additional
        tuple. Else, only the MMD estimate will be returned.
    Returns
    -------
    Maximum mean discrepancy value between `X` and `Y`. If
    `estimate_variance == True`, the estimate is returned,
    thus resulting in a tuple of `mmd, var`.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Following the original notation of the paper
    m = X.shape[0]
    n = Y.shape[0]

    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)

    # We could also skip diagonal elements in the calculation above but
    # this is more computationally efficient.
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    k_XX = np.sum(K_XX)
    k_YY = np.sum(K_YY)
    k_XY = np.sum(K_XY)

    mmd = 1 / (m * (m - 1)) * k_XX \
        + 1 / (n * (n - 1)) * k_YY \
        - 2 / (m * n) * k_XY

    if estimate_variance:
        var = mmd_variance_estimate(K_XX, K_YY, K_XY)
        return mmd, var

    return mmd


def mmd_linear_approximation(X, Y, kernel):
    """Calculate linear approximation to MMD between two sets of samples.
    This function calculates the maximum mean discrepancy between two
    distributions `X` and `Y` by means of a kernel function, which is
    assumed to be *compatible* with the input data. Notice that there
    are two caveats here:
    1. `X` and `Y` must have the same size
    2. Only an approximation to MMD is calculated, but the upshot is
       that this approximation can be calculated in linear time.
    Parameters
    ----------
    X : `array_like`
        First distribution; must match cardinality of `Y`
    Y : `array_like`
        Second distribution; must match cardinality of `X`
    kernel : callable following the sklearn.gaussian_process.Kernel API or
        a general callable which compares two samples from `X` and `Y`.
        The kernel function must support the *type* of `X` and `Y`,
        respectively. It is supposed to return a real value. On the technical
        side, the kernel must be PSD, i.e. it must be a positive semi-definite
        function.
    Returns
    -------
    Maximum mean discrepancy value between `X` and `Y`.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Following the original notation of the paper
    m = X.shape[0]
    n = Y.shape[0]

    assert m == n, RuntimeError('Cardinalities must coincide')

    n = (n // 2) * 2

    # The main functionality of this code is taken from Dougal
    # Sutherland's repository:
    #
    #  https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
    #
    # This is a more generic version.
    K_XX = np.sum(kernel.diag(X[:n:2], X[1:n:2]))
    K_YY = np.sum(kernel.diag(Y[:n:2], Y[1:n:2]))

    K_XY = np.sum(kernel.diag(X[:n:2], Y[1:n:2])) + \
        np.sum(kernel.diag(X[1:n:2], Y[:n:2]))

    mmd = (K_XX + K_YY - K_XY) / (n // 2)
    return mmd


def mmd_variance_estimate(K_XX, K_YY, K_XY, lam=1e-8):
    """Perform simple MMD variance estimation.
    This is a regularised variant of the MMD variance estimate given by
    Feng Liu [1]. It is based on the formula given in the paper [2] and
    requires both distributions to have the same cardinality. Using the
    terminology from below, we require `m == n`.
    Parameters
    ----------
    K_XX : matrix of shape `(m, m)`
        Matrix of kernel values between samples from the first
        distribution.
    K_YY : matrix of shape `(n, n)`
        Matrix of kernel values between samples from the second
        distribution.
    K_XY : matrix of shape `(m, n)`
        Matrix of kernel values between samples from either one of the
        distributions.
    lam : float, optional
        Optional regularisation parameter to ensure that the value does
        not become too small (or zero).
    Returns
    -------
    Regularised variance estimate as a single float.
    References
    ----------
    [1]: https://github.com/fengliu90/DK-for-TST/blob/b78b7115327c9bd15aac719988fa3c45909d8e65/utils.py#L84
    [2]: https://arxiv.org/pdf/2002.09116.pdf
    """
    m = K_XX.shape[0]
    n = K_YY.shape[1]

    assert m == n, RuntimeError('Cardinalities must coincide')

    # Following the usual Gretton, Smola, Sutherland notation. Normally,
    # this is defined on a per-sample level but here, `H` is a matrix.
    H = K_XX + K_YY - 2 * K_XY

    var1 = np.dot(H.sum(axis=1) / n, H.sum(axis=1) / n) / n
    var2 = H.sum() / (m ** 2)

    return 4 * (var1 - var2 ** 2) + lam


def mmd_power_estimate(X, Y, kernel):
    """Calculate MMD power estimate for two sets of samples, using a kernel.
    This function calculates a power estimate for the maximum mean
    discrepancy between two distributions `X` and `Y`. The call to
    this function follows the conventions of the `mmd()` function;
    the same limitations apply.
    Parameters
    ----------
    X : `array_like`
        First distribution
    Y : `array_like`
        Second distribution
    kernel : callable
        Kernel function for evaluating the similarity between samples
        from `X` and `Y`.
    Returns
    -------
    Tuple of `power, mmd, var`, i.e. the power estimate, the maximum
    mean discrepancy value between `X` and `Y`, and a variance estimate.
    `estimate_variance == True`, the estimate is returned,
    thus resulting in a tuple of `mmd, var`.
    """
    mmd_, var_ = mmd(X, Y, kernel, estimate_variance=True)

    # `var_` is regularised so it will never be zero. Hence, this
    # expression is always well-defined.
    return mmd_ / var_



