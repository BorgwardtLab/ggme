''' Calculate MMD for a given kernel choice, parameter choice, and
descriptor function choice '''

import networkx as nx

from metrics.mmd import mmd
from metrics.mmd import mmd_linear_approximation

from metrics.kernels import KernelDistributionWrapper



def evaluate_mmd(
    graphs_dist_1, graphs_dist_2, function, kernel,
    use_linear_approximation=False,
    **kwargs
):
    """Perform MMD evaluation based on a summary statistic function.
    Main evaluation method employing an arbitrary function for
    converting a graph into a different representation. Later,
    this representation will be used in a user-defined kernel,
    which is finally used to a obtain maximum mean discrepancy
    value.
    Parameters
    ----------
    graphs_dist_1: `iterable` of `networkx.Graph`
        Collection of graphs from the first distribution, i.e. graphs originating from some
        pre-defined distribution.
    graphs_dist_2 : `iterable` of `networkx.Graph`
        Collection of graphs from the second distribution, i.e. graphs originating from
        some kind of generative model.
    function : callable
        Function that is applied to both sets of graphs. The function
        will be presented with additional `kwwargs`, and may select a
        number of additional arguments. Its output type must be valid
        with respect to the specified kernel.
    kernel : callable
        Kernel function to use for the MMD calculations. Must be capable
        of accepting inputs of the type generated by `function`. Just as
        for `function`, this parameter will also be shown `kwargs` for a
        way to select additional configuration options.
    use_linear_approximation : bool
        If set, calculates the linear approximation of MMD, which is
        faster but only a lower bound on the 'true' MMD.
    kwargs:
        Further keyword arguments, which will primarily be used by the
        `function` and `kernel` callables.
    Returns
    -------
    Maximum mean discrepancy (MMD) value according to the function and
    the kernel.
    """
    fn_values_true = [function(G, **kwargs) for G in graphs_dist_1]
    fn_values_pred = [function(G, **kwargs) for G in graphs_dist_2]

    kernel = KernelDistributionWrapper(kernel, **kwargs)

    if use_linear_approximation:
        return mmd_linear_approximation(
            fn_values_true, fn_values_pred, kernel=kernel)
    else:
        return mmd(fn_values_true, fn_values_pred, kernel=kernel)



