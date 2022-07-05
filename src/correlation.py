import scipy as sp

from evaluate import evaluate_mmd

from metrics.kernels import linear_kernel
from metrics.kernels import laplacian_total_variation_kernel
from metrics.kernels import gaussian_emd
from metrics.kernels import gaussian_kernel

from metrics.descriptor_functions import degree_distribution
from metrics.descriptor_functions import clustering_coefficient
from metrics.descriptor_functions import normalised_laplacian_spectrum

from perturb_graphs import perturb_graphs

def compute_correlation(perturbation_values,
            list_of_mmd, correlation_type="pearson"):
    ''' 


    Parameters:
    ______________
    perturbation_values:        list of perturbation levels
    list_of_mdd:                list of mmd values
    correlation_type:           type of correlation (pearsons, spearmans)

    Returns:
    ______________
    corr:           the correlation between the mmd and the degree
                    of perturbation

    '''
    if correlation_type=="spearman":
        corr, _ = sp.stats.spearmanr(perturbation_values, list_of_mmd)
    elif correlation_type=="pearson":
        corr, _ = sp.stats.pearsonr(perturbation_values, list_of_mmd)
    return(corr)



def select_kernel_and_hyperparameters(
    test_graphs,
    perturbed_graphs,
    perturbation_values,
    kernels=[gaussian_kernel],
    sigmas=[0.1, 1, 10],
    descriptor_fn=degree_distribution,
    correlation_type="pearsonr"
    ):
    '''
    Find and return a kernel/hyperparameter combination that has
    the best correlation between mmd and increasing levels of
    perturbations.

    Parameters:
    ______________
    test_graphs:                list of test graphs
    perturbed_graphs:           list of perturbed graphs
    perturbation_values:        list of levels of perturbation
    kernels:                    kernels to iterate over (see kernels.py)
    sigmas:                     values of sigma
    descriptor_fn:              descriptor function (from
                                degree_distribution,
                                clustering_coefficient,
                                normalised_laplacian_spectrum,
                                or new function implemented in
                                descriptor_functions.py)
    correlation_type:           type of correlation (pearsons, spearmans)

    Returns:
    ______________
    best_params:                a dictionary containing the kernel,
                                value of sigma, and correlation of the best
                                configuration

    '''

    best_params = {
        "kernel": 0,
        "sigma": 0,
        "corr": 0
        }

    # iterate over different kernel/hyperparamter combinations. Note
    # that the linear kernel does not have any sigma, but in this config
    # it will just recalculate the same linear kernel for each value
    # of sigma.
    for kernel in kernels:
        for sigma in sigmas:

            list_of_mmd = [evaluate_mmd(
                graphs_dist_1=test_graphs,
                graphs_dist_2=graphs,
                function=descriptor_fn,
                kernel=kernel,
                use_linear_approximation=False,
                density=True,
                sigma=sigma
                ) for graphs in perturbed_graphs]

            # compute correlation between mmd and the level of perturbation
            # (for the given descriptor fn/kernel/hyperparameter combination)
            correlation = compute_correlation(
                    perturbation_values,
                    list_of_mmd,
                    correlation_type=correlation_type
                    )

            # if the correlation is better than the exist best param setup,
            # replace the best param setup with the new configuration
            if correlation > best_params["corr"]:
                best_params["kernel"] = kernel
                best_params["sigma"] = sigma
                best_params["corr"] = correlation

    return(best_params)


