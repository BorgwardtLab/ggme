

from evaluate import evaluate_mmd
from correlation import compute_correlation
from correlation import select_kernel_and_hyperparameters

from metrics.kernels import linear_kernel
from metrics.kernels import laplacian_total_variation_kernel
from metrics.kernels import gaussian_emd
from metrics.kernels import gaussian_kernel

from metrics.descriptor_functions import degree_distribution
from metrics.descriptor_functions import clustering_coefficient
from metrics.descriptor_functions import normalised_laplacian_spectrum

from perturb_graphs import perturb_graphs

from utils import load_graphs

if __name__ == "__main__":
    
    # take a set of graphs and perturb them for a given perturbation
    test_graphs = load_graphs("../data/test/CommunityGraphs.pickle")

    perturbation_values, perturbed_graphs = perturb_graphs(
            graphs=test_graphs,
            perturbation_type="AddEdges"
            )

    # OPTION 1: to find the best parameters for a grid of kernel/hyperparameter
    # combination, run the following function, which will return the
    # kernel, sigma, and correlation of the best configuration
    best_params = select_kernel_and_hyperparameters(
        test_graphs=test_graphs,
        perturbed_graphs=perturbed_graphs,
        perturbation_values=perturbation_values,
        kernels=[gaussian_kernel, linear_kernel],
        sigmas=[0.1, 1, 10],
        descriptor_fn=degree_distribution,
        correlation_type="pearson"
        )
    print("Option 1:")
    print("The best kernel-hyperparameter combination is the {} with sigma={}. This had a correlation of {:.4f}."
        .format(
        best_params["kernel"].__name__,
        best_params["sigma"],
        best_params["corr"]
        ))
    print("======================")

    # OPTION 2: to compute the correlation for a single kernel/param combo,
    # take perturbed graphs and true graphs and calculate MMD
    # for each level of perturbation (specify the kernel/sigma value)
    list_of_mmd = [evaluate_mmd(
        graphs_dist_1=test_graphs,
        graphs_dist_2=graphs,
        function=degree_distribution,
        kernel=gaussian_kernel,
        use_linear_approximation=False,
        density=True,
        sigma=0.1
        ) for graphs in perturbed_graphs]

    correlation = compute_correlation(
            perturbation_values,
            list_of_mmd,
            correlation_type="pearson"
            )

    print("Option 2:")
    print("The correlation is {:.6f}.".format(correlation))
    print("======================")

    # OPTION 3: just compute MMD between predicted graphs and test graphs
    predicted_graphs = load_graphs("../data/predictions/CommunityGraphs.pickle")
    test_graphs = load_graphs("../data/test/CommunityGraphs.pickle")

    mmd = evaluate_mmd(
            graphs_dist_1=test_graphs,
            graphs_dist_2=predicted_graphs,
            function=degree_distribution,
            kernel=gaussian_kernel,
            use_linear_approximation=False,
            density=True,
            sigma=0.1
            )

    print("Option 3:")
    print("The MMD between the test and predicted graphs is {:.4f}.".format(mmd))

