

from evaluate import evaluate_mmd
from correlation import compute_correlation

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
    
    # take a set of graphs and perturb them
    test_graphs = load_graphs("../data/test/CommunityGraphs.pickle")
    
    # take perturbed graphs and true graphs and calculate MMD
    perturbation_values, perturbed_graphs = perturb_graphs(
            graphs=test_graphs,
            perturbation_type="AddEdges"
            )
    
    list_of_mmd = [evaluate_mmd(
        graph_dist_1=test_graphs, 
        graph_dist_2=graphs,
        function=degree_distribution,
        kernel=gaussian_kernel,
        use_linear_approximation=False,
        density=True,
        sigma=0.1
        ) for graphs in perturbed_graphs]

    # compute correlation of mmd (should probably use fast
    # mmd calculation?
    best_kernel_params, mmd = compute_correlation(
            perturbation_values,
            perturbed_graphs,
            candidate_kernels=["tv", "gaussian", "linear"],
            candidate_descriptor_functions=["degree", "clustering", "laplacian"],
            candidate_sigma=[])


    # OR, just compute MMD between predicted graphs and perturbed graphs
    predicted_graphs = load_graphs("../data/predictions/CommunityGraphs.pickle")
    test_graphs = load_graphs("../data/test/CommunityGraphs.pickle")
    
    mmd = evaluate_mmd(
            graph_dist_1=test_graphs, 
            graph_dist_2=predicted_graphs, 
            function=degree_distribution,
            kernel=gaussian_kernel,
            use_linear_approximation=False,
            density=True,
            sigma=0.1
            )
