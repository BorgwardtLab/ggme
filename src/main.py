

from evaluate import evaluate_mmd

from metrics.kernels import linear_kernel
from metrics.kernels import laplacian_total_variation_kernel
from metrics.kernels import gaussian_emd
from metrics.kernels import gaussian_kernel

from metrics.descriptor_functions import degree_distribution
from metrics.descriptor_functions import clustering_coefficient
from metrics.descriptor_functions import normalised_laplacian_spectrum

from utils import load_graphs

if __name__ == "__main__":
    
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
    print(mmd)
