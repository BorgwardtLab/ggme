"""Module for perturbing a graph dataset."""
import argparse
import inspect
import pickle
from collections import OrderedDict
from itertools import product

import numpy as np
import networkx as nx

import perturbations as perturbations
from utils import load_graphs, networkx_to_adj



def perturb_graphs(graphs, perturbation_type, random_seed=42,
        n_repetitions=1, **kwargs):
    """
    Take a list of networkx graphs as input and perturb them according
    to a specified perturbation and degree of perturbation. 

    Parameters
    ----------
    graphs : a list of networkx graphs
    perturbation_type: one of [AddEdges, RemoveEdges, RewireEdges, AddConnectedNodes]
    random_seed: the desired random seed
    **kwargs: any perturbation parameter required for the chosen perturbation

    Returns
    -------
    perturbation_values: values of perturbation levels
    perturbed_graphs : a list of perturbed networkx graphs
    """

    random_state = np.random.RandomState(random_seed)
    parameters_for_perturbation = {}
    parameter_values = list(np.arange(0.15, 1.0, 0.05))
    if perturbation_type == "AddEdges":
        parameters = 'p_add'
    elif perturbation_type == "RemoveEdges":
        parameters = 'p_remove'
    elif perturbation_type == "RewireEdges":
        parameters = 'p_rewire'

    perturbation_class = getattr(perturbations, perturbation_type)
    perturbed_graphs = []
    perturbation_parameters = []
    parameter_dict = {parameters: parameter_values}
    
    for i in parameter_values:
        perturbation_dict = {parameters: [i]}
        perturbation_parameters.append(perturbation_dict)
        cur_perturbation = []
        perturbation = perturbation_class(
            random_state=random_state, **perturbation_dict)

        def perturb_and_convert(graph):
            return(perturbation(graph))
            # return nx.to_scipy_sparse_matrix(perturbation(graph))

        for _ in range(n_repetitions):
            cur_perturbation.extend(map(perturb_and_convert, graphs))

        perturbed_graphs.append(cur_perturbation)

    return(parameter_values, perturbed_graphs)


    
