"""Module with implementation of graph perturbations."""
from abc import ABCMeta

import networkx as nx
import numpy as np


class GraphPerturbation(metaclass=ABCMeta):
    def __init__(self, random_state):
        self.random_state = random_state

    def __call__(self, graph):
        """Apply perturbation."""


class RemoveEdges(GraphPerturbation):
    """Randomly remove edges."""

    def __init__(self, p_remove: float, **kwargs):
        """Remove edges with probability p_remove."""
        super().__init__(**kwargs)
        self.p_remove = p_remove

    def __call__(self, graph):
        """Apply perturbation."""
        graph = graph.copy()
        edges_to_remove = self.random_state.binomial(
            1, self.p_remove, size=graph.number_of_edges())
        edge_indices_to_remove = np.where(edges_to_remove == 1.)[0]
        edges = list(graph.edges())

        for edge_index in edge_indices_to_remove:
            edge = edges[edge_index]
            graph.remove_edge(*edge)

        return graph


class AddEdges(GraphPerturbation):
    """Randomly add edges."""

    def __init__(self, p_add: float, **kwargs):
        """Add edges with probability p_add."""
        super().__init__(**kwargs)
        self.p_add = p_add

    def __call__(self, graph):
        """Apply perturbation."""
        graph = graph.copy()
        nodes = list(graph.nodes())

        for i, node1 in enumerate(nodes):
            nodes_to_connect = self.random_state.binomial(
                1, self.p_add, size=len(nodes))
            nodes_to_connect[i] = 0  # Never introduce self connections
            node_idxs_to_connect = np.where(nodes_to_connect == 1)[0]
            for j in node_idxs_to_connect:
                node2 = nodes[j]
                graph.add_edge(node1, node2)

        return graph


class RewireEdges(GraphPerturbation):
    """Randomly rewire edges."""

    def __init__(self, p_rewire: float, **kwargs):
        """Rewire edges with probability p_rewire."""
        super().__init__(**kwargs)
        self.p_rewire = p_rewire

    def __call__(self, graph):
        """Apply perturbation."""
        graph = graph.copy()
        edges_to_rewire = self.random_state.binomial(
            1, self.p_rewire, size=graph.number_of_edges())
        edge_indices_to_rewire = np.where(edges_to_rewire == 1.)[0]
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        for edge_index in edge_indices_to_rewire:
            edge = edges[edge_index]
            graph.remove_edge(*edge)

            # Randomly pick one of the nodes which should be detached
            if self.random_state.random() > 0.5:
                keep_node, detach_node = edge
            else:
                detach_node, keep_node = edge

            # Pick a random node besides detach node and keep node to attach to
            possible_nodes = list(filter(
                lambda n: n not in [keep_node, detach_node], nodes))
            attach_node = self.random_state.choice(possible_nodes)
            graph.add_edge(keep_node, attach_node)
        return graph


class AddConnectedNodes(GraphPerturbation):
    """Randomly add nodes to graph."""

    def __init__(self, n_nodes: int, p_edge: float, **kwargs):
        """Add n_nodes nodes and attach edges to node with prob p_edge."""
        super().__init__(**kwargs)
        self.n_nodes = n_nodes
        self.p_edge = p_edge

    def __call__(self, graph):
        """Apply perturbation."""
        graph = graph.copy()

        for i in range(self.n_nodes):
            n_nodes = graph.number_of_nodes()
            # As the graphs are loaded from numpy arrays the nodes are simply
            # the index
            new_node = n_nodes
            graph.add_node(new_node)
            nodes_idxs_to_attach = np.where(
                self.random_state.binomial(1, self.p_edge, size=n_nodes))[0]
            for node in nodes_idxs_to_attach:
                graph.add_edge(new_node, node)

        return graph


__all__ = ["RemoveEdges", "AddEdges", "RewireEdges", "AddConnectedNodes"]
