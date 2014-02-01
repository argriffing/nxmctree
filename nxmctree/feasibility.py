"""
Markov chain algorithms on trees.

Regarding notation, sometimes a variable named 'v' (indicating a 'vertex' of
a graph) is used to represent a networkx graph node,
because this name is shorter than 'node' and looks less like a count than 'n'.
The 'edge_to_adjacency' is a map from a directed edge on the networkx graph
(in the direction from the root toward the tips) to a networkx
directed graph representing a sparse state transition feasibility matrix.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = ['get_node_to_posterior_feasible_set']


def get_node_to_posterior_feasible_set(T, edge_to_adjacency, root,
        root_prior_feasible_set, node_to_data_feasible_set):
    """
    For each node get the marginal posterior set of feasible states.

    The posterior feasible state set for each node is determined through
    several data sources: transition matrix sparsity,
    state restrictions due to observed data at nodes,
    and state restrictions at the root due to the support of its prior
    distribution.

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_feasible_set : set
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_feasible_set : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.

    Returns
    -------
    node_to_posterior_feasible_set : dict
        Map from node to set of posterior feasible states.

    """
    v_to_subtree_feasible_set = _backward(T, edge_to_adjacency, root,
            root_prior_feasible_set, node_to_data_feasible_set)
    v_to_posterior_feasible_set = _forward(T, edge_to_adjacency, root,
            v_to_subtree_feasible_set)
    return v_to_posterior_feasible_set


def _backward(T, edge_to_adjacency, root,
        root_prior_feasible_set, node_to_data_feasible_set):
    """
    Determine the subtree feasible state set of each node.
    This is the backward pass of a backward-forward algorithm.

    """
    v_to_subtree_feasible_set = {}
    for v in reversed(nx.topological_sort(T, [root])):
        cs = T[v]
        fset_data = node_to_data_feasible_set[v]
        if cs:
            fset = set()
            for s in fset_data:
                if _state_is_subtree_feasible(edge_to_adjacency,
                        v_to_subtree_feasible_set, v, cs, s):
                    fset.add(s)
        else:
            fset = set(fset_data)
        if v == root:
            fset &= root_prior_feasible_set
        v_to_subtree_feasible_set[v] = fset
    return v_to_subtree_feasible_set


def _forward(T, edge_to_adjacency, root,
        v_to_subtree_feasible_set):
    """
    Forward pass.

    """
    v_to_posterior_feasible_set = {}
    v_to_posterior_feasible_set[root] = set(v_to_subtree_feasible_set[root])
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        A = edge_to_adjacency[edge]
        fset = set()
        for s in v_to_posterior_feasible_set[va]:
            fset.update(set(A[s]) & v_to_subtree_feasible_set[vb])
        v_to_posterior_feasible_set[vb] = fset
    return v_to_posterior_feasible_set


def _state_is_subtree_feasible(edge_to_adjacency,
        v_to_subtree_feasible_set, v, cs, s):
    """
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    v_to_subtree_feasible_set : which states are allowed in child nodes
    v : node under consideration
    cs : child nodes of v
    s : state under consideration
    """
    for c in cs:
        edge = v, c
        A = edge_to_adjacency[edge]
        if s not in A:
            return False
        if not set(A[s]) & v_to_subtree_feasible_set[c]:
            return False
    return True

