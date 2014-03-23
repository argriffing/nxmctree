"""
Markov chain feasibility algorithms on trees using NetworkX graphs.

Regarding notation, sometimes a variable named 'v' (indicating a 'vertex' of
a graph) is used to represent a networkx graph node,
because this name is shorter than 'node' and looks less like a count than 'n'.
The 'edge_to_adjacency' is a map from a directed edge on
the networkx tree graph (in the direction from the root toward the tips)
to a networkx directed graph representing a sparse state transition
feasibility matrix.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from nxmctree.util import ddec

__all__ = [
        'get_feas',
        'get_node_to_fset',
        'get_edge_to_nxfset',
        ]


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_fset : set
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_fset : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""


@ddec(params=params)
def get_feas(T, edge_to_adjacency, root, root_prior_fset, node_to_data_fset):
    """
    Get the feasibility of this combination of parameters.

    Parameters
    ----------
    {params}

    Returns
    -------
    feas : bool
        True if the data is structurally supported by the model,
        otherwise False.

    """
    root_fset = _get_root_fset(T, edge_to_adjacency, root,
            root_prior_fset, node_to_data_fset)
    return True if root_fset else False


@ddec(params=params)
def get_node_to_fset(T, edge_to_adjacency, root,
        root_prior_fset, node_to_data_fset):
    """
    For each node get the marginal posterior set of feasible states.

    The posterior feasible state set for each node is determined through
    several data sources: transition matrix sparsity,
    state restrictions due to observed data at nodes,
    and state restrictions at the root due to the support of its prior
    distribution.

    Parameters
    ----------
    {params}

    Returns
    -------
    node_to_posterior_fset : dict
        Map from node to set of posterior feasible states.

    """
    v_to_subtree_fset = _backward(T, edge_to_adjacency, root,
            root_prior_fset, node_to_data_fset)
    v_to_posterior_fset = _forward(T, edge_to_adjacency, root,
            v_to_subtree_fset)
    return v_to_posterior_fset


@ddec(params=params)
def get_edge_to_nxfset(T, edge_to_adjacency, root,
        root_prior_fset, node_to_data_fset):
    """
    For each edge, get the joint feasibility of states at edge endpoints.

    Parameters
    ----------
    {params}

    Returns
    -------
    edge_to_nxfset : map from directed edge to networkx DiGraph
        For each directed edge in the rooted tree report the networkx DiGraph
        among states, for which presence/absence of an edge defines the
        posterior feasibility of the corresponding state transition
        along the edge.

    """
    if not T:
        return {}
    v_to_fset = get_node_to_fset(T, edge_to_adjacency, root,
            root_prior_fset, node_to_data_fset)
    edge_to_nxfset = {}
    for edge in nx.bfs_edges(T, root):
        A = edge_to_adjacency[edge]
        J = nx.DiGraph()
        va, vb = edge
        for sa in v_to_fset[va]:
            sbs = set(A[sa]) & v_to_fset[vb]
            J.add_edges_from((sa, sb) for sb in sbs)
        edge_to_nxfset[edge] = J
    return edge_to_nxfset


@ddec(params=params)
def _get_root_fset(T, edge_to_adjacency, root,
        root_prior_fset, node_to_data_fset):
    """
    Get the posterior set of feasible states at the root.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_fset = _backward(T, edge_to_adjacency, root,
            root_prior_fset, node_to_data_fset)
    return v_to_subtree_fset[root]


@ddec(params=params)
def _backward(T, edge_to_adjacency, root,
        root_prior_fset, node_to_data_fset):
    """
    Determine the subtree feasible state set of each node.
    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_fset = {}
    if T:
        postorder_nodes = reversed(nx.topological_sort(T, [root]))
    else:
        postorder_nodes = [root]
    for v in postorder_nodes:
        fset_data = node_to_data_fset[v]
        if T and T[v]:
            cs = T[v]
        else:
            cs = set()
        if cs:
            fset = set()
            for s in fset_data:
                if _state_is_subtree_feasible(edge_to_adjacency,
                        v_to_subtree_fset, v, cs, s):
                    fset.add(s)
        else:
            fset = set(fset_data)
        if v == root:
            fset &= set(root_prior_fset)
        v_to_subtree_fset[v] = fset
    return v_to_subtree_fset


def _forward(T, edge_to_adjacency, root, v_to_subtree_fset):
    """
    Forward pass.

    """
    v_to_posterior_fset = {}
    v_to_posterior_fset[root] = set(v_to_subtree_fset[root])
    if not T:
        return v_to_posterior_fset
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        A = edge_to_adjacency[edge]
        fset = set()
        for s in v_to_posterior_fset[va]:
            fset.update(set(A[s]) & v_to_subtree_fset[vb])
        v_to_posterior_fset[vb] = fset
    return v_to_posterior_fset


def _state_is_subtree_feasible(edge_to_adjacency,
        v_to_subtree_fset, v, cs, s):
    """
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    v_to_subtree_fset : which states are allowed in child nodes
    v : node under consideration
    cs : child nodes of v
    s : state under consideration
    """
    for c in cs:
        edge = v, c
        A = edge_to_adjacency[edge]
        if s not in A:
            return False
        if not set(A[s]) & v_to_subtree_fset[c]:
            return False
    return True


# function suite for testing
fnsuite = (get_feas, get_node_to_fset, get_edge_to_nxfset)

