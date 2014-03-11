"""
Brute force feasibility calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

import nxmctree
from nxmctree.docspam import ddec
from nxmctree.history import get_history_feas, gen_plausible_histories

__all__ = [
        'get_feas_brute',
        'get_node_to_fset_brute',
        'get_edge_to_nxfset_brute',
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
def get_feas_brute(T, edge_to_A, root, root_prior_fset, node_to_data_fset):
    """
    Get the feasibility of this combination of parameters.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    Returns
    -------
    feas : bool
        True if the data is structurally supported by the model,
        otherwise False.

    """
    for node_to_state in gen_plausible_histories(node_to_data_fset):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fset, node_to_state):
            return True
    return False


@ddec(params=params)
def get_node_to_fset_brute(T, edge_to_A, root,
        root_prior_fset, node_to_data_fset):
    """
    Get the map from node to state feasibility.

    Use brute force enumeration over all histories.

    Parameters
    ----------
    {params}

    Returns
    -------
    node_to_posterior_fset : dict
        Map from node to set of posterior feasible states.

    """
    nodes = set(node_to_data_fset)
    v_to_feas = dict((v, set()) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_fset):
        if get_history_feas(T, edge_to_A, root, root_prior_fset, node_to_state):
            for node, state in node_to_state.items():
                v_to_feas[node].add(state)
    return v_to_feas


@ddec(params=params)
def get_edge_to_nxfset_brute(T, edge_to_A, root,
        root_prior_fset, node_to_data_fset):
    """
    Use brute force enumeration over all histories.

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
    edge_to_d = dict((edge, nx.DiGraph()) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_fset):
        if get_history_feas(T, edge_to_A, root,
                root_prior_fset, node_to_state):
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                edge_to_d[tree_edge].add_edge(sa, sb)
    return edge_to_d


# function suite for testing
fnsuite = (get_feas_brute, get_node_to_fset_brute, get_edge_to_nxfset_brute)
