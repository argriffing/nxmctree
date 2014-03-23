"""
Brute force likelihood calculations.

This module is only for testing.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import networkx as nx

import nxmctree
from nxmctree.util import ddec, dict_distn
from nxmctree.history import (
        get_history_feas, get_history_lhood, gen_plausible_histories)

__all__ = [
        'get_lhood_brute',
        'get_node_to_distn_brute',
        'get_edge_to_nxdistn_brute',
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
    root_prior_distn : dict
        Prior state distribution at the root.
    node_to_data_fset : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""


@ddec(params=params)
def get_lhood_brute(T, edge_to_P, root, root_prior_distn, node_to_data_fset):
    """
    Get the likelihood of this combination of parameters.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    lk_total = None
    for node_to_state in gen_plausible_histories(node_to_data_fset):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


@ddec(params=params)
def get_node_to_distn_brute(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible states.

    Parameters
    ----------
    {params}

    """
    nodes = set(node_to_data_fset)
    v_to_d = dict((v, defaultdict(float)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_fset):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            for node, state in node_to_state.items():
                v_to_d[node][state] += lk
    v_to_posterior_distn = dict((v, dict_distn(d)) for v, d in v_to_d.items())
    return v_to_posterior_distn


@ddec(params=params)
def get_edge_to_nxdistn_brute(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """

    Parameters
    ----------
    {params}

    """
    edge_to_d = dict((edge, nx.DiGraph()) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        lk = get_history_lhood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                d = edge_to_d[tree_edge]
                if d.has_edge(sa, sb):
                    d[sa][sb]['weight'] += lk
                else:
                    d.add_edge(sa, sb, weight=lk)
    for tree_edge in T.edges():
        d = edge_to_d[tree_edge]
        total = d.size(weight='weight')
        for sa, sb in d.edges():
            d[sa][sb]['weight'] /= total
    return edge_to_d


# function suite for testing
fnsuite = (get_lhood_brute, get_node_to_distn_brute, get_edge_to_nxdistn_brute)

