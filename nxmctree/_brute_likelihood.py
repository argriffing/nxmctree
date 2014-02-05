"""
Brute force likelihood calculations.

This module is only for testing.

"""

from collections import defaultdict

import networkx as nx

import nxmctree
from nxmctree.util import dict_distn
from nxmctree.history import gen_plausible_histories

__all__ = [
        'get_history_likelihood',
        'get_likelihood',
        'get_node_to_posterior_distn',
        'get_edge_to_joint_posterior_distn',
        ]


def get_history_likelihood(T, edge_to_P, root,
        root_prior_distn, node_to_state):
    """
    Compute the probability of a single specific history.
    """
    root_state = node_to_state[root]
    if root_state not in root_prior_distn:
        return None
    lk = root_prior_distn[root_state]
    if not T:
        return lk
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]
        sa = node_to_state[va]
        sb = node_to_state[vb]
        if sa not in P:
            return None
        if sb not in P[sa]:
            return None
        lk *= P[sa][sb]['weight']
    return lk


def get_likelihood(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the likelihood of this combination of parameters.

    Use brute force enumeration over all possible states.
    The meanings of the parameters are the same as for the other functions.

    """
    lk_total = None
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        lk = get_history_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


def get_node_to_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible states.
    The meanings of the parameters are the same as for the other functions.

    """
    nodes = set(node_to_data_feasible_set)
    v_to_d = dict((v, defaultdict(float)) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        lk = get_history_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            for node, state in node_to_state.items():
                v_to_d[node][state] += lk
    v_to_posterior_distn = dict(
            (v, dict_distn(d)) for v, d in v_to_d.items())
    return v_to_posterior_distn


def get_edge_to_joint_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    """
    edge_to_d = dict((edge, nx.DiGraph()) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        lk = get_history_likelihood(T, edge_to_P, root,
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

