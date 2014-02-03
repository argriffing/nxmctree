"""
Brute force likelihood calculations.

This module is only for testing.

"""

import itertools
from collections import defaultdict

import networkx as nx

import nxmctree
from nxmctree.util import dict_distn

__all__ = [
        'get_history_likelihood',
        'get_likelihood_brute',
        'get_node_to_posterior_distn_brute',
        'get_edge_to_joint_posterior_distn_brute',
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


def gen_plausible_histories(T, root, node_to_data_feasible_set):
    """
    Yield histories compatible with observed data.
    Each history is a collection of (node, state) pairs.
    Some of these histories may have zero probability.
    """
    nodes = set(node_to_data_feasible_set)
    pairs = node_to_data_feasible_set.items()
    nodes, sets = zip(*pairs)
    for assignment in itertools.product(*sets):
        yield zip(nodes, assignment)


def get_likelihood_brute(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the likelihood of this combination of parameters.

    Use brute force enumeration over all possible states.
    The meanings of the parameters are the same as for the other functions.

    """
    lk_total = None
    for history in gen_plausible_histories(T, root, node_to_data_feasible_set):
        node_to_state = dict(history)
        lk = get_history_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            if lk_total is None:
                lk_total = lk
            else:
                lk_total += lk
    return lk_total


def get_node_to_posterior_distn_brute(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the map from node to state distribution.

    Use brute force enumeration over all possible states.
    The meanings of the parameters are the same as for the other functions.

    """
    nodes = set(node_to_data_feasible_set)
    v_to_d = dict((v, defaultdict(float)) for v in nodes)
    for history in gen_plausible_histories(T, root, node_to_data_feasible_set):
        node_to_state = dict(history)
        lk = get_history_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_state)
        if lk is not None:
            for v, s in history:
                v_to_d[v][s] += lk
    v_to_posterior_distn = dict(
            (v, dict_distn(d)) for v, d in v_to_d.items())
    return v_to_posterior_distn


def get_edge_to_joint_posterior_distn_brute(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    """
    edge_to_d = dict((edge, nx.DiGraph()) for edge in T.edges())
    for history in gen_plausible_histories(T, root, node_to_data_feasible_set):
        node_to_state = dict(history)
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

