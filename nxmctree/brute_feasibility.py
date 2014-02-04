"""
Brute force feasibility calculations.

This module is only for testing.

"""

import networkx as nx

import nxmctree
from nxmctree.history import gen_plausible_histories

__all__ = [
        'get_history_feasibility',
        'get_feasibility_brute',
        'get_node_to_posterior_feasible_set_brute',
        'get_edge_to_joint_posterior_feasibility_brute',
        ]


def get_history_feasibility(T, edge_to_A, root,
        root_prior_feasible_set, node_to_state):
    """
    Compute the feasibility of a single specific history.
    """
    root_state = node_to_state[root]
    if root_state not in root_prior_feasible_set:
        return False
    if not T:
        return True
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        A = edge_to_A[edge]
        sa = node_to_state[va]
        sb = node_to_state[vb]
        if sa not in P:
            return False
        if sb not in P[sa]:
            return False
    return True


def get_feasibility_brute(T, edge_to_A, root,
        root_prior_feasible_set, node_to_data_feasible_set):
    """
    Get the feasibility of this combination of parameters.

    Use brute force enumeration over all possible states.
    The meanings of the parameters are the same as for the other functions.

    """
    args = T, edge_to_P, root, root_prior_feasible_state, node_to_state
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        if not get_history_feasibility(T, edge_to_P, root,
                root_prior_feasible_state, node_to_state):
            return False
    return True


def get_node_to_posterior_feasible_set_brute(T, edge_to_A, root,
        root_prior_feasible_set, node_to_data_feasible_set):
    """
    Get the map from node to state feasibility.

    Use brute force enumeration over all histories.
    The meanings of the parameters are the same as for the other functions.

    """
    nodes = set(node_to_data_feasible_set)
    v_to_feas = dict((v, set()) for v in nodes)
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        if get_history_likelihood(T, edge_to_A, root,
                root_prior_feasible_set, node_to_state):
            for node, state in node_to_state.items():
                v_to_feas[node].add(state)
    return v_to_feas


def get_edge_to_joint_posterior_feasibility_brute(T, edge_to_A, root,
        root_prior_feasible_set, node_to_data_feasible_set):
    """
    Use brute force enumeration over all histories.
    The meanings of the parameters are the same as for the other functions.
    """
    edge_to_d = dict((edge, nx.DiGraph()) for edge in T.edges())
    for node_to_state in gen_plausible_histories(node_to_data_feasible_set):
        if get_history_feasibility(T, edge_to_A, root,
                root_prior_feasible_set, node_to_state):
            for tree_edge in T.edges():
                va, vb = tree_edge
                sa = node_to_state[va]
                sb = node_to_state[vb]
                edge_to_d[tree_edge].add_edge(sa, sb)
    return edge_to_d

