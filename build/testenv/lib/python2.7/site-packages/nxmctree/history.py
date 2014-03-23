"""
Functions related to histories on trees.

Every node in a Markov chain tree history has a known state.

"""

import itertools

import networkx as nx

__all__ = [
        'get_history_lhood',
        'get_history_feas',
        'gen_plausible_histories',
        ]


def get_history_feas(T, edge_to_A, root, root_prior_fset, node_to_state):
    """
    Compute the feasibility of a single specific history.
    """
    root_state = node_to_state[root]
    if root_state not in root_prior_fset:
        return False
    if not T:
        return True
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        A = edge_to_A[edge]
        sa = node_to_state[va]
        sb = node_to_state[vb]
        if sa not in A:
            return False
        if sb not in A[sa]:
            return False
    return True


def get_history_lhood(T, edge_to_P, root, root_prior_distn, node_to_state):
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


def gen_plausible_histories(node_to_data_fset):
    """
    Yield histories compatible with directly observed data.

    Each history is a map from node to state.
    Some of these histories may have zero probability when the
    shape of the tree and the structure of the transition matrices
    is taken into account.

    """
    nodes = set(node_to_data_fset)
    pairs = node_to_data_fset.items()
    nodes, sets = zip(*pairs)
    for assignment in itertools.product(*sets):
        yield dict(zip(nodes, assignment))

