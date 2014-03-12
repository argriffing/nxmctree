"""
Joint state sampling algorithm for a Markov chain on a NetworkX tree graph.

"""
from __future__ import division, print_function, absolute_import

import random

import networkx as nx

from nxmctree import dynamic_fset_lhood, dynamic_lmap_lhood

__all__ = [
        'sample_history',
        'sample_histories',
        ]


def dict_random_choice(d):
    total = sum(d.values())
    x = random.uniform(0, total)
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def sample_history_fset(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset):
    """
    Jointly sample states on a tree.
    This is called a history.

    """
    v_to_subtree_partial_likelihoods = dynamic_fset_lhood._backward(
            T, edge_to_P, root, root_prior_distn, node_to_data_fset)
    node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return node_to_state


def sample_histories_fset(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset, nhistories):
    """
    Sample multiple history.
    Each history is a joint sample of states on the tree.

    """
    v_to_subtree_partial_likelihoods = dynamic_fset_lhood._backward(
            T, edge_to_P, root, root_prior_distn, node_to_data_fset)
    for i in range(nhistories):
        node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
                v_to_subtree_partial_likelihoods)
        yield node_to_state


def sample_history(T, edge_to_P, root,
        root_prior_distn, node_to_data_lmap):
    """
    Jointly sample states on a tree.
    This is called a history.

    """
    v_to_subtree_partial_likelihoods = dynamic_lmap_lhood._backward(
            T, edge_to_P, root, root_prior_distn, node_to_data_lmap)
    node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return node_to_state


def sample_histories(T, edge_to_P, root,
        root_prior_distn, node_to_data_lmap, nhistories):
    """
    Sample multiple history.
    Each history is a joint sample of states on the tree.

    """
    v_to_subtree_partial_likelihoods = dynamic_lmap_lhood._backward(
            T, edge_to_P, root, root_prior_distn, node_to_data_lmap)
    for i in range(nhistories):
        node_to_state = _sample_states_preprocessed(T, edge_to_P, root,
                v_to_subtree_partial_likelihoods)
        yield node_to_state


def _sample_states_preprocessed(T, edge_to_P, root,
        v_to_subtree_partial_likelihoods):
    """
    Jointly sample states on a tree.

    This variant requires subtree partial likelihoods.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    if not root_partial_likelihoods:
        return None
    v_to_sampled_state = {}
    v_to_sampled_state[root] = dict_random_choice(root_partial_likelihoods)
    if not T:
        return v_to_sampled_state
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]

        # For the relevant parent state,
        # compute an unnormalized distribution over child states.
        sa = v_to_sampled_state[va]

        # Construct conditional transition probabilities.
        fset = set(P[sa]) & set(v_to_subtree_partial_likelihoods[vb])
        sb_weights = {}
        for sb in fset:
            a = P[sa][sb]['weight']
            b = v_to_subtree_partial_likelihoods[vb][sb]
            sb_weights[sb] = a * b

        # Sample the state using the unnormalized dictionary of weights.
        v_to_sampled_state[vb] = dict_random_choice(sb_weights)

    return v_to_sampled_state

