"""
Joint state sampling algorithm for a Markov chain on a NetworkX tree graph.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = [
        'sample_states',
        ]


def dict_random_choice(d):
    if not d:
        raise ValueError('empty support')
    total = sum(d.values())
    x = random.uniform(0, total)
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def sample_states(T, edge_to_P, root, v_to_subtree_partial_likelihoods):
    """
    Jointly sample states on a tree.

    NOTE: this is like raoteh/sampler/_sample_mc0.py

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    v_to_sampled_state = {}
    v_to_sampled_state[root] = dict_random_choice(root_partial_likelihoods)
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
        v_to_sampled_state[sb] = dict_random_choice(sb_weights)

    return v_to_sampled_state

