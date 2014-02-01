"""
Markov chain algorithms to compute likelihoods on trees.

The NetworkX digraph representing a sparse transition probability matrix
will be represented by the notation 'P'.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = [
        'get_likelihood',
        'get_root_posterior_partial_likelihoods',
        #'get_node_to_posterior_feasible_set',
        #'get_edge_to_joint_posterior_feasibility',

        # This function is for testing.
        #'get_feasibility_info_slow',
        ]



def get_likelihood(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the feasibility of this combination of parameters.

    The meanings of the parameters are the same as for the other functions.

    """
    root_post = get_root_posterior_partial_likelihoods(T, edge_to_P, root,
            root_prior_distn, node_to_data_feasible_set)
    if root_post:
        return sum(root_post.values())
    else:
        return None


def get_root_posterior_partial_likelihoods(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the posterior set of feasible states at the root.

    The meanings of the parameters are the same as for the other functions.

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_feasible_set)
    return v_to_subtree_partial_likelihoods[root]


def _backward(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Determine the subtree feasible state set of each node.
    This is the backward pass of a backward-forward algorithm.

    """
    v_to_subtree_partial_likelihoods = {}
    for v in reversed(nx.topological_sort(T, [root])):
        cs = T[v]
        fset_data = node_to_data_feasible_set[v]
        if cs:
            partial_likelihoods = {}
            for s in fset_data:
                prob =  _get_partial_likelihood(edge_to_P,
                        v_to_subtree_partial_likelihoods, v, cs, s)
                if prob is not None:
                    partial_likelihoods[s] = prob
        else:
            partial_likelihoods = dict((s, 1) for s in fset_data)
        if v == root:
            pnext = {}
            for s in set(partial_likelihoods) & set(root_prior_distn):
                pnext[s] = partial_likelihoods[s] * root_prior_distn[s]
            partial_likelihoods = pnext
        v_to_subtree_partial_likelihoods[v] = partial_likelihoods
    return v_to_subtree_partial_likelihoods


def _get_partial_likelihood(edge_to_P,
        v_to_subtree_partial_likelihoods, v, cs, s):
    """
    edge_to_P : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition probability.
    v_to_subtree_partial_likelihoods : map a node to dict of partial likelihoods
    v : node under consideration
    cs : child nodes of v
    s : state under consideration
    """
    prob = 1.0
    for c in cs:
        edge = v, c
        P = edge_to_P[edge]
        if s not in P:
            return None
        cstates = set(P[s]) & set(v_to_subtree_partial_likelihoods[c])
        if not cstates:
            return None
        p = 0.0
        for cstate in cstates:
            p_trans = P[s][cstate]['weight'] 
            p_subtree = v_to_subtree_partial_likelihoods[c][cstate]
            p += p_trans * p_subtree
        prob *= p
    return prob

