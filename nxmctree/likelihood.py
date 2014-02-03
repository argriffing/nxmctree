"""
Markov chain algorithms to compute likelihoods and distributions on trees.

The NetworkX digraph representing a sparse transition probability matrix
will be represented by the notation 'P'.
State distributions are represented by sparse dicts.
Joint endpoint state distributions are represented by networkx graphs.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import networkx as nx

import nxmctree
from nxmctree.util import dict_distn

__all__ = [
        'get_likelihood',
        'get_root_posterior_partial_likelihoods',
        'get_node_to_posterior_distn',
        'get_edge_to_joint_posterior_distn',
        ]


def get_likelihood(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the likelihood of this combination of parameters.

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
    Get the posterior distribution at the root.

    The meanings of the parameters are the same as for the other functions.

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_feasible_set)
    return v_to_subtree_partial_likelihoods[root]


def get_node_to_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the map from node to state distribution.

    The meanings of the parameters are the same as for the other functions.

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_feasible_set)
    v_to_posterior_distn = _forward(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return v_to_posterior_distn


def get_edge_to_joint_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set):
    """
    Get the map from edge to joint state distribution at endpoint nodes.

    The meanings of the parameters are the same as for the other functions.

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_feasible_set)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


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


def _forward(T, edge_to_P, root,
        v_to_subtree_partial_likelihoods):
    """
    Forward pass.

    Return a map from node to posterior state distribution.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    v_to_posterior_distn = {}
    v_to_posterior_distn[root] = dict_distn(root_partial_likelihoods)
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]

        # For each parent state, compute the distribution over child states.
        distn = defaultdict(float)
        parent_distn = v_to_posterior_distn[va]
        for sa, pa in parent_distn.items():

            # Construct conditional transition probabilities.
            fset = set(P[sa]) & set(v_to_subtree_partial_likelihoods[vb])
            sb_weights = {}
            for sb in fset:
                a = P[sa][sb]['weight']
                b = v_to_subtree_partial_likelihoods[vb][sb]
                sb_weights[sb] = a * b
            sb_distn = dict_distn(sb_weights)

            # Add to the marginal distribution.
            for sb, pb in sb_distn.items():
                distn[sb] += pa * pb

        v_to_posterior_distn[vb] = distn

    return v_to_posterior_distn


def _forward_edges(T, edge_to_P, root,
        v_to_subtree_partial_likelihoods):
    """
    Forward pass.

    Return a map from edge to joint state distribution.
    Also calculate the posterior state distributions at nodes,
    but do not return them.

    """
    root_partial_likelihoods = v_to_subtree_partial_likelihoods[root]
    v_to_posterior_distn = {}
    v_to_posterior_distn[root] = dict_distn(root_partial_likelihoods)
    edge_to_J = dict((edge, nx.DiGraph()) for edge in T.edges())
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        P = edge_to_P[edge]
        J = edge_to_J[edge]

        # For each parent state, compute the distribution over child states.
        distn = defaultdict(float)
        parent_distn = v_to_posterior_distn[va]
        for sa, pa in parent_distn.items():

            # Construct conditional transition probabilities.
            fset = set(P[sa]) & set(v_to_subtree_partial_likelihoods[vb])
            sb_weights = {}
            for sb in fset:
                a = P[sa][sb]['weight']
                b = v_to_subtree_partial_likelihoods[vb][sb]
                sb_weights[sb] = a * b
            sb_distn = dict_distn(sb_weights)

            # Add to the joint and marginal distribution.
            for sb, pb in sb_distn.items():
                p = pa * pb
                distn[sb] += p
                J.add_edge(sa, sb, weight=p)

        v_to_posterior_distn[vb] = distn

    return edge_to_J


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

