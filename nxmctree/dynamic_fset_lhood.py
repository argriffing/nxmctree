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
from nxmctree.util import ddec, dict_distn

__all__ = [
        'get_lhood',
        'get_node_to_distn',
        'get_edge_to_nxdistn',
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
def get_lhood(T, edge_to_P, root, root_prior_distn, node_to_data_fset):
    """
    Get the likelihood of this combination of parameters.

    Parameters
    ----------
    {params}

    Returns
    -------
    lhood : float or None
        If the data is structurally supported by the model then
        return the likelihood, otherwise None.

    """
    root_lhoods = _get_root_lhoods(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    return sum(root_lhoods.values()) if root_lhoods else None


@ddec(params=params)
def get_node_to_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset):
    """
    Get the map from node to state distribution.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    v_to_posterior_distn = _forward(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return v_to_posterior_distn


@ddec(params=params)
def get_edge_to_nxdistn(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset):
    """
    Get the map from edge to joint state distribution at endpoint nodes.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


@ddec(params=params)
def _get_root_lhoods(T, edge_to_P, root,
        root_prior_distn, node_to_data_fset):
    """
    Get the posterior likelihoods at the root, conditional on root state.

    These are also known as partial likelihoods.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    return v_to_subtree_partial_likelihoods[root]


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn, node_to_data_fset):
    """
    Determine the subtree feasible state set of each node.

    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = {}
    for v in nx.topological_sort(T, [root], reverse=True):
        fset_data = node_to_data_fset[v]
        if T and T[v]:
            cs = T[v]
        else:
            cs = set()
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


def _forward(T, edge_to_P, root, v_to_subtree_partial_likelihoods):
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


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn, get_edge_to_nxdistn)

