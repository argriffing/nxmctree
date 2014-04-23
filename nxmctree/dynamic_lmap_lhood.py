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
from nxmctree.dynamic_fset_lhood import (
        _get_partial_likelihood, _forward, _forward_edges)

__all__ = [
        'get_lhood',
        'get_node_to_distn',
        'get_edge_to_nxdistn',
        ]


params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_P : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition probability.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn : dict
        Prior state distribution at the root.
    node_to_data_lmap : dict
        For each node, a map from state to observation likelihood.
"""


@ddec(params=params)
def get_lhood(T, edge_to_P, root, root_prior_distn, node_to_data_lmap):
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
            root_prior_distn, node_to_data_lmap)
    return sum(root_lhoods.values()) if root_lhoods else None


@ddec(params=params)
def get_node_to_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_lmap):
    """
    Get the map from node to state distribution.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_lmap)
    v_to_posterior_distn = _forward(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return v_to_posterior_distn


@ddec(params=params)
def get_edge_to_nxdistn(T, edge_to_P, root,
        root_prior_distn, node_to_data_lmap):
    """
    Get the map from edge to joint state distribution at endpoint nodes.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_lmap)
    edge_to_J = _forward_edges(T, edge_to_P, root,
            v_to_subtree_partial_likelihoods)
    return edge_to_J


@ddec(params=params)
def _get_root_lhoods(T, edge_to_P, root,
        root_prior_distn, node_to_data_lmap):
    """
    Get the posterior likelihoods at the root, conditional on root state.

    These are also known as partial likelihoods.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = _backward(T, edge_to_P, root,
            root_prior_distn, node_to_data_lmap)
    return v_to_subtree_partial_likelihoods[root]


@ddec(params=params)
def _backward(T, edge_to_P, root, root_prior_distn, node_to_data_lmap):
    """
    Determine the subtree partial likelihoods for each node.

    This is the backward pass of a backward-forward algorithm.

    Parameters
    ----------
    {params}

    """
    v_to_subtree_partial_likelihoods = {}
    for v in nx.topological_sort(T, [root], reverse=True):
        lmap_data = node_to_data_lmap[v]
        if T and T[v]:
            cs = T[v]
        else:
            cs = set()
        if cs:
            partial_likelihoods = {}
            for s, lk_obs in lmap_data.items():
                prob =  _get_partial_likelihood(edge_to_P,
                        v_to_subtree_partial_likelihoods, v, cs, s)
                if prob is not None:
                    partial_likelihoods[s] = prob * lk_obs
        else:
            partial_likelihoods = lmap_data
        if v == root:
            pnext = {}
            for s in set(partial_likelihoods) & set(root_prior_distn):
                pnext[s] = partial_likelihoods[s] * root_prior_distn[s]
            partial_likelihoods = pnext
        v_to_subtree_partial_likelihoods[v] = partial_likelihoods
    return v_to_subtree_partial_likelihoods


# function suite for testing
fnsuite = (get_lhood, get_node_to_distn, get_edge_to_nxdistn)

