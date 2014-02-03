"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

import random

import networkx as nx
from numpy.testing import run_module_suite, assert_equal, assert_allclose

import nxmctree
from nxmctree.util import dict_distn
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree.likelihood import (
        get_likelihood,
        get_root_posterior_partial_likelihoods,
        get_node_to_posterior_distn,
        get_edge_to_joint_posterior_distn,
        )
from nxmctree.lkbrute import (
        get_likelihood_brute,
        get_node_to_posterior_distn_brute,
        get_edge_to_joint_posterior_distn_brute,
        )

def _random_transition_graph(states, pzero):
    """
    Return a random transition matrix as a networkx digraph.
    Some entries may be zero.
    states : set of states
    pzero : probability that any given transition has nonzero probability
    """
    P = nx.DiGraph()
    for sa in states:
        for sb in states:
            if random.random() > pzero:
                P.add_edge(sa, sb, weight=random.expovariate(1))
    for sa in states:
        if sa in P:
            total = sum(P[sa][sb]['weight'] for sb in P[sa])
            for sb in P[sa]:
                P[sa][sb]['weight'] /= total
    return P


def _random_dict_distn(states, pzero):
    """
    Return a random state distribution as a dict.
    Some entries may be zero.
    states : set of states
    pzero : probability that any given state has nonzero probability
    """
    fset = set(s for s in states if random.random() > pzero)
    d = dict((s, random.expovariate(1)) for s in fset)
    return dict_distn(d)


def _random_data_feasible_sets(nodes, states, pzero):
    """
    Return a map from node to feasible state set.
    states : set of states
    pzero : probability that any given state is infeasible
    """
    d = {}
    for v in nodes:
        fset = set(s for s in states if random.random() > pzero)
        d[v] = fset
    return d


def _gen_random_systems(pzero):
    """
    Sample whole systems, where pzero indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set).
    """
    nsamples = 10

    # Make some random systems with a single node.
    for i in range(nsamples):
        root = 42
        nodes = {root}
        states = set(['a', 'b', 'c'])
        T = nx.DiGraph()
        root_prior_distn = _random_dict_distn(states, pzero)
        edge_to_P = {}
        node_to_data_feasible_set = _random_data_feasible_sets(
                nodes, states, pzero)
        yield (T, edge_to_P, root,
                root_prior_distn, node_to_data_feasible_set)

    # Make some random systems with multiple nodes.
    nodes = set(range(4))
    states = set(['a', 'b', 'c'])

    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)

    for i in range(nsamples):

        for root in nodes:

            T = nx.dfs_tree(G, root)
            root_prior_distn = _random_dict_distn(states, pzero)
            edge_to_P = {}
            for edge in T.edges():
                P = _random_transition_graph(states, pzero)
                edge_to_P[edge] = P
            node_to_data_feasible_set = _random_data_feasible_sets(
                    nodes, states, pzero)

            yield (T, edge_to_P, root,
                    root_prior_distn, node_to_data_feasible_set)


def test_likelihood_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in _gen_random_systems(pzero):
        lk_dynamic = get_likelihood(T, e_to_P, r, r_prior, node_feas)
        lk_brute = get_likelihood_brute(T, e_to_P, r, r_prior, node_feas)
        if lk_dynamic is None or lk_brute is None:
            assert_equal(lk_dynamic, lk_brute)
        else:
            assert_allclose(lk_dynamic, lk_brute)


def test_unrestricted_likelihood():
    # When there is no data restriction the likelihood should be 1.
    pzero = 0
    for T, e_to_P, r, r_prior, node_feas in _gen_random_systems(pzero):
        lk = get_likelihood(T, e_to_P, r, r_prior, node_feas)
        assert_allclose(lk, 1)


def test_node_posterior_distns_dynamic_vs_brute():
    # Check that both methods give the same posterior distributions.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in _gen_random_systems(pzero):
        v_to_d = get_node_to_posterior_distn(
                T, e_to_P, r, r_prior, node_feas)
        v_to_d_brute = get_node_to_posterior_distn_brute(
                T, e_to_P, r, r_prior, node_feas)
        for v in set(node_feas):
            assert_dict_distn_allclose(v_to_d[v], v_to_d_brute[v])


def test_edge_joint_posterior_distn_dynamic_vs_brute():
    # Check that both methods give the same posterior distributions.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in _gen_random_systems(pzero):
        edge_to_J = get_edge_to_joint_posterior_distn(
                T, e_to_P, r, r_prior, node_feas)
        edge_to_J_brute = get_edge_to_joint_posterior_distn_brute(
                T, e_to_P, r, r_prior, node_feas)
        for edge in T.edges():
            assert_nx_distn_allclose(
                    edge_to_J[edge], edge_to_J_brute[edge])


if __name__ == '__main__':
    run_module_suite()

