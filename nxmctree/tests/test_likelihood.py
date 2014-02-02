"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

"""
from __future__ import division, print_function, absolute_import

import random

import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_equal, assert_allclose)

import nxmctree
from nxmctree.util import dict_distn
from nxmctree.nputil import assert_dict_distn_allclose
from nxmctree.likelihood import (
        get_likelihood,
        get_likelihood_brute,
        get_history_likelihood,
        get_root_posterior_partial_likelihoods,
        get_node_to_posterior_distn,
        get_node_to_posterior_distn_brute,
        #get_edge_to_joint_posterior_feasibility,
        #get_feasibility_info_slow,
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


class Test_Likelihood(TestCase):


    def test_likelihood_dynamic_vs_brute(self):
        # Compare dynamic programming vs. summing over all histories.

        nodes = set(range(4))
        states = set(['a', 'b', 'c'])

        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)

        nsamples = 20
        for i in range(nsamples):

            for root in nodes:

                pzero = 0.2
                T = nx.dfs_tree(G, root)
                root_prior_distn = _random_dict_distn(states, pzero)
                edge_to_P = {}
                for edge in T.edges():
                    P = _random_transition_graph(states, pzero)
                    edge_to_P[edge] = P
                node_to_data_feasible_set = _random_data_feasible_sets(
                        nodes, states, pzero)

                lk_dynamic = get_likelihood(
                        T, edge_to_P, root,
                        root_prior_distn, node_to_data_feasible_set)
                lk_brute = get_likelihood_brute(
                        T, edge_to_P, root,
                        root_prior_distn, node_to_data_feasible_set)

                if lk_dynamic is None or lk_brute is None:
                    assert_equal(lk_dynamic, lk_brute)
                else:
                    assert_allclose(lk_dynamic, lk_brute)


    def test_unrestricted_likelihood(self):
        # When there is no data restriction the likelihood should be 1.

        nsamples = 10
        for i in range(nsamples):

            nodes = set(range(4))
            states = set(['a', 'b', 'c'])

            G = nx.Graph()
            G.add_edge(0, 1)
            G.add_edge(0, 2)
            G.add_edge(0, 3)

            for root in nodes:

                pzero = 0
                T = nx.dfs_tree(G, root)
                root_prior_distn = _random_dict_distn(states, pzero)
                edge_to_P = {}
                for edge in T.edges():
                    P = _random_transition_graph(states, pzero)
                    edge_to_P[edge] = P
                node_to_data_feasible_set = _random_data_feasible_sets(
                        nodes, states, pzero)

                lk = get_likelihood(T, edge_to_P, root,
                        root_prior_distn, node_to_data_feasible_set)

                assert_allclose(lk, 1)

    def test_dynamic_history_likelihood(self):
        # In this test the history is completely specified.

        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)

        # Define the rooted tree.
        root = 0
        T = nx.dfs_tree(G, root)

        # The data completely restricts the set of states.
        node_to_data_feasible_set = {
                0 : {'a'},
                1 : {'a'},
                2 : {'a'},
                3 : {'a'},
                }

        # The root prior distribution is informative.
        root_prior_distn = {
                'a' : 0.5,
                'b' : 0.5,
                'c' : 0,
                'd' : 0,
                }

        # Define the transition matrix.
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            ('a', 'a', 0.5),
            ('a', 'b', 0.25),
            ('a', 'c', 0.25),
            ('b', 'b', 0.5),
            ('b', 'c', 0.25),
            ('b', 'a', 0.25),
            ('c', 'c', 0.5),
            ('c', 'a', 0.25),
            ('c', 'b', 0.25)])

        # Associate each edge with the transition matrix.
        edge_to_P = dict((edge, P) for edge in T.edges())

        # The likelihood is simple in this case.
        desired_likelihood = 0.5 ** 4

        # Compute the likelhood.
        actual_likelihood = get_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_data_feasible_set)

        # Check that the likelihood is correct.
        assert_equal(actual_likelihood, desired_likelihood)


    def test_node_posterior_distns_dynamic_vs_brute(self):
        # Check that both methods give the same posterior distributions.

        nodes = set(range(4))
        states = set(['a', 'b', 'c'])

        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)

        nsamples = 10
        for i in range(nsamples):

            for root in nodes:

                pzero = 0.2
                T = nx.dfs_tree(G, root)
                root_prior_distn = _random_dict_distn(states, pzero)
                edge_to_P = {}
                for edge in T.edges():
                    P = _random_transition_graph(states, pzero)
                    edge_to_P[edge] = P
                node_to_data_feasible_set = _random_data_feasible_sets(
                        nodes, states, pzero)

                v_to_d = get_node_to_posterior_distn(
                        T, edge_to_P, root,
                        root_prior_distn, node_to_data_feasible_set)
                v_to_d_brute = get_node_to_posterior_distn_brute(
                        T, edge_to_P, root,
                        root_prior_distn, node_to_data_feasible_set)

                for v in nodes:
                    assert_dict_distn_allclose(v_to_d[v], v_to_d_brute[v])


if __name__ == '__main__':
    run_module_suite()

