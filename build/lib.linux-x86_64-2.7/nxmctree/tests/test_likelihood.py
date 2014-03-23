"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

Test the likelihood calculation for a single specific example.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_allclose)

import nxmctree
from nxmctree import dynamic_fset_lhood, brute_fset_lhood
from nxmctree import dynamic_lmap_lhood, brute_lmap_lhood


def test_dynamic_history_likelihood():
    # In this test the history is completely specified.

    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)

    # Define the rooted tree.
    root = 0
    T = nx.dfs_tree(G, root)

    # The data completely restricts the set of states.
    node_to_data_fset = {
            0 : {'a'},
            1 : {'a'},
            2 : {'a'},
            3 : {'a'},
            }

    # The data completely restricts the set of states and includes likelihood.
    node_to_data_lmap = {
            0 : {'a' : 0.1},
            1 : {'a' : 0.2},
            2 : {'a' : 0.3},
            3 : {'a' : 0.4},
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
    desired_fset_likelihood = (0.5 ** 4)
    desired_lmap_likelihood = (0.5 ** 4) * (0.1 * 0.2 * 0.3 * 0.4)

    # Compare to brute fset likelihood.
    actual_likelihood = brute_fset_lhood.get_lhood_brute(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    assert_allclose(actual_likelihood, desired_fset_likelihood)

    # Compare to dynamic fset likelihood.
    actual_likelihood = dynamic_fset_lhood.get_lhood(T, edge_to_P, root,
            root_prior_distn, node_to_data_fset)
    assert_allclose(actual_likelihood, desired_fset_likelihood)

    # Compare to brute lmap likelihood.
    actual_likelihood = brute_lmap_lhood.get_lhood_brute(T, edge_to_P, root,
            root_prior_distn, node_to_data_lmap)
    assert_allclose(actual_likelihood, desired_lmap_likelihood)

    # Compare to dynamic lmap likelihood.
    actual_likelihood = dynamic_lmap_lhood.get_lhood(T, edge_to_P, root,
            root_prior_distn, node_to_data_lmap)
    assert_allclose(actual_likelihood, desired_lmap_likelihood)

