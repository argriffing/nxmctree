"""
Test joint state sampling on Markov chains on NetworkX tree graphs.
"""

from collections import defaultdict
import math

import networkx as nx

import numpy as np
from numpy.testing import (run_module_suite, assert_equal,
        assert_array_less, decorators)

import nxmctree
from nxmctree import (
        sample_history,
        sample_histories,
        get_likelihood,
        get_node_to_posterior_distn,
        get_edge_to_joint_posterior_distn,
        )
from nxmctree.puzzles import gen_random_systems
from nxmctree._sampling import dict_random_choice



def _sampling_helper(sqrt_nsamples):

    # Define an arbitrary tree.
    # The nodes 0, 1 are internal nodes.
    # The nodes 2, 3, 4, 5 are tip nodes.
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    G.add_edge(1, 4)
    G.add_edge(1, 5)

    # Define a symmetric 3-state path-like state transition matrix.
    P = nx.Graph()
    P.add_edge('a', 'a', weight=0.75)
    P.add_edge('a', 'b', weight=0.25)
    P.add_edge('b', 'a', weight=0.25)
    P.add_edge('b', 'b', weight=0.5)
    P.add_edge('b', 'c', weight=0.25)
    P.add_edge('c', 'b', weight=0.25)
    P.add_edge('c', 'c', weight=0.75)

    # Define an informative distribution at the root.
    root_prior_distn = {
            'a' : 0.6, 
            'b' : 0.4,
            }

    # Define data state restrictions at nodes.
    node_to_data_feasible_set_traditional = {
            0 : {'a', 'b', 'c'},
            1 : {'a', 'b', 'c'},
            2 : {'a'},
            3 : {'b'},
            4 : {'b'},
            5 : {'c'},
            }
    node_to_data_feasible_set_internal_constraint = {
            0 : {'a', 'b'},
            1 : {'b', 'c'},
            2 : {'a'},
            3 : {'b'},
            4 : {'b'},
            5 : {'c'},
            }

    # Try a couple of state restrictions.
    for node_to_data_feasible_set in (
            node_to_data_feasible_set_traditional,
            node_to_data_feasible_set_internal_constraint):

        # Try a couple of roots.
        for root in (0, 2):

            # Get the rooted tree.
            T = nx.dfs_tree(G, root)
            edge_to_P = dict((edge, P) for edge in T.edges())

            # Compute the exact joint distributions at edges.
            edge_to_J_exact = get_edge_to_joint_posterior_distn(
                    T, edge_to_P, root,
                    root_prior_distn, node_to_data_feasible_set)

            # Sample a bunch of joint states.
            nsamples = sqrt_nsamples * sqrt_nsamples
            edge_to_J_approx = dict(
                    (edge, nx.DiGraph()) for edge in T.edges())
            for node_to_state in sample_histories(T, edge_to_P, root,
                    root_prior_distn, node_to_data_feasible_set, nsamples):
                for tree_edge in T.edges():
                    va, vb = tree_edge
                    sa = node_to_state[va]
                    sb = node_to_state[vb]
                    J = edge_to_J_approx[tree_edge]
                    if J.has_edge(sa, sb):
                        J[sa][sb]['weight'] += 1.0
                    else:
                        J.add_edge(sa, sb, weight=1.0)
            edge_to_nx_distn = {}
            for v, J in edge_to_J_approx.items():
                total = J.size(weight='weight')
                for sa, sb in J.edges():
                    J[sa][sb]['weight'] /= total

            # Compare exact vs. approx joint state distributions on edges.
            # These should be similar up to finite sampling error.
            zstats = []
            for edge in T.edges():
                J_exact = edge_to_J_exact[edge]
                J_approx = edge_to_J_approx[edge]

                # Check that for each edge
                # the set of nonzero joint state probabilities is the same.
                # Technically this may not be true because of sampling error,
                # but we will assume that it is required.
                A = J_exact
                B = J_approx
                nodes = set(A) & set(B)
                assert_equal(set(A), nodes)
                assert_equal(set(B), nodes)
                edges = set(A.edges()) & set(B.edges())
                assert_equal(set(A.edges()), edges)
                assert_equal(set(B.edges()), edges)

                # Compute a z statistic for the error of each edge proportion.
                for sa, sb in edges:
                    p_observed = A[sa][sb]['weight']
                    p_exact = B[sa][sb]['weight']
                    num = sqrt_nsamples * (p_observed - p_exact)
                    den = math.sqrt(p_exact * (1 - p_exact))
                    z = num / den
                    zstats.append(z)

            # The z statistics should be smaller than a few standard deviations.
            assert_array_less(np.absolute(z), 4)


@decorators.slow
def test_sampling_slow():
    sqrt_nsamples = 400
    _sampling_helper(sqrt_nsamples)


def test_sampling_fast():
    sqrt_nsamples = 50
    _sampling_helper(sqrt_nsamples)


def test_puzzles():
    # Check for raised exceptions but do not check the answers.
    pzero = 0.2
    for system in gen_random_systems(pzero):
        node_to_state = sample_history(*system)


def test_empty_dict_random_choice():
    assert_equal(dict_random_choice({}), None)
