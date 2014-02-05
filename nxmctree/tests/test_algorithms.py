"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import run_module_suite, assert_equal, assert_allclose

import nxmctree
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree import (
        get_likelihood,
        get_node_to_posterior_distn,
        get_edge_to_joint_posterior_distn,
        )
from nxmctree.puzzles import gen_random_systems


def test_without_sparsity():
    # Test a special case of no data and no sparsity in the model.
    pzero = 0
    for args in gen_random_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        # likelihood
        dynamic = get_likelihood(*args)
        brute = get_likelihood(*args, efficiency='brute')
        assert_allclose(dynamic, 1)
        assert_allclose(brute, 1)

        # feasibility
        dynamic = get_likelihood(*args, variant='feasibility')
        brute = get_likelihood(*args, efficiency='brute', variant='feasibility')
        assert_equal(dynamic, True)
        assert_equal(brute, True)

        # state distributions at nodes
        dynamic = get_node_to_posterior_distn(*args)
        brute = get_node_to_posterior_distn(*args, efficiency='brute')
        for v in set(node_feas):
            assert_equal(set(dynamic[v]), node_feas[v])
            assert_equal(set(brute[v]), node_feas[v])

        # state feasibility at nodes
        dynamic = get_node_to_posterior_distn(*args, variant='feasibility')
        brute = get_node_to_posterior_distn(*args, efficiency='brute',
                variant='feasibility')
        for v in set(node_feas):
            assert_equal(dynamic[v], node_feas[v])
            assert_equal(brute[v], node_feas[v])

        # joint state distributions at edge endpoints
        dynamic = get_edge_to_joint_posterior_distn(*args)
        brute = get_edge_to_joint_posterior_distn(*args, efficiency='brute')
        for edge in T.edges():
            dynamic_edges = set(dynamic[edge].edges())
            brute_edges = set(dynamic[edge].edges())
            desired_edges = set(e_to_P[edge].edges())
            assert_equal(dynamic_edges, desired_edges)
            assert_equal(brute_edges, desired_edges)

        # joint state feasibility at edge endpoints
        dynamic = get_edge_to_joint_posterior_distn(*args,
                variant='feasibility')
        brute = get_edge_to_joint_posterior_distn(*args, efficiency='brute',
                variant='feasibility')
        for edge in T.edges():
            dynamic_edges = set(dynamic[edge].edges())
            brute_edges = set(dynamic[edge].edges())
            desired_edges = set(e_to_P[edge].edges())
            assert_equal(dynamic_edges, desired_edges)
            assert_equal(brute_edges, desired_edges)


def test_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for args in gen_random_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        # likelihood
        dynamic = get_likelihood(*args)
        brute = get_likelihood(*args, efficiency='brute')
        if dynamic is None or brute is None:
            assert_equal(dynamic, brute)
        else:
            assert_allclose(dynamic, brute)

        # feasibility
        dynamic = get_likelihood(*args, variant='feasibility')
        brute = get_likelihood(*args, efficiency='brute', variant='feasibility')
        assert_equal(dynamic, brute)

        # state distributions at nodes
        dynamic = get_node_to_posterior_distn(*args)
        brute = get_node_to_posterior_distn(*args, efficiency='brute')
        for v in set(node_feas):
            assert_dict_distn_allclose(dynamic[v], brute[v])

        # state feasibility at nodes
        dynamic = get_node_to_posterior_distn(*args, variant='feasibility')
        brute = get_node_to_posterior_distn(*args, efficiency='brute',
                variant='feasibility')
        for v in set(node_feas):
            assert_equal(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = get_edge_to_joint_posterior_distn(*args)
        brute = get_edge_to_joint_posterior_distn(*args, efficiency='brute')
        for edge in T.edges():
            assert_nx_distn_allclose(dynamic[edge], brute[edge])

        # joint state feasibility at edge endpoints
        dynamic = get_edge_to_joint_posterior_distn(*args,
                variant='feasibility')
        brute = get_edge_to_joint_posterior_distn(*args, efficiency='brute',
                variant='feasibility')
        for edge in T.edges():
            dynamic_edges = set(dynamic[edge].edges())
            brute_edges = set(brute[edge].edges())
            assert_equal(dynamic_edges, brute_edges)


if __name__ == '__main__':
    run_module_suite()

