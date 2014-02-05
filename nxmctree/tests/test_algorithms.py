"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import (run_module_suite, assert_,
        assert_equal, assert_allclose)

import nxmctree
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree import (
        get_likelihood,
        get_node_to_posterior_distn,
        get_edge_to_joint_posterior_distn,
        )
from nxmctree.puzzles import gen_random_systems


def test_full_sparsity():
    # Test a special case of full sparsity in the system.
    pzero = 1
    for args in gen_random_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        for efficiency in 'dynamic', 'brute':
            for variant in 'likelihood', 'feasibility':

                # likelihood or feasibility
                likelihood = get_likelihood(*args,
                        efficiency=efficiency, variant=variant)
                assert_(not likelihood)

                # state distributions or feasible sets at nodes
                d = get_node_to_posterior_distn(*args,
                        efficiency=efficiency, variant=variant)
                assert_(not any(d.values()))

                # joint state distributions at edge endpoints
                d = get_edge_to_joint_posterior_distn(*args,
                        efficiency=efficiency, variant=variant)
                for edge in T.edges():
                    assert_(not d[edge].edges())


def test_zero_sparsity():
    # Test a special case of no sparsity in the system.
    pzero = 0
    for args in gen_random_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        for efficiency in 'dynamic', 'brute':

            # likelihood
            likelihood = get_likelihood(*args, efficiency=efficiency)
            assert_allclose(likelihood, 1)

            # feasibility
            feasibility = get_likelihood(*args,
                efficiency=efficiency, variant='feasibility')
            assert_equal(feasibility, True)

            for variant in 'likelihood', 'feasibility':

                # state distributions at nodes
                d = get_node_to_posterior_distn(*args,
                        efficiency=efficiency, variant=variant)
                for v in set(node_feas):
                    assert_equal(set(d[v]), node_feas[v])

                # joint state distributions at edge endpoints
                d = get_edge_to_joint_posterior_distn(*args,
                        efficiency=efficiency, variant=variant)
                for edge in T.edges():
                    observed_edges = set(d[edge].edges())
                    desired_edges = set(e_to_P[edge].edges())
                    assert_equal(observed_edges, desired_edges)


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

