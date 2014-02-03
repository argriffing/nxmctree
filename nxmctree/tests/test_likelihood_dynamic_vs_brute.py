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
from nxmctree.likelihood import (
        get_likelihood,
        get_node_to_posterior_distn,
        get_edge_to_joint_posterior_distn,
        )
from nxmctree.lkbrute import (
        get_likelihood_brute,
        get_node_to_posterior_distn_brute,
        get_edge_to_joint_posterior_distn_brute,
        )
from nxmctree.puzzles import gen_random_systems


def test_likelihood_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in gen_random_systems(pzero):
        lk_dynamic = get_likelihood(T, e_to_P, r, r_prior, node_feas)
        lk_brute = get_likelihood_brute(T, e_to_P, r, r_prior, node_feas)
        if lk_dynamic is None or lk_brute is None:
            assert_equal(lk_dynamic, lk_brute)
        else:
            assert_allclose(lk_dynamic, lk_brute)


def test_unrestricted_likelihood():
    # When there is no data restriction the likelihood should be 1.
    pzero = 0
    for T, e_to_P, r, r_prior, node_feas in gen_random_systems(pzero):
        lk = get_likelihood(T, e_to_P, r, r_prior, node_feas)
        assert_allclose(lk, 1)


def test_node_posterior_distns_dynamic_vs_brute():
    # Check that both methods give the same posterior distributions.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in gen_random_systems(pzero):
        v_to_d = get_node_to_posterior_distn(
                T, e_to_P, r, r_prior, node_feas)
        v_to_d_brute = get_node_to_posterior_distn_brute(
                T, e_to_P, r, r_prior, node_feas)
        for v in set(node_feas):
            assert_dict_distn_allclose(v_to_d[v], v_to_d_brute[v])


def test_edge_joint_posterior_distn_dynamic_vs_brute():
    # Check that both methods give the same posterior distributions.
    pzero = 0.2
    for T, e_to_P, r, r_prior, node_feas in gen_random_systems(pzero):
        edge_to_J = get_edge_to_joint_posterior_distn(
                T, e_to_P, r, r_prior, node_feas)
        edge_to_J_brute = get_edge_to_joint_posterior_distn_brute(
                T, e_to_P, r, r_prior, node_feas)
        for edge in T.edges():
            assert_nx_distn_allclose(
                    edge_to_J[edge], edge_to_J_brute[edge])


if __name__ == '__main__':
    run_module_suite()

