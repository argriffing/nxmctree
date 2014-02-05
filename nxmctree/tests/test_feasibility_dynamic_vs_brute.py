"""
Test Markov chain algorithms to compute feasibility on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import run_module_suite, assert_equal, assert_allclose

import nxmctree
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree.feasibility import (
        get_feasibility,
        get_node_to_posterior_feasible_set,
        get_edge_to_joint_posterior_feasibility,
        )
from nxmctree.brute_feasibility import (
        get_feasibility_brute,
        get_node_to_posterior_feasible_set_brute,
        get_edge_to_joint_posterior_feasibility_brute,
        )
from nxmctree.puzzles import gen_random_feas_systems


def test_feasibility():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for T, e_to_A, r, r_prior, node_feas in gen_random_feas_systems(pzero):
        feas_dynamic = get_feasibility(T, e_to_A, r, r_prior, node_feas)
        feas_brute = get_feasibility_brute(T, e_to_A, r, r_prior, node_feas)
        assert_equal(feas_dynamic, feas_brute)


def test_node_posterior_feasible_sets():
    # Check that both methods give the same posterior feasible sets.
    pzero = 0.2
    for T, e_to_A, r, r_prior, node_feas in gen_random_feas_systems(pzero):
        v_to_d = get_node_to_posterior_feasible_set(
                T, e_to_A, r, r_prior, node_feas)
        v_to_d_brute = get_node_to_posterior_feasible_set_brute(
                T, e_to_A, r, r_prior, node_feas)
        for v in set(node_feas):
            assert_equal(v_to_d[v], v_to_d_brute[v])


def test_edge_joint_posterior_feasibility():
    # Check that both methods give the same posterior feasible sets.
    pzero = 0.2
    for T, e_to_A, r, r_prior, node_feas in gen_random_feas_systems(pzero):
        edge_to_J = get_edge_to_joint_posterior_feasibility(
                T, e_to_A, r, r_prior, node_feas)
        edge_to_J_brute = get_edge_to_joint_posterior_feasibility_brute(
                T, e_to_A, r, r_prior, node_feas)
        for edge in T.edges():
            J_edges = edge_to_J[edge].edges()
            J_brute_edges = edge_to_J_brute[edge].edges()
            assert_equal(set(J_edges), set(J_brute_edges))


if __name__ == '__main__':
    run_module_suite()

