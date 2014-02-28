"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import (
        run_module_suite, assert_, assert_equal, assert_allclose)

import nxmctree
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree.puzzles import gen_random_systems, gen_random_infeasible_systems
from nxmctree.brute_feasibility import *
from nxmctree.brute_likelihood import *
from nxmctree.dynamic_feasibility import *
from nxmctree.dynamic_likelihood import *


# function suites for testing
suites = (
        nxmctree.brute_feasibility.fnsuite,
        nxmctree.brute_likelihood.fnsuite,
        nxmctree.dynamic_feasibility.fnsuite,
        nxmctree.dynamic_feasibility.fnsuite)


def test_infeasible_systems():
    # Test systems that are structurally infeasible.
    for args in gen_random_infeasible_systems():
        T, e_to_P, r, r_prior, node_feas = args

        for f_overall, f_node, f_edge in suites:

            # overall likelihood or feasibility
            overall_info = f_overall(*args)
            edge_info = [(e, P.edges()) for e, P in e_to_P.items()]
            msg = str((T.edges(), edge_info, r, r_prior, node_feas))
            if overall_info:
                raise Exception(msg)
            assert_(not overall_info)

            # state distributions or feasible sets at nodes
            node_info = f_node(*args)
            assert_(not any(node_info.values()))

            # joint state distributions at edge endpoints
            edge_info = f_edge(*args)
            for edge in T.edges():
                assert_(not edge_info[edge].edges())


def test_complete_density():
    # Test the special case of a completely dense system.
    pzero = 0
    for args in gen_random_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        for feas_suite, lhood_suite in (
                (
                    nxmctree.dynamic_feasibility.fnsuite,
                    nxmctree.dynamic_likelihood.fnsuite),
                (
                    nxmctree.brute_feasibility.fnsuite,
                    nxmctree.brute_likelihood.fnsuite),
                ):
            f_feas, f_node_to_fset, f_edge_to_nxfset = feas_suite
            f_lhood, f_node_to_distn, f_edge_to_nxdistn = lhood_suite

            # Check overall likelihood and feasibility.
            assert_allclose(f_lhood(*args), 1)
            assert_equal(f_feas(*args), True)

            # Check node and edge distributions and feasibility.
            for f_node, f_edge in (
                    (f_node_to_fset, f_edge_to_nxfset),
                    (f_node_to_distn, f_edge_to_nxdistn)):

                # node info
                d = f_node(*args)
                for v in set(node_feas):
                    assert_equal(set(d[v]), node_feas[v])

                # edge info
                d = f_edge(*args)
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
        dynamic = get_lhood(*args)
        brute = get_lhood_brute(*args)
        if dynamic is None or brute is None:
            assert_equal(dynamic, brute)
        else:
            assert_allclose(dynamic, brute)

        # feasibility
        dynamic = get_feas(*args)
        brute = get_feas_brute(*args)
        assert_equal(dynamic, brute)

        # state distributions at nodes
        dynamic = get_node_to_distn(*args)
        brute = get_node_to_distn_brute(*args)
        for v in set(node_feas):
            assert_dict_distn_allclose(dynamic[v], brute[v])

        # state feasibility at nodes
        dynamic = get_node_to_fset(*args)
        brute = get_node_to_fset_brute(*args)
        for v in set(node_feas):
            assert_equal(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = get_edge_to_nxdistn(*args)
        brute = get_edge_to_nxdistn_brute(*args)
        for edge in T.edges():
            assert_nx_distn_allclose(dynamic[edge], brute[edge])

        # joint state feasibility at edge endpoints
        dynamic = get_edge_to_nxfset(*args)
        brute = get_edge_to_nxfset_brute(*args)
        for edge in T.edges():
            dynamic_edges = set(dynamic[edge].edges())
            brute_edges = set(brute[edge].edges())
            assert_equal(dynamic_edges, brute_edges)
