"""
Test Markov chain algorithms to compute likelihoods and distributions on trees.

This module compares dynamic programming implementations against
brute force implementations.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import (run_module_suite, assert_,
        assert_equal, assert_allclose, assert_array_less)

import nxmctree
from nxmctree.nputil import (
        assert_dict_distn_allclose, assert_nx_distn_allclose)
from nxmctree.puzzles import (
        gen_random_fset_systems, gen_random_infeasible_fset_systems,
        gen_random_lmap_systems, gen_random_infeasible_lmap_systems)
from nxmctree import brute_fset_feas, dynamic_fset_feas
from nxmctree import brute_fset_lhood, dynamic_fset_lhood
from nxmctree import brute_lmap_lhood, dynamic_lmap_lhood


# function suites for testing
fset_suites = (
        brute_fset_feas.fnsuite,
        brute_fset_lhood.fnsuite,
        dynamic_fset_feas.fnsuite,
        dynamic_fset_lhood.fnsuite)
lmap_suites = (
        brute_lmap_lhood.fnsuite,
        dynamic_lmap_lhood.fnsuite)
all_suites = fset_suites + lmap_suites


def test_infeasible_fset_systems():
    # Test systems that are structurally infeasible.
    for args in gen_random_infeasible_fset_systems():
        T, e_to_P, r, r_prior, node_feas = args

        for f_overall, f_node, f_edge in fset_suites:

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


def test_infeasible_lmap_systems():
    # Test systems that are structurally infeasible.
    for args in gen_random_infeasible_lmap_systems():
        T, e_to_P, r, r_prior, node_data = args

        for f_overall, f_node, f_edge in all_suites:

            # overall likelihood or feasibility
            overall_info = f_overall(*args)
            edge_info = [(e, P.edges()) for e, P in e_to_P.items()]
            msg = str((T.edges(), edge_info, r, r_prior, node_data))
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


def test_complete_fset_density():
    # Test the special case of a completely dense system.
    pzero = 0
    for args in gen_random_fset_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        for feas_suite, lhood_suite in (
                (dynamic_fset_feas.fnsuite, dynamic_fset_lhood.fnsuite),
                (brute_fset_feas.fnsuite, brute_fset_lhood.fnsuite),
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
                    assert_equal(set(d[v]), set(node_feas[v]))

                # edge info
                d = f_edge(*args)
                for edge in T.edges():
                    observed_edges = set(d[edge].edges())
                    desired_edges = set(e_to_P[edge].edges())
                    assert_equal(observed_edges, desired_edges)


def test_complete_lmap_density():
    # Test the special case of a completely dense system.
    pzero = 0
    for args in gen_random_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_data = args

        for feas_suite, lhood_suite, lmap_suite in (
                (
                    dynamic_fset_feas.fnsuite,
                    dynamic_fset_lhood.fnsuite,
                    dynamic_lmap_lhood.fnsuite),
                (
                    brute_fset_feas.fnsuite,
                    brute_fset_lhood.fnsuite,
                    brute_lmap_lhood.fnsuite),
                ):
            f_feas, f_node_to_fset, f_edge_to_nxfset = feas_suite
            f_lhood, f_node_to_distn, f_edge_to_nxdistn = lhood_suite
            f_lmap_overall, f_lmap_node, f_lmap_edge = lmap_suite

            # Check overall likelihood and feasibility.
            assert_array_less(0, f_lmap_overall(*args))
            assert_array_less(f_lmap_overall(*args), 1)
            assert_allclose(f_lhood(*args), 1)
            assert_equal(f_feas(*args), True)

            # Check node and edge distributions and feasibility.
            for f_node, f_edge in (
                    (f_lmap_node, f_lmap_edge),
                    (f_node_to_fset, f_edge_to_nxfset),
                    (f_node_to_distn, f_edge_to_nxdistn)):

                # node info
                d = f_node(*args)
                for v in set(node_data):
                    assert_equal(set(d[v]), set(node_data[v]))

                # edge info
                d = f_edge(*args)
                for edge in T.edges():
                    observed_edges = set(d[edge].edges())
                    desired_edges = set(e_to_P[edge].edges())
                    assert_equal(observed_edges, desired_edges)


def test_fset_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for args in gen_random_fset_systems(pzero):
        T, e_to_P, r, r_prior, node_feas = args

        # likelihood
        dynamic = dynamic_fset_lhood.get_lhood(*args)
        brute = brute_fset_lhood.get_lhood_brute(*args)
        if dynamic is None or brute is None:
            assert_equal(dynamic, None)
            assert_equal(brute, None)
        else:
            assert_allclose(dynamic, brute)

        # feasibility
        dynamic = dynamic_fset_feas.get_feas(*args)
        brute = brute_fset_feas.get_feas_brute(*args)
        assert_equal(dynamic, brute)

        # state distributions at nodes
        dynamic = dynamic_fset_lhood.get_node_to_distn(*args)
        brute = brute_fset_lhood.get_node_to_distn_brute(*args)
        for v in set(node_feas):
            assert_dict_distn_allclose(dynamic[v], brute[v])

        # state feasibility at nodes
        dynamic = dynamic_fset_feas.get_node_to_fset(*args)
        brute = brute_fset_feas.get_node_to_fset_brute(*args)
        for v in set(node_feas):
            assert_equal(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = dynamic_fset_lhood.get_edge_to_nxdistn(*args)
        brute = brute_fset_lhood.get_edge_to_nxdistn_brute(*args)
        for edge in T.edges():
            assert_nx_distn_allclose(dynamic[edge], brute[edge])

        # joint state feasibility at edge endpoints
        dynamic = dynamic_fset_feas.get_edge_to_nxfset(*args)
        brute = brute_fset_feas.get_edge_to_nxfset_brute(*args)
        for edge in T.edges():
            dynamic_edges = set(dynamic[edge].edges())
            brute_edges = set(brute[edge].edges())
            assert_equal(dynamic_edges, brute_edges)


def test_lmap_dynamic_vs_brute():
    # Compare dynamic programming vs. summing over all histories.
    pzero = 0.2
    for args in gen_random_lmap_systems(pzero):
        T, e_to_P, r, r_prior, node_lmap = args

        # likelihood
        dynamic = dynamic_lmap_lhood.get_lhood(*args)
        brute = brute_lmap_lhood.get_lhood_brute(*args)
        if dynamic is None or brute is None:
            assert_equal(dynamic, None)
            assert_equal(brute, None)
        else:
            assert_allclose(dynamic, brute)

        # state distributions at nodes
        dynamic = dynamic_lmap_lhood.get_node_to_distn(*args)
        brute = brute_lmap_lhood.get_node_to_distn_brute(*args)
        for v in set(node_lmap):
            assert_dict_distn_allclose(dynamic[v], brute[v])

        # joint state distributions at edge endpoints
        dynamic = dynamic_lmap_lhood.get_edge_to_nxdistn(*args)
        brute = brute_lmap_lhood.get_edge_to_nxdistn_brute(*args)
        for edge in T.edges():
            assert_nx_distn_allclose(dynamic[edge], brute[edge])

        # get simplified data without subtlety in the observations
        simple_node_data = dict(
                (v, dict((k, 1) for k in d)) for v, d in node_lmap.items())

        simple_args = (T, e_to_P, r, r_prior, simple_node_data)

        # simplified data likelihood
        dynamic_fset = dynamic_fset_lhood.get_lhood(*simple_args)
        brute_fset = brute_fset_lhood.get_lhood_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_lhood(*simple_args)
        brute_lmap = brute_lmap_lhood.get_lhood_brute(*simple_args)
        if None in (dynamic_fset, brute_fset, dynamic_lmap, brute_lmap):
            assert_equal(dynamic_fset, None)
            assert_equal(brute_fset, None)
            assert_equal(dynamic_lmap, None)
            assert_equal(brute_lmap, None)
        else:
            assert_allclose(dynamic_fset, brute_fset)
            assert_allclose(dynamic_lmap, brute_fset)
            assert_allclose(brute_lmap, brute_fset)

        # simplified data state distributions at nodes
        dynamic_fset = dynamic_fset_lhood.get_node_to_distn(*simple_args)
        brute_fset = brute_fset_lhood.get_node_to_distn_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_node_to_distn(*simple_args)
        brute_lmap = brute_lmap_lhood.get_node_to_distn_brute(*simple_args)
        for v in set(simple_node_data):
            assert_dict_distn_allclose(dynamic_fset[v], brute_fset[v])
            assert_dict_distn_allclose(dynamic_lmap[v], brute_fset[v])
            assert_dict_distn_allclose(brute_lmap[v], brute_fset[v])

        # simplified data joint state distributions at edge endpoints
        dynamic_fset = dynamic_fset_lhood.get_edge_to_nxdistn(*simple_args)
        brute_fset = brute_fset_lhood.get_edge_to_nxdistn_brute(*simple_args)
        dynamic_lmap = dynamic_lmap_lhood.get_edge_to_nxdistn(*simple_args)
        brute_lmap = brute_lmap_lhood.get_edge_to_nxdistn_brute(*simple_args)
        for edge in T.edges():
            assert_nx_distn_allclose(dynamic_fset[edge], brute_fset[edge])
            assert_nx_distn_allclose(dynamic_lmap[edge], brute_fset[edge])
            assert_nx_distn_allclose(brute_lmap[edge], brute_fset[edge])

