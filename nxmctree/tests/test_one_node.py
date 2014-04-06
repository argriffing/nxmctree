"""
Test calculations on a tree with only a single node.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import assert_equal

import nxmctree
from nxmctree import dynamic_fset_lhood, brute_fset_lhood
from nxmctree import dynamic_lmap_lhood, brute_lmap_lhood
from nxmctree import dynamic_fset_feas, brute_fset_feas
from nxmctree import nputil


def test_tree_with_one_node():

    # define the tree and the root
    T = nx.DiGraph()
    root = 'ROOT'
    T.add_node(root)

    # define the prior state distribution at the root
    prior_distn = {
            0 : 1/4,
            1 : 1/4,
            2 : 1/4,
            3 : 1/4,
            }
    prior_fset = set(prior_distn)

    # define the data
    node_to_data_fset = {root : {1, 3}}
    node_to_data_lmap = {root : {1 : 1/8, 3 : 3/8}}

    # define the desired posterior state distribution at the root
    desired_fset = {1, 3}
    desired_fset_distn = {1 : 1/2, 3 : 1/2}
    desired_lmap_distn = {1 : 1/4, 3 : 3/4}

    # define extra parameters
    edge_to_P = {}

    # brute feasibility
    actual_fset = brute_fset_feas.get_node_to_fset_brute(
            T, edge_to_P, root, prior_fset, node_to_data_fset)[root]
    assert_equal(actual_fset, desired_fset)

    # dynamic feasibility
    actual_fset = dynamic_fset_feas.get_node_to_fset(
            T, edge_to_P, root, prior_fset, node_to_data_fset)[root]
    assert_equal(actual_fset, desired_fset)

    # brute fset distribution
    actual_fset_distn = brute_fset_lhood.get_node_to_distn_brute(
            T, edge_to_P, root, prior_distn, node_to_data_fset)[root]
    nputil.assert_dict_distn_allclose(actual_fset_distn, desired_fset_distn)

    # dynamic fset distribution
    actual_fset_distn = dynamic_fset_lhood.get_node_to_distn(
            T, edge_to_P, root, prior_distn, node_to_data_fset)[root]
    nputil.assert_dict_distn_allclose(actual_fset_distn, desired_fset_distn)

    # brute lmap distribution
    actual_lmap_distn = brute_lmap_lhood.get_node_to_distn_brute(
            T, edge_to_P, root, prior_distn, node_to_data_lmap)[root]
    nputil.assert_dict_distn_allclose(actual_lmap_distn, desired_lmap_distn)

    # dynamic lmap distribution
    actual_lmap_distn = dynamic_lmap_lhood.get_node_to_distn(
            T, edge_to_P, root, prior_distn, node_to_data_lmap)[root]
    nputil.assert_dict_distn_allclose(actual_lmap_distn, desired_lmap_distn)

