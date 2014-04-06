"""
Test calculations on a tree with only a single node.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

import nxmctree
from nxmctree import dynamic_fset_lhood, brute_fset_lhood
from nxmctree import dynamic_lmap_lhood, brute_lmap_lhood
from nxmctree import nputil


def test_tree_with_one_node():

    # define the tree and the root
    T = nx.DiGraph()
    root = 'ROOT'
    T.add_node(root)

    # define the prior state distribution at the root
    prior = {
            0 : 1/4,
            1 : 1/4,
            2 : 1/4,
            3 : 1/4,
            }

    # define the data
    node_to_data_fset = {root : {1, 3}}
    node_to_data_lmap = {root : {1 : 1, 3 : 1}}

    # define the desired posterior state distribution at the root
    desired = {1 : 1/2, 3 : 1/2}

    # define extra parameters
    edge_to_P = {}

    actual = dynamic_fset_lhood.get_node_to_distn(
            T, edge_to_P, root, prior, node_to_data_fset)
    nputil.assert_dict_distn_allclose(actual, desired)

    actual = dynamic_lmap_lhood.get_node_to_distn(
            T, edge_to_P, root, prior, node_to_data_lmap)
    nputil.assert_dict_distn_allclose(actual, desired)

    actual = brute_fset_lhood.get_node_to_distn_brute(
            T, edge_to_P, root, prior, node_to_data_fset)
    nputil.assert_dict_distn_allclose(actual, desired)

    actual = dynamic_fset_lhood.get_node_to_distn(
            T, edge_to_P, root, prior, node_to_data_fset)
    nputil.assert_dict_distn_allclose(actual, desired)

