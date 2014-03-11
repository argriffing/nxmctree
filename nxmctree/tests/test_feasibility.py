"""
Test algorithms that compute NetworkX Markov tree feasibility.

"""
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_equal)

import nxmctree
from nxmctree.dynamic_fset_feas import get_feas, get_node_to_fset


class Test_ShortPathFeasibility(TestCase):

    def setUp(self):

        # Define the tree.
        # It is just a path on three nodes.
        G = nx.Graph()
        G.add_path([0, 1, 2])
        nodes = set(G)

        # Define the state transition adjacency matrix.
        # It is a bidirectional path on three states,
        # where self-transitions are allowed.
        A = nx.DiGraph()
        states = set(['a', 'b', 'c'])
        A.add_path(['a', 'b', 'c'])
        A.add_path(['c', 'b', 'a'])
        A.add_edges_from((s, s) for s in states)

        # Store the setup.
        self.G = G
        self.A = A
        self.nodes = nodes
        self.states = states

    def test_unrestricted(self):
        # For each root position, each state should be allowed at each node.

        # Use an uninformative prior state distribution at the root.
        root_prior_feasible_set = self.states

        # The data does not restrict the set of states.
        node_to_data_feasible_set = {
                0 : self.states,
                1 : self.states,
                2 : self.states,
                }

        # Check each possible root position.
        for root in self.nodes:
            T = nx.dfs_tree(self.G, root)
            edge_to_adjacency = dict((edge, self.A) for edge in T.edges())

            # Assert that the combination of input parameters is feasible.
            feas = get_feas(
                    T, edge_to_adjacency, root,
                    root_prior_feasible_set, node_to_data_feasible_set)
            assert_(feas)

            # Assert that the posterior feasibility is the same
            # as the feasibility imposed by the data.
            v_to_fset = get_node_to_fset(
                    T, edge_to_adjacency, root,
                    root_prior_feasible_set, node_to_data_feasible_set)
            assert_equal(v_to_fset, node_to_data_feasible_set)

    def test_restricted(self):
        # The data imposes restrictions that imply further restrictions.

        # Use an uninformative prior state distribution at the root.
        root_prior_feasible_set = self.states

        # Restrict the two endpoint states to 'a' and 'c' respectively.
        # Together with the tree graph and the state transition adjacency graph
        # this will imply that the middle node must have state 'b'.
        node_to_data_feasible_set = {
                0 : {'c'},
                1 : {'a', 'b', 'c'},
                2 : {'a'},
                }

        # Regardless of the root, the details of the tree topology and the
        # state transition matrix imply the following map from nodes
        # to feasible state sets.
        node_to_implied_feasible_set = {
                0 : {'c'},
                1 : {'b'},
                2 : {'a'},
                }

        # Check each possible root position.
        for root in self.nodes:
            T = nx.dfs_tree(self.G, root)
            edge_to_adjacency = dict((edge, self.A) for edge in T.edges())
            v_to_fset = get_node_to_fset(
                    T, edge_to_adjacency, root,
                    root_prior_feasible_set, node_to_data_feasible_set)
            assert_equal(v_to_fset, node_to_implied_feasible_set)


class Test_LongPathFeasibility(TestCase):

    def setUp(self):

        # Define the tree.
        # It is a path on three nodes.
        G = nx.Graph()
        G.add_path([0, 1, 2])
        nodes = set(G)

        # Define the state transition adjacency matrix.
        # It is a bidirectional path on four states,
        # where self-transitions are allowed.
        # Note that the two extreme states cannot both occur on the
        # particular tree that we have chosen.
        A = nx.DiGraph()
        states = set(['a', 'b', 'c', 'd'])
        A.add_path(['a', 'b', 'c', 'd'])
        A.add_path(['d', 'c', 'b', 'a'])
        A.add_edges_from((s, s) for s in states)

        # Store the setup.
        self.G = G
        self.A = A
        self.nodes = nodes
        self.states = states

    def test_long_path_infeasibility(self):
        
        for a, b, c in itertools.permutations(self.nodes):
            for root in self.nodes:

                # Use an uninformative prior state distribution at the root.
                root_prior_feasible_set = self.states

                # Let two of the states be endpoints of the transition path,
                # and let the other state be anything.
                # No state assignment will work for this setup.
                node_to_data_feasible_set = {
                        a : {'a'},
                        b : self.states,
                        c : {'d'},
                        }

                # This dict represents an infeasible combination
                # of prior and data and tree shape and transition matrix.
                node_to_implied_feasible_set = {
                        a : set(),
                        b : set(),
                        c : set(),
                        }

                T = nx.dfs_tree(self.G, root)
                edge_to_adjacency = dict((edge, self.A) for edge in T.edges())
                v_to_fset = get_node_to_fset(
                        T, edge_to_adjacency, root,
                        root_prior_feasible_set, node_to_data_feasible_set)
                assert_equal(v_to_fset, node_to_implied_feasible_set)
