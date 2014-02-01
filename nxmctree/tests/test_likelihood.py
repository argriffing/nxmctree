"""
Test algorithms that compute NetworkX Markov tree likelihoods.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import (run_module_suite, TestCase,
        decorators, assert_, assert_equal)

import nxmctree
from nxmctree.likelihood import (
        get_likelihood,
        get_root_posterior_partial_likelihoods,
        #get_node_to_posterior_feasible_set,
        #get_edge_to_joint_posterior_feasibility,
        #get_feasibility_info_slow,
        )


class Test_Likelihood(TestCase):

    def test_history_log_likelihood_P_default(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)

        # Define the rooted tree.
        root = 0
        T = nx.dfs_tree(G, root)

        # The data completely restricts the set of states.
        node_to_data_feasible_set = {
                0 : {'a'},
                1 : {'a'},
                2 : {'a'},
                3 : {'a'},
                }

        # The root prior distribution is informative.
        root_prior_distn = {
                'a' : 0.5,
                'b' : 0.5,
                'c' : 0,
                'd' : 0,
                }

        # Define the transition matrix.
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            ('a', 'a', 0.5),
            ('a', 'b', 0.25),
            ('a', 'c', 0.25),
            ('b', 'b', 0.5),
            ('b', 'c', 0.25),
            ('b', 'a', 0.25),
            ('c', 'c', 0.5),
            ('c', 'a', 0.25),
            ('c', 'b', 0.25)])

        # Associate each edge with the transition matrix.
        edge_to_P = dict((edge, P) for edge in T.edges())

        # The likelihood is simple in this case.
        desired_likelihood = 0.5 ** 4

        # Compute the likelhood.
        actual_likelihood = get_likelihood(T, edge_to_P, root,
                root_prior_distn, node_to_data_feasible_set)

        # Check that the likelihood is correct.
        assert_equal(actual_likelihood, desired_likelihood)


if __name__ == '__main__':
    run_module_suite()

