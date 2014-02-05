"""

"""
from __future__ import division, print_function, absolute_import

import nxmctree
from nxmctree import (
        _dynamic_likelihood, _brute_likelihood,
        _dynamic_feasibility, _brute_feasibility)

LIKELIHOOD = 'likelihood'
FEASIBILITY = 'feasibility'
DYNAMIC = 'dynamic'
BRUTE = 'brute'


#TODO add an informative docstring
def get_likelihood(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set,
        efficiency=DYNAMIC, variant=LIKELIHOOD):
    """
    """
    f = {
            (DYNAMIC, LIKELIHOOD) : _dynamic_likelihood.get_likelihood,
            (DYNAMIC, FEASIBILITY) : _dynamic_feasibility.get_feasibility,
            (BRUTE, LIKELIHOOD) : _brute_likelihood.get_likelihood,
            (BRUTE, FEASIBILITY) : _brute_feasibility.get_feasibility,
            }[efficiency, variant]
    return f(T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set)


#TODO add an informative docstring
def get_node_to_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set,
        efficiency=DYNAMIC, variant=LIKELIHOOD):
    """
    """
    f = {
            (DYNAMIC, LIKELIHOOD) : (
                _dynamic_likelihood.get_node_to_posterior_distn),
            (DYNAMIC, FEASIBILITY) : (
                _dynamic_feasibility.get_node_to_posterior_feasible_set),
            (BRUTE, LIKELIHOOD) : (
                _brute_likelihood.get_node_to_posterior_distn),
            (BRUTE, FEASIBILITY) : (
                _brute_feasibility.get_node_to_posterior_feasible_set),
            }[efficiency, variant]
    return f(T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set)


#TODO add an informative docstring
def get_edge_to_joint_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set,
        efficiency=DYNAMIC, variant=LIKELIHOOD):
    """
    """
    f = {
            (DYNAMIC, LIKELIHOOD) : (
                _dynamic_likelihood.get_edge_to_joint_posterior_distn),
            (DYNAMIC, FEASIBILITY) : (
                _dynamic_feasibility.get_edge_to_joint_posterior_feasibility),
            (BRUTE, LIKELIHOOD) : (
                _brute_likelihood.get_edge_to_joint_posterior_distn),
            (BRUTE, FEASIBILITY) : (
                _brute_feasibility.get_edge_to_joint_posterior_feasibility),
            }[efficiency, variant]
    return f(T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set)

