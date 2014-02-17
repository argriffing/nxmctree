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

__all__ = [
        'get_likelihood',
        'get_node_to_posterior_distn',
        'get_edge_to_joint_posterior_distn',
        ]


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


def get_node_to_posterior_distn(T, edge_to_P, root,
        root_prior_distn, node_to_data_feasible_set,
        efficiency=DYNAMIC, variant=LIKELIHOOD):
    """
    For each node get a posterior distribution or posterior feasible set.

    The posterior feasible state set for each node is determined through
    several data sources: transition matrix sparsity,
    state restrictions due to observed data at nodes,
    and state restrictions at the root due to the support of its prior
    distribution.

    Parameters
    ----------
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition.
        In likelihood mode the edge weights represent probabilities.
        In feasibility mode the edges are interpreted as unweighted.
        In both cases, missing edges are interpreted as transitions
        that are infeasible due to properties of the process or model.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_distn : dict or set
        In likelihood mode this is a prior distribution of states at the root.
        In feasibility mode this is a set of feasible prior root states.
    node_to_data_feasible_set : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
    efficiency : str in {'dynamic', 'brute'}, optional
        The dynamic mode uses dynamic programming, while the brute force mode
        uses enumeration.  The only reason to use the brute force mode
        is for testing. (default: 'dynamic')
    variant : str in {'likelihood', 'feasibility'}, optional
        The likelihood mode uses probabilities and distributions,
        whereas the feasibility mode is concerned only with
        possibility vs. impossibility. (default: 'likelihood')

    Returns
    -------
    node_to_posterior_distn : dict
        The returned dict gives posterior information about each node.
        In likelihood mode the per-node information consists of a posterior
        distribution over states.
        In feasibility mode the per-node information consists of the
        posterior set of feasible states.

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

