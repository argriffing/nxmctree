"""
Sample random systems for testing.

"""
from __future__ import division, print_function, absolute_import

import random

import networkx as nx

import nxmctree
from nxmctree.util import dict_distn


def sample_transition_graph(states, pzero):
    """
    Return a random transition matrix as a networkx digraph.
    Some entries may be zero.

    Parameters
    ----------
    states : set
        set of states
    pzero : float
        probability that any given transition has nonzero probability

    """
    P = nx.DiGraph()
    for sa in states:
        for sb in states:
            if random.random() > pzero:
                P.add_edge(sa, sb, weight=random.expovariate(1))
    for sa in states:
        if sa in P:
            total = sum(P[sa][sb]['weight'] for sb in P[sa])
            for sb in P[sa]:
                P[sa][sb]['weight'] /= total
    return P


def sample_dict_distn(states, pzero):
    """
    Return a random state distribution as a dict.
    Some entries may be zero.

    Parameters
    ----------
    states : set
        set of states
    pzero : float
        probability that any given state has nonzero probability

    """
    fset = set(s for s in states if random.random() > pzero)
    d = dict((s, random.expovariate(1)) for s in fset)
    return dict_distn(d)


def sample_data_fsets(nodes, states, pzero):
    """
    Return a map from node to feasible state set.

    Parameters
    ----------
    nodes : set
        nodes
    states : set
        set of states
    pzero : float
        probability that any given state is infeasible

    """
    d = {}
    for v in nodes:
        fset = set(s for s in states if random.random() > pzero)
        d[v] = fset
    return d


def sample_single_node_fset_system(pzero):
    """
    Sample a system with a single node.
    """
    root = 42
    nodes = {root}
    states = set(['a', 'b', 'c'])
    T = nx.DiGraph()
    root_prior_distn = sample_dict_distn(states, pzero)
    edge_to_P = {}
    node_to_data_fset = sample_data_fsets(nodes, states, pzero)
    return (T, edge_to_P, root, root_prior_distn, node_to_data_fset)


def sample_four_node_fset_system(pzero_transition, pzero_other):
    """
    Sample node states for a 4-node tree with a root and three leaves.

    """
    nnodes = 4
    root = random.randrange(nnodes)
    nodes = set(range(nnodes))
    states = set(['a', 'b', 'c'])
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)
    T = nx.dfs_tree(G, root)
    root_prior_distn = sample_dict_distn(states, pzero_other)
    edge_to_P = {}
    for edge in T.edges():
        P = sample_transition_graph(states, pzero_transition)
        edge_to_P[edge] = P
    node_to_data_fset = sample_data_fsets(nodes, states, pzero_other)
    return (T, edge_to_P, root, root_prior_distn, node_to_data_fset)


def gen_random_fset_systems(pzero, nsystems=40):
    """
    Sample whole systems for testing likelihood.
    The pzero parameter indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set).

    """
    for i in range(nsystems):
        if random.choice((0, 1)):
            yield sample_single_node_fset_system(pzero)
        else:
            yield sample_four_node_fset_system(pzero, pzero)


def gen_random_infeasible_fset_systems(nsystems=60):
    pzero = 1
    for i in range(nsystems):
        k = random.randrange(3)
        if k == 0:
            yield sample_single_node_fset_system(pzero)
        elif k == 1:
            yield sample_four_node_fset_system(pzero, pzero)
        else:
            pzero_transition = 1
            pzero_other = 0.2
            yield sample_four_node_fset_system(pzero_transition, pzero_other)

