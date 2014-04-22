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


def sample_fset(states, pzero):
    """
    Parameters
    ----------
    states : set
        set of states
    pzero : float
        probability that any given state is infeasible

    """
    return set(s for s in states if random.random() > pzero)


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
    d = dict((s, random.expovariate(1)) for s in sample_fset(states, pzero))
    return dict_distn(d)


def sample_lmap(states, pzero):
    """
    Parameters
    ----------
    states : set
        set of states
    pzero : float
        probability that any given state has nonzero probability

    """
    return sample_dict_distn(states, pzero)


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
    return dict((v, sample_fset(states, pzero)) for v in nodes)


def sample_data_lmaps(nodes, states, pzero):
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
    return dict((v, sample_lmap(states, pzero)) for v in nodes)


def _sample_single_node_system(pzero, fn_sample_data):
    T = nx.DiGraph()
    root = 42
    T.add_node(root)
    nodes = set(T)
    states = set(['a', 'b', 'c'])
    root_prior_distn = sample_dict_distn(states, pzero)
    edge_to_P = {}
    node_to_data = fn_sample_data(nodes, states, pzero)
    return (T, edge_to_P, root, root_prior_distn, node_to_data)


def _sample_four_node_system(pzero_transition, pzero_other, fn_sample_data):
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
    node_to_data = fn_sample_data(nodes, states, pzero_other)
    return (T, edge_to_P, root, root_prior_distn, node_to_data)


def _gen_random_systems(pzero, fn_sample_data, nsystems):
    for i in range(nsystems):
        if random.choice((0, 1)):
            yield _sample_single_node_system(pzero, fn_sample_data)
        else:
            yield _sample_four_node_system(pzero, pzero, fn_sample_data)


def _gen_random_infeasible_systems(fn_sample_data, nsystems):
    pzero = 1
    for i in range(nsystems):
        k = random.randrange(3)
        if k == 0:
            yield _sample_single_node_system(pzero, fn_sample_data)
        elif k == 1:
            yield _sample_four_node_system(pzero, pzero, fn_sample_data)
        else:
            pzero_transition = 1
            pzero_other = 0.2
            yield _sample_four_node_system(
                    pzero_transition, pzero_other, fn_sample_data)


def gen_random_fset_systems(pzero, nsystems=40):
    """
    Sample whole systems for testing likelihood.
    The pzero parameter indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_fset).

    """
    for x in _gen_random_systems(pzero, sample_data_fsets, nsystems):
        yield x


def gen_random_infeasible_fset_systems(nsystems=60):
    for x in _gen_random_infeasible_systems(sample_data_fsets, nsystems):
        yield x


def gen_random_lmap_systems(pzero, nsystems=40):
    """
    Sample whole systems for testing likelihood.
    The pzero parameter indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_lmap).

    """
    for x in _gen_random_systems(pzero, sample_data_lmaps, nsystems):
        yield x


def gen_random_infeasible_lmap_systems(nsystems=60):
    for x in _gen_random_infeasible_systems(sample_data_lmaps, nsystems):
        yield x

