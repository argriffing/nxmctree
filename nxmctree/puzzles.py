"""
"""

import random

import networkx as nx

import nxmctree
from nxmctree.util import dict_distn


def sample_transition_graph(states, pzero):
    """
    Return a random transition matrix as a networkx digraph.
    Some entries may be zero.
    states : set of states
    pzero : probability that any given transition has nonzero probability
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
    states : set of states
    pzero : probability that any given state has nonzero probability
    """
    fset = set(s for s in states if random.random() > pzero)
    d = dict((s, random.expovariate(1)) for s in fset)
    return dict_distn(d)


def sample_data_feasible_sets(nodes, states, pzero):
    """
    Return a map from node to feasible state set.
    states : set of states
    pzero : probability that any given state is infeasible
    """
    d = {}
    for v in nodes:
        fset = set(s for s in states if random.random() > pzero)
        d[v] = fset
    return d


def gen_random_systems(pzero):
    """
    Sample whole systems, where pzero indicates sparsity.
    Yield (T, edge_to_P, root, root_prior_distn, node_to_data_feasible_set).
    """
    nsamples = 10

    # Make some random systems with a single node.
    for i in range(nsamples):
        root = 42
        nodes = {root}
        states = set(['a', 'b', 'c'])
        T = nx.DiGraph()
        root_prior_distn = sample_dict_distn(states, pzero)
        edge_to_P = {}
        node_to_data_feasible_set = sample_data_feasible_sets(
                nodes, states, pzero)
        yield (T, edge_to_P, root,
                root_prior_distn, node_to_data_feasible_set)

    # Make some random systems with multiple nodes.
    nodes = set(range(4))
    states = set(['a', 'b', 'c'])

    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    G.add_edge(0, 3)

    for i in range(nsamples):

        for root in nodes:

            T = nx.dfs_tree(G, root)
            root_prior_distn = sample_dict_distn(states, pzero)
            edge_to_P = {}
            for edge in T.edges():
                P = sample_transition_graph(states, pzero)
                edge_to_P[edge] = P
            node_to_data_feasible_set = sample_data_feasible_sets(
                    nodes, states, pzero)

            yield (T, edge_to_P, root,
                    root_prior_distn, node_to_data_feasible_set)

