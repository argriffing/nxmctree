"""
Some tests from Biological Sequence Analysis by Durbin et al.

This is the "dishonest casino" model in which a casino has two dice,
a fair die and a loaded die, and the casino surrupticiously switches
between them over the course of a sequence of rolls.

The fair die is known to emit one of {1, 2, 3, 4, 5, 6} uniformly at random,
whereas the loaded die is known to roll 6 with probability 0.5 versus rolling
one of {1, 2, 3, 4, 5} each with probability 0.1.
Between consecutive rolls, the casino may secretly switch between the two dice
with a known probability that depends on which die is currently being used.
If the current die is fair then the casino will switch to the loaded die
with probability 0.05.  If the current die is loaded then the casino will
switch to the fair die with probability 0.10.  Note that at stationarity
these switching dynamics imply that the current die state is fair with
probability 2/3 and is loaded with probability 1/3.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

import nxmctree.pathgraph

ROLLS = (
        '315116246446644245311321631164'
        '152133625144543631656626566666'
        '651166453132651245636664631636'
        '663162326455236266666625151631'
        '222555441666566563564324364131'
        '513465146353411126414626253356'
        '366163666466232534413661661163'
        '252562462255265252266435353336'
        '233121625364414432335163243633'
        '665562466662632666612355245242')

FAIR = 'fair'
LOADED = 'loaded'


def get_dishonest_casino_P():
    """
    Define the way that the casino switches the dice between rolls.

    """
    P = nx.DiGraph()
    P.add_weighted_edges_from([
            (FAIR, FAIR, 0.95),
            (FAIR, LOADED, 0.05),
            (LOADED, FAIR, 0.10),
            (LOADED, LOADED, 0.90),
            ])
    return P


def get_dishonest_casino_model():

    # Define the way that the casino switches the dice between rolls.
    P = get_dishonest_casino_P()

    # Define the prior distribution over die choice.
    prior_distn = {FAIR : 2/3, LOADED : 1/3}

    # Construct the tree graph.
    # In this case the tree graph is just a linear path without any branching.
    # The node label is just the index of the roll.
    T = nx.DiGraph()
    nrolls = len(ROLLS)
    vertices = range(nrolls)
    edge_to_P = {}
    for va, vb in zip(vertices[:-1], vertices[1:]):
        T.add_edge(va, vb)
        edge_to_P[va, vb] = P

    # In this simplified model the root of the tree graph
    # is just the index of the first roll.
    root = 0

    # For each possible hidden state for each node in the tree, 
    # record the likelihood of the observed die roll.
    node_to_data_lmap = {}
    for i, roll in enumerate(ROLLS):
        node_to_data_lmap[i] = {
                FAIR : 1/6,
                LOADED : 1/2 if roll == '6' else 1/10,
                }

    # Return the information about the model and the observations.
    return T, edge_to_P, root, prior_distn, node_to_data_lmap


def test_dishonest_casino_likelihood():

    # define the model and data
    P = get_dishonest_casino_P()
    args = get_dishonest_casino_model()
    T, edge_to_P, root, prior_distn, node_to_data_lmap = args

    # compute likelihood using a path algorithm with scaling
    f, s = nxmctree.pathgraph.scaled_forward_durbin(
            P, prior_distn, node_to_data_lmap)
    likelihood_a = np.prod(s)

    # compute likelihood using a path algorithm without scaling
    f, total_prob = nxmctree.pathgraph.naive_forward_durbin(
            P, prior_distn, node_to_data_lmap)
    likelihood_b = total_prob

    # compute likelihood using the algorithm generalized to branching chains
    likelihood_c = nxmctree.get_lhood(*args)

    # check that all of the likelihoods are all the same
    assert_allclose(likelihood_b, likelihood_a)
    assert_allclose(likelihood_c, likelihood_a)


def test_dishonest_casino_posterior_state_distributions():

    # define the model and data
    P = get_dishonest_casino_P()
    args = get_dishonest_casino_model()
    T, edge_to_P, root, prior_distn, node_to_data_lmap = args

    # compute distributions using a path algorithm with scaling
    distn_a = nxmctree.pathgraph.scaled_posterior_durbin(
            P, prior_distn, node_to_data_lmap)

    # compute distributions using a path algorithm without scaling
    distn_b = nxmctree.pathgraph.naive_posterior_durbin(
            P, prior_distn, node_to_data_lmap)

    # compute distributions using the algorithm generalized to branching chains
    states = sorted(set(prior_distn) | set(P))
    node_to_distn = nxmctree.get_node_to_distn(*args)
    distn_c = []
    for v in sorted(node_to_distn):
        distn = node_to_distn[v]
        distribution = [distn.get(s, 0) for s in states]
        distn_c.append(distribution)

    # check that all of the distributions are the same
    assert_allclose(distn_b, distn_a)
    assert_allclose(distn_c, distn_a)

