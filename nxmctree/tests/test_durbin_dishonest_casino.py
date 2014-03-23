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

import nxmctree

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


def naive_forward_durbin(P, prior_distn, node_to_data_lmap):
    """
    Implement the forward algorithm directly from the book.
    The book is Biological Sequence Analysis by Durbin et al.
    @param observations: the sequence of observations
    @return: list of lists, total probability
    """
    nhidden = len(prior_distn)
    nobs = len(node_to_data_lmap)
    f = [[0]*nhidden for i in range(nobs)]
    # define the initial f variable
    for sink_index, (sink_state, p) in enumerate(prior_distn.items()):
        f[0][sink_index] = node_to_data_lmap[0][sink_state] * p
    # define the subsequent f variables
    for i in range(1, nobs):
        lmap = node_to_data_lmap[i]
        for sink_index, sink_state in enumerate(P):
            f[i][sink_index] = lmap[sink_state]
            p = 0
            for source_index, source_state in enumerate(P):
                ptrans = P[source_state][sink_state]['weight']
                p += f[i-1][source_index] * ptrans
            f[i][sink_index] *= p
    total_probability = 0
    for source_index, source_state in enumerate(P):
        total_probability += f[nobs-1][source_index]
    return f, total_probability


def scaled_forward_durbin(P, prior_distn, node_to_data_lmap):
    """
    Implement the scaled forward algorithm directly from the book.
    The book is Biological Sequence Analysis by Durbin et al.
    At each position, the sum over states of the f variable is 1.
    @param observations: the sequence of observations
    @return: the list of lists of scaled f variables, and the scaling variables
    """
    nhidden = len(prior_distn)
    nobs = len(node_to_data_lmap)
    f = [[0]*nhidden for i in range(nobs)]
    s = [0]*nobs
    # define the initial unscaled f variable
    for sink_index, (sink_state, p) in enumerate(prior_distn.items()):
        f[0][sink_index] = node_to_data_lmap[0][sink_state] * p
    # define the initial scaling factor
    s[0] = sum(f[0])
    # define the initial scaled f variable
    for sink_index in range(nhidden):
        f[0][sink_index] /= s[0]
    # define the subsequent f variables and scaling factors
    for i in range(1, nobs):
        lmap = node_to_data_lmap[i]
        # define an unscaled f variable at this position
        for sink_index, sink_state in enumerate(P):
            f[i][sink_index] = lmap[sink_state]
            p = 0
            for source_index, source_state in enumerate(P):
                ptrans = P[source_state][sink_state]['weight']
                p += f[i-1][source_index] * ptrans
            f[i][sink_index] *= p
        # define the scaling factor at this position
        s[i] = sum(f[i])
        # define the scaled f variable at this position
        for sink_index in range(nhidden):
            f[i][sink_index] /= s[i]
    return f, s


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
    f, s = scaled_forward_durbin(P, prior_distn, node_to_data_lmap)
    likelihood_a = np.prod(s)

    # compute likelihood using a path algorithm without scaling
    f, total_prob = naive_forward_durbin(P, prior_distn, node_to_data_lmap)
    likelihood_b = total_prob

    # compute likelihood using the algorithm generalized to branching chains
    likelihood_c = nxmctree.get_lhood(*args)

    # check that the likelihoods are all the same
    assert_allclose(likelihood_b, likelihood_a)
    assert_allclose(likelihood_c, likelihood_a)

