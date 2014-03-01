"""
This toy model is described in raoteh/examples/codon2x3.

"""
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
import numpy as np
import scipy.linalg

from nxmctree import get_edge_to_nxdistn


def get_Q_primary():
    # this is like a symmetric codon rate matrix
    rate = 1
    Q_primary = nx.DiGraph()
    Q_primary.add_weighted_edges_from((
        (0, 1, rate),
        (0, 2, rate),
        (1, 0, rate),
        (1, 3, rate),
        (2, 0, rate),
        (2, 3, rate),
        (2, 4, rate),
        (3, 1, rate),
        (3, 2, rate),
        (3, 5, rate),
        (4, 4, rate),
        (4, 5, rate),
        (5, 3, rate),
        (5, 4, rate),
        ))
    return Q_primary


def get_primary_to_tol():
    # this is like a genetic code mapping codons to amino acids
    primary_to_tol = {
            0 : 0,
            1 : 0,
            2 : 1,
            3 : 1,
            4 : 2,
            5 : 2,
            }
    return primary_to_tol


def get_node_to_data_fset(compound_states):
    # this accounts for both the alignment data and the disease data
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 1))},
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : {
                (4, (0, 0, 1)),
                (4, (0, 1, 1)),
                (4, (1, 0, 1)),
                (4, (1, 1, 1))},
            'N4' : {
                (5, (0, 0, 1)),
                (5, (0, 1, 1)),
                (5, (1, 0, 1)),
                (5, (1, 1, 1))},
            'N5' : {
                (1, (1, 0, 0)),
                (1, (1, 0, 1)),
                (1, (1, 1, 0)),
                (1, (1, 1, 1))},
            }
    return node_to_data_fset


def get_T_and_root():
    # rooted tree, deliberately without branch lengths
    T = nx.DiGraph()
    T.add_edges_from([
        ('N1', 'N0'),
        ('N1', 'N2'),
        ('N1', 'N5'),
        ('N2', 'N3'),
        ('N2', 'N4'),
        ])
    return T, 'N1'


def get_edge_to_blen():
    edge_to_blen = {
            ('N1', 'N0') : 0.5,
            ('N1', 'N2') : 0.5,
            ('N1', 'N5') : 0.5,
            ('N2', 'N3') : 0.5,
            ('N2', 'N4') : 0.5,
            }
    return edge_to_blen


def hamming_distance(va, vb):
    return sum(1 for a, b in zip(va, vb) if a != b)


def get_compound_states():
    nprimary = len(primary_to_tol)
    all_tols = list(itertools.product((0, 1), repeat=3))
    compound_states = list(itertools.product(range(nprimary), all_tols))
    return compound_states


def define_compound_process(compound_states, primary_to_tol):
    """
    Compute indicator matrices for the compound process.

    """
    n = len(compound_states)

    # define some dense indicator matrices
    I_syn = np.zeros((n, n), dtype=float)
    I_non = np.zeros((n, n), dtype=float)
    I_on = np.zeros((n, n), dtype=float)
    I_off = np.zeros((n, n), dtype=float)

    for i, sa in enumerate(compound_states):

        # skip compound states that have zero probability
        prim_a, tols_a = sa
        tclass_a = primary_to_tol[prim_a]
        if not tols_a[tclass_a]:
            continue

        for j, sb in enumerate(compound_states):

            # skip compound states that have zero probability
            prim_b, tols_b = sb
            tclass_b = primary_to_tol[prim_b]
            if not tols_b[tclass_b]:
                continue

            # if both the primary state and tolerance change then skip
            if hamming_distance(sa, sb) != 1:
                continue

            # if the tolerances change at more than one position then skip
            if hamming_distance(tols_a, tols_b) > 1:
                continue

            # set the indicator according to the transition type
            if prim_a != prim_b and tclass_a == tclass_b:
                I_syn[i, j] = 1
            elif prim_a != prim_b and tclass_a != tclass_b:
                I_non[i, j] = 1
            elif sum(tols_b) - sum(tols_a) == 1:
                I_on[i, j] = 1
            elif sum(tols_b) - sum(tols_a) == -1:
                I_off[i, j] = 1
            else:
                raise Exception

    return I_syn, I_non, I_on, I_off


def get_expected_rate(Q_dense, dense_distn):
    return -np.dot(np.diag(Q_dense), dense_distn)


def nx_to_np_rate_matrix(Q_nx, ordered_states):
    state_to_idx = dict((s, i) for i, s in enumerate(ordered_states))
    nstates = len(ordered_states)
    Q_np = np.zeros((nstates, nstates))
    for sa, sb in Q_nx.edges():
        i = state_to_idx[sa]
        j = state_to_idx[sb]
        Q_np[i, j] = Q_nx[i][j]['weight']

    # set negative diagonal entries so that rows sum to zero
    row_sums = np.sum(Q_np, axis=1)
    Q_np = Q_np - np.diag(row_sums)

    # return the dense rate matrix
    return Q_np


def main():

    # Get the primary rate matrix and convert it to a dense ndarray.
    nprimary = 6
    Q_primary_nx = get_Q_primary()
    Q_primary_dense = nx_to_np_rate_matrix(Q_primary_nx, range(nprimary))
    primary_distn_dense = np.ones(nprimary, dtype=float) / nprimary
    expected_rate = get_expected_rate(Q_primary_dense, primary_distn_dense)
    print('pure primary process expected rate:')
    print(expected_rate)
    print

    # The expected rate of the pure primary process
    # will be used for normalization.

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()



main()
