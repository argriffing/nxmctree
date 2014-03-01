"""
This toy model is described in raoteh/examples/codon2x3.

"""
from __future__ import division, print_function, absolute_import

import itertools

from nxmctree import get_edge_to_nxdistn


def get_Q_primary(rate):
    # this is like a symmetric codon rate matrix
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
            'N0' : {(0, (1, 0, 1))},
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


def get_Q_compound(
        compound_states, primary_to_tol, primary_rate, rate_on, rate_off):
    """
    Compute the compound rate matrix.

    """
    Q_compound = nx.DiGraph()
    for sa in itertools.product(compound_states):

        # skip compound states that have zero probability
        prim_a, tols_a = sa
        tclass_a = primary_to_tol[prim_a]
        if not tols_a[tclass_a]:
            continue

        for sb in itertools.product(compound_states):

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

            # use a rate that depends on the transition type
            if prim_a != prim_b:
                rate = primary_rate
            elif sum(tols_b) - sum(tols_a) == 1:
                rate = rate_on
            elif sum(tols_b) - sum(tols_a) == -1:
                rate = rate_off
            else:
                raise Exception

            # set the compound process transition rate
            Q_compound.add_edge((sa, sb, weight=rate))

    # return the rate matrix of the compound process
    return Q_compound


def main():
    pass

