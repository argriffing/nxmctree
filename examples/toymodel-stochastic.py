"""
This toy model is described in raoteh/examples/codon2x3.

mini glossary
tol -- tolerance
traj -- trajectory

The tree is rooted and edges are directed.
For each substate track, each permanent node maps to a list of events.
Each event is a handle mapping to some event info giving the
time of the event along the branch and the nature of the transition,
if any, associated with the event.

"""
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
import numpy as np
import scipy.linalg

from nxmctree import get_lhood, get_edge_to_nxdistn, sample_history
from util import get_total_rates, get_uniformized_P_nx


def get_edge_tree(T, root):
    """
    Nodes in the edge tree are edges in the original tree.

    The new tree will have a node (None, root) which does not correspond
    to any edge in the original tree.

    """
    T_dual = nx.DiGraph()
    if not T:
        return T_dual
    for c in T[root]:
        T_dual.add_edge((None, root), (root, c))
    for v in T:
        for c in T[v]:
            for g in T[c]:
                T.dual.add_edge((v, c), (c, g))
    return T


def gen_poisson_events(ephemeral_edges):
    """
    Yield event times.

    Parameters
    ----------
    ephemeral_edges : tuples (edge, rate, offset, blen)
        Information about the ephemeral edges.
        The enclosing permanent edge, the poisson rate of the ephemeral edge,
        the offset of the beginning of the ephemeral edge from the
        enclosing permanent edge, and the length of the ephemeral edge.

    """
    for edge, rate, offset, blen in ephemeral_edges:
        n = np.random.poisson(rate * blen)
        times = np.random.uniform(low=offset, high=offset+blen, size=n)
        for tm in times:
            yield edge, tm


def sample_primary_trajectory(T, root, edge_to_blen, node_to_fset, Q_nx):
    pass


def blinking_model_rao_teh(T, root, edge_to_blen, primary_to_tol, Q_primary,
        rate_on, rate_off, track_to_data, uniformization_factor):
    """

    Parameters
    ----------
    T : x
        x
    root : x
        x
    edge_to_blen : x
        x
    primary_to_tol : x
        x
    Q_primary : x
        x
    rate_on : float
        x
    rate_off : float
        x
    track_to_data : x
        x
    uniformization_factor : float
        Somewhat arbitrary constant greater than 1.  Two is good.

    """
    
    # Initialize the tolerance trajectories to all tolerated
    # with no events on the trajectory.

    # Find a feasible initial primary trajectory.

    pass



def get_initial_tol_traj(T, root):
    pass

#TODO copypaste after here...


def foo():
    pass


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
        (4, 2, rate),
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


def get_compound_states(primary_to_tol):
    nprimary = len(primary_to_tol)
    all_tols = list(itertools.product((0, 1), repeat=3))
    compound_states = list(itertools.product(range(nprimary), all_tols))
    return compound_states


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False


def define_compound_process(Q_primary, compound_states, primary_to_tol):
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

            # if neither or both primary state and tolerance change then skip
            if hamming_distance(sa, sb) != 1:
                continue

            # if the tolerances change at more than one position then skip
            if hamming_distance(tols_a, tols_b) > 1:
                continue

            # if a primary transition is not allowed then skip
            if prim_a != prim_b and not Q_primary.has_edge(prim_a, prim_b):
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


def nx_to_np(M_nx, ordered_states):
    state_to_idx = dict((s, i) for i, s in enumerate(ordered_states))
    nstates = len(ordered_states)
    M_np = np.zeros((nstates, nstates))
    for sa, sb in M_nx.edges():
        i = state_to_idx[sa]
        j = state_to_idx[sb]
        M_np[i, j] = M_nx[sa][sb]['weight']
    return M_np


def nx_to_np_rate_matrix(Q_nx, ordered_states):
    Q_np = nx_to_np(Q_nx, ordered_states)
    row_sums = np.sum(Q_np, axis=1)
    Q_np = Q_np - np.diag(row_sums)
    return Q_np


def np_to_nx_transition_matrix(P_np, ordered_states):
    P_nx = nx.DiGraph()
    for i, sa in enumerate(ordered_states):
        for j, sb in enumerate(ordered_states):
            p = P_np[i, j]
            if p:
                P_nx.add_edge(sa, sb, weight=p)
    return P_nx


def compute_edge_expectation(Q, P, J, indicator, t):
    # Q is the rate matrix
    # P is the conditional transition matrix
    # J is the joint distribution matrix
    ncompound = Q.shape[0]
    E = Q * indicator
    interact = scipy.linalg.expm_frechet(Q*t, E*t, compute_expm=False)
    total = 0
    for i in range(ncompound):
        for j in range(ncompound):
            if J[i, j]:
                total += J[i, j] * interact[i, j] / P[i, j]
    return total


def run(primary_to_tol, compound_states, node_to_data_fset):

    # Get the primary rate matrix and convert it to a dense ndarray.
    nprimary = 6
    Q_primary_nx = get_Q_primary()
    Q_primary_dense = nx_to_np_rate_matrix(Q_primary_nx, range(nprimary))
    primary_distn_dense = np.ones(nprimary, dtype=float) / nprimary

    # The expected rate of the pure primary process
    # will be used for normalization.
    expected_primary_rate = get_expected_rate(
            Q_primary_dense, primary_distn_dense)
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()

    # Define the compound process through some indicators.
    indicators = define_compound_process(
            Q_primary_nx, compound_states, primary_to_tol)
    I_syn, I_non, I_on, I_off = indicators

    # Define the dense compound transition rate matrix through the indicators.
    syn_rate = 1.0
    non_rate = 1.0
    on_rate = 1.0
    off_rate = 1.0
    Q_compound = (
            syn_rate * I_syn / expected_primary_rate +
            non_rate * I_non / expected_primary_rate +
            #syn_rate * I_syn +
            #non_rate * I_non +
            on_rate * I_on +
            off_rate * I_off)
    #Q_compound = Q_compound / expected_primary_rate
    row_sums = np.sum(Q_compound, axis=1)
    Q_compound = Q_compound - np.diag(row_sums)
    
    # Define a sparse stationary distribution over compound states.
    # This should use the rates but for now it will just be
    # uniform over the ok compound states because of symmetry.
    compound_distn = {}
    for state in compound_states:
        if compound_state_is_ok(primary_to_tol, state):
            compound_distn[state] = 1.0
    total = sum(compound_distn.values())
    compound_distn = dict((k, v/total) for k, v in compound_distn.items())
    #print('compound distn:')
    #print(compound_distn)
    #print()

    # Make the np and nx transition probability matrices.
    # Map each branch to the transition matrix.
    edge_to_P_np = {}
    edge_to_P_nx = {}
    for edge in T.edges():
        t = edge_to_blen[edge]
        P_np = scipy.linalg.expm(Q_compound * t)
        P_nx = np_to_nx_transition_matrix(P_np, compound_states)
        edge_to_P_np[edge] = P_np
        edge_to_P_nx[edge] = P_nx

    # Compute the likelihood
    lhood = get_lhood(T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    print('likelihood:')
    print(lhood)
    print()

    # Compute the map from edge to posterior joint state distribution.
    # Convert the nx transition probability matrices back into dense ndarrays.
    edge_to_nxdistn = get_edge_to_nxdistn(
            T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    edge_to_J = {}
    for edge, J_nx in edge_to_nxdistn.items():
        J_np = nx_to_np(J_nx, compound_states)
        edge_to_J[edge] = J_np

    # Compute labeled transition count expectations
    # using the rate matrix, the joint posterior state distribution matrices,
    # the indicator matrices, and the conditional transition probability
    # distribution matrix.
    primary_expectation = 0
    blink_expectation = 0
    for edge in T.edges():
        va, vb = edge
        Q = Q_compound
        J = edge_to_J[edge]
        P = edge_to_P_np[edge]
        t = edge_to_blen[edge]

        # primary transition event count expectations
        syn_total = compute_edge_expectation(Q, P, J, I_syn, t)
        non_total = compute_edge_expectation(Q, P, J, I_non, t)
        primary_expectation += syn_total
        primary_expectation += non_total
        print('edge %s -> %s syn expectation %s' % (va, vb, syn_total))
        print('edge %s -> %s non expectation %s' % (va, vb, non_total))

        # blink transition event count expectations
        on_total = compute_edge_expectation(Q, P, J, I_on, t)
        off_total = compute_edge_expectation(Q, P, J, I_off, t)
        blink_expectation += on_total
        blink_expectation += off_total
        print('edge %s -> %s on expectation %s' % (va, vb, on_total))
        print('edge %s -> %s off expectation %s' % (va, vb, off_total))
        
        print()

    print('primary expectation:')
    print(primary_expectation)
    print()

    print('blink expectation:')
    print(blink_expectation)
    print()



def main():

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()

    # Define the ordering of the compound states.
    compound_states = get_compound_states(primary_to_tol)

    # No data.
    print ('expectations given no alignment or disease data')
    print()
    node_to_data_fset = {
            'N0' : set(compound_states),
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : set(compound_states),
            'N4' : set(compound_states),
            'N5' : set(compound_states),
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment data only.
    print ('expectations given only alignment data but not disease data')
    print()
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 0)),
                (0, (1, 0, 1)),
                (0, (1, 1, 0)),
                (0, (1, 1, 1))},
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
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment and disease data.
    print ('expectations given alignment and disease data')
    print()
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
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment and fully observed disease data.
    print ('expectations given alignment and fully observed disease data')
    print ('(all leaf disease states which were previously considered to be')
    print ('unobserved are now considered to be tolerated (blinked on))')
    print()
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 1))},
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : {
                (4, (1, 1, 1))},
            'N4' : {
                (5, (1, 1, 1))},
            'N5' : {
                (1, (1, 1, 1))},
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()


main()

