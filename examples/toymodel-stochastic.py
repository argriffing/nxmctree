"""
This toy model is described in raoteh/examples/codon2x3.

mini glossary
tol -- tolerance
traj -- trajectory
fg -- foreground track
bg -- background track

The tree is rooted and edges are directed.
For each substate track, each permanent node maps to a list of events.
Each event is a handle mapping to some event info giving the
time of the event along the branch and the nature of the transition,
if any, associated with the event.

We can use the jargon that 'events' are associated with
locations in the tree defined by a directed edge
and a distance along that edge.
Events will usually be associated with a state transition,
but 'incomplete events' will not have such an association.

The process is separated into multiple 'tracks' -- a primary process
track and one track for each of the tolerance processes.
The track trajectories are not independent of each other.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from functools import partial
from itertools import product
import warnings

import networkx as nx
import numpy as np
import scipy.linalg

from nxmctree import get_lhood, get_edge_to_nxdistn, sample_history
from util import get_total_rates, get_uniformized_P_nx


class TrackInfo(object):
    def __init__(self, name=None, data=None, history=None, events=None,
            Q_nx=None):
        """

        Parameters
        ----------
        name : hashable, optional
            track label
        data : dict, optional
            map from permanent node to set of states compatible with data
        history : dict, optional
            Map from permanent node to current state.
            Note that this is not the same as a trajectory.
        events : dict, optional
            map from permanent edge to list of events

        """
        self.name = name
        self.data = data
        self.history = history
        self.events = events
        self.Q_nx = Q_nx

    def clear_state_labels(self):
        """
        Clear the state labels but not the event times.

        """
        nodes = set(self.history)
        edges = set(self.events)
        for v in nodes:
            self.history[v] = None
        for edge in edges:
            for ev in self.events[edge]:
                ev.sa = None
                ev.sb = None

    def add_poisson_events(self, T, edge_to_blen, poisson_rates):
        """
        Add incomplete events to all edges.

        Parameters
        ----------
        T : directed nx tree
            tree
        edge_to_blen : dict
            maps structural edges to branch lengths
        poisson_rates : dict
            maps foreground states to poisson sampling rates

        """
        for edge in T.edges():
            va, vb = edge

            # build triples defining the foreground trajectory along the branch
            triples = []
            for ev in self.events[edge]:
                triple = (ev.tm, ev.sa, ev.sb)
                triples.append(triple)
            initial_triple = (0, None, self.history[va])
            final_triple = (edge_to_blen[edge], self.history[vb], None)
            triples = sorted([initial_triple] + triples + [final_triple])

            # sample new poisson events along the branch
            poisson_events = []
            for ta, tb in zip(triples[:-1], triples[1:]):
                ta_tm, ta_initial, ta_final = ta
                tb_tm, tb_initial, tb_final = tb
                if ta_tm == tb_tm:
                    warnings.warn('multiple events occur simultaneously')
                if tb_tm < ta_tm:
                    raise Exception('times are not monotonically increasing')
                if None in (ta_final, tb_initial):
                    raise Exception('trajectory has incomplete events')
                if ta_final != tb_initial:
                    raise Exception('trajectory has incompatible transitions')
                state = ta_final
                rate = poisson_rates[state]
                blen = tb_tm - ta_tm
                n = np.random.poisson(rate * blen)
                times = np.random.uniform(low=ta_tm, high=tb_tm, size=n)
                for tm in times:
                    ev = Event(track=track_label, edge=edge, tm=tm)
                    poisson_events.append(ev)

            # add the sampled poisson events to the track info for the branch
            self.events[edge].extend(poisson_events)


class Event(object):
    def __init__(self, track=None, edge=None, tm=None, sa=None, sb=None):
        """

        Parameters
        ----------
        track : hashable track label, optional
            label of the track on which the event occurs
        edge : ordered pair of nodes, optional
            structural edge on which the event occurs
        tm : float, optional
            time along the edge at which the event occurs
        sa : hashable, optional
            initial state of the transition
        sb : hashable, optional
            final state of the transition

        """
        self.track = track
        self.edge = edge
        self.tm = tm
        self.sa = sa
        self.sb = sb

    def init_sa(self, state):
        if self.sa is not None:
            raise Exception('the initial state is already set')
        self.sa = state

    def init_sb(self, state):
        if self.sb is not None:
            raise Exception('the final state is already set')
        self.sb = state

    def __lt__(self, other):
        """
        Give events a partial order.

        """
        if self.edge != other.edge:
            raise Exception('cannot compare events on different edges')
        if self.tm == other.tm:
            warnings.warn('found two events with identical times, '
                    'but this should rarely occur with double precision')
        return self.tm < other.tm


class MetaNode(object):
    """
    This is hashable so it can be a node in a networkx graph.

    """
    def __init__(self, P_nx=None,
            initial_data_fset=None, final_data_fset=None,
            set_sa=None, set_sb=None):
        self.P_nx = P_nx
        self.initial_data_fset = initial_data_fset
        self.final_data_fset = final_data_fset
        self.set_sa = set_sa
        self.set_sb = set_sb
    def __eq__(self, other):
        return id(self) == id(other)
    def __hash__(self):
        return id(self)



###############################################################################
# Primary track and blink track initialization.



def init_blink_history(T, edge_to_blen, track_info):
    """
    Initial blink history is True where consistent with the data.

    """
    for v in T:
        track_info.history[v] = (True in track_info.data[v])


def init_complete_blink_events(T, edge_to_blen, track_info):
    """
    Init blink track.

    """
    track_label = track_info.track
    for edge in T:
        va, vb = edge
        sa = track_info.history[va]
        sb = track_info.history[vb]
        blen = edge_to_blen[edge]
        tma = np.random.uniform(0, 1/3)
        tmb = np.random.uniform(2/3, 1)
        eva = Event(track=track_label, edge=edge, tm=tma, sa=sa, sb=True)
        evb = Event(track=track_label, edge=edge, tm=tmb, sa=True, sb=sb)
        track_info.events[edge] = [eva, evb]


def init_incomplete_primary_events(T, edge_to_blen, track_info, diameter):
    """
    Parameters
    ----------
    T : nx tree
        tree
    edge_to_blen : dict
        maps edges to branch lengths
    track_info : TrackInfo
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    """
    track_label = track_info.track
    for edge in T:
        va, vb = edge
        times = np.random.uniform(low=1/3, high=2/3, size=diameter-1)
        events = [Event(track=track_label, edge=edge, tm=tm) for tm in times]
        track_info.events[edge] = events


###############################################################################
# Functions implementing steps of Rao Teh iteration.


def do_nothing():
    """
    Helper function as a placeholder callback.

    """
    pass


def set_or_confirm_history_state(node_to_state, node, state):
    """
    Helper function for updating history within a trajectory.

    """
    if node_to_state.get(node, None) not in (state, None):
        raise Exception('found a history incompatibility')
    node_to_state[node] = state


def get_edge_tree(T, root):
    """
    Nodes in the edge tree are edges in the original tree.

    The new tree will have a node (None, root) which does not correspond
    to any edge in the original tree.

    """
    dual_root = (None, root)
    T_dual = nx.DiGraph()
    if not T:
        return T_dual, dual_root
    for c in T[root]:
        T_dual.add_edge(dual_root, (root, c))
    for v in T:
        for c in T[v]:
            for g in T[c]:
                T.dual.add_edge((v, c), (c, g))
    return T_dual, dual_root




def sample_transitions(T, root, fg_distn, P_fg, P_fg_identity,
        fg_info, bg_infos, bg_to_fg_fset):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    primary_state_set = set(primary_distn)

    # Define the map from blink track to set of primary states.
    tol_to_primary_states = defaultdict(set)
    for primary, tol in primary_to_tol.items():
        tol_to_primary_states[tol].add(primary)

    # Construct a primary process identity transition matrix.
    primary_states = set(primary_to_tol)
    P_identity = nx.DiGraph()
    for s in primary_states:
        P_identity.add_edge(s, s, weight=1)

    # Construct a meta node for each structural node.
    node_to_meta = {}
    for v in T:
        f = partial(set_or_confirm_history_state, primary_info.history, v)
        fset = primary_info.data[v]
        m = MetaNode(P_nx=P_identity,
                initial_data_fset=fset, final_data_fset=fset,
                set_sa=f, set_sb=f)
        node_to_meta[v] = m
        if v == root:
            mroot = m

    # Build the tree whose vertices are meta nodes,
    # and map edges of this tree to sets of feasible foreground states,
    # accounting for data at structural nodes and background context
    # along edge segments.
    meta_node_tree = nx.DiGraph()
    for edge in T.edges():
        va, vb = edge

        # Initialize the background states.
        for 

        # Sequence meta nodes from three sources:
        # the two structural endpoint nodes,
        # the nodes representing transitions in background blinking tracks,
        # and nodes representing transitions in the foreground primary track.
        seq = []
        for v in edge:
            m = node_to_meta[v]
            seq.append(m)
        for blink_info in blink_infos:
            for ev in blink_info.events[edge]:
                m = MetaNode(P_nx=P_identity,
                        initial_data_fset=tol_to_primary_states[ev.sa],
                        final_data_fset=tol_to_primary_states[ev.sb],
                        set_sa=do_nothing, set_sb=do_nothing)
                seq.append(m)
        for ev in primary_info.events[edge]:
            m = MetaNode(P_nx=P_primary,
                    initial_data_fset=primary_state_set,
                    final_data_fset=primary_state_set,
                    set_sa=ev.init_sa, set_sb=ev.init_sb)
            seq.append(m)
        seq = sorted([ma] + seq + [mb])

        # Add edges to the meta node tree.
        for ma, mb in zip(seq[:-1], seq[1:]):
            meta_node_tree.add_edge(ma, mb)

    # Build the tree whose vertices are edges of the meta node tree.
    meta_edge_tree, meta_edge_root = get_edge_tree(meta_node_tree, mroot)

    # Create the map from nodes of the meta edge tree
    # to sets of primary states not directly contradicted by data or context.
    node_to_data_fset = {}
    for meta_edge in meta_edge_tree:
        ma, mb = meta_edge
        fset = ma.final_data_fset & mb.initial_data_fset
        node_to_data_fset[meta_edge] = fset

    # Create the map from edges of the meta edge tree
    # to primary state transition matrices.
    edge_to_P = {}
    for pair in meta_edge_tree.edges():
        (ma, mb), (mb2, mc) = pair
        if mb != mb2:
            raise Exception('incompatibly constructed meta edge tree')
        edge_to_P[pair] = mb.P_nx

    # Use nxmctree to sample a history on the meta edge tree.
    h = sample_history(meta_edge_tree, edge_to_P, meta_edge_root,
            primary_distn, node_to_data_fset)

    # Use the sampled history to update the primary history at structural nodes
    # and to update the primary event transitions.
    for meta_edge in meta_edge_tree:
        ma, mb = meta_edge
        state = h[meta_edge]
        ma.set_sb(state)
        mb.set_sa(state)



###############################################################################
# Main Rao-Teh-Gibbs sampling function.


def blinking_model_rao_teh(T, root, edge_to_blen, primary_to_tol, Q_primary,
        rate_on, rate_off, uniformization_factor, event_map,
        primary_track, tolerance_tracks, track_to_data):
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
    uniformization_factor : float
        Somewhat arbitrary constant greater than 1.  Two is good.
    event_registry : dict
        map from event id to event object
    primary_track : hashable
        label of the primary track
    tolerance_tracks : collection of hashables
        labels of tolerance tracks
    track_to_data : x
        x

    """
    Q_blink = nx.DiGraph()
    Q_blink.add_edge(False, True, weight=rate_on)
    Q_blink.add_edge(True, False, weight=rate_off)

    # Partially initialize track info.
    # This does not intialize a history or a trajectory.
    track_to_info = dict((t, TrackInfo(t, d)) for t, d in track_to_data.items())

    # Initialize the tolerance trajectories.
    # Add incomplete events to the first and last thirds of the branch
    # with the eventual goal to force the middle third of each branch
    # to be in the True tolerance state regardless of the endpoint data.
    for track in tolerance_tracks:
        for edge in T.edges():
            blen = edge_to_blen[edge]
            for low, high in ((0, 1/3), (2/3, 1)):
                tm = random.uniform(low, high) * blen
                ev = Event(track=track, edge=edge, tm=tm, trans=None)
                track.events[edge].append(ev)
                event_registry[id(ev)] = ev

    # Initialize the primary trajectory.
    #

    # Define 'incomplete events' associated with the primary track.

    for i in range(nsamples):
        for foreground_track in tracks:

            # TODO
            # Add poisson events as incomplete events, using the ephemeral
            # edges defined by the foreground track.
            poisson_rates = x
            add_poisson_events(T, edge_to_blen, poisson_rates, track_info)

            # TODO
            # Remove the transitions associated with foreground events
            # and remove the foreground history.

            # TODO
            # Use the background and foreground events and node states and data
            # to define state restrictions on ephemeral edges and to define the
            # transitions between the states at these ephemeral edges.
            # Then use this construction to sample transitions associated
            # with foreground events and to assign a corresponding foreground
            # history to the structural nodes.
            sample_primary(T, root, primary_to_tol, primary_distn,
                    P_primary, primary_info, blink_infos):

            # TODO
            # Remove all foreground events that correspond to self-transitions.

            pass



###############################################################################
# Copypasted model specification code etc.


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
    all_tols = list(product((0, 1), repeat=3))
    compound_states = list(product(range(nprimary), all_tols))
    return compound_states


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False


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

