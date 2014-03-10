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

from util import get_total_rates
from trajectory import Trajectory, Event

#TODO use 'segment' vs. 'edge' jargon


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
        tma = blen * np.random.uniform(0, 1/3)
        tmb = blen * np.random.uniform(2/3, 1)
        eva = Event(track=track_label, edge=edge, tm=tma, sa=sa, sb=True)
        evb = Event(track=track_label, edge=edge, tm=tmb, sa=True, sb=sb)
        track_info.events[edge] = [eva, evb]


def init_incomplete_primary_events(T, edge_to_blen, traj, diameter):
    """
    Parameters
    ----------
    T : nx tree
        tree
    edge_to_blen : dict
        maps edges to branch lengths
    traj : Trajectory
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    """
    for edge in T:
        blen = edge_to_len(edge)
        times = blen * np.random.uniform(low=1/3, high=2/3, size=diameter-1)
        events = [Event(traj=traj, tm=tm) for tm in times]
        track_info.events[edge] = events


###############################################################################
# Classes and functions for steps of Rao Teh iteration.


class MetaNode(object):
    """
    This is hashable so it can be a node in a networkx graph.

    """
    def __init__(self, P_nx=None,
            set_sa=None, set_sb=None, fset=None, transition=None):
        """

        Parameters
        ----------
        P_nx : nx transition matrix, optional
            the node is associated with this transition matrix
        set_sa : callback, optional
            report the sampled initial state
        set_sb : callback, optional
            report the sampled final state
        fset : set, optional
            Set of foreground state restrictions or None.
            None is interpreted as no restriction rather than
            lack of feasible states.
        transition : triple, optional
            A transition like (trajectory_name, sa, sb) or None.
            None is interpreted as absence of background transition.

        """
        self.P_nx = P_nx
        self.set_sa = set_sa
        self.set_sb = set_sb
        self.fset = fset
        self.transition = transition

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


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


def sample_transitions(T, root, fg_track, bg_tracks, bg_to_fg_fset):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    P_nx = fg_track.P_nx
    P_nx_identity = fg_track.P_nx_identity

    # Construct a meta node for each structural node.
    node_to_meta = {}
    for v in T:
        f = partial(set_or_confirm_history_state, primary_info.history, v)
        fset = fg_track.data[v]
        m = MetaNode(P_nx=P_nx_identity,
                set_sa=f, set_sb=f, fset=fset)
        node_to_meta[v] = m

    # Define the meta node corresponding to the root.
    mroot = node_to_meta[root]

    # Build the tree whose vertices are meta nodes,
    # and map edges of this tree to sets of feasible foreground states,
    # accounting for data at structural nodes and background context
    # along edge segments.
    #
    # Also create the map from edges of this tree
    # to sets of primary states not directly contradicted by data or context.
    #
    meta_node_tree = nx.DiGraph()
    node_to_data_fset = {}
    for edge in T.edges():
        va, vb = edge

        # Sequence meta nodes from three sources:
        # the two structural endpoint nodes,
        # the nodes representing transitions in background blinking tracks,
        # and nodes representing transitions in the foreground primary track.
        seq = []
        for v in edge:
            m = node_to_meta[v]
            seq.append(m)
        for bg_track in bg_tracks:
            for ev in bg_track.events[edge]:
                m = MetaNode(P_nx=P_nx_identity,
                        set_sa=do_nothing, set_sb=do_nothing,
                        transition=(bg_track.name, ev.sa, ev.sb))
                seq.append(m)
        for ev in primary_info.events[edge]:
            m = MetaNode(P_nx=P_nx,
                    set_sa=ev.init_sa, set_sb=ev.init_sb)
            seq.append(m)
        seq = sorted([ma] + seq + [mb])

        # Initialize background states at the beginning of the edge.
        bg_track_to_state = {}
        for bg_track in bg_tracks:
            bg_track_to_state[bg_track.name] = bg_track.history[va]

        # Add segments of the edge as edges of the meta node tree.
        # Track the state of each background track at each segment.
        for segment in zip(seq[:-1], seq[1:]):
            ma, mb = segment

            # Keep the state of each background track up to date.
            if ma.transition is not None:
                track_name, sa, sb = ma.transition
                if bg_track_to_state[name] != sa:
                    raise Exception('incompatible transition')
                bg_track_to_state[track_name] = sb

            # Use the states of the background tracks,
            # together with fsets of the two meta nodes if applicable,
            # to define the set of feasible foreground states at this segment.
            fsets = []
            for m in segment:
                if m.fset is not None:
                    fsets.append(m.fset)
            for name, state in bg_track_to_state.items():
                fsets.append(bg_to_fg_fset[name][state])
            fset = set.intersection(*fsets)

            # Map the segment to the fset.
            # Segments will be nodes of the tree whose history will be sampled.
            node_to_data_fset[segment] = fset

            # Add the meta node to the meta node tree.
            meta_node_tree.add_edge(ma, mb)

    # Build the tree whose vertices are edges of the meta node tree.
    meta_edge_tree, meta_edge_root = get_edge_tree(meta_node_tree, mroot)

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


def blinking_model_rao_teh(
        T, root, edge_to_blen, primary_to_tol, Q_primary, Q_blink,
        event_map, primary_track, tolerance_tracks, track_to_data):
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

    init_blink_history(T, edge_to_blen, track_info)

    # Define the map from blink track to set of primary states.
    tol_to_primary_states = defaultdict(set)
    for primary, tol in primary_to_tol.items():
        tol_to_primary_states[tol].add(primary)

    # Partially initialize track info.
    # This does not intialize a history or a trajectory.
    track_to_info = dict((t, TrackInfo(t, d)) for t, d in track_to_data.items())

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
                    P_primary, primary_info, blink_infos)

            # TODO
            # Remove all foreground events that correspond to self-transitions.

            pass



###############################################################################
# Model specification code etc.


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


def get_Q_blink():
    rate_on = 1
    rate_off = 1
    Q_blink = nx.DiGraph()
    Q_blink.add_weighted_edges_from((
        (False, True, rate_on),
        (True, False, rate_off),
        ))
    return Q_blink


def get_primary_to_tol():
    # This is like a genetic code mapping codons to amino acids.
    # It is a map from primary state to tolerance track name.
    primary_to_tol = {
            0 : 'T0',
            1 : 'T0',
            2 : 'T1',
            3 : 'T1',
            4 : 'T2',
            5 : 'T2',
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


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False


def run(primary_to_tol, compound_states, node_to_data_fset):

    # Define the primary rate matrix.
    Q_primary_nx = get_Q_primary()

    # Define the prior primary state distribution.
    #TODO do not use hardcoded uniform distribution
    nprimary = 6
    primary_distn = dict((s, 1/nprimary) for s in range(nprimary))

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary_nx.edges():
        p = primary_distn[sa]
        rate = Q_primary_nx[sa][sb]['weight']
        expected_primary_rate += p * rate
    #
    print('pure primary process expected rate:')
    print(expected_primary_rate)
    print()
    #
    for sa, sb in Q_primary_nx.edges():
        Q_primary_nx[sa][sb]['weight'] /= expected_rate

    # Define the rate matrix for a single blinking trajectory.
    rate_on = 1.0
    rate_off = 1.0
    Q_blink = nx.DiGraph()
    Q_blink.add_edge(False, True, weight=rate_on)
    Q_blink.add_edge(True, False, weight=rate_off)

    # Define the prior blink state distribution.
    #TODO do not use hardcoded uniform distribution
    blink_distn = {False : 0.5, True : 0.5}

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()

    # TODO ...
    uniformization_factor = 2

    # sample correlated trajectories using rao teh on the blinking model
    for foo in blinking_model_rao_teh(
            T, root, edge_to_blen, primary_to_tol, Q_primary, Q_blink,
            event_map, primary_track, tolerance_tracks, track_to_data):
        print(foo)


def main():

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()

    # Define track interactions.
    # This is analogous to the creation of the compound rate matrices.
    interaction_map = {
            'P' : {
                'T0' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {2, 3, 4, 5},
                    },
                'T1' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 4, 5},
                    }
                'T2' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 2, 3},
                    }
                }
            'T0' : {
                'P' : {
                    0 : {True},
                    1 : {True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                }
            'T1' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {True},
                    3 : {True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                }
            'T2' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {True},
                    5 : {True},
                    }
                }
            }

    # Define primary and blinking trajectories.



    t = Trajectory(name=None, data=None, history=None, events=None,
            prior_root_distn=None, Q_nx=None, uniformization_factor=None)

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
    run(primary_to_tol, node_to_data_fset)
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

