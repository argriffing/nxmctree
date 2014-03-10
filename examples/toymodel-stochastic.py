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



#TODO unused
def init_blink_history(T, edge_to_blen, track):
    """
    Initial blink history is True where consistent with the data.

    """
    for v in T:
        track.history[v] = (True in track.data[v])


#TODO unused
def init_complete_blink_events(T, edge_to_blen, track):
    """
    Init blink track.

    """
    for edge in T.edges():
        va, vb = edge
        sa = track.history[va]
        sb = track.history[vb]
        blen = edge_to_blen[edge]
        tma = blen * np.random.uniform(0, 1/3)
        tmb = blen * np.random.uniform(2/3, 1)
        eva = Event(track=track, edge=edge, tm=tma, sa=sa, sb=True)
        evb = Event(track=track, edge=edge, tm=tmb, sa=True, sb=sb)
        track.events[edge] = [eva, evb]


def init_incomplete_primary_events(T, edge_to_blen, primary_track, diameter):
    """
    Parameters
    ----------
    T : nx tree
        tree
    edge_to_blen : dict
        maps edges to branch lengths
    primary_track : Trajectory
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    """
    for edge in T.edges():
        blen = edge_to_blen[edge]
        times = blen * np.random.uniform(low=1/3, high=2/3, size=diameter-1)
        events = [Event(track=primary_track, tm=tm) for tm in times]
        primary_track.events[edge] = events


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


def do_nothing(state):
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
                T_dual.add_edge((v, c), (c, g))
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
    print('building meta nodes corresponding to structural nodes')
    for v in T:
        f = partial(set_or_confirm_history_state, fg_track.history, v)
        fset = fg_track.data[v]
        m = MetaNode(P_nx=P_nx_identity, set_sa=f, set_sb=f, fset=fset)
        node_to_meta[v] = m
        #print('adding meta node', v, m)

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
    node_to_data_fset = dict()
    for edge in T.edges():
        va, vb = edge

        # Initialize background states at the beginning of the edge.
        bg_track_to_state = {}
        for bg_track in bg_tracks:
            bg_track_to_state[bg_track.name] = bg_track.history[va]

        # Sequence meta nodes from three sources:
        # the two structural endpoint nodes,
        # the nodes representing transitions in background tracks,
        # and nodes representing transitions in the foreground track.
        # Note that meta nodes are not meaningfully sortable,
        # but events are sortable.
        events = []
        for bg_track in bg_tracks:
            for ev in bg_track.events[edge]:
                events.append(ev)
        for ev in fg_track.events[edge]:
            events.append(ev)

        # Construct the meta nodes corresponding to sorted events.
        seq = []
        for ev in sorted(events):
            if ev.track is fg_track:
                m = MetaNode(P_nx=P_nx, set_sa=ev.init_sa, set_sb=ev.init_sb)
            else:
                m = MetaNode(P_nx=P_nx_identity,
                        set_sa=do_nothing, set_sb=do_nothing,
                        transition=(bg_track.name, ev.sa, ev.sb))
            seq.append(m)
        ma = node_to_meta[va]
        mb = node_to_meta[vb]
        seq = [ma] + seq + [mb]

        # Add segments of the edge as edges of the meta node tree.
        # Track the state of each background track at each segment.
        #print('processing edge', va, vb)
        for segment in zip(seq[:-1], seq[1:]):
            ma, mb = segment

            # Keep the state of each background track up to date.
            if ma.transition is not None:
                name, sa, sb = ma.transition
                if bg_track_to_state[name] != sa:
                    raise Exception('incompatible transition: '
                            'current state on track %s is %s '
                            'but encountered a transition event from '
                            'state %s to state %s' % (
                                name, bg_track_to_state[name], sa, sb))
                bg_track_to_state[name] = sb

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
            #print('adding segment', ma, mb)
            meta_node_tree.add_edge(ma, mb)

    # Build the tree whose vertices are edges of the meta node tree.
    meta_edge_tree, meta_edge_root = get_edge_tree(meta_node_tree, mroot)
    #print('size of meta_edge_tree:')
    #print(len(meta_edge_tree))
    #print()
    #print('meta edge root:')
    #print(meta_edge_root)
    #print()

    # Create the map from edges of the meta edge tree
    # to primary state transition matrices.
    edge_to_P = {}
    for pair in meta_edge_tree.edges():
        (ma, mb), (mb2, mc) = pair
        if mb != mb2:
            raise Exception('incompatibly constructed meta edge tree')
        edge_to_P[pair] = mb.P_nx

    # Use nxmctree to sample a history on the meta edge tree.
    node_to_data_fset[meta_edge_root] = fg_track.data[root]
    meta_edge_to_sampled_state = sample_history(
            meta_edge_tree, edge_to_P, meta_edge_root,
            fg_track.prior_root_distn, node_to_data_fset)
    #print('size of sampled history:')
    #print(len(meta_edge_to_sampled_state))
    #print()

    # Use the sampled history to update the primary history at structural nodes
    # and to update the primary event transitions.
    for meta_edge in meta_edge_tree:
        ma, mb = meta_edge
        state = meta_edge_to_sampled_state[meta_edge]
        if ma is not None:
            ma.set_sb(state)
        if mb is not None:
            mb.set_sa(state)



###############################################################################
# Main Rao-Teh-Gibbs sampling function.


def blinking_model_rao_teh(
        T, root, edge_to_blen, primary_to_tol,
        Q_primary, Q_blink,
        primary_track, tolerance_tracks, interaction_map, track_to_data):
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
    Q_blink : x
        x
    primary_track : hashable
        label of the primary track
    tolerance_tracks : collection of hashables
        labels of tolerance tracks
    interaction_map : dict
        x
    track_to_data : x
        x

    """
    #TODO go back to this when disease data is used
    # Initialize blink history and events.
    #for track in tolerance_tracks:
        #init_blink_history(T, edge_to_blen, track)
        #init_complete_blink_events(T, edge_to_blen, track)

    # For now use a custom initialization of the blinking process.
    # Assume that all states are initially blinked on.
    # TODO change this when we begin using disease data
    for track in tolerance_tracks:
        for va in T:
            track.history[va] = True
        for edge in T.edges():
            track.events[edge] = []

    # Initialize the primary trajectory with many incomplete events.
    # TODO change this when we begin using disease data
    diameter = 4
    init_incomplete_primary_events(T, edge_to_blen, primary_track, diameter)
    #
    # Sample the state of the primary track.
    sample_transitions(T, root,
            primary_track, tolerance_tracks, interaction_map['P'])
    #
    # Remove self-transition events from the primary track.
    primary_track.remove_self_transitions()

    # Outer loop of the Rao-Teh-Gibbs sampler.
    while True:

        # Update the primary track.
        primary_track.add_poisson_events(T, edge_to_blen)
        primary_track.clear_state_labels()
        sample_transitions(T, root,
                primary_track, tolerance_tracks, interaction_map['P'])
        primary_track.remove_self_transitions()

        # Update each blinking track.
        for track in tolerance_tracks:
            track.add_poisson_events(T, edge_to_blen)
            track.clear_state_labels()
            sample_transitions(T, root,
                    track, [primary_track], interaction_map[track.name])
            track.remove_self_transitions()

        # Summarize the sample.
        expected_on = 0
        expected_off = 0
        for track in tolerance_tracks:
            for edge in T.edges():
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if transition == (False, True):
                        expected_on += 1
                    elif transition == (True, False):
                        expected_off += 1

        yield expected_on, expected_off



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


def run(primary_to_tol, interaction_map, track_to_node_to_data_fset):

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()

    # Define the uniformization factor.
    uniformization_factor = 2

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
        Q_primary_nx[sa][sb]['weight'] /= expected_primary_rate

    # Define primary trajectory.
    primary_track = Trajectory(
            name='P', data=track_to_node_to_data_fset['P'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary_nx,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    rate_on = 1.0
    rate_off = 1.0
    Q_blink = nx.DiGraph()
    Q_blink.add_edge(False, True, weight=rate_on)
    Q_blink.add_edge(True, False, weight=rate_off)

    # Define the prior blink state distribution.
    #TODO do not use hardcoded uniform distribution
    blink_distn = {False : 0.5, True : 0.5}

    # Define tolerance process trajectories.
    name_to_blink_track = dict()
    for name in ('T0', 'T1', 'T2'):
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        name_to_blink_track[name] = track
    tolerance_tracks = list(name_to_blink_track.values())

    # sample correlated trajectories using rao teh on the blinking model
    expected_on = 0
    expected_off = 0
    for i, info in enumerate(blinking_model_rao_teh(
            T, root, edge_to_blen, primary_to_tol,
            Q_primary_nx, Q_blink,
            primary_track, tolerance_tracks, interaction_map,
            track_to_node_to_data_fset)):
        n = i + 1
        e_on, e_off = info
        expected_on += e_on
        expected_off += e_on
        avg_on = expected_on / n
        avg_off = expected_off / n
        print('n:')
        print(n)
        print()
        print('expected off->on:')
        print(avg_on)
        print()
        print('expected on->off:')
        print(avg_off)
        print()


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
                    },
                'T2' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 2, 3},
                    }
                },
            'T0' : {
                'P' : {
                    0 : {True},
                    1 : {True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T1' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {True},
                    3 : {True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
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


    # No data.
    print ('expectations given no alignment or disease data')
    print()
    data = {
            'P' : {
                'N0' : {0, 1, 2, 3, 4, 5},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {0, 1, 2, 3, 4, 5},
                'N4' : {0, 1, 2, 3, 4, 5},
                'N5' : {0, 1, 2, 3, 4, 5},
                },
            'T0' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            }
    run(primary_to_tol, interaction_map, data)
    print()

    #TODO unfinished after here...

    """
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
    """

    """
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
    """

    """
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
    """


main()

