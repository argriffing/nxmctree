"""
Functions related to sampling the poisson events.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from util import get_total_rates, get_omega
from trajectory import Event


__all__ = [
        'sample_segment_poisson_events', 'gen_segments',
        'sample_primary_poisson_events', 'sample_blink_poisson_events',
        ]


def sample_segment_poisson_events(track, rate, tma, tmb):
    """
    Sample poisson events on a segment.

    Parameters
    ----------
    track : Trajectory
        trajectory object for which the poisson events should be sampled
    rate : float
        poisson rate of events
    tma : float
        initial segment time
    tmb : float
        final segment time

    Returns
    -------
    events : list
        list of event objects

    """
    blen = tmb - tma
    nevents = np.random.poisson(rate * blen)
    times = np.random.uniform(low=tma, high=tmb, size=nevents)
    events = []
    for tm in times:
        ev = Event(track=track, tm=tm)
        events.append(ev)
    return events


def gen_segments(edge, node_to_tm, fg_track, bg_tracks, bg_to_fg_fset):
    """
    Iterate over segments, tracking the background state.

    This is a helper function for sampling poisson events.
    On each segment, neither the background nor the foreground state changes.

    """
    va, vb = edge
    edge_tma = node_to_tm[va]
    edge_tmb = node_to_tm[vb]

    events = []
    events.extend(fg_track.events[edge])
    for bg_track in bg_tracks:
        events.extend(bg_track.events[edge])

    # Construct tuples corresponding to sorted events or nodes.
    # No times should coincide.
    seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in sorted(events)]
    info_a = (edge_tma, None, None, None)
    info_b = (edge_tmb, None, None, None)
    seq = [info_a] + seq + [info_b]

    # Initialize foreground state and background states
    # at the beginning of the edge.
    track_to_state = {}
    for bg_track in bg_tracks:
        track_to_state[bg_track.name] = bg_track.history[va]
    track_to_state[fg_track.name] = fg_track.history[va]

    # Iterate over segments of the edge.
    # Within each segment the foreground and background tracks
    # maintain the same state, and therefore the rate
    # of new foreground poisson events is constant within the segment.
    poisson_events = []
    for segment in zip(seq[:-1], seq[1:]):
        info_a, info_b = segment
        tma, tracka, saa, sba = info_a
        tmb, trackb, sab, sbb = info_b
        blen = tmb - tma

        # Keep the state of each track up to date.
        if tracka is not None:
            tm, track, sa, sb = info_a
            name = tracka.name
            if track_to_state[name] != sa:
                raise Exception('incompatible transition: '
                        'current state on track %s is %s '
                        'but encountered a transition event from '
                        'state %s to state %s' % (
                            name, track_to_state[name], sa, sb))
            track_to_state[name] = sb

        yield tma, tmb, track_to_state


def sample_primary_poisson_events(edge, node_to_tm,
        primary_track, blink_tracks, blink_to_primary_fset):
    """
    Sample poisson events on an edge of the primary codon-like trajectory.

    This function is a specialization of an earlier function
    named sample_poisson_events which had intended to not care about
    primary vs. blinking tracks except through their roles as
    foreground vs. background tracks.
    Some specific details of our model causes this aggregation
    to not work so well in practice, but such a unification will make more
    sense when a fully general CTBN sampler is implemented.

    Parameters
    ----------
    edge : x
        x
    node_to_tm : x
        x
    primary_track : x
        x
    blink_tracks : x
        x
    blink_to_primary_fset : x
        x

    """
    fg_track = primary_track
    bg_tracks = blink_tracks
    bg_to_fg_fset = blink_to_primary_fset

    poisson_events = []
    for tma, tmb, track_to_state in gen_segments(edge, node_to_tm,
            fg_track, bg_tracks, bg_to_fg_fset):

        # Get the set of foreground states allowed by the background.
        fsets = []
        for bg_track in bg_tracks:
            bg_state = track_to_state[bg_track.name]
            fsets.append(bg_to_fg_fset[bg_track.name][bg_state])
        fg_allowed = set.intersection(*fsets)

        # Get the local transition rate matrix determined by background.
        Q_local = nx.DiGraph()
        for s in fg_track.Q_nx:
            Q_local.add_node(s)
        for sa, sb in fg_track.Q_nx.edges():
            if sb in fg_allowed:
                rate = fg_track.Q_nx[sa][sb]['weight']
                Q_local.add_edge(sa, sb, weight=rate)

        # Compute the total local rates.
        local_rates = get_total_rates(Q_local)
        local_omega = get_omega(local_rates, 2)

        # Compute the poisson rate.
        fg_state = track_to_state[fg_track.name]
        poisson_rate = local_omega - local_rates[fg_state]

        # Sample some poisson events on the segment.
        segment_events = sample_segment_poisson_events(
                fg_track, poisson_rate, tma, tmb)
        poisson_events.extend(segment_events)

    # Add the poisson events into the list of foreground
    # track events for this edge.
    fg_track.events[edge].extend(poisson_events)


def sample_blink_poisson_events(edge, node_to_tm,
        fg_track, bg_tracks, bg_to_fg_fset):
    """

    """
    poisson_events = []
    for tma, tmb, track_to_state in gen_segments(edge, node_to_tm,
            fg_track, bg_tracks, bg_to_fg_fset):

        # Compute the poisson rate.
        fg_sa = track_to_state[fg_track.name]
        poisson_rate = fg_track.poisson_rates[fg_sa]

        # Sample some poisson events on the segment.
        segment_events = sample_segment_poisson_events(
                fg_track, poisson_rate, tma, tmb)
        poisson_events.extend(segment_events)

    # Add the poisson events into the list of foreground
    # track events for this edge.
    fg_track.events[edge].extend(poisson_events)

