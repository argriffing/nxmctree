"""
Functions related to navigation of the tree structure.

"""


def gen_segments(edge, node_to_tm, tracks):
    """
    Iterate over segments, tracking the background state.

    This is a helper function for sampling poisson events.
    On each segment, neither the background nor the foreground state changes.

    """
    va, vb = edge
    edge_tma = node_to_tm[va]
    edge_tmb = node_to_tm[vb]

    # Concatenate events from all tracks of interest.
    events = [ev for track in tracks for ev in track.events[edge]]

    # Construct tuples corresponding to sorted events or nodes.
    # No times should coincide.
    seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in sorted(events)]
    info_a = (edge_tma, None, None, None)
    info_b = (edge_tmb, None, None, None)
    seq = [info_a] + seq + [info_b]

    # Initialize track states at the beginning of the edge.
    track_to_state = dict((t.name, t.history[va]) for t in tracks)

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

