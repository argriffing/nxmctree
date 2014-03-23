"""
Algorithms specialized to homogeneous hidden state paths, for testing.

All algorithms assume that the history is not branching and that the hidden
state transition matrix is the same for each neighboring pair of time points.
The implementation may also assume feasibility of all paths.
These algorithms are from Biological Sequence Analysis by Durbin et al.

"""
from __future__ import division, print_function, absolute_import

from nxmctree.util import ddec

params = """\
    P : networkx directed graph
        A sparse transition matrix as a networkx directed graph.
    prior_distn : dict
        Prior state distribution at the root.
    node_to_data_lmap : dict
        For each node, a map from state to observation likelihood.
"""


@ddec(params=params)
def naive_forward_durbin(P, prior_distn, node_to_data_lmap):
    """
    naive forward durbin

    Parameters
    ----------
    {params}

    Returns
    -------
    ret : list of lists, total probability
        x

    """
    states = sorted(set(P) | set(prior_distn))
    nhidden = len(states)
    nobs = len(node_to_data_lmap)
    f = [[0]*nhidden for i in range(nobs)]
    for sink_index, sink_state in enumerate(states):
        p = prior_distn[sink_state]
        f[0][sink_index] = node_to_data_lmap[0][sink_state] * p
    for i in range(1, nobs):
        lmap = node_to_data_lmap[i]
        for sink_index, sink_state in enumerate(states):
            f[i][sink_index] = lmap[sink_state]
            p = 0
            for source_index, source_state in enumerate(states):
                ptrans = P[source_state][sink_state]['weight']
                p += f[i-1][source_index] * ptrans
            f[i][sink_index] *= p
    total_probability = 0
    for source_index, source_state in enumerate(states):
        total_probability += f[nobs-1][source_index]
    return f, total_probability


@ddec(params=params)
def naive_backward_durbin(P, prior_distn, node_to_data_lmap):
    """
    naive backward durbin

    Parameters
    ----------
    {params}

    Returns
    -------
    ret : list of lists, total probability
        x

    """
    states = sorted(set(P) | set(prior_distn))
    nhidden = len(prior_distn)
    nobs = len(node_to_data_lmap)
    b = [[0]*nhidden for i in range(nobs)]
    b[nobs-1] = [1]*nhidden
    for i in reversed(range(nobs-1)):
        for source_index, source_state in enumerate(states):
            for sink_index, sink_state in enumerate(states):
                p = 1.0
                p *= P[source_state][sink_state]['weight']
                p *= node_to_data_lmap[i+1][sink_state]
                p *= b[i+1][sink_index]
                b[i][source_index] += p
    total_probability = 0
    for sink_index, sink_state in enumerate(states):
        p = 1.0
        p *= prior_distn[sink_state]
        p *= node_to_data_lmap[0][sink_state]
        p *= b[0][sink_index]
        total_probability += p
    return b, total_probability


@ddec(params=params)
def naive_posterior_durbin(P, prior_distn, node_to_data_lmap):
    """

    Parameters
    ----------
    {params}

    Returns
    -------
    ret : distributions
        x

    """
    f, total_f = naive_forward_durbin(P, prior_distn, node_to_data_lmap)
    b, total_b = naive_backward_durbin(P, prior_distn, node_to_data_lmap)
    total = (total_f + total_b) / 2
    distributions = []
    for fs, bs in zip(f, b):
        distribution = [x*y/total for x, y in zip(fs, bs)]
        distributions.append(distribution)
    return distributions


@ddec(params=params)
def scaled_forward_durbin(P, prior_distn, node_to_data_lmap):
    """
    At each position, the sum over states of the f variable is 1.

    Parameters
    ----------
    {params}

    Returns
    -------
    ret : x
        the list of lists of scaled f variables, and the scaling variables
    """
    states = sorted(set(P) | set(prior_distn))
    nhidden = len(prior_distn)
    nobs = len(node_to_data_lmap)
    f = [[0]*nhidden for i in range(nobs)]
    s = [0]*nobs
    # define the initial unscaled f variable
    for sink_index, sink_state in enumerate(states):
        p = prior_distn[sink_state]
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
        for sink_index, sink_state in enumerate(states):
            f[i][sink_index] = lmap[sink_state]
            p = 0
            for source_index, source_state in enumerate(states):
                ptrans = P[source_state][sink_state]['weight']
                p += f[i-1][source_index] * ptrans
            f[i][sink_index] *= p
        # define the scaling factor at this position
        s[i] = sum(f[i])
        # define the scaled f variable at this position
        for sink_index in range(nhidden):
            f[i][sink_index] /= s[i]
    return f, s


@ddec(params=params)
def scaled_backward_durbin(P, prior_distn, node_to_data_lmap, scaling_factors):
    """
    The scaled forward algorithm will have computed the scaling factors.

    Parameters
    ----------
    {params}
    scaling_factors : list
        the scaling factor for each position

    Returns
    -------
    ret : x
        the list of lists of scaled b variables

    """
    states = sorted(set(P) | set(prior_distn))
    nhidden = len(prior_distn)
    nobs = len(node_to_data_lmap)
    b = [[0]*nhidden for i in range(nobs)]
    b[nobs-1] = [1/scaling_factors[nobs-1]]*nhidden
    for i in reversed(range(nobs-1)):
        for source_index, source_state in enumerate(states):
            accum = 0
            for sink_index, sink_state in enumerate(states):
                p = 1.0
                p *= P[source_state][sink_state]['weight']
                p *= node_to_data_lmap[i+1][sink_state]
                p *= b[i+1][sink_index]
                accum += p
            b[i][source_index] = accum / scaling_factors[i]
    return b


@ddec(params=params)
def scaled_posterior_durbin(P, prior_distn, node_to_data_lmap):
    """
    scaled posterior durbin

    Parameters
    ----------
    {params}

    Returns
    -------
    ret : x
        the list of position-specific posterior hidden state distributions

    """
    f, s = scaled_forward_durbin(P, prior_distn, node_to_data_lmap)
    b = scaled_backward_durbin(P, prior_distn, node_to_data_lmap, s)
    distributions = []
    for fs, bs, si in zip(f, b, s):
        distribution = [x*y*si for x, y in zip(fs, bs)]
        distributions.append(distribution)
    return distributions

