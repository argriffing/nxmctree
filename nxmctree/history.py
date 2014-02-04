"""
Functions related to histories on trees.

Every node in a Markov chain tree history has a known state.

"""

import itertools

__all__ = [
        'gen_plausible_histories',
        ]


def gen_plausible_histories(node_to_data_feasible_set):
    """
    Yield histories compatible with observed data.
    Each history is a collection of (node, state) pairs.
    Some of these histories may have zero probability.
    """
    nodes = set(node_to_data_feasible_set)
    pairs = node_to_data_feasible_set.items()
    nodes, sets = zip(*pairs)
    for assignment in itertools.product(*sets):
        yield dict(zip(nodes, assignment))

