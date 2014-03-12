"""
Utility functions.

"""
from __future__ import division, print_function, absolute_import

import operator


def prod(seq):
    return reduce(operator.mul, seq, 1)


def ddec(**kwargs):
    """
    A decorator that puts some named string substitutions into a docstring.

    """
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj
    return dec


def dict_distn(d):
    """
    
    Parameters
    ----------
    d : dict
        Map from a key to a non-negative floating point number.

    Returns
    -------
    ret : dict
        Map from a key to a probability where the probabilities sum to 1
        unless the dict is empty.

    """
    total = sum(d.values())
    return dict((k, v / total) for k, v in d.items())


def generalize_fset(fset):
    return dict((s, 1) for s in fset)


def generalize_node_to_data_fset(node_to_data_fset):
    node_to_data_lmap = dict(
            (v, generalize_fset(fset)) for v, fset in node_to_data_fset.items())
    return node_to_data_lmap

