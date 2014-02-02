"""

"""
from __future__ import division, print_function, absolute_import

class FeasibilityError(Exception):
    pass

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

def dict_elementwise_sum(da, db):
    """
    Sparse vector addition using dicts.
    """
    ret = dict((k, 0) for k in set(da) | set(db))
    for d in da, db:
        for k, v in d.items():
            ret[k] += v
    return ret

