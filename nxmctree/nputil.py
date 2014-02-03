"""
Utility functions that depend on numpy.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal, assert_allclose

import networkx as nx


def assert_dict_distn_allclose(da, db, **kwargs):
    """
    The extra keyword args are passed to assert_allclose().
    """
    keys = set(da) & set(db)
    assert_equal(set(da), keys)
    assert_equal(set(da), keys)
    if keys:
        va = np.array([da[k] for k in keys])
        vb = np.array([db[k] for k in keys])
        assert_allclose(va, vb, **kwargs)


def assert_nx_distn_allclose(A, B, **kwargs):
    """
    The extra keyword args are passed to assert_allclose().
    """
    nodes = set(A) & set(B)
    assert_equal(set(A), nodes)
    assert_equal(set(B), nodes)
    edges = set(A.edges()) & set(B.edges())
    assert_equal(set(A.edges()), edges)
    assert_equal(set(B.edges()), edges)
    if edges:
        va = np.array([A[a][b]['weight'] for a, b in edges])
        vb = np.array([B[a][b]['weight'] for a, b in edges])
        assert_allclose(va, vb, **kwargs)

