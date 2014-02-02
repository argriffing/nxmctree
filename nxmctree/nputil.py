"""
Utility functions that depend on numpy.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal, assert_allclose

def assert_dict_distn_allclose(da, db):
    keys = set(da) & set(db)
    assert_equal(set(da), keys)
    assert_equal(set(da), keys)
    if keys:
        va = np.array([da[k] for k in keys])
        vb = np.array([db[k] for k in keys])
        assert_allclose(va, vb)

