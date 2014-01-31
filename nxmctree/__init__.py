from __future__ import division, print_function, absolute_import

__all__ = ['test', 'bench']

print('hello from nxmctree')

# This idiom is used by scipy to check if it is running during the setup.
try:
    __NXMCTREE_SETUP__
except NameError:
    __NXMCTREE_SETUP__ = False


if __NXMCTREE_SETUP__:
    import sys as _sys
    _sys.stderr.write('Running from the nxmctree source directory.\n')
    del _sys
else:
    from numpy.testing import Tester
    test = Tester().test
    bench = Tester().bench

