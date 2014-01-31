#!/usr/bin/env python

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

setup(
        name='nxmctree',
        version='0.1',
        description='Markov chain algorithms on a Python NetworkX tree graph',
        author='alex',
        url='https://github.com/argriffing/nxmctree',
        packages=['nxmctree'],
        )


