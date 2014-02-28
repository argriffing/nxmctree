Markov chain algorithms on a Python NetworkX tree.

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [NetworkX](http:/networkx.lanl.gov/) (graph data types and algorithms)
   - `$ pip install --user git+https://github.com/networkx/networkx`

Optional dependencies:
 * [nose](https://nose.readthedocs.org/) (testing)
 * [numpy](http://www.numpy.org/) (more testing infrastructure and assertions)
 * [coverage](http://nedbatchelder.com/code/coverage/) (test coverage)
   - `$ apt-get install python-coverage`


User
----

Install:

    $ pip install --user git+https://github.com/argriffing/nxmctree

Test:

    $ python -c "import nxmctree; nxmctree.test()"

Uninstall:

    $ pip uninstall nxmctree


Developer
---------

Install:

    $ git clone git@github.com:argriffing/nxmctree.git

Test:

    $ python runtests.py

Coverage:

    $ python-coverage run --branch runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Build docs:

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

