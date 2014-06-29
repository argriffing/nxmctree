Markov chain algorithms on a Python NetworkX tree.

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [git](http://git-scm.com/) (installation)
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

    $ rm -rf htmlcov/
    $ python-coverage run runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Build docs locally:

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

Subsequently update online docs:

    $ git checkout gh-pages
    $ cp /tmp/nxdocs/. ./ -R
    $ git add .
    $ git commit -am "update gh-pages"
    $ git push

