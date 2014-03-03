"""
Docstring spam that will be copypasted into multiple docstrings.

"""


# put some named string substitutions into a docstring
def ddec(**kwargs):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(**kwargs)
        return obj
    return dec


common_params = """\
    T : directed networkx tree graph
        Edge and node annotations are ignored.
    edge_to_adjacency : dict
        A map from directed edges of the tree graph
        to networkx graphs representing state transition feasibility.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_fset : set
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_fset : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.
"""
