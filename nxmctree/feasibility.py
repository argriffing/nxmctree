"""
Markov chain algorithms on trees.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

def get_node_to_posterior_feasible_set(T, root,
        root_prior_feasible_set,
        node_to_data_feasible_set,
        ):
    """
    For each node get the marginal posterior set of feasible states.

    The posterior feasible state set for each node is determined through
    several data sources: transition matrix sparsity,
    state restrictions due to observed data at nodes,
    and state restrictions at the root due to the support of its prior
    distribution.

    Parameters
    ----------
    T : directed networkx tree graph
        Edges are annotated with adjacency matrices A.
        These adjacency matrices are treated as unweighted for the purposes
        of feasibility.
        If the edges of the adjacency matrices are weighted,
        the weights are ignored here.
    root : hashable
        This is the root node.
        Following networkx convention, this may be anything hashable.
    root_prior_feasible_set : set
        The set of feasible prior root states.
        This may be interpreted as the support of the prior state
        distribution at the root.
    node_to_data_feasible_set : dict
        Map from node to set of feasible states.
        The feasibility could be interpreted as due to restrictions
        caused by observed data.

    Returns
    -------
    node_to_posterior_feasible_set : dict
        Map from node to set of posterior feasible states.

    """
    v_to_subtree_feasible_set = _tips_to_root(
            T, root, root_prior_feasible_set, node_to_data_feasible_set)
    _root_to_tips()


def _state_is_subtree_feasible(T, v_to_subtree_feasible_set, v, cs, s):
    """
    T : directed tree
    v_to_subtree_feasible_set : which states are allowed in child nodes
    v : node under consideration
    cs : child nodes of v
    s : state under consideration
    """
    for c in cs:
        A = T[v][c].A
        if s not in A:
            return False
        if not set(A[s].successors()) & v_to_subtree_feasible_set[c]:
            return False
    return True


def _tips_to_root(T, root, root_prior_feasible_set, node_to_data_feasible_set):
    """
    Backward pass.

    """
    # Initialize the map from node to subtree feasibility state set.
    v_to_subtree_feasible_set = {}

    # Determine the subtree feasible set for every node but the root.
    for v in reversed(nx.topological_sort(T, [root])):
        cs = T[v].successors()
        fset_data = set(node_to_data_feasible_set[v])
        if cs:
            fset = set()
            for s in set(fset):
                if _state_is_subtree_feasible(
                        T, v_to_subtree_feasible_set, v, cs, s):
                    fset.add(s)
        else:
            fset = fset_data
        if v == root:
            fset &= root_prior_feasible_set
        v_to_subtree_feasible_set[v] = fset

    # Return the map from node to subtree feasibility state set.
    return v_to_subtree_feasible_set


def _root_to_tips():
    """
    Forward pass.

    """
    pass
