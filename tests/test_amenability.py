import numpy as np
import pytest
from tigramite.graphs import Graphs


def make_empty_tsg_graph(N=3, tau_max=1):
    """Create an empty time-series DMG of shape (N, N, tau_max+1, tau_max+1)."""
    g = np.zeros((N, N, tau_max + 1, tau_max + 1), dtype="<U3")
    g[:] = ""
    return g

def test_no_path():
    """If there is no path between the end nodes it is always amenable."""
    N = 3
    graph = make_empty_tsg_graph(N, tau_max=1)

    G = Graphs(graph, graph_type="tsg_mag")
    assert G.is_amenable([(0, 0)], [(1, -1)]) is True


def test_dag_is_always_amenable():
    """DAGs, tsg_dags, and stationary_dags should always be amenable."""
    N = 3
    graph = np.zeros((N, N), dtype="<U3")
    graph[:] = ""

    graph[0, 1] = "-->"
    graph[1, 0] = "<--"

    G = Graphs(graph, graph_type="dag")
    assert G.is_amenable([(0, 0)], [(1, 0)]) is True


def test_visible_first_edge():
    """A simple MAG where the first edge is visible should be amenable."""
    N = 3
    graph = make_empty_tsg_graph(N, tau_max=1)

    # Make X -> Y a visible edge
    graph[0, 1, 0, 0] = "-->"
    graph[1, 0, 0, 0] = "<--"

    #edge Z -> X making the other edge visible
    graph[2, 0, 0, 0] = "-->"
    graph[0, 2, 0, 0] = "<--"

    G = Graphs(graph, graph_type="tsg_mag", tau_max=1)
    assert G.is_amenable([(0, 0)], [(1, 0)]) is True


def test_invisible_first_edge():
    """
    Construct an MAG where X -> Y is NOT visible:

    X -> Y
    X <-> Z  (spouse)
    Z -> Y   (parent of Y)

    According to the definition, the first edge X->Y is invisible.
    """
    N = 3
    graph = make_empty_tsg_graph(N, tau_max=0)

    # X0 -> X1
    graph[0, 1, 0, 0] = "-->"
    graph[1, 0, 0, 0] = "<--"

    # Spouse: X0 <-> X2
    graph[0, 2, 0, 0] = "<->"
    graph[2, 0, 0, 0] = "<->"

    # Z -> Y (0<-2)
    graph[2, 1, 0, 0] = "-->"
    graph[1, 2, 0, 0] = "<--"

    G = Graphs(graph, graph_type="tsg_mag", tau_max=0)

    assert G.is_amenable([(0, 0)], [(1, 0)]) is False


def test_multiple_paths_first_edges_checked():
    """
    Test a structure with multiple proper possibly-directed paths.
    All first edges must be visible.
    """
    N = 4
    graph = make_empty_tsg_graph(N, tau_max=1)

    # Path 1: 0 -> 1 -> 3
    graph[0, 1, 0, 0] = "-->"
    graph[1, 0, 0, 0] = "<--"

    graph[1, 3, 0, 0] = "-->"
    graph[3, 1, 0, 0] = "<--"

    # Path 2: 0 -> 2 -> 3
    graph[0, 2, 0, 0] = "-->"
    graph[2, 0, 0, 0] = "<--"

    graph[2, 3, 0, 0] = "-->"
    graph[3, 2, 0, 0] = "<--"

    #edge into 0 not adjacent to 3 making to first edges visible
    graph[0, 0, 1, 0] = "<->"
    graph[0, 0, 0, 1] = "<->"

    G = Graphs(graph, graph_type="tsg_mag", tau_max=1)

    assert G.is_amenable([(0, 0)], [(3, 0)]) is True
