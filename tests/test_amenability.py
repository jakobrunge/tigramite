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
    
cf = pytest.importorskip("ciflypy")
from tigramite.graphs import Graphs

def generate_random_mg(N, tau_max, edge_prob=0.5, bidirected_fraction=0.33, seed=42):
    """
    Generate a random mixed graph with N nodes. It is acyclic but may contain almost-cycles.
    It returns the graph once in the form used in tigramite and once in the form used in cifly.

    Steps:
    1. Start with an empty directed graph.
    2. Add directed edges i -> j for i > j with probability edge_prob.
    3. Convert around 1/3 of directed edges to bidirected edges.
    4. Use a seed for full reproducibility.
    """
    rng = np.random.default_rng(seed)

    # MG uses shape (N, N, tau_max + 1, tau_max + 1) in Tigramite
    graph = np.zeros((N, N, tau_max + 1, tau_max + 1), dtype="<U3")
    graph[:] = ""
    test_graph = {'-->': [], '<->': []}

    # Step 1–2: Create a DAG (edges only from lower index to higher)
    for i in range(N):
        for j in range(N):
            for lagi in range(tau_max + 1):
                for lagj in range(lagi + 1):
                    if (i + lagi * N > j + lagj * N) and rng.random() < edge_prob:
                        # Step 3: Convert 1/3 of directed edges into bidirected edges
                        if rng.random() < bidirected_fraction:
                            graph[i, j, lagi, lagj] = "<->"
                            graph[j, i, lagj, lagi] = "<->"
                            test_graph['<->'].append((i + lagi * N, j + lagj * N))
                        else:
                            graph[i, j, lagi, lagj] = "-->"
                            graph[j, i, lagj, lagi] = "<--"
                            test_graph['-->'].append((i + lagi * N, j + lagj * N))
    return graph, test_graph

def reference_is_visible(g, X, Y):
        sets = {"X": X, "Y": Y}
        is_visible_table = """
                    EDGES --> <--, <->
                    SETS X, Y
                    COLORS mediator, collider, deadend, Y
                    START  ... [Y] AT Y
                    OUTPUT ... [collider]
                    
                    ... [Y]        | -->    [mediator] | next in X
                    ... [mediator] | <->    [collider] | next not in Y and next not in X
                    ... [mediator] | <--    [collider] | next not in Y and next not in X
                    ... [Y]        | -->    [mediator] | next not in X
                    ... [Y]        | <--,<-> [deadend] | next not in X
                    """
        reached = cf.reach(g, sets, is_visible_table, table_as_string=True)
        return not len(reached) == 0

def reference_is_amenable(g, X, Y):
        invisibles = set()
        for (x, y) in g["-->"]:
            if (x in X) and not (y in X):
                if not reference_is_visible(g, [x], [y]):
                    invisibles.add(y)
        sets = {"X": X, "Invisibles": invisibles}
        is_not_amenable_table = """
                    EDGES --> <--, <->
                    SETS X, Y, Invisibles
                    COLORS yield
                    START ... [yield] AT Invisibles
                    OUTPUT ... [yield]

                    ... [yield]  | -->  [yield] | next not in X and next not in Invisibles
                    """
        reached = cf.reach(g, sets, is_not_amenable_table, table_as_string=True)
        if set(reached).intersection(Y) == set():
            return True
        return False

@pytest.mark.parametrize("N,tau_max,edge_prob,bidirected_fraction,realisations,node_set_size,seed", [
    (5, 2, 0.5, 0.33, 100, 3, 42),  # default quick check
])
def test_is_amenable_randomized(N, tau_max, edge_prob, bidirected_fraction, realisations, node_set_size, seed):
    rng_global = np.random.default_rng(seed)
    skips = 0
    for i in range(realisations):
        tries = 0
        while tries < 1000:  # try 100 times to find a graph that does not contain an almost-cycle
            try:
                graph, test_graph = generate_random_mg(N, tau_max, edge_prob, bidirected_fraction, seed=seed + i)
                causal_graph = Graphs(graph, graph_type='tsg_mag', tau_max=tau_max, verbosity=0)
                tries = 100000000
            except:
                tries += 1
                continue  # skip invalid graphs
        if tries == 100000000:
            X = rng_global.choice(N * (tau_max + 1) - node_set_size, size=node_set_size, replace=False)
            source_nodes = [(n % N, -1 * (n // N)) for n in X]
            Y = rng_global.choice(np.arange(max(X), N * (tau_max + 1)), size=node_set_size, replace=False)
            target_nodes = [(n % N, -1 * (n // N)) for n in Y]
            ref = reference_is_amenable(test_graph, list(X), list(Y))
            got = causal_graph.is_amenable(source_nodes, target_nodes)
            assert ref == got
        else:
            raise ValueError("Could not generate a valid graph without almost-cycles after 1000 tries.")