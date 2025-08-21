import networkx as nx
import random 
from typing import List, Tuple, Literal, Set

Edge = Tuple[int, int]


def generate_connected_graph(
    n_nodes: int,
    mode:
    Literal["undirected", "directed", "mixed"] = "undirected",
    extra_edge_prob: float = 0.3,
    rng: random.Random | None = None,
) -> List[Edge]:
    """
    Return a list of edges (u, v) on nodes 1..n_nodes.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (≥ 2).
    mode : {"undirected", "directed", "mixed"}
        *undirected* – every edge is stored once (u < v).  
        *directed*   – edges have orientation.  
        *mixed*      – some edges appear once (directed) and some twice
                       (u→v and v→u) to emulate “double-directed”.
    extra_edge_prob : float
        Probability of adding each *possible* extra edge beyond the
        spanning structure.
    rng : random.Random | None
        Optional seeded RNG for reproducibility.

    Returns
    -------
    List[Edge]
        List of (u, v) tuples.
    """
    if n_nodes < 2:
        raise ValueError("Need at least 2 nodes.")

    rng = rng or random
    edges: Set[Edge] = set()

    # 1) Build a spanning structure that guarantees connectivity
    if mode == "undirected":
       
        for v in range(2, n_nodes + 1):
            u = rng.randint(1, v - 1)
            edges.add(tuple(sorted((u, v))))
    else:  # directed or mixed 
        for v in range(1, n_nodes):
            edges.add((v, v + 1))
        edges.add((n_nodes, 1))

    # 2) Add extra random edges
    for u in range(1, n_nodes + 1):
        for v in range(u + 1, n_nodes + 1):
            if rng.random() < extra_edge_prob:
                if mode == "undirected":
                    edges.add(tuple(sorted((u, v))))
                elif mode == "directed":
                    edges.add((u, v) if rng.random() < 0.5 else (v, u))
                else:  # mixed
                    if rng.random() < 0.5:
                        edges.add((u, v) if rng.random() < 0.5 else (v, u))
                    else:
                        edges.add((u, v))
                        edges.add((v, u))

    return sorted(edges)

def generate_binary_tree(
    n_nodes: int,
    rng: random.Random | None = None,
    as_directed: bool = True,
) -> List[Edge]:
    """
    Produce a random binary tree with ≤ 2 children per node.

    Parameters
    ----------
    n_nodes : int
        Number of nodes (≥ 1).
    rng : random.Random | None
        Optional seeded RNG.
    as_directed : bool
        If True, edges point parent→child; otherwise undirected (u < v).

    Returns
    -------
    List[Edge]
        Edge list representing the tree.
    """
    if n_nodes < 1:
        raise ValueError("Tree needs ≥ 1 node.")
    if n_nodes == 1:
        return []

    rng = rng or random
    edges: List[Edge] = []
    children_count = {1: 0}  

    for child in range(2, n_nodes + 1):
        # pick a parent that still has < 2 children
        possible_parents = [p for p, c in children_count.items() if c < 2]
        parent = rng.choice(possible_parents)
        children_count[parent] += 1
        children_count[child] = 0  

        if as_directed:
            edges.append((parent, child))
        else:  # undirected version
            edges.append(tuple(sorted((parent, child))))

    return edges

def build_graph(edges, mode):
    """Return a NetworkX Graph or DiGraph from edge list."""
    if mode == "directed":
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    G.add_edges_from(edges)
    return G

def report_metrics(G, name):
    as_undir = G.to_undirected()
    tw, _ = nx.approximation.treewidth_min_degree(as_undir)
    print(f"\n{name}")
    print("  diameter:                ", nx.diameter(as_undir))
    print("  edge density:            ", nx.density(G))
    print("  tw: ", tw)
    print("  acc:   ", nx.average_clustering(as_undir))
    print("  avgspl:  ", nx.average_shortest_path_length(as_undir))


if __name__ == "__main__":
    rng = random.Random(25)
    configs = [
        ("undirected", [6, 10, 14, 18, 22]),
        ("directed",   [6, 10, 14, 18, 22]),
        ("mixed",      [6, 10, 14, 18, 22]),
    ]

    for mode, sizes in configs:
        for n in sizes:
            edges = generate_connected_graph(n, mode=mode, rng=rng)
            print(edges)
            G = build_graph(edges, mode if mode != "mixed" else "undirected")
            report_metrics(G, f"{mode.capitalize()} graph (n={n})")

