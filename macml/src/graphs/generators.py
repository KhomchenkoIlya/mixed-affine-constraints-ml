import networkx as nx


def make_default_graphs(n: int = 60, seed: int = 0):
    """Набор стандартных графов коммуникации для сравнения."""
    return {
        "Ring": nx.cycle_graph(n),
        "Grid_6x10": nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 10)),
        "ER_p0.08": nx.erdos_renyi_graph(n, 0.08, seed=seed),
        "BA_m2": nx.barabasi_albert_graph(n, 2, seed=seed),
        "WS_k6_p0.1": nx.watts_strogatz_graph(n, 6, 0.1, seed=seed),
    }
