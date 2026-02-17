import networkx as nx

def build_synthetic_city_graph():
    # UNDIRECTED GRAPH so blocking works both ways
    G = nx.Graph()

    nodes = {
        "Hospital": (0, 1),
        "A": (1, 2), "B": (2, 2), "C": (3, 2),
        "D": (1, 1), "E": (2, 1), "F": (3, 1),
        "G": (1, 0), "H": (2, 0),
        "Patient": (3, 0)
    }

    for name, pos in nodes.items():
        G.add_node(name, pos=pos)

    # Road network (undirected)
    edges = [
        ("Hospital", "A"), ("A", "B"), ("B", "C"),
        ("Hospital", "D"), ("D", "E"), ("E", "F"), ("F", "C"),
        ("D", "G"), ("G", "H"), ("H", "Patient"),
        ("E", "H"), ("F", "Patient")
    ]

    for u, v in edges:
        G.add_edge(u, v, distance=1, dynamic_weight=1, incident=False)

    positions = {n: G.nodes[n]["pos"] for n in G.nodes()}
    return G, positions
