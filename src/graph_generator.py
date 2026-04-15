import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

FUEL_PRICE_PER_KM = 0.15

FORBIDDEN_VERTICES_RATIO = 0.05

PRECEDENCE_NODES_RATIO = 0.15

TOLL_RATIO = 0.2
TOLL_MIN_COST = 5
TOLL_MAX_COST = 30

def compute_toll_cost(a: int, b: int) -> int:
    return np.random.randint(a, b)

def compute_weight(distance: float) -> float:
    has_toll = np.random.rand() < TOLL_RATIO
    if has_toll:
        return compute_toll_cost(TOLL_MIN_COST, TOLL_MAX_COST) + (distance * FUEL_PRICE_PER_KM)
    return distance * FUEL_PRICE_PER_KM

def is_forbidden_vertex() -> bool:
    return np.random.rand() < FORBIDDEN_VERTICES_RATIO

def is_precedence_constraint() -> bool:
    return np.random.rand() < PRECEDENCE_NODES_RATIO

def get_random_node_preference(graph_size: int, precedences: dict[int, int], current_node: int) -> int:
    """Get a random node that is not already a key in the precedences dict, and does not have current_node as a value in the precedences dict (to avoid cycles). Fallback to -1 if no valid node found after 1000 attempts."""
    for _ in range(100):
        node = np.random.randint(0, graph_size)
        # reject if already a key, or if inverse exists: {node: current_node} would conflict with {current_node: node}
        if node not in precedences and node != current_node:
            return node

    return -1 # fallback if no valid node found after 1000 attempts

def generate_graph(n: int) -> nx.Graph:
    graph = nx.complete_graph(n)
    precedences: dict[int, int] = {} # first key is the node, second key is the precedence node
    for node in graph.nodes():
        if is_precedence_constraint():
            precedence_node = get_random_node_preference(n, precedences, node)
            if precedence_node != -1:
                precedences[node] = precedence_node
                graph.nodes[node]['precedence'] = precedence_node
            else:
                graph.nodes[node]['precedence'] = None
        else:
            graph.nodes[node]['precedence'] = None

    for (u, v) in graph.edges():
        is_forbidden :bool = is_forbidden_vertex()
        distance = np.random.randint(1, 100)
        weight = -1 if is_forbidden else compute_weight(distance)

        graph.edges[u, v]['weight'] = weight
    return graph

def display_graph(graph: nx.Graph) -> None:
    precedences_edges = [(u, precedence) for u, precedence in graph.nodes(data='precedence') if precedence is not None]
    normal = []
    forbidden = []

    for (u, v) in graph.edges():
        forbidden.append((u, v)) if (graph.edges[u, v]['weight'] == -1) else normal.append((u, v))

    edge_labels = {
        (u, v): f"{graph.edges[u, v]['weight']:.2f}"
        for (u, v) in normal
    }

    plt.figure(figsize=(10, 10))

    pos = nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=500)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edgelist=normal, edge_color="b", alpha=0.5)
    nx.draw_networkx_edges(graph, pos, edgelist=forbidden, edge_color="r", alpha=0.5, style="dashed", width=1.5)
    nx.draw_networkx_edges(graph, pos, edgelist=precedences_edges, edge_color="g", alpha=1, connectionstyle="arc3,rad=0.3", arrows=True, arrowstyle='->', width=2)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
