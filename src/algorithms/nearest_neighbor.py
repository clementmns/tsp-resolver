import networkx as nx
import numpy as np

from graph_generator import generate_graph


PENALTY = 1_000_000


def _node_can_be_visited(graph: nx.Graph, node: int, visited: set[int]) -> bool:
    """Check if a node can be visited based on its precedence constraint."""
    precedence = graph.nodes[node].get('precedence')
    return precedence is None or precedence in visited


def _valid_next_nodes(graph: nx.Graph, current: int, visited: set[int]) -> list[int]:
    """Get a list of valid next nodes from current, respecting forbidden edges and precedence constraints."""
    return [
        node for node in graph.nodes()
        if node not in visited
        and _node_can_be_visited(graph, node, visited)
        and graph.edges[current, node]['weight'] != -1
    ]


def calculate_tour_cost(graph: nx.Graph, tour: list[int]) -> float:
    """
    Calculate the total cost of a closed tour.
    Applies a heavy penalty for violated constraints (forbidden edges or precedence).
    """
    cost = 0.0

    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        weight = graph.edges[u, v].get('weight', -1)
        cost += PENALTY if weight == -1 else weight

    for i, node in enumerate(tour):
        prec_node = graph.nodes[node].get('precedence')
        if prec_node is not None:
            prec_index = tour.index(prec_node)
            if prec_index > i:
                cost += PENALTY

    return cost


def nearest_neighbor_tour(graph: nx.Graph, start: int) -> tuple[list[int], float]:
    """
    Build a tour using the nearest neighbour heuristic from a given start node.
    At each step the closest unvisited reachable node (lowest edge weight) is chosen.
    If no valid neighbour exists the algorithm falls back to any unvisited node,
    accumulating a penalty so the result remains comparable with other solvers.
    Returns the closed tour and its total cost.
    """
    node_count = graph.number_of_nodes()
    tour = [start]
    visited = {start}

    while len(tour) < node_count:
        current = tour[-1]
        candidates = _valid_next_nodes(graph, current, visited)

        if candidates:
            # Pick the nearest (cheapest) valid neighbour
            next_node = min(candidates, key=lambda n: graph.edges[current, n]['weight'])
        else:
            # Fallback: any unvisited node that satisfies precedence (ignoring forbidden edges)
            fallback = [
                n for n in graph.nodes()
                if n not in visited and _node_can_be_visited(graph, n, visited)
            ]
            if not fallback:
                # Last resort: any unvisited node
                fallback = [n for n in graph.nodes() if n not in visited]
            if not fallback:
                break
            next_node = fallback[0]

        tour.append(next_node)
        visited.add(next_node)

    cost = calculate_tour_cost(graph, tour)
    return tour + [tour[0]], cost


def resolve_by_nearest_neighbor(graph: nx.Graph, multi_start: bool = True,) -> tuple[list[int], float]:
    """
    Resolve the TSP using the nearest neighbour heuristic.

    When *multi_start* is True (default) the algorithm is run once from every
    node and the best closed tour is returned, which greatly reduces sensitivity
    to the arbitrary choice of starting point.
    When *multi_start* is False a single run is performed from a random node
    that has no precedence constraint.
    """
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return [], 0.0

    if multi_start:
        starts = list(graph.nodes())
    else:
        free_starts = [n for n in graph.nodes() if graph.nodes[n].get('precedence') is None]
        starts = [int(np.random.choice(free_starts if free_starts else list(graph.nodes())))]

    best_tour: list[int] = []
    best_cost = float('inf')

    for start in starts:
        tour, cost = nearest_neighbor_tour(graph, start)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost
