import numpy as np

from ..helper import *

def _nearest_neighbor_tour(graph: nx.Graph, start: int) -> tuple[list[int], float]:
    """
    Build a tour using the nearest neighbour heuristic from a given start node.
    At each step the closest unvisited reachable node (lowest edge weight) is chosen.
    If no valid neighbour exists the algorithm falls back to any unvisited node,
    accumulating a penalty so the result remains comparable with other solvers.
    Returns the closed tour and its total cost.
    """
    node_count: int = graph.number_of_nodes()
    tour: list[int] = [start]
    visited: set[int] = {start}

    while len(tour) < node_count:
        current: int = tour[-1]
        candidates: list[int] = valid_next_nodes(graph, current, visited)

        if candidates:
            # Pick the nearest cheapest valid neighbour
            next_node = min(candidates, key=lambda n: graph.edges[current, n]["weight"])
        else:
            # Fallback: any unvisited node that satisfies precedence (ignoring forbidden edges)
            fallback: list[int] = [
                n
                for n in graph.nodes()
                if n not in visited and node_can_be_visited(graph, n, visited)
            ]
            if not fallback:
                # Last resort: any unvisited node
                fallback = [n for n in graph.nodes() if n not in visited]
            if not fallback:
                break
            next_node = fallback[0]

        tour.append(next_node)
        visited.add(next_node)

    cost: float = calculate_tour_cost_with_penalty(graph, tour)
    return tour + [tour[0]], cost


def resolve_by_nearest_neighbor(
    graph: nx.Graph,
    multi_start: bool = True,
) -> tuple[list[int], float]:
    """
    Resolve the TSP using the nearest neighbour heuristic.

    When *multi_start* is True (default) the algorithm is run once from every
    node and the best closed tour is returned, which greatly reduces sensitivity
    to the arbitrary choice of starting point.
    When *multi_start* is False a single run is performed from a random node
    that has no precedence constraint.
    """
    node_count: int = graph.number_of_nodes()
    if node_count == 0:
        return [], 0.0

    if multi_start:
        starts: list[int] = list(graph.nodes())
    else:
        free_starts: list[int] = [
            n for n in graph.nodes() if graph.nodes[n].get("precedence") is None
        ]
        starts: list[int] = [
            int(np.random.choice(free_starts if free_starts else list(graph.nodes())))
        ]

    best_tour: list[int] = []
    best_cost: float = float("inf")

    for start in starts:
        tour: list[int]
        cost: float
        tour, cost = _nearest_neighbor_tour(graph, start)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour, best_cost
