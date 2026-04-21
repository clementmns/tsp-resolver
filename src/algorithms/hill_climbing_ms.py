import random
from ..helper import *

def _generate_random_tour(nodes: list) -> list:
    """Generates an initial random solution (permutation of nodes)."""
    tour: list = list(nodes)
    depot: int = tour[0]
    rest_of_tour: list = tour[1:]
    random.shuffle(rest_of_tour)
    return [depot] + rest_of_tour


def _get_neighbors(tour: list) -> list:
    """Generates the neighborhood using a simple 2-node swap."""
    neighbors: list = []
    for i in range(1, len(tour) - 1):
        for j in range(i + 1, len(tour)):
            neighbor: list = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors


def resolve_by_hill_climbing(graph: nx.Graph, initial_tour: list) -> tuple:
    """Executes a single hill climbing pass."""
    current_tour: list = initial_tour
    current_cost: float = calculate_tour_cost_with_penalty(graph, current_tour)

    while True:
        neighbors: list = _get_neighbors(current_tour)
        best_neighbor: list | None = None
        best_neighbor_cost: float = current_cost

        for neighbor in neighbors:
            cost: float = calculate_tour_cost_with_penalty(graph, neighbor)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost

        if best_neighbor is not None:
            current_tour = best_neighbor
            current_cost = best_neighbor_cost
        else:
            break

    return current_tour, current_cost


def resolve_by_ms_hill_climbing(graph: nx.Graph, iterations: int = 50) -> tuple:
    """Runs the Hill Climbing algorithm multiple times from different starting points."""
    nodes: list = list(graph.nodes())
    global_best_tour: list | None = None
    global_best_cost: float = float("inf")

    for _ in range(iterations):
        initial_tour: list = _generate_random_tour(nodes)

        final_tour: list
        final_cost: float
        final_tour, final_cost = resolve_by_hill_climbing(graph, initial_tour)

        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_tour = final_tour

    if global_best_tour is None or global_best_cost >= PENALTY_COST:
        print("No valid tour is possible with the current constraints.")
        return [], float("inf")

    return global_best_tour, global_best_cost
