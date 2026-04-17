import random
import networkx as nx

def _calculate_path_cost(graph: nx.Graph, path: list) -> float:
    """
    Calculates the total cost of a path.
    Applies heavy penalties if constraints are not respected.
    """
    cost = 0
    PENALTY = 1000000 # Strong penalty for violated constraints

    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]

        weight = graph.edges[u, v].get('weight', -1)

        if weight == -1:
            cost += PENALTY
        else:
            cost += weight

    for i, node in enumerate(path):
        prev_node = graph.nodes[node].get('precedence')
        if prev_node is not None:
            prev_index = path.index(prev_node)
            if prev_index > i:
                cost += PENALTY

    return cost

def _generate_random_path(nodes: list) -> list:
    """Generates an initial random solution (permutation of nodes)."""
    path = list(nodes)
    deposit = path[0]
    rest_of_path = path[1:]
    random.shuffle(rest_of_path)
    return [deposit] + rest_of_path

def _get_neighbors(path: list) -> list:
    """Generates the neighborhood using a simple 2-node swap."""
    neighbors = []
    for i in range(1, len(path) - 1):
        for j in range(i + 1, len(path)):
            neighbor = path.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def _hill_climbing(graph: nx.Graph, initial_path: list) -> tuple:
    """Executes a single hill climbing pass."""
    current_path = initial_path
    current_cost = _calculate_path_cost(graph, current_path)

    while True:
        neighbors = _get_neighbors(current_path)
        best_neighbor = None
        best_neighbor_cost = current_cost

        for neighbor in neighbors:
            cost = _calculate_path_cost(graph, neighbor)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost

        if best_neighbor is not None:
            current_path = best_neighbor
            current_cost = best_neighbor_cost
        else:
            break

    return current_path, current_cost

def resolve_by_ms_hill_climbing(graph: nx.Graph, iterations: int = 50) -> tuple:
    """Runs the Hill Climbing algorithm multiple times from different starting points."""
    nodes = list(graph.nodes())
    global_best_path = None
    global_best_cost = float('inf')

    for _ in range(iterations):
        initial_path = _generate_random_path(nodes)
        final_path, final_cost = _hill_climbing(graph, initial_path)

        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_path = final_path

    return global_best_path, global_best_cost
