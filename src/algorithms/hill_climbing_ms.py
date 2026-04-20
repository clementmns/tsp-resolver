import random
import networkx as nx

PENALTY: float = 1e6

def _calculate_path_cost(graph: nx.Graph, path: list) -> float:
    """
    Calculates the total cost of a path.
    Applies heavy penalties if constraints are not respected.
    """
    cost: float = 0.0
    position: dict = {node: i for i, node in enumerate(path)}

    for i in range(len(path)):
        u = path[i]
        v = path[(i + 1) % len(path)]

        edge_data = graph.edges.get((u, v))
        weight = edge_data.get('weight', -1) if edge_data is not None else -1

        if weight == -1:
            cost += PENALTY
        else:
            cost += weight

    for i, node in enumerate(path):
        prec_node = graph.nodes[node].get('precedence')
        if prec_node is None:
            continue

        if prec_node not in position:
            cost += PENALTY
        else:
            if position[prec_node] > i:
                cost += PENALTY

    return cost

def _generate_random_path(nodes: list, depot = None) -> list:
    """Generates an initial random solution (permutation of nodes)."""
    if depot is not None:
        rest = [n for n in nodes if n != depot]
        random.shuffle(rest)
        return [depot] + rest
    else:
        path = list(nodes)
        random.shuffle(path)
        return path

def _get_neighbors(path: list, fix_depot: bool = True) -> list:
    """Generates the neighborhood using a simple 2-node swap."""
    neighbors = []
    start = 1 if fix_depot else 0

    for i in range(start, len(path) - 1):
        for j in range(i + 1, len(path)):
            neighbor = path.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)

    return neighbors

def _validate_path(path: list) -> None:
    """
    Validates that the path is valid.
    """
    if len(path) != len(set(path)):
        duplicates = [n for n in path if path.count(n) > 1]
        raise ValueError(
            f"The path contains node twice : {set(duplicates)}"
        )

def resolve_by_hill_climbing(graph: nx.Graph, initial_path: list, fix_depot: bool = True) -> tuple:
    """Executes a single hill climbing pass."""
    _validate_path(initial_path)

    current_tour = initial_path
    current_cost = _calculate_path_cost(graph, current_tour)

    while True:
        neighbors = _get_neighbors(current_tour, fix_depot=fix_depot)
        best_neighbor = None
        best_neighbor_cost = current_cost

        for neighbor in neighbors:
            cost = _calculate_path_cost(graph, neighbor)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost

        if best_neighbor is not None:
            current_tour = best_neighbor
            current_cost = best_neighbor_cost
        else:
            break

    return current_tour, current_cost

def resolve_by_ms_hill_climbing(graph: nx.Graph, iterations: int = 50, depot = None, fix_depot: bool = True) -> tuple:
    """Runs the Hill Climbing algorithm multiple times from different starting points."""
    nodes = list(graph.nodes())

    if iterations <= 0 or not nodes:
        return [], float('inf')

    global_best_path = None
    global_best_cost = float('inf')

    for _ in range(iterations):
        initial_tour = _generate_random_path(nodes, depot=depot)
        final_tour, final_cost = resolve_by_hill_climbing(graph, initial_tour, fix_depot=fix_depot)

        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_path = final_tour

    return global_best_path, global_best_cost
