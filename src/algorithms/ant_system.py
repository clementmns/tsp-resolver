import networkx as nx
import numpy as np

from graph_generator import generate_graph

def _node_can_be_visited(graph: nx.Graph, node: int, visited: set[int]) -> bool:
    """Check if a node can be visited based on its precedence constraint."""
    precedence = graph.nodes[node].get('precedence')
    return precedence is None or precedence in visited

def _valid_next_nodes(graph: nx.Graph, current: int, visited: set[int]) -> list[int]:
    """Get a list of valid next nodes that can be visited from the current node, based on the graph's edges, weights, and precedence constraints."""
    return [
        node for node in graph.nodes()
        if node not in visited
        and _node_can_be_visited(graph, node, visited)
        and graph.edges[current, node]['weight'] != -1
    ]

def resolve_by_ant_system(
    graph: nx.Graph,
    n_iterations: int = 200,
    n_ants: int | None = None,
    alpha: float = 1.0,
    beta: float = 5.0,
    evaporation_rate: float = 0.5,
    initial_pheromone: float = 1.0,
) -> tuple[list[int], float]:
    """Resolve the graph using a simple ant colony optimization algorithm."""
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return [], 0.0

    if n_ants is None:
        n_ants = node_count

    start_candidates = [node for node in graph.nodes() if graph.nodes[node].get('precedence') is None]
    if not start_candidates:
        start_candidates = list(graph.nodes())

    pheromones = np.full((node_count, node_count), initial_pheromone, dtype=float)
    best_tour: list[int] = []
    best_cost = float('inf')

    for _ in range(n_iterations):
        iteration_solutions: list[tuple[list[int], float]] = []

        for _ in range(n_ants):
            start = int(np.random.choice(start_candidates))
            tour = [start]
            visited = {start}

            while len(tour) < node_count:
                current = tour[-1]
                neighbors = _valid_next_nodes(graph, current, visited)
                if not neighbors:
                    break

                scores = []
                for neighbor in neighbors:
                    weight = graph.edges[current, neighbor]['weight']
                    score = (pheromones[current, neighbor] ** alpha) * ((1.0 / weight) ** beta)
                    scores.append(score)

                scores = np.array(scores, dtype=float)
                if scores.sum() <= 0:
                    next_node = int(np.random.choice(neighbors))
                else:
                    probabilities = scores / scores.sum()
                    next_node = int(np.random.choice(neighbors, p=probabilities))

                tour.append(next_node)
                visited.add(next_node)

            if len(tour) != node_count:
                continue

            total_cost = 0.0
            feasible = True
            for u, v in zip(tour, tour[1:]):
                weight = graph.edges[u, v]['weight']
                if weight == -1:
                    feasible = False
                    break
                total_cost += weight

            if not feasible:
                continue

            return_weight = graph.edges[tour[-1], tour[0]]['weight']
            if return_weight == -1:
                continue

            total_cost += return_weight
            closed_tour = tour + [tour[0]]

            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = closed_tour

            iteration_solutions.append((tour, total_cost))

        pheromones *= (1.0 - evaporation_rate)
        for tour, cost in iteration_solutions:
            deposit = 1.0 / cost
            for u, v in zip(tour, tour[1:]):
                pheromones[u, v] += deposit
                pheromones[v, u] += deposit
            pheromones[tour[-1], tour[0]] += deposit
            pheromones[tour[0], tour[-1]] += deposit

    if not best_tour:
        return [], float('inf')

    return best_tour, best_cost
