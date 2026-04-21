import numpy as np

from ..helper import *


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
    node_count: int = graph.number_of_nodes()
    if node_count == 0:
        return [], 0.0

    if n_ants is None:
        n_ants = node_count

    start_candidates: list[int] = [node for node in graph.nodes() if graph.nodes[node].get("precedence") is None]
    if not start_candidates:
        start_candidates = list(graph.nodes())

    pheromones: np.ndarray = np.full((node_count, node_count), initial_pheromone, dtype=float)
    best_tour: list[int] = []
    best_cost: float = float("inf")

    for _ in range(n_iterations):
        iteration_solutions: list[tuple[list[int], float]] = []

        for _ in range(n_ants):
            start: int = int(np.random.choice(start_candidates))
            tour: list[int] = [start]
            visited: set[int] = {start}

            while len(tour) < node_count:
                current: int = tour[-1]
                neighbors: list[int] = valid_next_nodes(graph, current, visited)
                if not neighbors:
                    break

                scores: list[float] = []
                for neighbor in neighbors:
                    weight: float = graph.edges[current, neighbor]["weight"]
                    score: float = (pheromones[current, neighbor] ** alpha) * ((1.0 / weight) ** beta)
                    scores.append(score)

                scores_array: np.ndarray = np.array(scores, dtype=float)
                if scores_array.sum() <= 0:
                    next_node: int = int(np.random.choice(neighbors))
                else:
                    probabilities: np.ndarray = scores_array / scores_array.sum()
                    next_node: int = int(np.random.choice(neighbors, p=probabilities))

                tour.append(next_node)
                visited.add(next_node)

            if len(tour) != node_count:
                continue

            total_cost: float = 0.0
            feasible: bool = True
            for u, v in zip(tour, tour[1:]):
                weight: float = graph.edges[u, v]["weight"]
                if weight == -1:
                    feasible = False
                    break
                total_cost += weight

            if not feasible:
                continue

            return_weight: float = graph.edges[tour[-1], tour[0]]["weight"]
            if return_weight == -1:
                continue

            total_cost += return_weight
            closed_tour: list[int] = tour + [tour[0]]

            if total_cost < best_cost:
                best_cost = total_cost
                best_tour = closed_tour

            iteration_solutions.append((tour, total_cost))

        pheromones *= (1.0 - evaporation_rate)
        for tour, cost in iteration_solutions:
            deposit: float = 1.0 / cost
            for u, v in zip(tour, tour[1:]):
                pheromones[u, v] += deposit
                pheromones[v, u] += deposit
            pheromones[tour[-1], tour[0]] += deposit
            pheromones[tour[0], tour[-1]] += deposit

    if not best_tour:
        return [], float("inf")

    return best_tour, best_cost
