from ..helper import *

def _apply_swap(tour: list[int], i: int, j: int) -> list[int]:
    """Return a new tour with positions i and j swapped."""
    new_tour: list[int] = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def resolve_by_tabu_search(
    graph: nx.Graph, n_iterations: int = 500, tabu_tenure: int | None = None
) -> tuple[list[int], float]:
    """Resolve the TSP using a simple Tabu Search with node-swap neighborhood."""
    node_count: int = graph.number_of_nodes()
    if node_count < 2:
        nodes: list[int] = list(graph.nodes())
        return nodes + nodes[:1], 0.0

    # check params
    if tabu_tenure is None:
        tabu_tenure = max(7, node_count // 5)

    current_tour: list[int] = build_greedy_tour(graph)
    if not current_tour:
        return [], float("inf")

    current_cost: float = closed_tour_cost(graph, current_tour)
    best_tour: list[int] = current_tour[:]
    best_cost: float = current_cost

    # tabu_list[frozenset({city_a, city_b})] = iteration until which the swap is forbidden
    tabu_list: dict[frozenset, int] = {}

    for iteration in range(n_iterations):
        best_candidate: list[int] | None = None
        best_candidate_cost: float = float("inf")
        best_move: frozenset | None = None

        n: int = len(current_tour)

        # Explore all possible swaps (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                candidate: list[int] = _apply_swap(current_tour, i, j)

                if not is_tour_feasible(graph, candidate):
                    continue

                cost: float = closed_tour_cost(graph, candidate)
                move_key: frozenset = frozenset({current_tour[i], current_tour[j]})

                is_tabu: bool = tabu_list.get(move_key, 0) > iteration

                # Accept if not tabu, OR aspiration criterion (beats global best)
                if (not is_tabu or cost < best_cost) and cost < best_candidate_cost:
                    best_candidate = candidate
                    best_candidate_cost = cost
                    best_move = move_key

        if best_candidate is None:
            continue

        current_tour = best_candidate
        current_cost = best_candidate_cost

        if best_move is not None:
            tabu_list[best_move] = iteration + tabu_tenure

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour[:]

    if not best_tour:
        return [], float("inf")

    # Return a closed tour
    return best_tour + [best_tour[0]], best_cost
