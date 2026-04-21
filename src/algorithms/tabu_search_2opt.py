import numpy as np

from ..helper import *

def _apply_2opt_swap(tour: list[int], i: int, j: int) -> list[int]:
    """Return a new tour after reversing the segment between indices i+1 and j (2-opt)."""
    return tour[: i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]


def resolve_by_tabu_search_with_2opt(
    graph: nx.Graph,
    n_iterations: int = 500,
    tabu_tenure: int | None = None,
    neighborhood_size: int | None = None,
) -> tuple[list[int], float]:
    """Resolve the TSP using a Tabu Search with 2-opt neighborhood"""
    node_count: int = graph.number_of_nodes()
    if node_count < 2:
        nodes: list[int] = list(graph.nodes())
        return nodes + nodes[:1], 0.0

    # check params
    if tabu_tenure is None:  # number of iterations during which a move remains tabu
        tabu_tenure = max(7, node_count // 5)
    if neighborhood_size is None:
        neighborhood_size = min(node_count * (node_count - 1) // 2, 200)

    current_tour: list[int] = build_greedy_tour(graph)
    if not current_tour:
        return [], float("inf")

    current_cost: float = closed_tour_cost(graph, current_tour)
    best_tour: list[int] = current_tour[:]
    best_cost: float = current_cost

    # each entry is a frozenset({i, j}) representing the swapped edge pair.
    tabu_list: dict[frozenset, int] = {}

    for iteration in range(n_iterations):
        # generate neighborhood: all 2-opt swaps (i, j) with i < j
        n: int = len(current_tour)
        all_moves: list[tuple[int, int]] = [(i, j) for i in range(n - 1) for j in range(i + 2, n)]

        # randomly sample if neighborhood is too large
        if len(all_moves) > neighborhood_size:
            all_moves = [
                all_moves[k]
                for k in np.random.choice(
                    len(all_moves), neighborhood_size, replace=False
                )
            ]

        best_candidate: list[int] | None = None
        best_candidate_cost: float = float("inf")
        best_move: tuple[int, int] | None = None

        for i, j in all_moves:
            candidate: list[int] = _apply_2opt_swap(current_tour, i, j)

            if not is_tour_feasible(graph, candidate):
                continue

            cost: float = closed_tour_cost(graph, candidate)
            move_key: frozenset = frozenset(
                {
                    frozenset({current_tour[i], current_tour[i + 1]}),
                    frozenset({current_tour[j], current_tour[(j + 1) % n]}),
                }
            )

            is_tabu: bool = tabu_list.get(move_key, 0) > iteration

            # accept move if: not tabu, OR aspiration criterion (beats global best)
            if (not is_tabu or cost < best_cost) and cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = cost
                best_move = (i, j)

        if best_candidate is None:
            # all neighbors are tabu and none satisfies aspiration → skip
            continue

        # move to best candidate
        current_tour = best_candidate
        current_cost = best_candidate_cost

        # register the move as tabu
        if best_move is not None:
            i, j = best_move
            n = len(current_tour)
            move_key: frozenset = frozenset(
                {
                    frozenset({current_tour[i], current_tour[i + 1]}),
                    frozenset({current_tour[j], current_tour[(j + 1) % n]}),
                }
            )
            tabu_list[move_key] = iteration + tabu_tenure

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour[:]

    if not best_tour:
        return [], float("inf")

    # return a closed tour
    return best_tour + [best_tour[0]], best_cost
