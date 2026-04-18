# tabu search with 2-opt optimization
import networkx as nx
import numpy as np

def _node_can_be_visited(graph: nx.Graph, node: int, visited: set[int]) -> bool:
    """Check if a node can be visited based on its precedence constraint."""
    precedence = graph.nodes[node].get('precedence')
    return precedence is None or precedence in visited


def _build_greedy_tour(graph: nx.Graph) -> list[int]:
    """Build an initial tour using a greedy nearest-neighbor heuristic."""
    nodes = list(graph.nodes())
    start_candidates = [n for n in nodes if graph.nodes[n].get('precedence') is None]
    if not start_candidates:
        start_candidates = nodes

    best_tour: list[int] = []
    best_cost = float('inf')

    for start in start_candidates:
        tour = [start]
        visited = {start}

        while len(tour) < len(nodes):
            current = tour[-1]
            valid = [
                n for n in nodes
                if n not in visited
                and _node_can_be_visited(graph, n, visited)
                and graph.edges[current, n]['weight'] != -1
            ]
            if not valid:
                break
            nearest = min(valid, key=lambda n: graph.edges[current, n]['weight'])
            tour.append(nearest)
            visited.add(nearest)

        if len(tour) != len(nodes):
            continue
        if graph.edges[tour[-1], tour[0]]['weight'] == -1:
            continue

        cost = sum(graph.edges[u, v]['weight'] for u, v in zip(tour, tour[1:]))
        cost += graph.edges[tour[-1], tour[0]]['weight']

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour


def _tour_cost(graph: nx.Graph, tour: list[int]) -> float:
    """Compute the total cost of a closed tour (last node → first node included)."""
    cost = sum(graph.edges[u, v]['weight'] for u, v in zip(tour, tour[1:]))
    cost += graph.edges[tour[-1], tour[0]]['weight']
    return cost


def _apply_2opt_swap(tour: list[int], i: int, j: int) -> list[int]:
    """Return a new tour after reversing the segment between indices i+1 and j (2-opt)."""
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


def _is_tour_feasible(graph: nx.Graph, tour: list[int]) -> bool:
    """Check that every edge in the closed tour exists and respects precedences."""
    visited: set[int] = set()
    for idx, node in enumerate(tour):
        if not _node_can_be_visited(graph, node, visited):
            return False
        visited.add(node)
        next_node = tour[(idx + 1) % len(tour)]
        if graph.edges[node, next_node]['weight'] == -1:
            return False
    return True

def resolve_by_tabu_search_with_2opt(graph: nx.Graph, n_iterations: int = 500, tabu_tenure: int | None = None, neighborhood_size: int | None = None) -> tuple[list[int], float]:
    """Resolve the TSP using a Tabu Search with 2-opt neighborhood"""
    node_count = graph.number_of_nodes()
    if node_count < 2:
        nodes = list(graph.nodes())
        return nodes + nodes[:1], 0.0

    # check params
    if tabu_tenure is None: # nombre d'itérations ou un mouvement reste interdit
        tabu_tenure = max(7, node_count // 5)
    if neighborhood_size is None:
        neighborhood_size = min(node_count * (node_count - 1) // 2, 200)

    current_tour = _build_greedy_tour(graph)
    if not current_tour:
        return [], float('inf')

    current_cost = _tour_cost(graph, current_tour)
    best_tour = current_tour[:]
    best_cost = current_cost

    # Each entry is a frozenset({i, j}) representing the swapped edge pair.
    tabu_list: dict[frozenset, int] = {}

    for iteration in range(n_iterations):

        # Generate neighborhood: all 2-opt swaps (i, j) with i < j
        n = len(current_tour)
        all_moves = [(i, j) for i in range(n - 1) for j in range(i + 2, n)]

        # Randomly sample if neighborhood is too large
        if len(all_moves) > neighborhood_size:
            all_moves = [
                all_moves[k]
                for k in np.random.choice(len(all_moves), neighborhood_size, replace=False)
            ]

        best_candidate: list[int] | None = None
        best_candidate_cost = float('inf')
        best_move: tuple[int, int] | None = None

        for i, j in all_moves:
            candidate = _apply_2opt_swap(current_tour, i, j)

            if not _is_tour_feasible(graph, candidate):
                continue

            cost = _tour_cost(graph, candidate)
            move_key = frozenset({
                frozenset({current_tour[i], current_tour[i + 1]}),
                frozenset({current_tour[j], current_tour[(j + 1) % n]})
            })

            is_tabu = tabu_list.get(move_key, 0) > iteration

            # Accept move if: not tabu, OR aspiration criterion (beats global best)
            if (not is_tabu or cost < best_cost) and cost < best_candidate_cost:
                best_candidate = candidate
                best_candidate_cost = cost
                best_move = (i, j)

        if best_candidate is None:
            # All neighbors are tabu and none satisfies aspiration → skip
            continue

        # Move to best candidate
        current_tour = best_candidate
        current_cost = best_candidate_cost

        # Register the move as tabu
        if best_move is not None:
            i, j = best_move
            n = len(current_tour)
            move_key = frozenset({
                frozenset({current_tour[i], current_tour[i + 1]}),
                frozenset({current_tour[j], current_tour[(j + 1) % n]})
            })
            tabu_list[move_key] = iteration + tabu_tenure

        if current_cost < best_cost:
            best_cost = current_cost
            best_tour = current_tour[:]

    if not best_tour:
        return [], float('inf')

    # Return a closed tour
    return best_tour + [best_tour[0]], best_cost