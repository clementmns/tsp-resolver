# tabu search simple
import networkx as nx

def _node_can_be_visited(graph: nx.Graph, node: int, visited: set[int]) -> bool:
    """Check if a node can be visited based on its precedence constraint."""
    precedence = graph.nodes[node].get('precedence')
    return precedence is None or precedence in visited


def _build_greedy_tour(graph: nx.Graph) -> list[int]:
    """Build an initial tour using a greedy nearest-neighbor heuristic"""
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


def _apply_swap(tour: list[int], i: int, j: int) -> list[int]:
    """Return a new tour with positions i and j swapped."""
    new_tour = tour[:]
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


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

def resolve_by_tabu_search(graph: nx.Graph, n_iterations: int = 500, tabu_tenure: int | None = None) -> tuple[list[int], float]:
    """Resolve the TSP using a simple Tabu Search with node-swap neighborhood."""
    node_count = graph.number_of_nodes()
    if node_count < 2:
        nodes = list(graph.nodes())
        return nodes + nodes[:1], 0.0

    # check params
    if tabu_tenure is None:
        tabu_tenure = max(7, node_count // 5)

    current_tour = _build_greedy_tour(graph)
    if not current_tour:
        return [], float('inf')

    current_cost = _tour_cost(graph, current_tour)
    best_tour = current_tour[:]
    best_cost = current_cost

    # tabu_list[frozenset({city_a, city_b})] = iteration until which the swap is forbidden
    tabu_list: dict[frozenset, int] = {}

    for iteration in range(n_iterations):

        best_candidate: list[int] | None = None
        best_candidate_cost = float('inf')
        best_move: frozenset | None = None

        n = len(current_tour)

        # Explore all possible swaps (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                candidate = _apply_swap(current_tour, i, j)

                if not _is_tour_feasible(graph, candidate):
                    continue

                cost = _tour_cost(graph, candidate)
                move_key = frozenset({current_tour[i], current_tour[j]})

                is_tabu = tabu_list.get(move_key, 0) > iteration

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
        return [], float('inf')

    # Return a closed tour
    return best_tour + [best_tour[0]], best_cost