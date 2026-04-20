import networkx as nx

PENALTY_COST = 1_000_000


def node_can_be_visited(graph: nx.Graph, node: int, visited: set[int]) -> bool:
    """Check if a node can be visited based on its precedence constraint."""
    precedence = graph.nodes[node].get("precedence")
    return precedence is None or precedence in visited


def valid_next_nodes(graph: nx.Graph, current: int, visited: set[int]) -> list[int]:
    """Get a list of valid next nodes that can be visited from the current node, based on the graph's edges, weights, and precedence constraints."""
    return [
        node
        for node in graph.nodes()
        if node not in visited
        and node_can_be_visited(graph, node, visited)
        and graph.edges[current, node]["weight"] != -1
    ]


def calculate_tour_cost_with_penalty(
    graph: nx.Graph,
    tour: list[int],
    penalty_cost: float = PENALTY_COST,
) -> float:
    """Calculate a closed tour cost, penalizing forbidden edges and precedence violations."""
    cost = 0.0

    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        if graph.has_edge(u, v):
            weight = graph.edges[u, v].get("weight", -1)
        else:
            weight = -1

        if weight == -1:
            cost += penalty_cost
        else:
            cost += weight

    for index, node in enumerate(tour):
        precedence = graph.nodes[node].get("precedence")
        if precedence is not None:
            if precedence not in tour:
                cost += penalty_cost
                continue

            precedence_index = tour.index(precedence)
            if precedence_index > index:
                cost += penalty_cost

    return cost


def build_greedy_tour(graph: nx.Graph) -> list[int]:
    """Build an initial tour using a nearest-neighbor heuristic."""
    nodes = list(graph.nodes())
    start_candidates = [n for n in nodes if graph.nodes[n].get("precedence") is None]
    if not start_candidates:
        start_candidates = nodes

    best_tour: list[int] = []
    best_cost = float("inf")

    for start in start_candidates:
        tour = [start]
        visited = {start}

        while len(tour) < len(nodes):
            current = tour[-1]
            valid = [
                n
                for n in nodes
                if n not in visited
                and node_can_be_visited(graph, n, visited)
                and graph.edges[current, n]["weight"] != -1
            ]
            if not valid:
                break
            nearest = min(valid, key=lambda n: graph.edges[current, n]["weight"])
            tour.append(nearest)
            visited.add(nearest)

        if len(tour) != len(nodes):
            continue
        if graph.edges[tour[-1], tour[0]]["weight"] == -1:
            continue

        cost = closed_tour_cost(graph, tour)

        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_tour


def closed_tour_cost(graph: nx.Graph, tour: list[int]) -> float:
    """Compute the total cost of a closed tour (last node to first node included)."""
    cost = sum(graph.edges[u, v]["weight"] for u, v in zip(tour, tour[1:]))
    cost += graph.edges[tour[-1], tour[0]]["weight"]
    return cost


def is_tour_feasible(graph: nx.Graph, tour: list[int]) -> bool:
    """Check that every edge in a closed tour exists and respects precedences."""
    visited: set[int] = set()
    for idx, node in enumerate(tour):
        if not node_can_be_visited(graph, node, visited):
            return False
        visited.add(node)

        next_node = tour[(idx + 1) % len(tour)]
        if not graph.has_edge(node, next_node):
            return False
        if graph.edges[node, next_node]["weight"] == -1:
            return False

    return True
