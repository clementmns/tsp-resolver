import random
import networkx as nx

def calculate_tour_cost(graph: nx.Graph, tour: list) -> float:
    """
    Calculates the total cost of a tour.
    Applies heavy penalties if constraints are not respected.
    """
    cost = 0
    PENALTY = 1000000 # Strong penalty for violated constraints
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)] 
        
        if graph.has_edge(u, v):
            weight = graph.edges[u, v].get('weight', -1)
        else:
            weight = -1
        
        if weight == -1:
            cost += PENALTY
        else:
            cost += weight
            
    for i, node in enumerate(tour):
        prec_node = graph.nodes[node].get('precedence')
        if prec_node is not None:
            if prec_node not in tour:
                cost += PENALTY
            else:
                prec_index = tour.index(prec_node)
                if prec_index > i:
                    cost += PENALTY
                
    return cost

def generate_random_tour(nodes: list) -> list:
    """Generates an initial random solution (permutation of nodes)."""
    tour = list(nodes)
    depot = tour[0]
    rest_of_tour = tour[1:]
    random.shuffle(rest_of_tour)
    return [depot] + rest_of_tour

def get_neighbors(tour: list) -> list:
    """Generates the neighborhood using a simple 2-node swap."""
    neighbors = []
    for i in range(1, len(tour) - 1):
        for j in range(i + 1, len(tour)):
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def hill_climbing(graph: nx.Graph, initial_tour: list) -> tuple:
    """Executes a single hill climbing pass."""
    current_tour = initial_tour
    current_cost = calculate_tour_cost(graph, current_tour)
    
    while True:
        neighbors = get_neighbors(current_tour)
        best_neighbor = None
        best_neighbor_cost = current_cost
        
        for neighbor in neighbors:
            cost = calculate_tour_cost(graph, neighbor)
            if cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = cost
                
        if best_neighbor is not None:
            current_tour = best_neighbor
            current_cost = best_neighbor_cost
        else:
            break
            
    return current_tour, current_cost

def multi_start_hill_climbing(graph: nx.Graph, iterations: int = 50) -> tuple:
    """Runs the Hill Climbing algorithm multiple times from different starting points."""
    nodes = list(graph.nodes())
    global_best_tour = None
    global_best_cost = float('inf')
    
    for _ in range(iterations):
        initial_tour = generate_random_tour(nodes)
        final_tour, final_cost = hill_climbing(graph, initial_tour)
        
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_tour = final_tour
    
    if global_best_tour is None:
        return [], float('inf')
            
    return global_best_tour, global_best_cost