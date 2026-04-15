import numpy as np
import random
import networkx as nx
import time
import matplotlib.pyplot as plt

def calculate_tour_cost(graph: nx.Graph, tour: list) -> float:
    """
    Calcule le coût total d'une tournée.
    Applique de fortes pénalités si les contraintes ne sont pas respectées.
    """
    cost = 0
    PENALTY = 1000000 # Pénalité dissuasive pour les contraintes violées
    
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)] 
        
        weight = graph.edges[u, v].get('weight', -1)
        
        if weight == -1:
            cost += PENALTY
        else:
            cost += weight
            
    for i, node in enumerate(tour):
        prec_node = graph.nodes[node].get('precedence')
        if prec_node is not None:
            prec_index = tour.index(prec_node)
            if prec_index > i:
                cost += PENALTY
                
    return cost

def generate_random_tour(nodes: list) -> list:
    """Génère une solution initiale aléatoire (permutation des nœuds)."""
    tour = list(nodes)
    depot = tour[0]
    rest_of_tour = tour[1:]
    random.shuffle(rest_of_tour)
    return [depot] + rest_of_tour

def get_neighbors(tour: list) -> list:
    """Génère le voisinage en utilisant un simple échange (Swap) de 2 nœuds."""
    neighbors = []
    for i in range(1, len(tour) - 1):
        for j in range(i + 1, len(tour)):
            neighbor = tour.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            neighbors.append(neighbor)
    return neighbors

def hill_climbing(graph: nx.Graph, initial_tour: list) -> tuple:
    """Exécute une seule ascension de colline."""
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
    """Lance le Hill Climbing plusieurs fois depuis des points de départ différents."""
    nodes = list(graph.nodes())
    global_best_tour = None
    global_best_cost = float('inf')
    
    for _ in range(iterations):
        initial_tour = generate_random_tour(nodes)
        final_tour, final_cost = hill_climbing(graph, initial_tour)
        
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_tour = final_tour
            
    return global_best_tour, global_best_cost


if __name__ == "__main__":
    # Import du générateur qui se trouve dans le même dossier (src/)
    from graph_generator import generate_graph, display_graph

    # 1. Génération de l'environnement (Contexte Boucherie)
    NB_VILLES = 30
    print(f"--- Génération du réseau routier pour {NB_VILLES} villes ---")
    G = generate_graph(NB_VILLES)

    # 2. Résolution du problème
    NB_ITERATIONS = 50
    print(f"--- Lancement de l'optimisation (Multi-Start : {NB_ITERATIONS} itérations) ---")
    start_time = time.time()

    best_tour, best_cost = multi_start_hill_climbing(G, iterations=NB_ITERATIONS)

    execution_time = time.time() - start_time

    villes_noms = {0:"Paris", 1:"Rennes", 2:"Rouen", 3:"Bordeaux", 4:"Toulouse", 5:"Lyon"}

    if best_cost >= 1000000:
        print("\n ÉCHEC : Le réseau routier est trop contraint (Trop d'arêtes interdites ou de précédences).")
        print("L'algorithme a terminé dans un cul-de-sac. Relancez le script.")
    else:
        print(f"\n SUCCÈS : Tournée optimale trouvée en {execution_time:.4f} secondes.")
        chemin_villes = [villes_noms.get(ville, f"Ville {ville}") for ville in best_tour]
        chemin_complet = " -> ".join(chemin_villes) + f" -> {villes_noms.get(best_tour[0])}"
        
        print(f" Itinéraire de livraison recommandé : {chemin_complet}")
        print(f" Coût total estimé (carburant + péages) : {best_cost:.2f} €")

    # 3. Étude Statistique
    print("\n--- Calcul de l'étude expérimentale en cours... ---")
    iterations_tests = [1, 5, 10, 20, 50, 100]
    couts_obtenus = []

    for nb_iter in iterations_tests:
        _, cout = multi_start_hill_climbing(G, iterations=nb_iter)
        couts_obtenus.append(cout)

    # Préparation du graphique de convergence
    plt.figure(figsize=(8, 4))
    plt.plot(iterations_tests, couts_obtenus, marker='o', color='purple', linewidth=2)
    plt.title("Convergence de l'algorithme Multi-Start Hill Climbing")
    plt.xlabel("Nombre de départs aléatoires (Itérations)")
    plt.ylabel("Meilleur Coût Trouvé (€)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    print("\nAffichage des graphiques. (Fermez les fenêtres pour arrêter le programme)")
    # Affichage du graphe réseau et de la courbe de stats
    display_graph(G)
    plt.show()