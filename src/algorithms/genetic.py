import random
from ..helper import *


def _generate_random_individual(nodes: list[int]) -> list[int]:
    """Generate a random valid individual for the population."""
    individual: list[int] = nodes.copy()
    random.shuffle(individual)
    return individual


def _ordered_crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """Perform ordered crossover (OX) between two parents."""
    size: int = len(parent1)
    start: int
    end: int
    start, end = sorted(random.sample(range(size), 2))
    child: list[int] = [-1] * size
    child[start : end + 1] = parent1[start : end + 1]

    pointer: int = end + 1
    for gene in parent2[end + 1 :] + parent2[: end + 1]:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1

    return child


def _swap_mutation(individual: list[int], mutation_rate: float) -> list[int]:
    """Perform swap mutation on an individual."""
    if random.random() < mutation_rate:
        i: int
        j: int
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual


def _tournament_selection(
    population: list[list[int]], costs: list[float], tournament_size: int
) -> list[int]:
    """Select an individual using tournament selection."""
    participants: list[tuple[list[int], float]] = random.sample(list(zip(population, costs)), tournament_size)
    participants.sort(key=lambda item: item[1])
    return participants[0][0].copy()


def resolve_by_genetic(
    graph: nx.Graph,
    population_size: int = 50,
    generations: int = 200,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.2,
    tournament_size: int = 3,
    elite_size: int = 2,
) -> tuple[list[int], float]:
    """Resolve the graph using a simple genetic algorithm."""
    nodes: list[int] = list(graph.nodes())
    node_count: int = len(nodes)

    if node_count == 0:
        return [], 0.0

    population: list[list[int]] = [_generate_random_individual(nodes) for _ in range(population_size)]
    best_tour: list[int] = []
    best_cost: float = float("inf")

    for _ in range(generations):
        costs: list[float] = [
            calculate_tour_cost_with_penalty(graph, individual)
            for individual in population
        ]

        for individual, cost in zip(population, costs):
            if cost < best_cost:
                best_cost = cost
                best_tour = individual.copy()

        new_population: list[list[int]] = []
        sorted_population: list[list[int]] = [
            individual
            for _, individual in sorted(
                zip(costs, population), key=lambda item: item[0]
            )
        ]
        new_population.extend(sorted_population[:elite_size])

        while len(new_population) < population_size:
            parent1: list[int] = _tournament_selection(population, costs, tournament_size)
            parent2: list[int] = _tournament_selection(population, costs, tournament_size)

            if random.random() < crossover_rate:
                child: list[int] = _ordered_crossover(parent1, parent2)
            else:
                child = parent1.copy()

            child = _swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    if not best_tour:
        return [], float("inf")

    return best_tour + [best_tour[0]], best_cost
