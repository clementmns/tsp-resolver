import math
import time
import numpy as np

from ..helper import *

def _tour_cost(graph: nx.Graph, tour: list[int]) -> float:
    total: float = 0.0
    for u, v in zip(tour, tour[1:]):
        weight: float = graph.edges[u, v]["weight"]
        if weight == -1:
            return float("inf")
        total += weight
    return total


def _is_feasible(graph: nx.Graph, tour: list[int]) -> bool:
    if len(tour) < 2 or tour[0] != tour[-1]:
        return False

    open_tour: list[int] = tour[:-1]
    if len(set(open_tour)) != len(open_tour):
        return False

    return is_tour_feasible(graph, open_tour)


def _two_opt_swap(tour: list[int], i: int, j: int) -> list[int]:
    return tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]


def _initial_tour(graph: nx.Graph, rng: np.random.Generator) -> list[int]:
    n: int = graph.number_of_nodes()
    visited: set[int] = {0}
    tour: list[int] = [0]

    # Greedy nearest neighbor from node 0
    while len(tour) < n:
        current: int = tour[-1]
        candidates: list[int] = valid_next_nodes(graph, current, visited)
        if not candidates:
            break
        next_node = min(candidates, key=lambda v: graph.edges[current, v]["weight"])
        tour.append(next_node)
        visited.add(next_node)

    tour.append(0)

    if _is_feasible(graph, tour):
        return tour

    # Fallback: random shuffles
    for _ in range(50):
        order: list[int] = list(range(1, n))
        rng.shuffle(order)
        candidate: list[int] = [0] + order + [0]
        if _is_feasible(graph, candidate):
            return candidate

    return [0] + list(range(1, n)) + [0]


def _initial_temperature(graph: nx.Graph, tour: list[int], rng: np.random.Generator) -> float:
    # Calibrate T0 so that ~80% of degradations are accepted initially
    n: int = len(tour)
    base_cost: float = _tour_cost(graph, tour)
    degradations: list[float] = []

    for _ in range(n * 20):
        i: int = int(rng.integers(1, n - 2))
        j: int = int(rng.integers(i + 1, n - 1))
        neighbor: list[int] = _two_opt_swap(tour, i, j)
        if not _is_feasible(graph, neighbor):
            continue
        delta: float = _tour_cost(graph, neighbor) - base_cost
        if delta > 0:
            degradations.append(delta)
        if len(degradations) >= 100:
            break

    if not degradations:
        return 1.0
    return -float(np.mean(degradations)) / math.log(0.8)


def resolve_by_recuit_simule(
    graph: nx.Graph,
    max_iterations: int = 10000,
    max_time_seconds: float | None = None,
    seed: int | None = None,
    alpha: float = 0.995,
) -> tuple[list[int], float]:
    rng: np.random.Generator = np.random.default_rng(seed)
    start: float = time.perf_counter()

    current: list[int] = _initial_tour(graph, rng)
    current_cost: float = _tour_cost(graph, current)
    best: list[int] = list(current)
    best_cost: float = current_cost

    t: float = _initial_temperature(graph, current, rng)
    n: int = len(current)
    iteration: int = 0

    while True:
        if max_time_seconds is not None and (time.perf_counter() - start) >= max_time_seconds:
            break
        if iteration >= max_iterations:
            break

        # Find a feasible 2-opt neighbor
        neighbor: list[int] | None = None
        for _ in range(100):
            i: int = int(rng.integers(1, n - 2))
            j: int = int(rng.integers(i + 1, n - 1))
            candidate: list[int] = _two_opt_swap(current, i, j)
            if _is_feasible(graph, candidate):
                neighbor = candidate
                break

        if neighbor is None:
            break

        delta: float = _tour_cost(graph, neighbor) - current_cost

        # Metropolis criterion
        if delta <= 0 or rng.random() < math.exp(-delta / t):
            current = neighbor
            current_cost += delta
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost

        # Geometric cooling
        t *= alpha
        iteration += 1

    return best, best_cost


def resolve_by_ms_recuit_simule(
    graph: nx.Graph,
    n_restarts: int = 5,
    max_iterations_per_restart: int = 2000,
    max_time_seconds: float | None = None,
    seed: int | None = None,
    alpha: float = 0.995,
) -> tuple[list[int], float]:
    rng: np.random.Generator = np.random.default_rng(seed)
    start: float = time.perf_counter()
    best: list[int] = []
    best_cost: float = float("inf")

    for _ in range(n_restarts):
        # Check global time budget
        if max_time_seconds is not None:
            elapsed: float = time.perf_counter() - start
            if elapsed >= max_time_seconds:
                break
            remaining: float | None = max_time_seconds - elapsed
        else:
            remaining = None

        tour: list[int]
        cost: float
        tour, cost = resolve_by_recuit_simule(
            graph,
            max_iterations=max_iterations_per_restart,
            max_time_seconds=remaining,
            seed=int(rng.integers(0, 2**31)),
            alpha=alpha,
        )

        if cost < best_cost:
            best = tour
            best_cost = cost

    return best, best_cost
