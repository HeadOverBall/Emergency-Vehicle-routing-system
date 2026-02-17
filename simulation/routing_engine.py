import heapq
import math


def a_star_route(G, start, goal):
    """
    A* implementation:
    - No infinite loops
    - Handles infinite / huge weights
    - Guarantees termination
    - Skips unusable edges
    """

    if start not in G or goal not in G:
        return None, float("inf")

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {node: float("inf") for node in G.nodes()}
    g_score[start] = 0

    f_score = {node: float("inf") for node in G.nodes()}
    f_score[start] = heuristic(G, start, goal)

    visited = set()
    max_iterations = 10000
    iterations = 0

    while open_set:
        iterations += 1
        if iterations > max_iterations:
            print("⚠ WARNING: A* reached safety iteration limit. Terminating.")
            break

        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current), g_score[current]

        if current in visited:
            continue
        visited.add(current)

        for neighbor in G.neighbors(current):
            # Use dynamic_weight if present, else fallback to distance
            weight = G[current][neighbor].get(
                "dynamic_weight",
                G[current][neighbor].get("distance", 1)
            )

            # Skip unusable edges
            if weight == float("inf") or weight > 1e12:
                continue

            tentative_g = g_score[current] + weight

            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(G, neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # No path found
    return None, float("inf")


def heuristic(G, a, b):
    (x1, y1) = G.nodes[a]["pos"]
    (x2, y2) = G.nodes[b]["pos"]
    return math.dist((x1, y1), (x2, y2))


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]
