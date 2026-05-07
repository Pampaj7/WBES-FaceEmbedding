import numpy as np
from scipy.spatial import KDTree, distance
from facebench.mesh_quality.utils.adaptive_radius import suggest_radius, suggest_density_based_radius


def suggest_adaptive_radius(points: np.ndarray, factor: float = 0.02, k: int = 8) -> float:
    r_bbox = suggest_radius(points, factor=factor)
    r_density = suggest_density_based_radius(points, k=k)
    return (r_bbox + r_density) / 2


def count_connected_components(points: np.ndarray, radius: float) -> int:
    tree = KDTree(points)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    n_components = 0

    for i in range(n_points):
        if not visited[i]:
            n_components += 1
            queue = [i]
            while queue:
                current = queue.pop()
                if not visited[current]:
                    visited[current] = True
                    neighbors = tree.query_ball_point(points[current], r=radius)
                    queue.extend(neighbors)

    return n_components


def component_surface_stats(points: np.ndarray, radius: float) -> dict:
    tree = KDTree(points)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    components = []

    for i in range(n_points):
        if not visited[i]:
            component = []
            queue = [i]
            while queue:
                current = queue.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(current)
                    neighbors = tree.query_ball_point(points[current], r=radius)
                    queue.extend(neighbors)
            components.append(np.array(component))

    sizes = [len(c) for c in components]
    return {
        "num_components": len(sizes),
        "max_size": max(sizes),
        "min_size": min(sizes),
        "mean_size": np.mean(sizes),
        "std_size": np.std(sizes)
    }


def component_centroid_dispersion(points: np.ndarray, radius: float) -> float:
    tree = KDTree(points)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    centroids = []

    for i in range(n_points):
        if not visited[i]:
            component = []
            queue = [i]
            while queue:
                current = queue.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(current)
                    neighbors = tree.query_ball_point(points[current], r=radius)
                    queue.extend(neighbors)
            comp_points = points[component]
            centroids.append(np.mean(comp_points, axis=0))

    if len(centroids) <= 1:
        return 0.0

    centroids = np.array(centroids)
    return np.std(distance.pdist(centroids))


def component_size_ratio(points: np.ndarray, radius: float) -> float:
    tree = KDTree(points)
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    sizes = []

    for i in range(n_points):
        if not visited[i]:
            component = []
            queue = [i]
            while queue:
                current = queue.pop()
                if not visited[current]:
                    visited[current] = True
                    component.append(current)
                    neighbors = tree.query_ball_point(points[current], r=radius)
                    queue.extend(neighbors)
            sizes.append(len(component))

    return max(sizes) / sum(sizes) if sizes else 0.0


def print_topology_report(name: str, points: np.ndarray) -> dict:
    radius = suggest_adaptive_radius(points)
    n_components = count_connected_components(points, radius=radius)
    size_ratio = component_size_ratio(points, radius=radius)
    surface_stats = component_surface_stats(points, radius=radius)

    return {
        "name": name,
        "radius": radius,
        "connected_components": n_components,
        "size_ratio": size_ratio,
        "num_components": surface_stats["num_components"],
        "max_size": surface_stats["max_size"],
        "min_size": surface_stats["min_size"],
        "mean_size": surface_stats["mean_size"],
        "std_size": surface_stats["std_size"]
    }
