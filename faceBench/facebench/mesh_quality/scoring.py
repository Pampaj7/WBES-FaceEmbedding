# facebench/scoring.py
import numpy as np
from global_quality.connected_components import (
    suggest_adaptive_radius,
    count_connected_components,
    component_size_ratio,
    component_centroid_dispersion
)
from local_topology.aspect_ratio import (
    local_planarity_stats,
    local_density_stats,
    knn_distance_stats
)
from shape_quality.basic_shape_stats import (
    bounding_box_ratios,
    pca_shape_energy,
    shape_spread_stats
)


def score_mesh(points: np.ndarray) -> dict:
    radius = suggest_adaptive_radius(points)

    # Topology
    n_components = count_connected_components(points, radius)
    size_ratio = component_size_ratio(points, radius)
    dispersion = component_centroid_dispersion(points, radius)

    # Local geometry
    planarity = local_planarity_stats(points)
    density = local_density_stats(points, radius)
    knn = knn_distance_stats(points)

    # Shape
    bbox = bounding_box_ratios(points)
    pca = pca_shape_energy(points)
    spread = shape_spread_stats(points)

    # Simple heuristic scoring system TODO need to fix based on real datas
    quality_flags = []
    if n_components > 10:
        quality_flags.append("⚠️ High fragmentation")
    if size_ratio < 0.95:
        quality_flags.append("⚠️ Low main structure ratio")
    if density["mean_density"] < 5:
        quality_flags.append("⚠️ Very sparse point cloud")
    if knn["mean_knn_distance"] > 0.05:
        quality_flags.append("⚠️ Low resolution spacing")
    if planarity["mean_planarity"] < 0.35:
        quality_flags.append("⚠️ Poor local planarity")

    # Final score heuristic [0–1], 1 = perfect
    penalties = 0
    penalties += min(n_components / 100, 0.5)  # limit fragmentation penalty
    penalties += min((1.0 - size_ratio), 0.3)  # moderate size drop impact
    penalties += max((5 - density["mean_density"]) / 10, 0)  # scaled
    penalties += min(knn["mean_knn_distance"] * 5, 0.5)  # gentler
    if planarity["mean_planarity"] < 0.35:
        penalties += (0.35 - planarity["mean_planarity"]) * 2.5

    score = max(0.0, 1.0 - penalties)

    return {
        "score": round(score, 3),
        "warnings": quality_flags,
        "topology": {
            "n_components": n_components,
            "size_ratio": round(size_ratio, 4),
            "centroid_dispersion": round(dispersion, 4)
        },
        "local_geometry": {**planarity, **density, **knn},
        "shape": {**bbox, **pca, **spread}
    }
