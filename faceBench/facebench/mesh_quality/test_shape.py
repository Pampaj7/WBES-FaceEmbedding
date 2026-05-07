# facebench/mesh_quality/test_shape.py
import os
import numpy as np
from shape_quality.basic_shape_stats import (
    bounding_box_ratios,
    pca_shape_energy,
    shape_spread_stats
)

# Ground truth path
gt_path = "../../data/BFMsynth/Gmeshes/id0000.txt"

# Degraded versions
degraded_versions = [
    "../../data/degradations/id0000/id0000_D_20.txt",
    "../../data/degradations/id0000/id0000_D_40.txt",
    "../../data/degradations/id0000/id0000_D_60.txt"
]

# Load and evaluate ground truth
points_G = np.loadtxt(gt_path) / 1e6
print(f"\n===== GROUND TRUTH =====")
print("Bounding Box:", bounding_box_ratios(points_G))
print("PCA Shape Energy:", pca_shape_energy(points_G))
print("Spread Stats:", shape_spread_stats(points_G))

# Evaluate degraded point clouds
for path in degraded_versions:
    name = os.path.basename(path).replace(".txt", "")
    points = np.loadtxt(path)
    print(f"\n===== {name} =====")
    print("Bounding Box:", bounding_box_ratios(points))
    print("PCA Shape Energy:", pca_shape_energy(points))
    print("Spread Stats:", shape_spread_stats(points))
