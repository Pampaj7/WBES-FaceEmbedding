import os
import trimesh
import numpy as np
from matplotlib import pyplot as plt

from global_quality.connected_components import print_topology_report
from local_topology.aspect_ratio import local_planarity_stats, local_density_stats, knn_distance_stats
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
print_topology_report("Ground Truth", points_G)
print("\n🔍 LOCAL GEOMETRY – Ground Truth")
print(local_planarity_stats(points_G))
print(local_density_stats(points_G))
print(knn_distance_stats(points_G))

# Evaluate degraded point clouds
for path in degraded_versions:
    name = os.path.basename(path).replace(".txt", "")
    points = np.loadtxt(path)
    print(f"\n===== {name} =====")
    print_topology_report(name, points)
    print(f"\n🔍 LOCAL GEOMETRY – {name}")
    print(local_planarity_stats(points))
    print(local_density_stats(points))
    print(knn_distance_stats(points))