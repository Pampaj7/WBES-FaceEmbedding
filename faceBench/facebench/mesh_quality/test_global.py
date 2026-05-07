import os
import trimesh
import numpy as np
from matplotlib import pyplot as plt

from global_quality.connected_components import print_topology_report

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
print_topology_report("Ground Truth", points_G)

# Evaluate degraded point clouds
for path in degraded_versions:
    name = os.path.basename(path).replace(".txt", "")
    points = np.loadtxt(path)
    print_topology_report(name, points)
