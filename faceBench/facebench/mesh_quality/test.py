# facebench/mesh_quality/plot_scoring_summary.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scoring import score_mesh
import open3d as o3d

# Paths
gt_path = "../../data/BFMsynth/Gmeshes/id0000.txt"
degraded_versions = [
    "../../data/degradations/id0000/id0000_D_20.txt",
    "../../data/degradations/id0000/id0000_D_40.txt",
    "../../data/degradations/id0000/id0000_D_60.txt",
    "../../data/degradations/id0000/id0000_D_80.txt",
    "../../data/degradations/id0000/id0000_D_95.txt"
]

# Score collection
data = []
all_paths = [("Ground Truth", gt_path)] + [
    (os.path.basename(p).replace(".txt", ""), p) for p in degraded_versions
]

for name, path in all_paths:
    points = np.loadtxt(path)
    if name == "Ground Truth":
        points = points / 1e6

    result = score_mesh(points)
    flat_metrics = {
        "name": name,
        "score": result["score"],
        **result["topology"],
        **result["local_geometry"],
        **result["shape"]
    }
    data.append(flat_metrics)

# DataFrame
scores_df = pd.DataFrame(data)
print(scores_df.to_markdown(index=False))

# Plot
plt.figure(figsize=(10, 6))
plt.bar(scores_df["name"], scores_df["score"], color="skyblue")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Mesh Quality Score by Degradation Level")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


def visualize_pointcloud(path: str, downsample_voxel: float = 0.001):
    points = np.loadtxt(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if downsample_voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel)
    o3d.visualization.draw_geometries([pcd], window_name=os.path.basename(path))


for path in degraded_versions:
    visualize_pointcloud(path)
