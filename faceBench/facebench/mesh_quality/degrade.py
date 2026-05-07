import os
import numpy as np
from typing import List


def degrade_pointcloud(subject_paths: List[str], drop_percents: List[int], output_root: str):
    """
    Generate degraded versions of .txt point cloud files by randomly removing
    a percentage of points from each input file.

    Parameters:
        subject_paths (List[str]): List of paths to .txt files (one per subject)
        drop_percents (List[int]): Percentages of points to remove (e.g., [20, 40, 60])
        output_root (str): Root directory where degraded files will be saved
    """
    os.makedirs(output_root, exist_ok=True)

    for path in subject_paths:
        base_name = os.path.splitext(os.path.basename(path))[0]  # e.g., id0000
        points = np.loadtxt(path)

        subject_dir = os.path.join(output_root, base_name)
        os.makedirs(subject_dir, exist_ok=True)

        for percent in drop_percents:
            n_total = points.shape[0]
            n_drop = int(n_total * percent / 100)

            idx_keep = np.random.choice(n_total, n_total - n_drop, replace=False)
            degraded = points[idx_keep]

            save_path = os.path.join(subject_dir, f"{base_name}_D_{percent}.txt")
            np.savetxt(save_path, degraded, fmt="%.6f")
            print(f"✅ Saved: {save_path} ({percent}% drop)")


"""paths = ["/Users/pampaj/PycharmProjects/3Dfacebenchmark/data/BFMsynth/Rmeshes/BFM/p23470/3DDFAv2-m/id0000.txt",
         "/Users/pampaj/PycharmProjects/3Dfacebenchmark/data/BFMsynth/Rmeshes/BFM/p23470/3DIv2-m/id0003.txt",
         "/Users/pampaj/PycharmProjects/3Dfacebenchmark/data/BFMsynth/Rmeshes/BFM/p23470/Deep3DFace-m/id0004.txt",
         "/Users/pampaj/PycharmProjects/3Dfacebenchmark/data/BFMsynth/Rmeshes/BFM/p23470/INORig-m/id0002.txt"]
drops = [20, 40, 60, 80, 95]
output_dir = "/Users/pampaj/PycharmProjects/3Dfacebenchmark/data/degradations"

degrade_pointcloud(paths, drops, output_dir)
"""
