import facebench as fb
import numpy as np
import os
import json

# Assumi di avere: R (ricostruzione), G (ground-truth), Rlmks, Glmks, errori...

subject_id = 0
method = "Deep3DFace-m"

r_path = f"../../data/BFMsynth/Rmeshes/BFM/p23470/{method}/id{subject_id:04d}.txt"
g_path = f"../../data/BFMsynth/Gmeshes/id{subject_id:04d}.txt"
g_lmk_path = f"../../data/BFMsynth/Gmeshes/id{subject_id:04d}.lmks"

R = np.loadtxt(r_path)
G = np.loadtxt(g_path) / 1e6
Glmks = np.loadtxt(g_lmk_path) / 1e6

with open("../../info/BFM-p23470.json") as f:
    mm = json.load(f)

Rlmks = R[mm["lmk_indices"]]
eye_indices = (36, 45)  # Left/right eye landmarks

#R_aligned, _ = fb.icp_align(R, G, prealign="bbox")
R, _ = fb.landmark_based_align(R, G, Rlmks, Glmks)

#R = fb.nonrigid_icp_align(R, G)

R = fb.landmark_elastic_align(R, G, Glmks, lmk_indices=mm["lmk_indices"])

corr = fb.chamfer_correspondence(R, G)

G = fb.topology_consistency_corrector(R, G, corr, mm)

errors = fb.p2tri_distance(R, G, corr)

fb.generate_results_table(errors)

# Plot
fb.plot_face_heatmap(
    vertices=R,
    errors=errors,
    landmarks=Rlmks,
    landmark_indices=eye_indices,
    title=f"Error Heatmap - {method} (Subject {subject_id})",
    save_path="../docs/assets/error_heatmap.png"
)

fb.plot_face_3d_error(R, errors, elev=45, azim=45)

fb.plot_face_3d_error_interactive(R, errors)

fb.plot_point_clouds_with_error_interactive(R, G, errors)

