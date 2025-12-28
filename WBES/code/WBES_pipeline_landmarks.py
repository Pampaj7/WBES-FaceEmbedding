import os, math, itertools
from glob import glob
from collections import defaultdict
from random import sample, seed
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import WBES_helper as wh

# === CONFIG ===
STUDY_ROOT = ""
METHOD_DIRS = {
    "3DDFAv2":     "3DDFAV2_23470_neutral",
    "3DDFAv3":     "3DDFAV3_neutral",
    "Deep3DFace":  "Deep3DFace_23470_neutral",
    "SynergyNet":  "SynergyNet_neutral",
    "INORig":      "INORig_23470_neutral",
    "3DI":         "3DI_neutral",
    "Smirk":       "Smirk_cropped_neutral",
    "FaceVerse":   "Faceverse_cropped_neutral"
}
GROUP_SIZES_BY_METHOD = {
    "3DDFAv2":     [1, 3, 5, 10, 15],
    "3DDFAv3":     [1, 3, 5, 10, 15],
    "Deep3DFace":  [1, 3, 5, 10, 15],
    "SynergyNet":  [1, 3, 5, 10, 15],
    "INORig":      [1, 2, 3],
    "3DI":         [1],
    "Smirk":       [1, 3, 5, 10, 15],
    "FaceVerse":   [1, 3, 5, 10, 15]
}
F_SHOW = [10] + [1, 3, 5, 10, 15]
N_REPS = 3
OUT_ROOT = "results_landmarks"
BANNED_SUBJECTS = {"id0099"} #nan
COLOR_W = "#f6a600"
COLOR_B = "#0080c9"
VAR_EPS = 1e-6
BFM_METHODS = {"3DDFAv2", "3DDFAv3", "Deep3DFace", "SynergyNet", "INORig", "3DI"}
LANDMARK_JSON = "/Users/pampaj/PycharmProjects/data/utils/BFM-p23470.json"
# === PIPELINE ===
F_SCALE_BY_METHOD = {
    "INORig": 1,
    "3DI": 1,
}
landmark_indices = wh.load_landmark_indices(LANDMARK_JSON)

density_data = {}

for pretty, subdir in tqdm(METHOD_DIRS.items(), desc="Methods"):
    group_sizes = GROUP_SIZES_BY_METHOD.get(pretty, [])
    scale = F_SCALE_BY_METHOD.get(pretty, 1)
    print(f"\n=== {pretty} ===")

    out_dir = os.path.join(OUT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{subdir}-wbes_landmark_inter_F.csv")
    """
    if os.path.exists(csv_path):
        print(f"[✓] Skipping {pretty} (CSV already exists)")
        continue
    """
    meshes_per_subj = wh.load_landmarks_npy(os.path.join(STUDY_ROOT, subdir))
    print(f"Loaded {len(meshes_per_subj)} subjects in {subdir}")

    reps_byF = {}
    for F in group_sizes:
        adjusted_F = F * scale
        reps_byF[F] = {}

        for sid, m in meshes_per_subj.items():
            if sid in BANNED_SUBJECTS:
                continue
            reps = wh.reps_disjoint(m, adjusted_F, N_REPS)
            if len(reps) == N_REPS:
                reps_byF[F][sid] = reps


    geom_errors_byF = {}

    if all(len(reps_byF[F]) == 0 for F in group_sizes):
        print(f"[!] WARNING: No valid subjects found for {pretty}. Skipping...")
        continue

    rows_inter, rows_intra = [], []
    density_data[pretty] = {}

    for F in group_sizes:
        within, between = [], []
        subs = list(reps_byF[F])

        for sid in subs:
            for r1, r2 in itertools.combinations(reps_byF[F][sid], 2):
                dist = np.linalg.norm(r1 - r2)
                if np.isnan(dist):
                    print(f"[DEBUG] NaN in within: {sid}")

        
        for sid in subs:
            for r1, r2 in itertools.combinations(reps_byF[F][sid], 2):
                within.append(np.linalg.norm(r1 - r2))

        for i, sa in enumerate(subs):
            for sb in subs[i+1:]:
                for r1 in reps_byF[F][sa]:
                    for r2 in reps_byF[F][sb]:
                        between.append(np.linalg.norm(r1 - r2))

        if len(within) and len(between):
            within = np.array(within)
            between = np.array(between)

            if np.any(np.isnan(within)) or np.any(np.isnan(between)):
                print(f"[!] Skipping F={F} for {pretty} — NaNs in distances")
                wbse_val = np.nan
                w_mean = np.nan
                b_mean = np.nan
            else:
                try:
                    wbse_val = wh.cohens_d(within, between)
                    w_mean = within.mean()
                    b_mean = between.mean()
                except Exception as e:
                    print(f"[!] Failed WBES at F={F} for {pretty}: {e}")
                    wbse_val = np.nan
                    w_mean = np.nan
                    b_mean = np.nan

            rows_inter.append(dict(
                F=F,
                wbse=wbse_val,
                within_mean=w_mean,
                between_mean=b_mean,
                n_subjects=len(subs)
            ))


            if F in F_SHOW:
                density_data[pretty][F] = (within, between, wh.cohens_d(within, between))

            out_npy_prefix = os.path.join(out_dir, f"wbse_landmark_inter_F{F}")
            np.save(out_npy_prefix + "_within.npy", within)
            np.save(out_npy_prefix + "_between.npy", between)

        if pretty in BFM_METHODS:
            geom_errors = []

            for sid, reps in reps_byF[F].items():
                gt_path = os.path.join(
                    "/Users/pampaj/PycharmProjects/data/GT/GT_BFM", f"{sid}.id.txt")
                if not os.path.exists(gt_path):
                    continue
                try:
                    gt = np.loadtxt(gt_path) / 1e6
                    gt_lmks = wh.extract_landmarks_from_mesh(gt, landmark_indices)
                except:
                    continue

                mean_rep = np.mean(np.stack(reps), axis=0)
                rep_lmks = wh.extract_landmarks_from_mesh(mean_rep, landmark_indices)

                try:
                    aligned_rep, _ = wh.landmark_based_align(mean_rep, gt, rep_lmks, gt_lmks)
                    aligned_rep_lmks = wh.extract_landmarks_from_mesh(aligned_rep, landmark_indices)
                    err = wh.vertexwise_l2_error(aligned_rep_lmks, gt_lmks)
                    geom_errors.append(err)
                except:
                    continue

            geom_errors_byF[F] = geom_errors




    for F1, F2 in itertools.combinations(group_sizes, 2):
        subs = set(reps_byF[F1]) & set(reps_byF[F2])
        within, between = [], []

        for sid in subs:
            d = np.linalg.norm(np.mean(reps_byF[F1][sid], 0) -
                               np.mean(reps_byF[F2][sid], 0))
            within.append(d)

        for sa, sb in itertools.combinations(subs, 2):
            d = np.linalg.norm(np.mean(reps_byF[F1][sa], 0) -
                               np.mean(reps_byF[F2][sb], 0))
            between.append(d)

        if within and between:
            rows_intra.append(dict(
                F1=F1, F2=F2,
                wbse=wh.cohens_d(np.array(within), np.array(between)),
                within_mean=np.mean(within),
                between_mean=np.mean(between),
                n_subjects=len(subs)
            ))

    pd.DataFrame(rows_inter).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_landmark_inter_F.csv"), index=False)
    pd.DataFrame(rows_intra).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_landmark_intra_F1vsF2.csv"), index=False)
    pd.concat([
        pd.DataFrame(rows_inter).assign(type="inter", compare="F"),
        pd.DataFrame(rows_intra ).assign(type="intra", compare="F1_vs_F2")
    ]).to_csv(os.path.join(out_dir,
        f"{subdir}-wbes_landmark_summary.csv"), index=False)

    if pretty in BFM_METHODS:
        with open(os.path.join(out_dir, f"{subdir}-geom_error_vs_F_landmark.csv"), "w") as f:
            f.write("F,mean_landmark_error,n_subjects\n")
            for F, errors in geom_errors_byF.items():
                if len(errors):
                    f.write(f"{F},{np.mean(errors)},{len(errors)}\n")
                else:
                    f.write(f"{F},nan,0\n")


    print(f"CSV saved to {out_dir}")
