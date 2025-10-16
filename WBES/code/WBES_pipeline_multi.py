"""
wbes_pipeline.py
================
• Compute WBES (inter & intra) for each reconstruction method
• Save CSVs under results/<method>/
• Plot a 2×3 density grid (FaceVerse vs Smirk, F = 1 | 15 | 16)
  with variance‑zero spikes handled gracefully
"""

import os
import math
import itertools
from glob import glob
from collections import defaultdict
from random import sample, seed
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Literal
from utils import WBES_helper as wh

# ------------------------------------------------------------------
# folder containing cropped_* dirs
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

F_SHOW = [10, 1, 3, 5, 15]
N_REPS = 3
OUT_ROOT = "results"
BANNED_SUBJECTS = {"id0099"}
COLOR_W = "#f6a600"
COLOR_B = "#0080c9"
VAR_EPS = 1e-6
BFM_METHODS = {"3DDFAv2", "3DDFAv3",
               "Deep3DFace", "SynergyNet", "INORig", "3DI"}
LANDMARK_JSON = "/Users/pampaj/PycharmProjects/data/utils/BFM-p23470.json"

# ---------- main -------------------------------------------------------------
density_data = {}          # {method: {F: (within_arr, between_arr, wbes)}}
F_SCALE_BY_METHOD = {
    "INORig": 1,   # ogni mesh è già una media su 5 frame, quindi la tratti come unità
    "3DI": 1,      # idem, ogni mesh è F=10 → 1 mesh == 1 unità logica
    # default per gli altri è 1: 1 mesh = 1 frame
}

landmark_indices = wh.load_landmark_indices(LANDMARK_JSON)

for pretty, subdir in tqdm(METHOD_DIRS.items(), desc="Methods", total=len(METHOD_DIRS)):
    group_sizes = GROUP_SIZES_BY_METHOD.get(pretty, [])
    scale = F_SCALE_BY_METHOD.get(pretty, 1)

    print(f"\n=== {pretty} ===")
    out_dir = os.path.join(OUT_ROOT, subdir)
    os.makedirs(out_dir, exist_ok=True)

    meshes_per_subj = wh.load_meshes_txt(os.path.join(STUDY_ROOT, subdir))
    print(f"Loaded {len(meshes_per_subj)} subjects in {subdir}")

    reps_byF = {}
    geom_errors_byF = {}

    for F in group_sizes:
        reps_byF[F] = {}
        geom_errors = []

        print(f"  [F={F}] Processing subjects...")

        adjusted_F = F * scale

        pbar = tqdm(meshes_per_subj.items(), total=len(meshes_per_subj),
                    desc=f"    {pretty} F={F}", ncols=100, leave=False)
        for sid, m in pbar:
            if sid in BANNED_SUBJECTS:
                continue
            reps = wh.reps_disjoint(m, adjusted_F, N_REPS)
            if len(reps) == N_REPS:
                reps_byF[F][sid] = reps

        if pretty in BFM_METHODS:
            for sid, reps in reps_byF[F].items():
                gt_path = os.path.join(
                    "/Users/pampaj/PycharmProjects/data/GT/GT_BFM", f"{sid}.id.txt")
                try:
                    gt = np.loadtxt(gt_path) / 1e6
                except:
                    continue

                mean_rep = np.mean(np.stack(reps), axis=0)
                if mean_rep.shape != gt.shape:
                    continue

                try:
                    gt_lmks = wh.extract_landmarks_from_mesh(
                        gt, landmark_indices)
                    rep_lmks = wh.extract_landmarks_from_mesh(
                        mean_rep, landmark_indices)
                except IndexError:
                    continue

                aligned_rep, _ = wh.landmark_based_align(
                    mean_rep, gt, rep_lmks, gt_lmks)
                err = wh.vertexwise_l2_error(aligned_rep, gt)
                geom_errors.append((sid, err))

            geom_errors_byF[F] = geom_errors

    if pretty in BFM_METHODS:
        csv_path = os.path.join(out_dir, f"{subdir}-geom_error_per_subject.csv")
        with open(csv_path, "w") as f:
            f.write("method,F,subject,error\n")
            for F, errors in geom_errors_byF.items():
                for sid, err in errors:
                    f.write(f"{pretty},{F},{sid},{err}\n")
        print(f"✅ Saved geom_error_vs_F.csv for {pretty}")

    if all(len(reps_byF[F]) == 0 for F in group_sizes):
        print(
            f"[!] WARNING: No valid subjects found for {pretty}. Skipping...")
        continue
    rows_inter, rows_intra = [], []
    density_data[pretty] = {}
    per_subject_within = []  
    per_subject_wbes = []  
    
    for F in group_sizes:
        within, between = [], []
        subs = list(reps_byF[F])
        # 🔧 Calcolo WBES per soggetto
        subj_means = {
            sid: np.mean(np.stack(reps_byF[F][sid]), axis=0)
            for sid in subs
        }

        for sid in subs:
            this = subj_means[sid]
            rest = [subj_means[s] for s in subs if s != sid]
            if not rest:
                continue
            within_dists = [np.linalg.norm(
                r1 - r2) for r1, r2 in itertools.combinations(reps_byF[F][sid], 2)]
            between_dists = [np.linalg.norm(subj_means[sid] - subj_means[other])
                            for other in subs if other != sid]

            if within_dists and between_dists:
                wbse_s = wh.cohens_d(np.array(within_dists), np.array(between_dists))
                per_subject_wbes.append({
                    "method": pretty,
                    "F": F,
                    "subject": sid,
                    "wbse": wbse_s
                })
                
        for sid in subs:
            subject_withins = []
            for r1, r2 in itertools.combinations(reps_byF[F][sid], 2):
                d = np.linalg.norm(r1 - r2)
                within.append(d)
                subject_withins.append(d)
            if subject_withins:
                per_subject_within.append({
                    "method": pretty,
                    "F": F,
                    "subject": sid,
                    "within": np.mean(subject_withins)
                })

        for i, sa in enumerate(subs):
            for sb in subs[i+1:]:
                for r1 in reps_byF[F][sa]:
                    for r2 in reps_byF[F][sb]:
                        between.append(np.linalg.norm(r1 - r2))

        if len(within) and len(between):
            within = np.array(within)
            between = np.array(between)
            w_mean, w_std = within.mean(), within.std() + VAR_EPS
            rows_inter.append(dict(
                F=F,
                wbse=wh.cohens_d(within, between),
                within_mean=w_mean,
                within_std=w_std,
                within_cv=w_mean / w_std,
                between_mean=between.mean(),
                n_subjects=len(subs)
            ))

            if F in F_SHOW:
                density_data[pretty][F] = (
                    within, between, wh.cohens_d(within, between))

            out_npy_prefix = os.path.join(out_dir, f"wbse_inter_F{F}")
            np.save(out_npy_prefix + "_within.npy", within)
            np.save(out_npy_prefix + "_between.npy", between)

    # === Salva CSV per soggetto (con nome unico) ===
    if per_subject_within:
        df_within = pd.DataFrame(per_subject_within)
        out_csv = os.path.join(out_dir, f"{subdir}-within_per_subject_v2.csv")
        df_within.to_csv(out_csv, index=False)
        print(f"✅ Saved {out_csv}")
    
    if per_subject_wbes:
        df_wbes = pd.DataFrame(per_subject_wbes)
        out_csv = os.path.join(
            out_dir, f"{subdir}-wbes_per_subject_v2.csv")  # 🔧 nuovo nome
        df_wbes.to_csv(out_csv, index=False)
        print(f"✅ Saved {out_csv}")
        
    for F1, F2 in itertools.combinations(group_sizes, 2):
        subs = set(reps_byF[F1]) & set(reps_byF[F2])
        within, between = [], []

        for sid in subs:
            d = np.linalg.norm(
                np.mean(reps_byF[F1][sid], 0) - np.mean(reps_byF[F2][sid], 0))
            within.append(d)

        for sa, sb in itertools.combinations(subs, 2):
            d = np.linalg.norm(
                np.mean(reps_byF[F1][sa], 0) - np.mean(reps_byF[F2][sb], 0))
            between.append(d)

        if within and between:
            rows_intra.append(dict(
                F1=F1, F2=F2,
                wbse=wh.cohens_d(np.array(within), np.array(between)),
                within_mean=np.mean(within),
                between_mean=np.mean(between),
                n_subjects=len(subs)
            ))

    pd.DataFrame(rows_inter).to_csv(os.path.join(
        out_dir, f"{subdir}-wbes_inter_F.csv"), index=False)
    pd.DataFrame(rows_intra).to_csv(os.path.join(
        out_dir, f"{subdir}-wbes_intra_F1vsF2.csv"), index=False)
    pd.concat([
        pd.DataFrame(rows_inter).assign(type="inter", compare="F"),
        pd.DataFrame(rows_intra).assign(type="intra", compare="F1_vs_F2")
    ]).to_csv(os.path.join(out_dir, f"{subdir}-wbes_summary.csv"), index=False)

    print(f"CSV saved to {out_dir}")
