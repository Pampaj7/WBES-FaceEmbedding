import os
import numpy as np
import pandas as pd
from glob import glob
from random import sample, seed
from tqdm import tqdm
import scipy.stats
from typing import List
import sys

# CONFIG
DATA_ROOT = "../"
GT_ROOT = os.path.join(DATA_ROOT, "GT", "GT_BFM")
LANDMARK_JSON = "/Users/pampaj/PycharmProjects/data/utils/BFM-p23470.json"
sys.path.append("/Users/pampaj/data")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import WBES_helper as wh

seed(42)
landmark_indices = wh.load_landmark_indices(LANDMARK_JSON)

METHODS = [
    "3DDFAV2_23470_neutral",
    "3DDFAV3_neutral",
    "Deep3DFace_23470_neutral",
    "SynergyNet_neutral",
    "INORig_23470_neutral",
    "3DI_neutral",
]
BANNED_SUBJECTS = {"id0099"}

F_BY_METHOD = {
    "INORig_23470_neutral": [1, 2, 3],
    "3DI_neutral": [1],
    "default": [1, 3, 5, 10, 15]
}

N_REPS = 3

def load_all_meshes(method_dir, sid):
    folder = os.path.join(DATA_ROOT, method_dir)
    if "INORig" in method_dir:
        mesh_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.startswith(sid + "_") and f.endswith("group0.txt")
        ])
    else:
        mesh_files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.startswith(sid + "_") and f.endswith(".txt") and "lmk" not in f
        ])
    return [np.loadtxt(f) for f in mesh_files if os.path.getsize(f) > 0]

def load_gt_mesh(sid):
    path = os.path.join(GT_ROOT, f"{sid}.id.txt")
    return np.loadtxt(path) / 1e6

def compute_corr(x, y):
    return np.abs(scipy.stats.pearsonr(x, y)[0]) if len(x) and len(y) else np.nan

def compute_corr_evangelos_complex(x, y):
    """Pearson combinato reale/immaginario su valori complessi."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    r_real = scipy.stats.pearsonr(x.real, y.real)[0]
    r_imag = scipy.stats.pearsonr(x.imag, y.imag)[0]
    return np.sqrt(r_real**2 + r_imag**2)

# PRELOAD
gt_meshes = {}
raw_meshes = {m: {} for m in METHODS}
valid_subjects_by_method = {m: [] for m in METHODS}

for sid in tqdm([f"id{n:04d}" for n in range(100)], desc="Subjects"):
    if sid in BANNED_SUBJECTS:
        continue
    try:
        gt_meshes[sid] = load_gt_mesh(sid)
    except:
        continue
    for method in METHODS:
        try:
            subject_meshes = load_all_meshes(method, sid)
            f_list = F_BY_METHOD.get(method, F_BY_METHOD["default"])
            if len(subject_meshes) >= max(f_list) * N_REPS:
                raw_meshes[method][sid] = subject_meshes
                valid_subjects_by_method[method].append(sid)
        except:
            continue

# ANALYSIS
all_results = []

for method in METHODS:
    print(f"\n--- {method} ---")
    F_list = F_BY_METHOD.get(method, F_BY_METHOD["default"])
    valid_subjects = valid_subjects_by_method[method]

    for F in F_list:
        reps = {}
        for sid in valid_subjects:
            try:
                reps[sid] = wh.reps_disjoint(raw_meshes[method][sid], F, N_REPS)
            except:
                continue

        rs_cosine, gs_cosine = [], []
        rs_l2, gs_l2 = [], []
        rs_complex, gs_complex = [], []

        subs = list(reps.keys())
        for i, s1 in enumerate(subs):
            if s1 not in gt_meshes: continue
            r1 = np.mean(np.stack(reps[s1]), axis=0).flatten()
            g1 = gt_meshes[s1].flatten()
            for s2 in subs[i+1:]:
                if s2 not in gt_meshes: continue
                r2 = np.mean(np.stack(reps[s2]), axis=0).flatten()
                g2 = gt_meshes[s2].flatten()
                try:
                    theta_r = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
                    theta_g = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2))
                    l2_r = np.linalg.norm(r1 - r2)
                    l2_g = np.linalg.norm(g1 - g2)
                    angle_r = np.arccos(np.clip(theta_r, -1.0, 1.0))
                    angle_g = np.arccos(np.clip(theta_g, -1.0, 1.0))
                    complex_r = l2_r * np.exp(1j * angle_r)
                    complex_g = l2_g * np.exp(1j * angle_g)

                    if not np.isnan(theta_r) and not np.isnan(theta_g):
                        rs_cosine.append(theta_r)
                        gs_cosine.append(theta_g)
                    rs_l2.append(l2_r)
                    gs_l2.append(l2_g)
                    rs_complex.append(complex_r)
                    gs_complex.append(complex_g)
                except:
                    continue

        c_cosine = compute_corr(rs_cosine, gs_cosine)
        c_l2 = compute_corr(rs_l2, gs_l2)
        c_complex = compute_corr_evangelos_complex(rs_complex, gs_complex)
        all_results.append({
            "method": method,
            "F": F,
            "cosine_corr": c_cosine,
            "l2_corr": c_l2,
            "complex_corr": c_complex,
            "n_comparisons": len(rs_cosine)
        })

df = pd.DataFrame(all_results)
df.to_csv("results/cosine_wbesstyle_disjoint.csv", index=False)
