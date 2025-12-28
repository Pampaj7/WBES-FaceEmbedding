# utils/WBES_helper.py (completo e sistemato)

import os
import json
import numpy as np
import seaborn as sns
from glob import glob
from typing import List, Tuple, Literal
from collections import defaultdict
from random import sample

# === Costanti ===
VAR_EPS = 1e-6
BANNED_SUBJECTS = set()

# === Landmark and alignment utils ===

def load_landmark_indices(path: str, n: int = 51) -> List[int]:
    with open(path, 'r') as f:
        data = json.load(f)
    indices = data["lmk_indices"]
    if len(indices) < n:
        raise ValueError(f"Only {len(indices)} indices found, expected at least {n}")
    return indices[:n]


def extract_landmarks_from_mesh(mesh: np.ndarray, indices: List[int]) -> np.ndarray:
    if mesh.shape[0] == len(indices):
        # Caso: è già una mesh con solo i landmark
        return mesh
    elif mesh.shape[0] > max(indices):
        return mesh[indices]
    else:
        raise ValueError(f"Mesh shape {mesh.shape} incompatible with landmark indices")



def landmark_based_align(
        X: np.ndarray,
        Y: np.ndarray,
        Xlmks: np.ndarray,
        Ylmks: np.ndarray,
        ref_lmk_indices: List[int] = [13, 19, 28, 31, 37]
) -> Tuple[np.ndarray, np.ndarray]:

    b, R, t = _procrustes(Ylmks[ref_lmk_indices], Xlmks[ref_lmk_indices])
    X_aligned = b * (X @ R) + t
    Xlmks_aligned = b * (Xlmks @ R) + t
    return X_aligned, Xlmks_aligned


def _procrustes(
        X: np.ndarray,
        Y: np.ndarray,
        scaling: bool = True,
        reflection: Literal["best", True, False] = "best",
        tol: float = 1e-8
) -> Tuple[float, np.ndarray, np.ndarray]:

    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X {X.shape} and Y {Y.shape} must be the same.")

    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - muX, Y - muY

    normX = np.linalg.norm(X0)
    normY = max(np.linalg.norm(Y0), tol)

    X0 /= normX
    Y0 /= normY

    U, s, Vt = np.linalg.svd(X0.T @ Y0)
    V = Vt.T
    R = V @ U.T

    if reflection != "best" and (np.linalg.det(R) < 0) != reflection:
        V[:, -1] *= -1
        R = V @ U.T

    traceTA = s.sum()
    scale = traceTA * normX / normY if scaling else 1.0
    translation = muX - scale * (muY @ R)

    return scale, R, translation


# === Error metrics ===

def vertexwise_l2_error(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean per-vertex L2 error between two aligned meshes."""
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: {X.shape} vs {Y.shape}")
    return np.linalg.norm(X - Y, axis=1).mean()


def cohens_d(w: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled sigma (w = within, b = between)."""
    pooled = np.sqrt(((len(w)-1)*w.var() + (len(b)-1)*b.var()) /
                     (len(w)+len(b)-2))
    return (b.mean() - w.mean()) / (pooled + 1e-12)


# === Data loading ===

def load_meshes_txt(folder: str) -> dict:
    """Carica solo .txt per le mesh. Return {subject_id: [mesh, ...]}"""
    d = defaultdict(list)
    for fp in sorted(glob(os.path.join(folder, "*.txt"))):
        sid = os.path.basename(fp).split("_")[0]
        if sid in BANNED_SUBJECTS:
            continue
        try:
            d[sid].append(np.loadtxt(fp))
        except Exception as e:
            print(f"[ERROR] mesh {fp}: {e}")
    return d

def load_landmarks_npy(folder: str) -> dict:
    """Carica solo .npy per i landmark. Return {subject_id: [landmarks, ...]}"""
    d = defaultdict(list)
    for fp in sorted(glob(os.path.join(folder, "*.npy"))):
        sid = os.path.basename(fp).split("_")[0]
        if sid in BANNED_SUBJECTS:
            continue
        try:
            d[sid].append(np.load(fp))
        except Exception as e:
            print(f"[ERROR] landmark {fp}: {e}")
    return d



def reps_disjoint(meshes: List[np.ndarray], F: int, n_rep: int) -> List[np.ndarray]:
    """n_rep means built from disjoint F-frame subsets when possible."""
    tot = len(meshes)
    if tot < F:
        return []
    if F * n_rep <= tot:
        idx = sample(range(tot), F * n_rep)
        return [np.mean(np.stack([meshes[i] for i in idx[k*F:(k+1)*F]]), 0)
                for k in range(n_rep)]
    # fallback (may overlap)
    return [np.mean(np.stack(sample(meshes, F)), 0) for _ in range(n_rep)]


# === Visualization ===

def safe_kde(arr: np.ndarray, color: str, label: str, ax, var_eps: float = VAR_EPS):
    """Draw KDE or spike if var ~ 0."""
    if arr.size == 0:
        return
    if np.std(arr) < var_eps:
        ax.axvline(arr.mean(), color=color, lw=2, label=label)
    else:
        sns.kdeplot(arr, ax=ax, color=color, bw_adjust=0.6, label=label)
