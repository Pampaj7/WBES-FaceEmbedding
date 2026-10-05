#!/usr/bin/env python3
"""Da quale base BFM vengono le 500 identita' REMESH? Misura, prima di generarne altre.

    srun -p cpu -c 4 --mem=16G -t 00:20:00 env AAU_NV= aau/run.sh aau/data_scale/probe_bfm_basis.py

Il generatore originale (``render3d_Leonardo/data_creator/synthetic_meshes`` ->
``prepare_GT_ready.py``, CODEBASE_GUIDE.md:1076) e i coefficienti delle 500 mesh non sono
sul cluster ne' nello snapshot HF.  Sappiamo solo che ``original`` e' il crop p23470
(``WBES/utils/ix_23470_relative_to_53215.txt``) della topologia densa a 53215 vertici del
BFM di 3DDFA, allineato con una similarita' (trimesh procrustes) a una mesh di riferimento.

L'unico BFM in quella topologia sul cluster e' quello di 3DDFA v1 (``train.configs``):
media ``u_shp + u_exp``, 40 modi di forma ``w_shp_sim`` e 10 di espressione ``w_exp_sim``.
Se le 500 mesh stanno nel suo sottospazio, a meno di una similarita', si possono
campionare identita' nuove dallo STESSO modello; altrimenti no, e il lato BFM va
generato da un'altra base (e allora e' un dominio diverso, da dichiarare).

Per ogni mesh: alternanza similarita' (Umeyama) <-> minimi quadrati sui coefficienti,
residuo RMS per vertice riportato in rapporto alla variazione d'identita' della mesh
(RMS di ``Y - media``).  Due fit: solo forma (40) e forma+espressione (50).  Un residuo
relativo ~1e-3 o meno vuol dire "dentro il sottospazio"; ~0.3 vuol dire "fuori".
Riporta anche media e deviazione dei coefficienti stimati, per capire con che
distribuzione erano stati campionati.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CFG = REPO_ROOT / "external/3DDFA_v1/train.configs"
IDX = REPO_ROOT / "WBES/utils/ix_23470_relative_to_53215.txt"
REMESH = REPO_ROOT / "datasets/REMESH/npz_data_topo_500"


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t che minimizzano ||s R src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean())
    return s, R, mu_d - s * R @ mu_s


def fit(X: np.ndarray, mean: np.ndarray, W: np.ndarray, n_iter: int = 30):
    """X (nv,3); mean (nv,3); W (nv*3, k). Ritorna coefficienti, residuo relativo, scala."""
    coef = np.zeros(W.shape[1])
    pinv = np.linalg.pinv(W)
    for _ in range(n_iter):
        M = mean + (W @ coef).reshape(-1, 3)
        s, R, t = umeyama(M, X)
        Y = ((X - t) @ R) / s                       # X riportata nel frame del modello
        coef = pinv @ (Y - mean).ravel()
    M = mean + (W @ coef).reshape(-1, 3)
    res = np.sqrt(((Y - M) ** 2).sum(1).mean())
    ref = np.sqrt(((Y - mean) ** 2).sum(1).mean())
    return coef, float(res / ref), s


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    ix = np.loadtxt(IDX, dtype=np.int64)
    rows = (3 * ix[:, None] + np.arange(3)[None, :]).ravel()   # vertice i -> x,y,z interlacciati
    u = (np.load(CFG / "u_shp.npy") + np.load(CFG / "u_exp.npy")).ravel()
    Ws = np.load(CFG / "w_shp_sim.npy")
    We = np.load(CFG / "w_exp_sim.npy")
    mean = u[rows].reshape(-1, 3).astype(np.float64)
    Ws_c, We_c = Ws[rows].astype(np.float64), We[rows].astype(np.float64)
    print(f"base: u {u.shape}, w_shp {Ws.shape}, w_exp {We.shape}; crop {len(ix)} vertici", flush=True)

    files = sorted(REMESH.glob("id*_GTready_original.npz"))[:n]
    out = {"shape40": [], "shape40_exp10": []}
    coefs = {"shape40": [], "shape40_exp10": []}
    for p in files:
        with np.load(p) as d:
            X = np.asarray(d["V"], dtype=np.float64)
        if X.shape != mean.shape:
            raise SystemExit(f"{p.name}: {X.shape} contro {mean.shape}: non e' il crop p23470")
        for tag, W in (("shape40", Ws_c), ("shape40_exp10", np.hstack([Ws_c, We_c]))):
            c, r, _ = fit(X, mean, W)
            out[tag].append(r)
            coefs[tag].append(c)

    # controllo: una mesh del modello con coefficienti casuali deve dare residuo ~0
    rng = np.random.default_rng(0)
    c_true = rng.normal(0, 1, Ws.shape[1]) * np.std(np.array(coefs["shape40"]), 0)
    Xs = mean + (Ws_c @ c_true).reshape(-1, 3)
    _, r_ctrl, _ = fit(Xs * 0.37 + 5.0, mean, Ws_c)

    summary = {"n_meshes": len(files), "control_residual_rel": r_ctrl}
    for tag in out:
        r = np.array(out[tag])
        C = np.array(coefs[tag])
        summary[tag] = {
            "residual_rel": {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())},
            "coef_mean_first8": [float(x) for x in C.mean(0)[:8]],
            "coef_std_first8": [float(x) for x in C.std(0)[:8]],
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
