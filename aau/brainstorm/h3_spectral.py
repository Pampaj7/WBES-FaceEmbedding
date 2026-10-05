#!/usr/bin/env python3
"""H3 (BOARD_DIARY, brainstorm sugli operatori): spettri e HKS per variante di operatore.

Per ogni soggetto held-out BFM e ognuna delle 6 topologie calcola le prime K_EIG autocoppie
del problema generalizzato L phi = lambda M phi in cinque varianti:

    a  cotangente come DiffusionNet  (pp3d.cotan_laplacian + pp3d.vertex_areas, coordinate grezze)
    b  Laplaciano robusto            (robust_laplacian.mesh_laplacian, mollify di default)
    c  come a, su mesh ad area totale 1 (Weyl)
    d  come b, su mesh ad area totale 1
    e  pozzo di potenziale a 0.55    (v2_work/potential/potential_operators.py, usato cosi'
                                      com'e', alpha_mode global, scala comune 127507)

Le autocoppie non si salvano intere (sarebbero GB): si salvano gli autovalori e phi^2 nei
punti di campionamento, che bastano a ricostruire la HKS a qualunque tempo in
``h3_table.py``.  I punti sono N_SAMPLES vertici dell'``original``, gli stessi indici per
tutti i soggetti (gli original BFM condividono la connettivita', 23470 vertici: l'indice e'
quindi gia' una corrispondenza semantica fra soggetti).  In ogni altra topologia il punto e'
il vertice piu' vicino nel frame dato, che e' comune a tutte le topologie dello stesso
soggetto; ``valid`` marca i punti la cui distanza sta sotto VALID_EDGES lunghezze d'edge
mediane dell'original, cioe' esclude la parte tagliata dal crop (il crop e' un sottoinsieme
esatto dei vertici dell'original, distanza 0; down8k/noisy restano sotto 3 edge, sonda
``_probe.py``).

Il costo e' per mesh, ~1 min per soggetto con le cinque varianti: si sharda per soggetti.

    AAU_NV="" srun -p cpu -c 2 --mem=16G aau/run.sh aau/brainstorm/h3_spectral.py --shard 0/10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "potential"))

OUT_DIR = common.AAU_DIR / "runs" / "brainstorm" / "h3_parts"

# 64 modi non banali piu' quello costante: la dispersione guarda lambda_1..lambda_64.
K_EIG = 65
N_SUBJECTS = 50
N_SAMPLES = 2000
SAMPLE_SEED = 0
VALID_EDGES = 3.0
EPS = 1e-8  # come compute_operators di diffusion_net

VARIANTS = ("a", "b", "c", "d", "e")

# Pozzo: alpha del ginocchio dello sweep (STATUS.md, 2026-08-17 23:10) e scala comune della
# calibrazione collection-wide (STATUS.md, "scala comune 127507"), coordinate grezze come i
# bracci pot_w55/m55 (shard_job_global.sh non passa --area-normalize).
WELL_ALPHA = 0.55
WELL_SCALE = 127507.0


def mesh_area(V: np.ndarray, F: np.ndarray) -> float:
    tri = V[F]
    return float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                                      axis=1).sum())


def unit_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Centro e scala ad area totale 1, come areanorm_operators / potential_operators."""
    return (V - V.mean(0)) / np.sqrt(mesh_area(V, F))


def eig_pairs(L, massvec: np.ndarray):
    """Stesso solver di diffusion_net.geometry.compute_operators (L + eps I, sigma = eps)."""
    L_eigsh = (L + sp.identity(L.shape[0]) * EPS).tocsc()
    M = sp.diags(massvec)
    for fail in range(5):
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=K_EIG, M=M, sigma=EPS)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  eigsh fallito ({exc}); aggiungo eps", flush=True)
            L_eigsh = L_eigsh + sp.identity(L.shape[0]) * (EPS * 10 ** (fail + 1))
    else:
        raise ValueError("eigsh fallito 5 volte")
    order = np.argsort(evals)
    return np.clip(evals[order], 0.0, None), evecs[:, order]


def cotan_pairs(V, F):
    import potpourri3d as pp3d

    L = pp3d.cotan_laplacian(V, F, denom_eps=1e-10)
    m = pp3d.vertex_areas(V, F)
    m = m + EPS * np.mean(m)
    return eig_pairs(L, m)


def robust_pairs(V, F):
    import robust_laplacian

    L, M = robust_laplacian.mesh_laplacian(V, F)
    m = np.asarray(M.diagonal(), dtype=np.float64)
    m = m + EPS * np.mean(m)
    return eig_pairs(L, m)


def well_pairs(V, F):
    from potential_operators import potential_operators

    out = potential_operators(V, F.astype(np.int32), k_eig=K_EIG, alpha_mode="global",
                              alpha_global=WELL_ALPHA, scale_global=WELL_SCALE)
    evals, evecs, meta = out[3].numpy().astype(np.float64), out[4].numpy().astype(np.float64), out[7]
    return evals, evecs, meta["roi"]


def sample_indices() -> np.ndarray:
    n = 23470  # vertici dell'original BFM, verificato in main()
    return np.sort(np.random.default_rng(SAMPLE_SEED).choice(n, size=N_SAMPLES, replace=False))


def process_subject(subject: str, out_path: Path) -> None:
    Vo, Fo = common.load_verts_faces(subject, "original")
    if len(Vo) != 23470:
        raise ValueError(f"{subject}: original con {len(Vo)} vertici, attesi 23470")
    idx_o = sample_indices()
    edge = float(np.median(np.linalg.norm(Vo[Fo[:, 0]] - Vo[Fo[:, 1]], axis=1)))
    P = Vo[idx_o]

    rec: dict[str, np.ndarray] = {"sample_idx": idx_o, "edge_median": np.array(edge)}
    for topo in common.TOPOLOGIES:
        t0 = time.time()
        V, F = common.load_verts_faces(subject, topo)
        dist, nn = cKDTree(V).query(P)
        rec[f"{topo}/valid"] = dist < VALID_EDGES * edge
        rec[f"{topo}/area"] = np.array(mesh_area(V, F))
        rec[f"{topo}/nverts"] = np.array(len(V))
        Vn = unit_area(V, F)
        for var in VARIANTS:
            if var == "a":
                evals, evecs = cotan_pairs(V, F)
            elif var == "b":
                evals, evecs = robust_pairs(V, F)
            elif var == "c":
                evals, evecs = cotan_pairs(Vn, F)
            elif var == "d":
                evals, evecs = robust_pairs(Vn, F)
            else:
                evals, evecs, roi = well_pairs(V, F)
                rec[f"{topo}/e/roi"] = np.asarray(roi, dtype=np.float32)[nn]
            rec[f"{topo}/{var}/evals"] = evals
            rec[f"{topo}/{var}/phi2"] = (evecs[nn] ** 2).astype(np.float32)
        print(f"  {subject} {topo:8s} nV={len(V):6d} valid={rec[f'{topo}/valid'].mean():.3f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    tmp = out_path.with_name(f".{out_path.name}.tmp.npz")
    np.savez_compressed(tmp, **rec)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", default="0/1", help="i/n: questo processo prende i soggetti i::n")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subjects = common.subject_set("heldout")[:N_SUBJECTS]
    si, sn = (int(x) for x in args.shard.split("/"))
    mine = subjects[si::sn]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[h3] shard {si}/{sn}: {len(mine)} soggetti di {len(subjects)}", flush=True)
    t0 = time.time()
    for subject in mine:
        out_path = args.out_dir / f"{subject}.npz"
        if out_path.exists() and not args.overwrite:
            print(f"  {subject} gia' presente, salto", flush=True)
            continue
        process_subject(subject, out_path)
    print(f"[h3] finito in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
