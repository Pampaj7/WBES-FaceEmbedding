#!/usr/bin/env python
"""Un solo frame per tutto l'insieme, invece di uno per mesh.

PROBLEMA. Il loader (`dataset_gtready.py:167-171`, e in copia a :299-306) normalizza OGNI mesh
per conto suo: `V -= V.mean(0); V /= max|V|`. Il bersaglio D_GT invece e' calcolato sulle
coordinate grezze (`compute_gt_distance_matrix_normalized.py`: il docstring dichiara di
normalizzare, il codice non lo fa -- verificato, rho(salvato, raw) = 1.000000). Ingresso e
bersaglio vivono quindi in frame diversi, e il divisore per-mesh varia del 14% fra `original`
(99933) e `crop` (86072), perche' il crop toglie proprio il vertice che fissa il divisore.

SOLUZIONE. Applicare una SINGOLA similarita' a tutte le mesh:

    V -> (V - c0) / s0        con c0, s0 costanti dell'insieme

Sotto una similarita' globale, d(i,j) = media_v ||V_i[v] - V_j[v]|| diventa d(i,j)/s0. E' un
fattore moltiplicativo costante: **nessun rango cambia**. Il bersaglio e' preservato
esattamente, non approssimativamente -- a differenza di rms, area o global_rms, che centrano
per-mesh e costano fra 0.068 e 0.075 di auto-consistenza cross-topologia.

    frame          rho cross-topologia del bersaglio
    raw / globale              0.8652
    global_rms                 0.7976
    area                       0.7959
    rms                        0.7906
    maxabs  (attuale)          0.5734

E l'ingresso resta allenabile: maxabs mediano ~1.57, span ~3.03.

LEAKAGE. c0 e s0 sono statistiche dei dati. Vanno stimate SOLO sulle identita' di training,
altrimenti il frame porta informazione sul test. Lo split e' ricostruito con la stessa funzione
del trainer, cosi' non c'e' modo che divergano.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))


def vertex_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    tri = V[F]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    out = np.zeros(len(V))
    for c in range(3):
        np.add.at(out, F[:, c], a / 3.0)
    return out


def rms_radius(V: np.ndarray, F: np.ndarray) -> float:
    a = vertex_areas(V, F)
    w = a / a.sum()
    X = V - (w[:, None] * V).sum(0, keepdims=True)
    return float(np.sqrt((w * (X * X).sum(1)).sum()))


def fit(mesh_dir: Path, train_subjects: list[str], topo: str = "original") -> dict:
    """c0 = centroide medio, s0 = mediana dei raggi rms. Entrambi sul solo training."""
    cs, rs = [], []
    for sid in train_subjects:
        p = mesh_dir / f"{sid}_GTready_{topo}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            V = (z["verts"] if "verts" in z else z["V"]).astype(np.float64)
            F = (z["faces"] if "faces" in z else z["F"]).astype(np.int64)
        cs.append(V.mean(0))
        rs.append(rms_radius(V, F))
    if len(cs) < 10:
        raise SystemExit(f"trovate solo {len(cs)} mesh di training in {mesh_dir}")
    return {"c0": np.mean(cs, axis=0).tolist(), "s0": float(np.median(rs)),
            "n_train_meshes": len(cs), "topology": topo, "mesh_dir": str(mesh_dir)}


def apply(V: np.ndarray, c0, s0: float) -> np.ndarray:
    return (V - np.asarray(c0)) / s0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-dir", type=Path,
                    default=REPO / "datasets" / "REMESH" / "npz_data_topo_500_withops")
    ap.add_argument("--eval-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--out", type=Path, default=THIS / "gt_matrices" / "global_frame.json")
    args = ap.parse_args()

    from robustness.data_utils import rebuild_subject_split

    all_sids = sorted({p.name.split("_GTready")[0]
                       for p in args.mesh_dir.glob("*_GTready_original.npz")})
    train, evl = rebuild_subject_split(all_sids, args.eval_fraction, args.seed, args.max_subjects)
    assert not (set(train) & set(evl)), "split non disgiunto"
    print(f"{len(all_sids)} identita' -> {len(train)} train / {len(evl)} eval  (seed {args.seed})")

    fr = fit(args.mesh_dir, train)
    fr.update(eval_fraction=args.eval_fraction, seed=args.seed,
              n_train_subjects=len(train), n_eval_subjects=len(evl))
    print(f"c0 = {np.round(fr['c0'], 1)}   s0 = {fr['s0']:.1f}")

    # --- il controllo che rende il frame utilizzabile: i ranghi del bersaglio sopravvivono ---
    from scipy.stats import spearmanr
    sub = evl[:60]                       # sul lato EVAL: e' li' che il bersaglio deve reggere
    V, F = [], None
    for sid in sub:
        with np.load(args.mesh_dir / f"{sid}_GTready_original.npz") as z:
            V.append((z["verts"] if "verts" in z else z["V"]).astype(np.float64))
            F = (z["faces"] if "faces" in z else z["F"]).astype(np.int64) if F is None else F

    def pdist(vs):
        n = len(vs)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = np.linalg.norm(vs[i] - vs[j], axis=1).mean()
        return D[np.triu_indices(n, 1)]

    d_raw = pdist(V)
    d_glob = pdist([apply(x, fr["c0"], fr["s0"]) for x in V])
    rho = float(spearmanr(d_raw, d_glob).statistic)
    ratio = d_glob / d_raw
    spread = float(ratio.max() / ratio.min() - 1.0)
    print(f"controllo su {len(sub)} identita' eval, {len(d_raw)} coppie:")
    print(f"  rho(D_GT raw, D_GT frame globale) = {rho:.10f}")
    print(f"  rapporto d_glob/d_raw costante entro {spread:.2e}  (atteso ~1e-15, e' una similarita')")
    assert rho > 1 - 1e-9, f"il frame globale NON preserva i ranghi del bersaglio (rho={rho})"
    assert spread < 1e-9, f"il rapporto non e' costante ({spread:.2e}): non e' una similarita' pura"

    # e per contrasto, quanto costa il maxabs per-mesh sullo stesso insieme
    def maxabs(x):
        y = x - x.mean(0, keepdims=True)
        return y / np.abs(y).max()
    rho_max = float(spearmanr(d_raw, pdist([maxabs(x) for x in V])).statistic)
    print(f"  per confronto, maxabs per-mesh:      rho = {rho_max:.4f}  (perde {1-rho_max:.4f})")

    fr.update(check_rho_global_vs_raw=rho, check_rho_maxabs_vs_raw=rho_max,
              check_n_eval_subjects=len(sub))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fr, indent=2))
    print(f"scritto in {args.out}")


if __name__ == "__main__":
    main()
