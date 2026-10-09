#!/usr/bin/env python3
"""Studio umano v2, passo 1: le GT sulle 100 identita' GNM, con le funzioni di E12 (``v3_work/canonical_gt/cgt.py``).

    v3_work/unified_gt/run.sh aau/human_study_v2/gt_v2.py          (run.sbatch, passo ``gt``)

Punti della regione unificata (1478, ``sp.map``) di ogni testa GNM, nel frame di GT-F in mm (X0), poi la rigida
ROBUSTA per identita' verso mu (``rigid_robust``: IRLS Tukey, E12 Emendamento 1) -> a_i. GT dello studio:
  - **F**  = distanza RMS pesata fra le a_i (forma metrica in mm, rigida robusta: la GT form di riferimento di E12);
  - **S**  = la stessa forma senza taglia: a_i centrata sul suo centroide pesato e scalata alla centroid size di mu
             (``shape_S`` dopo ``centered``): Procrustes pieno con la rotazione robusta. La S di E12 a punto fisso
             NON e' usata: non toglie la traslazione;
  - **EDM**, **EDM_s** = distanze interne fra 400 punti (invarianti alla rigida, ``edm_features`` su X0);
  - **maxabs** = la GT legacy della pipeline zero-shot: patch ``hockey_mask`` di ``v3_work.mm``, normalizzazione
             maxabs per mesh, media per vertice della distanza L2 (``gt.maxabs_matrix``);
  - secondarie: **unified** (Procrustes di similarita' LS, la GT di E8), **F_rig_ls** (rigida LS), **F_pure** (X0
             senza rigida per identita'), baseline **size_only** e **height_only** (|dlog| di centroid size e
             altezza di a_i).
Matrici in ``datasets/HUMAN_STUDY_V2/gt/<gt>.npz`` (``D``, ``names``), ``manifest.json``, ``gt_corr.csv`` (Spearman
sulle 4950 coppie), ``size.csv`` e ``rigid.csv`` (rigida robusta per identita').
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import hs2

UNITS = {"F": "mm", "S": "mm (taglia di mu)", "EDM": "mm", "EDM_s": "mm (taglia di mu)", "unified": "mm",
         "F_rig_ls": "mm", "F_pure": "mm", "size_only": "|dlog CS|", "height_only": "|dlog H|",
         "maxabs": "unita' maxabs"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def study_gts(cn, X0: np.ndarray, rob: dict, patches: np.ndarray) -> dict:
    import cgt
    import gt
    a = rob["a"]
    m = cn.W @ cn.mu
    D = {"F": cn.distances(a), "S": cn.distances(cn.shape_S(cn.centered(a) + m)),
         "EDM": cgt.edm_distances(cn.edm_features(X0)), "EDM_s": cgt.edm_distances(cn.edm_features(X0, True)),
         "maxabs": gt.maxabs_matrix(patches), "unified": cn.distances(cn.unified(X0)),
         "F_rig_ls": cn.distances(cn.rigid_ls(X0)["a"]), "F_pure": cn.distances(X0),
         "size_only": cgt.scalar_distances(cn.centroid_size(a)), "height_only": cgt.scalar_distances(cn.height(a))}
    for g, M in D.items():
        M = 0.5 * (M + M.T)
        np.fill_diagonal(M, 0.0)
        D[g] = M
    return D


def main() -> None:
    t0 = time.time()
    subj = hs2.subjects()
    cn, frame = hs2.canon()
    rob = hs2.robust_rigid(cn, subj)
    patches = np.stack([hs2.face_patch(s) for s in subj])
    D = study_gts(cn, rob["X0"], rob, patches)
    print(f"[hs2-gt] {len(subj)} identita' {hs2.DOMAIN}, frame {frame['source']} ({time.time() - t0:.0f}s)", flush=True)
    hs2.GT_DIR.mkdir(parents=True, exist_ok=True)
    for old in hs2.GT_DIR.glob("*.npz"):
        old.unlink()
    iu = np.triu_indices(len(subj), 1)
    man = {"domain": hs2.DOMAIN, "n": len(subj), "seed": hs2.SEED, "frame": frame,
           "sampling": "z ~ N(0, 1), 170 modi head_*, v3_work.mm sample_identity(tails=False, trunc=0)",
           "definition": "aau/human_study_v2/gt_v2.py (funzioni di cgt.py, E12 Emendamento 1); F = rigida robusta",
           "gts": {}}
    for g, M in D.items():
        np.savez(hs2.GT_DIR / f"{g}.npz", D=M, names=np.asarray(subj))
        man["gts"][g] = {"units": UNITS[g], "median": float(np.median(M[iu])), "min_offdiag": float(M[iu].min()),
                         "max": float(M.max())}
    np.savez(hs2.GT_DIR / "identities.npz", z=hs2.identity_weights(), names=np.asarray(subj))
    rows = []
    names = list(D)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"gt_a": a, "gt_b": b, "spearman": spearman(D[a][iu], D[b][iu]), "n_pairs": len(iu[0])})
    corr = pd.DataFrame(rows)
    corr.to_csv(hs2.GT_DIR / "gt_corr.csv", index=False)
    a = rob["a"]
    size = pd.DataFrame({"subject": subj, "centroid_size_mm": cn.centroid_size(a), "height_mm": cn.height(a),
                         "width_mm": np.ptp(a[..., 0], axis=1), "depth_mm": np.ptp(a[..., 2], axis=1)})
    size.to_csv(hs2.GT_DIR / "size.csv", index=False)
    pd.DataFrame({"subject": subj, "angle_deg": rob["angle"], "offset_mm": rob["offset"],
                  "iterations": rob["iterations"], "converged": rob["converged"],
                  "downweighted_area": rob["downweighted_area"]}).to_csv(hs2.GT_DIR / "rigid.csv", index=False)
    np.savez(hs2.GT_DIR / "rigid.npz", R=rob["R"], t=rob["t"], names=np.asarray(subj))
    man["size_cv"] = {k: float(size[k].std(ddof=1) / size[k].mean()) for k in size.columns[1:]}
    man["rigid"] = {"angle_deg_median": float(np.median(rob["angle"])), "angle_deg_max": float(rob["angle"].max()),
                    "converged_fraction": float(np.mean(rob["converged"]))}
    man["seconds"] = time.time() - t0
    hs2.write_json(hs2.GT_DIR / "manifest.json", man)
    main_gts = ["F", "S", "EDM", "maxabs", "unified", "size_only"]
    print(corr[corr["gt_a"].isin(main_gts) & corr["gt_b"].isin(main_gts)].to_string(index=False), flush=True)
    print(f"[hs2-gt] CV della dimensione: {man['size_cv']}; rigida robusta: {man['rigid']}", flush=True)
    print(f"[hs2-gt] fatto in {man['seconds']:.0f}s -> {hs2.GT_DIR}", flush=True)


if __name__ == "__main__":
    main()
