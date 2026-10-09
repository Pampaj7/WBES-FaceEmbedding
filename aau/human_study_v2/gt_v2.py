#!/usr/bin/env python3
"""Studio umano v2, passo 1: le GT di E12 sui 100 soggetti BFM REMESH held-out (provvisorie finche' E12 non consegna).

    v3_work/unified_gt/run.sh aau/human_study_v2/gt_v2.py          (run.sbatch, passo ``gt``)

E12 calcola le matrici solo per i set di valutazione (hifi3d, faceverse, facescape, famos), non per BFM: qui si
chiama lo STESSO codice (``cgt.all_gts``) sui punti della regione unificata delle 100 ``original`` REMESH, con il
frame di GT-F di E12 se esiste (``hs2.canon``). Dentro un dominio la trasformazione e' comune a tutti, quindi
l'ordine delle distanze (e le triplette) non dipende dal frame; cambia solo se E12 corregge il codice delle GT,
e per questo il manifest porta l'impronta di ``cgt.py`` e ``gt.py``.

GT salvate in ``datasets/HUMAN_STUDY_V2/gt/<gt>.npz`` (``D`` piena, ``names``), unita' in ``manifest.json``:
  F, S, EDM, EDM_s, unified, F_centered, F_rig_ls, F_rig_rob, F_json (mm), size_only, height_only (log),
  maxabs = la GT legacy della v1 (``common.load_gt_submatrix``: la matrice del paper sulle REMESH, divisa per il
  suo massimo, quindi senza unita').
Piu' gli Spearman fra le GT sulle 4950 coppie (``gt_corr.csv``) e la dimensione del volto (centroid size,
altezza) per soggetto (``size.csv``), che servono al controllo dei render.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import hs2

UNITS = {"F": "mm", "S": "mm", "EDM": "mm", "EDM_s": "mm", "unified": "mm", "F_centered": "mm", "F_rig_ls": "mm",
         "F_rig_rob": "mm", "F_json": "mm (scala del json)", "size_only": "|dlog CS|", "height_only": "|dlog H|",
         "maxabs": "GT legacy / suo massimo"}


def symmetrize(D: np.ndarray) -> np.ndarray:
    """Matrice piena da una piena o riempita solo per i<j (come ``aau/human_study/select_triplets.py``)."""
    lower = np.tril(np.ones_like(D, dtype=bool), -1)
    if np.isfinite(D[lower]).all() and np.abs(D[lower]).sum() > 0:
        if not np.allclose(D[lower], D.T[lower]):
            raise ValueError("matrice piena ma non simmetrica")
        S = np.array(D, dtype=np.float64)
    else:
        upper = np.triu(np.ones_like(D, dtype=bool), 1)
        S = np.zeros_like(D, dtype=np.float64)
        S[upper] = D[upper]
        S = S + S.T
    np.fill_diagonal(S, 0.0)
    return S


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main() -> None:
    import cgt
    import common

    t0 = time.time()
    subj = hs2.subjects()
    cn, frame = hs2.canon()
    print(f"[hs2-gt] {len(subj)} soggetti, frame {frame['source']} ({time.time() - t0:.0f}s)", flush=True)
    P = np.stack([cn.sp.map(hs2.DOMAIN, hs2.load_mesh(s)[0]) for s in subj])
    D, diag = cgt.all_gts(cn, hs2.DOMAIN, P, real=False)
    D["maxabs"] = np.asarray(common.load_gt_submatrix(subj), dtype=np.float64)
    hs2.GT_DIR.mkdir(parents=True, exist_ok=True)
    iu = np.triu_indices(len(subj), 1)
    man = {"domain": hs2.DOMAIN, "topology": hs2.TOPOLOGY, "subject_set": hs2.SUBJECT_SET, "n": len(subj),
           "frame": frame, "definition": "aau/runs/evidence/e12/protocol.md (Emendamento 1), cgt.all_gts",
           "maxabs": f"common.load_gt_submatrix ({common.GT_MATRIX.relative_to(hs2.REPO_ROOT)})", "gts": {}}
    for g in list(D):
        D[g] = M = symmetrize(D[g])
        np.savez(hs2.GT_DIR / f"{g}.npz", D=M, names=np.asarray(subj))
        man["gts"][g] = {"units": UNITS.get(g, ""), "median": float(np.median(M[iu])),
                         "min_offdiag": float(M[iu].min()), "max": float(M.max())}
    rows = []
    names = list(D)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"gt_a": a, "gt_b": b, "spearman": spearman(D[a][iu], D[b][iu]), "n_pairs": len(iu[0])})
    pd.DataFrame(rows).to_csv(hs2.GT_DIR / "gt_corr.csv", index=False)
    Xf = diag["Xf"]
    size = pd.DataFrame({"subject": subj, "centroid_size_mm": diag["centroid_size_mm"], "height_mm": diag["height_mm"],
                         "width_mm": Xf[..., 0].max(1) - Xf[..., 0].min(1),
                         "depth_mm": Xf[..., 2].max(1) - Xf[..., 2].min(1)})
    size.to_csv(hs2.GT_DIR / "size.csv", index=False)
    man["size_cv"] = {k: float(size[k].std(ddof=1) / size[k].mean()) for k in size.columns[1:]}
    man["seconds"] = time.time() - t0
    hs2.write_json(hs2.GT_DIR / "manifest.json", man)
    main_gts = ["F", "S", "EDM", "EDM_s", "unified", "maxabs"]
    corr = pd.DataFrame(rows)
    print(corr[corr["gt_a"].isin(main_gts) & corr["gt_b"].isin(main_gts)].to_string(index=False), flush=True)
    print(f"[hs2-gt] CV della dimensione: {man['size_cv']}", flush=True)
    print(f"[hs2-gt] fatto in {man['seconds']:.0f}s -> {hs2.GT_DIR}", flush=True)


if __name__ == "__main__":
    main()
