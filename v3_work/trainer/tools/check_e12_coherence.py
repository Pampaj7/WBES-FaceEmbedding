#!/usr/bin/env python3
"""La distanza form della testa fattorizzata e' coerente con le GT di E12? Controllo sulle matrici di E12 su disco.

Per i set di valutazione di E12 (HIFI3D, FaceVerse, FaceScape dev; pool di 500), dai punti nativi della regione e
dal frame di GT-F (v3_work/canonical_gt/cgt.py, importato): f_i = u_d R_d p_i + t_d, S_i = centroid size pesata,
z_i = (f_i - m_i) / S_i, dP = corda fra pre-forme (la GT shape del training, factorized_v3.py). Poi:
  * d_F = sqrt((S_i - S_j)^2 + S_i S_j dP^2) contro ``F_centered`` di E12: deve coincidere (stessi punti, pesi);
  * d_F contro ``F`` (con la traslazione per identita'): Spearman;
  * CS(mu) * dP contro ``S`` di E12 (che scala attorno al centroide fisso di mu, senza togliere la traslazione per
    identita'): Spearman e rapporto mediano; dP contro ``unified`` (Procrustes completo);
  * baseline "solo taglia" |log S_i - log S_j| contro F.

    v3_work/unified_gt/run.sh v3_work/trainer/tools/check_e12_coherence.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
sys.path.insert(0, str(REPO / "v3_work/canonical_gt"))
sys.path.insert(0, str(THIS.parent))

import cgt  # noqa: E402  (E12, sola lettura)


def load(path: Path, ids: list[str]) -> np.ndarray:
    with np.load(path, allow_pickle=True) as z:
        names = [str(n) for n in z["names"]]
        D = np.asarray(z["D_orig"], np.float64)
    pos = {n: i for i, n in enumerate(names)}
    ii = np.asarray([pos[s] for s in ids])
    return D[np.ix_(ii, ii)]


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="hifi3d,faceverse,facescape")
    a = ap.parse_args()
    t0 = time.time()
    import factorized_v3 as fz
    cn = cgt.Canon()
    W = cn.W
    mu_c = cn.mu - W @ cn.mu
    cs_mu = float(np.sqrt(W @ (mu_c ** 2).sum(1)))
    out = {"cs_mu_mm": cs_mu, "frames": str(cgt.FRAMES_JSON), "gt_dir": str(cgt.OUT_DIR), "sets": {}}
    for name in a.sets.split(","):
        nat = cgt.native_points(name, cn)
        X = cn.to_F(nat["domain"], nat["P"])
        m = np.einsum("v,nvd->nd", W, X)
        Xc = X - m[:, None]
        S = np.sqrt(np.einsum("v,nv->n", W, (Xc ** 2).sum(-1)))
        Zf = (np.sqrt(W)[None, :, None] * (Xc / S[:, None, None])).reshape(len(X), -1)
        dP = np.sqrt(np.clip(2.0 - 2.0 * Zf @ Zf.T, 0.0, None))
        iu = np.triu_indices(len(X), 1)
        dF = fz.form_distance(S[iu[0]], S[iu[1]], dP[iu])
        g = {v: load(cgt.OUT_DIR / f"{name}_{v}.npz", nat["ids"])[iu] for v in ("F", "F_centered", "S", "unified")
             if (cgt.OUT_DIR / f"{name}_{v}.npz").exists()}
        r = {"n": len(X), "S_mm_mean": float(S.mean()), "S_cv": float(S.std(ddof=1) / S.mean()),
             "median_dP": float(np.median(dP[iu])), "median_dF_mm": float(np.median(dF))}
        if "F_centered" in g:
            r["dF_vs_F_centered_max_abs_mm"] = float(np.abs(dF - g["F_centered"]).max())
        if "F" in g:
            r["spearman_dF_vs_F"] = spearman(dF, g["F"])
            r["spearman_size_only_vs_F"] = spearman(np.abs(np.log(S)[iu[0]] - np.log(S)[iu[1]]), g["F"])
        if "S" in g:
            r["spearman_dP_vs_S"] = spearman(dP[iu], g["S"])
            r["median_S_over_csmu_dP"] = float(np.median(g["S"] / (cs_mu * dP[iu])))
        if "unified" in g:
            r["spearman_dP_vs_unified"] = spearman(dP[iu], g["unified"])
        out["sets"][name] = r
        print(name, json.dumps(r), flush=True)
    out["seconds"] = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
