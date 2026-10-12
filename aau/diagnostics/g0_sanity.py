#!/usr/bin/env python3
"""G0, controllo di sanita' POST HOC (non nel protocollo, scritto dopo i numeri): l'insegnante riconosce ancora le
identita' su ogni insieme di G0? Se si', il suo Spearman basso non viene da una pipeline rotta (render, frame, crop).

    aau/baselines/run_bl.sh aau/diagnostics/g0_sanity.py

Per insieme, sulle mesh senza crop di ``g0/teacher/<insieme>/teacher.npz``, coppie di mesh con etichette diverse: AUC di
-d_T (stessa persona contro persone diverse, Mann-Whitney con i ranghi medi), rank-1 (query nelle etichette diverse da
original, galleria = le original), mediane e IQR di d_T fra stessa persona e persone diverse. Descrittivo, nessun
bootstrap. Uscita: ``g0/sanity_posthoc.json``.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata

import diag

G0 = diag.EV / "g0"
SETS = ("facescape", "hifi3d", "faceverse_neutral", "flame2023_s1", "famos", "bfm", "ict", "gnm")


def main() -> None:
    out = {"definition": "POST HOC, descrittivo (non nel protocollo G0)"}
    for s in SETS:
        with np.load(G0 / "teacher" / s / "teacher.npz", allow_pickle=True) as z:
            keep = np.asarray([str(t) != "crop" for t in z["topologies"]])
            Z = np.asarray(z["Z"], np.float64)[keep]
            sub, top = np.asarray(z["subjects"])[keep], np.asarray(z["topologies"])[keep]
        D = 1.0 - Z @ Z.T
        i, j = np.triu_indices(len(Z), 1)
        k = top[i] != top[j]
        same, d = (sub[i] == sub[j])[k], D[i, j][k]
        r = rankdata(-d)
        n1, n0 = int(same.sum()), int((~same).sum())
        subs = sorted(set(sub.tolist()))
        gal = np.asarray([np.flatnonzero((sub == x) & (top == "original"))[0] for x in subs])
        q = np.flatnonzero(top != "original")
        out[s] = {"auc_same_vs_diff": float((r[same].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)),
                  "rank1_vs_original": float(np.mean([subs[np.argmin(D[x, gal])] == sub[x] for x in q])),
                  "d_same_median": float(np.median(d[same])), "d_same_p95": float(np.percentile(d[same], 95)),
                  "d_diff_median": float(np.median(d[~same])),
                  "d_diff_iqr": [float(x) for x in np.percentile(d[~same], [25, 75])], "n_same": n1, "n_diff": n0}
        print(f"[g0-sanity] {s}: {out[s]}", flush=True)
    diag.atomic_json(G0 / "sanity_posthoc.json", out)


if __name__ == "__main__":
    main()
