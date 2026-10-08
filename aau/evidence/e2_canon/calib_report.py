#!/usr/bin/env python3
"""Rapporto della calibrazione di canon.py e soglia di fallimento, PRIMA di ogni eval sui domini di test.

    aau/run.sh aau/evidence/e2_canon/calib_report.py --dir aau/runs/evidence/e2/calibration

Soglia (regola fissata qui, prima di vedere residui sui domini di test): T = 2 x il 99-esimo percentile
del residuo trimmed sulle canonicalizzazioni CORRETTE della calibrazione (mesh non ruotate dei tre domini
di training, angolo dalla convenzione del dominio < 15 gradi). Una mesh di test con residuo > T conta come
fallimento. Scrive ``calibration.md`` e ``threshold.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OK_DEG = 15.0


def rot(df: pd.DataFrame, p: str) -> np.ndarray:
    return df[[f"{p}{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)


def angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    tr = np.einsum("nij,nij->n", A, B)   # trace(A B^T)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dir", type=Path, required=True)
    a = p.parse_args()
    df = pd.read_csv(a.dir / "calibration.csv")
    extra = json.loads((a.dir / "calibration_extra.json").read_text())
    base = df[~df["perturbed"]].copy()
    ok = base["angle_from_convention"] < OK_DEG
    T = 2.0 * float(np.percentile(base.loc[ok, "residual"], 99))

    # robustezza: R stimata sulla mesh ruotata, composta con la rotazione applicata, contro R sulla mesh com'e'
    pert = df[df["perturbed"]].merge(base, on=["domain", "subject", "topology"], suffixes=("", "_base"))
    Rp, P = rot(pert, "R"), rot(pert, "P")
    Rb = pert[[f"R{i}{j}_base" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)
    pert["recovery_err_deg"] = angles(Rp @ P, Rb)

    lines = ["# E2, calibrazione della canonicalizzazione (domini di training, soggetti held-out)\n",
             f"Parametri: {json.dumps(extra['params'])}; start: {', '.join(extra['starts'])}.",
             f"Soggetti: {len(extra['subjects']['ict'])} per dominio (ICT, BFM, GNM held-out del run grande), 6 topologie.\n",
             "## Mesh come sono (frame del dominio: ICT e GNM = identita', BFM = Rx180)\n",
             "| dominio | topologia | n | residuo mediano | p99 | max | angolo dalla convenzione mediano | max | start scelti | s/mesh mediani |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, t), g in base.groupby(["domain", "topology"]):
        lines.append(f"| {d} | {t} | {len(g)} | {g.residual.median():.4f} | {np.percentile(g.residual, 99):.4f} | "
                     f"{g.residual.max():.4f} | {g.angle_from_convention.median():.2f} | {g.angle_from_convention.max():.2f} | "
                     f"{dict(g.start.value_counts())} | {g.seconds.median():.2f} |")
    # consistenza fra topologie dello stesso soggetto: angolo fra R(t) e R(original)
    orig = base[base.topology == "original"].set_index(["domain", "subject"])
    cons = []
    for (d, s), g in base.groupby(["domain", "subject"]):
        Ro = rot(orig.loc[[(d, s)]], "R")[0]
        for _, r in g.iterrows():
            if r.topology != "original":
                cons.append({"domain": d, "topology": r.topology,
                             "deg": float(angles(rot(pd.DataFrame([r]), "R"), Ro[None])[0])})
    cons = pd.DataFrame(cons)
    lines += ["\n## Consistenza: angolo fra la rotazione di una topologia e quella della `original` dello stesso soggetto\n",
              "| dominio | topologia | mediana (gradi) | p95 | max |", "| --- | --- | --- | --- | --- |"]
    for (d, t), g in cons.groupby(["domain", "topology"]):
        lines.append(f"| {d} | {t} | {g.deg.median():.2f} | {np.percentile(g.deg, 95):.2f} | {g.deg.max():.2f} |")
    lines += ["\n## Robustezza: mesh ruotate (flip casuale fra i 4 di 180 gradi, poi fino a 30 gradi attorno a un asse casuale)\n",
              "Errore = angolo fra (R stimata sulla mesh ruotata) x (rotazione applicata) e R stimata sulla mesh com'e'.\n",
              "| dominio | topologia | n | errore mediano (gradi) | p95 | max | > 5 gradi | residuo mediano |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, t), g in pert.groupby(["domain", "topology"]):
        lines.append(f"| {d} | {t} | {len(g)} | {g.recovery_err_deg.median():.2f} | {np.percentile(g.recovery_err_deg, 95):.2f} | "
                     f"{g.recovery_err_deg.max():.2f} | {(g.recovery_err_deg > 5).sum()} | {g.residual.median():.4f} |")
    bad = pert[pert.recovery_err_deg > 5]
    lines += [f"\nFalliti nella robustezza (> 5 gradi): {len(bad)} su {len(pert)}; residuo min dei falliti "
              f"{bad.residual.min() if len(bad) else float('nan'):.4f}, sopra la soglia: {int((bad.residual > T).sum())}.",
              f"\nMesh come sono con angolo dalla convenzione >= {OK_DEG} gradi: {int((~ok).sum())} su {len(base)}.",
              f"\nFaccia media GNM (held-out, original) contro faccia media ICT: {json.dumps(extra['gnm_mean_vs_ict_mean'])}.",
              f"\n## Soglia di fallimento dichiarata\n",
              f"T = 2 x p99 del residuo trimmed delle canonicalizzazioni corrette = 2 x "
              f"{np.percentile(base.loc[ok, 'residual'], 99):.4f} = **{T:.4f}** (unita': raggio RMS della faccia media).",
              f"Tempo per mesh (calibrazione, 1 thread, 8 start): mediana {df.seconds.median():.2f} s, p95 "
              f"{np.percentile(df.seconds, 95):.2f} s."]
    (a.dir / "calibration.md").write_text("\n".join(lines) + "\n")
    (a.dir / "threshold.json").write_text(json.dumps({"T": T, "rule": "2 x p99 residuo trimmed, calibrazione corretta",
                                                      "p99": float(np.percentile(base.loc[ok, "residual"], 99)),
                                                      "ict_original_area_over_rms2_median":
                                                          extra["ict_original_area_over_rms2_median"]}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
