#!/usr/bin/env python3
"""Figura di H1 (FaMoS come confine di dominio): AUC di verifica e rank-1 in funzione dell'intensita' dell'espressione,
per metodo. Matplotlib: container.

    aau/run.sh aau/runs/evidence/famos_dominio/fig_famos_expr.py        # su un nodo di calcolo (fd.sbatch)

Fonti qui accanto: ``strata.csv`` (righe ``metrica``, insiemi ``expr:L|M|H``) e ``summary.json`` (terzili di
``expr_mm``), entrambi di ``aau/famos_dominio/fd.py analyze``. AUC_k: coppie della stessa persona dello strato k
(della scansione piu' espressiva) contro tutte le coppie di persone diverse; rank-1_k: query dello strato, galleria
completa. Barre = IC 95% per soggetto (15 soggetti). Uscita ``fig_famos_expr.pdf`` qui accanto. Stile delle figure
del paper (``make_figures.py``) ricopiato: niente import da ``paper/``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
COL_W = 3.25  # larghezza di una colonna CVPR in pollici
# (colonna, etichetta, colore, marcatore, linea)
METHODS = [("factorized_s1234|form", "ours (seed 1)", "#5b2c83", "o", "-"),
           ("factorized_s2345|form", "ours (seed 2)", "#9e7cc1", "o", "--"),
           ("icp|mm", "rigid ICP (mm)", "#1f77b4", "s", "-"),
           ("icp|cs", "rigid ICP (size-norm.)", "#1f77b4", "s", ":"),
           ("now_metrical|mm", "NoW metrical", "#d95f02", "^", "-"),
           ("ulrssm|pair_fr", "ULRSSM (mm)", "#1b9e77", "v", "-")]
STRATA = ("L", "M", "H")


def main() -> None:
    with open(HERE / "strata.csv", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["riga"] == "metrica"]
    t1, t2 = json.loads((HERE / "summary.json").read_text())["strati"]["expr"]["tagli"]
    ticks = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.7), sharex=True)
    for ax, met, ylab in zip(axes, ("AUC", "rank1"), ("verification AUC", "rank-1 (queries in stratum)")):
        for j, (key, label, color, marker, ls) in enumerate(METHODS):
            pts = [next(r for r in rows if r["colonna"] == key and r["insieme"] == f"expr:{s}" and r["metrica"] == met)
                   for s in STRATA]
            x = [k + 0.08 * (j - (len(METHODS) - 1) / 2) for k in range(len(STRATA))]
            y = [float(p["valore"]) for p in pts]
            err = [[y[k] - float(p["ic_lo"]) for k, p in enumerate(pts)], [float(p["ic_hi"]) - y[k]
                                                                           for k, p in enumerate(pts)]]
            ax.errorbar(x, y, yerr=err, fmt=marker, ls=ls, ms=2.5, lw=0.7, elinewidth=0.5, capsize=0, color=color,
                        mfc="white" if ls != "-" else color, label=label)
        ax.set_xticks(range(len(STRATA)), ticks)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", lw=0.3, alpha=0.5)
    fig.supxlabel(f"expression intensity, registration vs. neutral (low $\\leq${t1:.1f} mm, high >{t2:.1f} mm;"
                  " AUC: pair maximum, rank-1: query)", fontsize=6, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), frameon=False,
               handlelength=1.8, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(HERE / "fig_famos_expr.pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"[fig] {HERE / 'fig_famos_expr.pdf'}", flush=True)


if __name__ == "__main__":
    main()
