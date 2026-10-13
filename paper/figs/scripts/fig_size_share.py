#!/usr/bin/env python3
"""Figura della quota della taglia: per dominio, quanto pesa la taglia nella GT FR (asse x) contro quanto guadagna,
con FR, un metodo che tiene la taglia rispetto allo stesso metodo che la toglie (asse y). Matplotlib: container.

    aau/run.sh paper/figs/scripts/fig_size_share.py          # su un nodo di calcolo (size_share.sbatch)

Fonti: ``aau/runs/evidence/size_share/gt_size.csv`` (quota della taglia, IC per soggetto) e ``methods.csv`` (delta
ICP mm - ICP cs e NoW metrical - NoW originale, con IC; file:riga d'origine nella colonna ``fonte``), entrambi di
``aau/size_share/size_share.py``. FaMoS TEST (15 soggetti) con marcatori vuoti: descrittivo. Uscita
``paper/figs/fig_size_share.pdf`` con ``fig_size_share.sources.txt`` accanto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EV, find, read_csv, rel  # noqa: E402
from make_figures import COL_W, save  # noqa: E402  (stile e salvataggio delle altre figure)

SS = EV / "size_share"
GT, MET = SS / "gt_size.csv", SS / "methods.csv"
DOMS = [("hifi3d", "HIFI3D"), ("facescape", "FaceScape"), ("faceverse", "FaceVerse"),
        ("faceverse_neutral", "FaceVerse neutral"), ("famos", "FaMoS")]
PAIRS = [("d_icp", "rigid ICP: mm $-$ size-normalized", "#1f77b4", "o", -1),
         ("d_now", "NoW: metrical $-$ original (scaled)", "#d95f02", "s", 1)]
# FaceVerse e la neutra hanno la stessa GT (stessa x): piccolo scarto orizzontale per leggerle
NUDGE = {"faceverse": -0.12, "faceverse_neutral": 0.12}


def main() -> None:
    gt, met = read_csv(GT), read_csv(MET)
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    src = []
    for key, label, color, marker, side in PAIRS:
        for d, name in DOMS:
            i, g = find(gt, dominio=d, grandezza="quota_taglia")
            j, m = find(met, dominio=d, chiave=key)
            x = 100 * float(g["punto"]) + NUDGE.get(d, 0.0) + 0.05 * side
            xl, xh = 100 * float(g["ic_lo"]), 100 * float(g["ic_hi"])
            y, yl, yh = float(m["punto"]), float(m["ic_lo"]), float(m["ic_hi"])
            x0 = x - NUDGE.get(d, 0.0) - 0.05 * side
            ax.errorbar(x, y, xerr=[[x0 - xl], [xh - x0]], yerr=[[y - yl], [yh - y]], fmt=marker, ms=3, lw=0.6,
                        color=color, mfc="white" if d == "famos" else color, capsize=0,
                        label=label if d == "hifi3d" else None)
            src.append(f"{label} | {d}: quota taglia {rel(GT)}:{i}; delta {rel(MET)}:{j} (origine {m['fonte']})")
            if key == "d_icp":
                ax.annotate(name, (x, y), xytext=(3, -7 if d in ("faceverse", "famos") else 3),
                            textcoords="offset points", fontsize=5.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("size term share of FR$^2$, mean$[(S_i-S_j)^2]$ / mean[FR$^2$] (%)")
    ax.set_ylabel(r"$\Delta\rho$ with FR, keeps size $-$ removes it")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="upper left", fontsize=5.5, handletextpad=0.3)
    ax.text(0.99, 0.02, "open markers: FaMoS (15 subjects, descriptive)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=5)
    save(fig, "fig_size_share", "DATI", src)


if __name__ == "__main__":
    main()
