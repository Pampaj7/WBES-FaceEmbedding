#!/usr/bin/env python3
"""Figure 3-6 del paper CVPR, dai CSV di evidenza. Matplotlib: gira nel container (sbatch make_figs.sbatch).

    aau/run.sh paper/figs/scripts/make_figures.py          # su un nodo di calcolo

Uscite in ``paper/figs/``: ``fig_scale.pdf``, ``fig_perturbation.pdf``, ``fig_fewshot.pdf``, ``fig_size.pdf``, ognuna
con ``<nome>.sources.txt`` accanto (file:riga di ogni numero disegnato, e lo stato: DATI, FALLBACK, PLACEHOLDER).

- fig_scale (Fig. 3, scala dell'allineamento): con ``full_matrix/paired.csv`` disegna i delta appaiati contro ICP
  rigido in mm, con IC. Senza, FALLBACK: punti e IC NON appaiati dalle tabelle esistenti (righe diverse fra la
  Chamfer centrata / NICP per coppia in mm e il resto), con un timbro rosso sulla figura.
- fig_perturbation (Fig. 4): con ``full_matrix/perturbation.csv`` disegna lo Spearman per perturbazione e metodo.
  Senza, PLACEHOLDER: per i concorrenti non esiste nessuna scomposizione per perturbazione con FR/SR.
- fig_fewshot (Fig. 5, D5): ``diagnostics/d5_delta.csv``, variante EQ preregistrata, media dei 4 bersagli.
- fig_size (Fig. 6): CV della taglia e Spearman FR-SR della GT (``baselines_param/controls_e2.json``), Spearman con FR
  della sola taglia oracolo, di ICP rigido in mm e dei nostri (``baselines_param/spearman_e2.csv``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EV, FIGS, GROUP, find, full_matrix, json_line, read_csv, rel  # noqa: E402

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})
COL_W = 3.25  # larghezza di una colonna CVPR in pollici

E2 = EV / "baselines_param" / "spearman_e2.csv"
MM = EV / "baselines_mm" / "spearman.csv"
D5 = EV / "diagnostics" / "d5_delta.csv"
CTRL = EV / "baselines_param" / "controls_e2.json"

DOMS = [("hifi3d", "HIFI3D"), ("facescape", "FaceScape"), ("faceverse", "FaceVerse"), ("famos", "FaMoS")]


def stamp(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=7, color="red", alpha=0.55,
            rotation=12, fontweight="bold", bbox=dict(facecolor="white", edgecolor="red", alpha=0.6))


def save(fig, name: str, status: str, sources: list[str]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    if os.environ.get("FIG_PREVIEW"):  # anteprima PNG fuori dal repo
        fig.savefig(Path(os.environ["FIG_PREVIEW"]) / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (FIGS / f"{name}.sources.txt").write_text(f"stato: {status}\n" + "\n".join(sources) + "\n")
    print(f"scritto paper/figs/{name}.pdf ({status}, {len(sources)} fonti)")


def e2_point(rows, domain, gt, arm, distance):
    hit = find(rows, domain=domain, group=GROUP[domain], gt=gt, kind="rho", arm=arm, distance=distance, baseline="-")
    i, r = hit
    return (float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"])), f"{rel(E2)}:{i}"


def mm_point(rows, domain, gt, method):
    hit = find(rows, domain=domain, group=GROUP[domain], gt=gt, method=method)
    i, r = hit
    return (float(r["point"]), float(r["ci_low"]), float(r["ci_high"])), f"{rel(MM)}:{i}"


# ---------------------------------------------------------------------------------------------------------------
# Fig. 3: scala dell'allineamento

LADDER = [  # (etichetta, sorgente, chiave, nome nella matrice, colore, gruppo)
    ("Chamfer, centred", "mm", "mm_chamfer", "mm_chamfer", "#999999", "pair"),
    ("rigid ICP (pose)", "e2", ("baseline", "ICP + Chamfer in mm"), "mm_rigid_icp_chamfer", "#1f77b4", "pair"),
    ("+ size normalization", "e2", ("baseline", "ICP + Chamfer (cs)"), "cs_rigid_icp_chamfer", "#9ecae1", "pair"),
    ("+ non-rigid (pairwise NICP)", "mm", "mm_nicp_p2tri", "mm_nicp_p2tri", "#d62728", "pair"),
    ("+ size norm. + pairwise NICP", "e2", ("baseline", "NICP per coppia (cs)"), "cs_nicp_p2tri", "#ff9896", "pair"),
    ("NICP to a template", "e2", ("baseline", "NICP su template in mm"), "mm_nicp_template", "#2ca02c", "mesh"),
    ("FLAME fit (loop), id. mesh", "e2", ("baseline", "FLAME 2023 Open vB, mesh d'identita' FR"), "flame2023_vb_fr",
     "#98df8a", "mesh"),
    ("Ours, $d_F$ uncal.", "e2", ("factorized_s1234", "form"), "factorized_s1234|form", "#9467bd", "mesh"),
    ("Ours, $d_F$ cal. (post hoc)", "e2", ("factorized_s1234", "form_cal"), "factorized_s1234|form_cal", "#5b2c83",
     "mesh"),
]


def fig_scale() -> None:
    paired = full_matrix("paired.csv")
    e2, mm = read_csv(E2), read_csv(MM)
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    src = []
    n = len(LADDER)
    w = 0.8 / n
    for j, (label, kind, key, fm_name, color, grp) in enumerate(LADDER):
        xs, ys, lo, hi = [], [], [], []
        for k, (d, _) in enumerate(DOMS):
            if paired:
                if fm_name == "mm_rigid_icp_chamfer":
                    continue
                hit = find(paired, domain=d, group=GROUP[d], gt="fr", method=fm_name, ref="mm_rigid_icp_chamfer")
                if hit is None:
                    continue
                v = (float(hit[1]["delta"]), float(hit[1]["ci_low"]), float(hit[1]["ci_high"]))
                s = f"aau/runs/evidence/full_matrix/paired.csv:{hit[0]}"
            else:
                try:
                    v, s = mm_point(mm, d, "fr", key) if kind == "mm" else e2_point(e2, d, "fr", *key)
                except TypeError:  # cella inesistente (NICP su template su FaMoS)
                    continue
            xs.append(k - 0.4 + w * (j + 0.5))
            ys.append(v[0]); lo.append(v[0] - v[1]); hi.append(v[2] - v[0])
            src.append(f"{label} | {d}: {s}")
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o" if grp == "pair" else "s", ms=2.6, lw=0.7, color=color,
                    label=label, capsize=0)
    ax.set_xticks(range(len(DOMS)))
    ax.set_xticklabels([n for _, n in DOMS])
    if paired:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(r"$\Delta\rho$ vs. rigid ICP (mm), FR")
        status = "DATI (matrice unificata)"
    else:
        ax.set_ylabel(r"Spearman $\rho$ with FR")
        stamp(ax, "FALLBACK: unpaired existing numbers\nawaiting unified matrix")
        status = "FALLBACK (numeri esistenti, non appaiati; manca aau/runs/evidence/full_matrix/paired.csv)"
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3,
              columnspacing=0.8)
    save(fig, "fig_scale", status, src)


# ---------------------------------------------------------------------------------------------------------------
# Fig. 4: robustezza per perturbazione

PERTS = ["remesh", "down8k", "up60k", "noisy", "crop"]


def fig_perturbation() -> None:
    pert = full_matrix("perturbation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.6), sharey=True)
    src = []
    if pert:
        methods = []
        for _, r in pert:
            if r["method"] not in methods:
                methods.append(r["method"])
        for ax, gt in zip(axes, ("fr", "sr")):
            for j, m in enumerate(methods):
                xs, ys, lo, hi = [], [], [], []
                for k, p in enumerate(PERTS):
                    hit = find(pert, domain="hifi3d", gt=gt, method=m, perturbation=p)
                    if hit is None:
                        continue
                    r = hit[1]
                    xs.append(k + (j - len(methods) / 2) * 0.08); ys.append(float(r["point"]))
                    lo.append(float(r["point"]) - float(r["ci_low"])); hi.append(float(r["ci_high"]) - float(r["point"]))
                    src.append(f"{m} | {gt} | {p}: aau/runs/evidence/full_matrix/perturbation.csv:{hit[0]}")
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=2, lw=0.6, label=m)
            ax.set_xticks(range(len(PERTS))); ax.set_xticklabels(PERTS, rotation=30)
            ax.set_title(f"HIFI3D, {gt.upper()}")
        axes[0].set_ylabel(r"Spearman $\rho$")
        axes[1].legend(frameon=False, fontsize=5)
        status = "DATI (matrice unificata)"
    else:
        for ax, gt in zip(axes, ("FR", "SR")):
            ax.set_xticks(range(len(PERTS))); ax.set_xticklabels(PERTS, rotation=30)
            ax.set_xlim(-0.5, len(PERTS) - 0.5); ax.set_ylim(0, 1); ax.set_title(f"HIFI3D, {gt}")
        axes[0].set_ylabel(r"Spearman $\rho$")
        fig.text(0.55, 0.5, "PLACEHOLDER\nper-perturbation results of all methods\nawait the unified matrix",
                 ha="center", va="center", color="red", fontsize=7, fontweight="bold",
                 bbox=dict(facecolor="white", edgecolor="red"))
        status = "PLACEHOLDER (manca aau/runs/evidence/full_matrix/perturbation.csv)"
    save(fig, "fig_perturbation", status, src)


# ---------------------------------------------------------------------------------------------------------------
# Fig. 5: adattamento few-shot (D5)

def fig_fewshot() -> None:
    rows = read_csv(D5)
    ks = [0, 5, 10, 20, 50, 100, 200]
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    src = []
    styles = {"factorized_s1234": ("#5b2c83", "seed 1"), "factorized_s2345": ("#d95f02", "seed 2")}
    for arm, (color, name) in styles.items():
        for gt, what, ls, lab in (("sr", "SR, mean of 4 targets", "-", ""), ("fr", "FR (constraint)", ":", "")):
            ys, lo, hi = [], [], []
            for k in ks:
                hit = find(rows, arm=arm, target="mean4", k=str(k), variant="eq", what="vs_dP", gt=gt)
                i, r = hit
                ys.append(float(r["delta"])); lo.append(float(r["ci_low"])); hi.append(float(r["ci_high"]))
                src.append(f"{arm} | {gt} | k={k}: {rel(D5)}:{i}")
            ax.plot(ks, ys, ls, color=color, lw=1.0, marker="o" if gt == "sr" else None, ms=2.2,
                    label=f"{name}, {what}")
            if gt == "sr":
                ax.fill_between(ks, lo, hi, color=color, alpha=0.15, lw=0)
    ax.axhline(0.05, color="k", lw=0.6, ls="--")
    ax.text(200, 0.052, "pre-registered threshold", ha="right", va="bottom", fontsize=5.5)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.set_xlim(0, 215)
    ax.set_xlabel("labelled identities of the new generator, $k$")
    ax.set_ylabel(r"$\Delta\rho$ vs. $d_P$")
    ax.legend(frameon=False, loc="upper left", fontsize=5.5)
    save(fig, "fig_fewshot", "DATI", src)


# ---------------------------------------------------------------------------------------------------------------
# Fig. 6: taglia

SIZE_DOMS = [("hifi3d", "HIFI3D"), ("famos", "FaMoS"), ("facescape", "FaceScape"), ("faceverse", "FaceVerse")]


def fig_size() -> None:
    e2 = read_csv(E2)
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.55), gridspec_kw={"width_ratios": [1, 1.35]})
    src = []
    for d, name in SIZE_DOMS:
        l1, cv = json_line(CTRL, "S_cv", after=('"gt": {', f'"{d}": {{'))
        l2, rho = json_line(CTRL, "rho_fr_sr_gt", after=('"gt": {', f'"{d}": {{'))
        src.append(f"CV taglia {d}: {rel(CTRL)}:{l1}; rho FR-SR: {rel(CTRL)}:{l2}")
        a.plot(100 * cv, rho, "o", ms=3, color="#5b2c83")
        a.annotate(name, (100 * cv, rho), xytext=(3, -2 if d != "faceverse" else 2), textcoords="offset points",
                   fontsize=5.5)
    a.set_xlabel("CV of centroid size (%)")
    a.set_ylabel(r"$\rho$(FR, SR)")
    a.set_xlim(0.8, 5.6); a.set_ylim(0.55, 1.0)
    a.set_title("(a) size variation vs. FR/SR", loc="left")
    bars = [("Size only (oracle)", ("baseline", "taglia oracolo"), "#bbbbbb"),
            ("Rigid ICP (mm)", ("baseline", "ICP + Chamfer in mm"), "#1f77b4"),
            ("Ours $d_F$ cal.", ("factorized_s1234", "form_cal"), "#5b2c83")]
    w = 0.26
    for j, (lab, key, color) in enumerate(bars):
        for k, (d, _) in enumerate(SIZE_DOMS):
            v, s = e2_point(e2, d, "fr", *key)
            b.bar(k + (j - 1) * w, v[0], w, color=color, label=lab if k == 0 else None)
            b.errorbar(k + (j - 1) * w, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color="k", lw=0.5, capsize=0)
            src.append(f"{lab} | {d}: {s}")
    b.set_xticks(range(len(SIZE_DOMS))); b.set_xticklabels([n for _, n in SIZE_DOMS], rotation=20)
    b.set_ylabel(r"$\rho$ with FR")
    b.set_ylim(0, 1.2); b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    b.set_title("(b) FR ranking", loc="left")
    b.legend(frameon=False, fontsize=4.8, loc="upper center", bbox_to_anchor=(0.58, 1.0), ncol=3, handlelength=0.8, columnspacing=0.6, handletextpad=0.3)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_size", "DATI", src)


def main() -> None:
    fig_scale()
    fig_perturbation()
    fig_fewshot()
    fig_size()


if __name__ == "__main__":
    main()
