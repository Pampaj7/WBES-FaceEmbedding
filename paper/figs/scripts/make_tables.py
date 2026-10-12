#!/usr/bin/env python3
"""Tabelle del paper CVPR (main e supplementary) generate dai CSV di evidenza, con la fonte di ogni numero.

    python3 paper/figs/scripts/make_tables.py          # dalla radice del repo, anche sul frontend (solo stdlib)

Uscite in ``paper/tables/*.tex`` (incluse con \\input). Ogni riga di tabella e' seguita da un commento
``% fonte: file:riga, ...`` con una voce per cella, nell'ordine delle celle (``-`` = cella vuota o \\todo).

Fonti:
- ``aau/runs/evidence/baselines_param/spearman_e2.csv``: righe senza crop con la maschera COMUNE per dominio (stesse
  righe e repliche per tutti i metodi di una colonna; HIFI3D 98.224, FaceScape 88.725, FaceVerse 99.000, FaMoS 105).
  Per i fit 3DMM la tabella principale riporta la lettura DICHIARATA (protocollo, sez. 7): coefficienti con entrambe
  le GT, mesh d'identita' ``_fr`` con FR e ``_sr`` con SR; NON il massimo fra le letture (la bozza precedente lo
  faceva). Le letture non corrispondenti stanno nelle tabelle complete del supplementary.
- ``aau/runs/evidence/baselines_mm/spearman.csv``: Chamfer centrata e NICP per coppia in mm (non stanno in
  spearman_e2): 99.000 righe senza maschera comune, altro insieme di righe (segnato con il pugnale).
- ``aau/runs/evidence/baselines_param/spearman_e3.csv``: crop (righe col crop, tutte le coppie).
- ``aau/runs/evidence/baselines_param/topo_pairs_e3.csv``: per coppia di topologie (solo i nostri bracci).
- ``aau/runs/evidence/diagnostics/d5_delta.csv``: adattamento few-shot (D5).
- ``aau/runs/evidence/full_matrix/*.csv`` (se esiste): delta appaiati della scala dell'allineamento.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (EV, GROUP, REPO, TABLES, ci_cell, find, full_matrix, num, read_csv, rel,  # noqa: E402
                    write)

E2 = EV / "baselines_param" / "spearman_e2.csv"
E3 = EV / "baselines_param" / "spearman_e3.csv"
MM = EV / "baselines_mm" / "spearman.csv"
TOPO = EV / "baselines_param" / "topo_pairs_e3.csv"
D5 = EV / "diagnostics" / "d5_delta.csv"

HEADER = ("% GENERATO da paper/figs/scripts/make_tables.py: non modificare a mano, rigenerare.\n"
          "% Ogni riga di tabella e' seguita da '% fonte:' con file:riga di ogni cella, nell'ordine.\n")


# ---------------------------------------------------------------------------------------------------------------
# lettura

class Src:
    """Cache dei CSV e risoluzione di una cella in (testo LaTeX, fonte)."""

    def __init__(self):
        self.cache = {}

    def rows(self, path: Path):
        if path not in self.cache:
            self.cache[path] = read_csv(path)
        return self.cache[path]

    def e2(self, domain: str, gt: str, arm: str, distance: str, ci: bool = True):
        if arm == PENDING:  # concorrente in corso: nessun numero ancora (12 ottobre 2026)
            return r"\todo{run}", "-"
        hit = find(self.rows(E2), domain=domain, group=GROUP[domain], gt=gt, kind="rho", arm=arm, distance=distance,
                   baseline="-")
        if hit is None:
            return "--", "-"
        i, r = hit
        p, lo, hi = float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"])
        return (ci_cell(p, lo, hi) if ci else num(p)), f"{rel(E2)}:{i}"

    def mm(self, domain: str, gt: str, method: str, ci: bool = True):
        hit = find(self.rows(MM), domain=domain, group=GROUP[domain] if domain != "faceverse" else "mesh_pair_nocrop",
                   gt=gt, method=method)
        if hit is None:
            return "--", "-"
        i, r = hit
        p, lo, hi = float(r["point"]), float(r["ci_low"]), float(r["ci_high"])
        return (ci_cell(p, lo, hi) if ci else num(p)), f"{rel(MM)}:{i}"

    def e3(self, domain: str, group: str, gt: str, column: str, region: str):
        hit = find(self.rows(E3), domain=domain, group=group, gt=gt, kind="rho", column=column, region=region, ref="-")
        if hit is None:
            return "--", "-"
        i, r = hit
        return ci_cell(float(r["point"]), float(r["ci_low"]), float(r["ci_high"])), f"{rel(E3)}:{i}"


S = Src()


def row(label: str, cells: list[tuple[str, str]]) -> str:
    txt = " & ".join([label] + [c for c, _ in cells]) + r" \\"
    src = ", ".join(s for _, s in cells)
    return f"{txt}\n% fonte: {src}\n"


def tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------------------------------------------------------------------------------------------------------------
# tabella principale: FR / SR senza crop, 4 domini, lettura dichiarata

B = "baseline"
PENDING = "in_corso"  # concorrenti lanciati da altri agenti, risultati non ancora nel repo
FIT = {  # etichetta interna dei fit nei CSV
    "gnm": "GNM (visto)", "flame": "FLAME 2023 Open",
}


def fit(model: str, variant: str, reading: str) -> str:
    v = {"orig": "", "vB": " vB"}[variant]
    return f"{FIT[model]}{v}, {reading}"


MAIN_ROWS = [
    # (blocco, etichetta, (arm, distance) per FR, (arm, distance) per SR); None = lettura non dichiarata
    ("Pairwise alignment (per pair of meshes)", [
        ("Rigid ICP + Chamfer (mm)", (B, "ICP + Chamfer in mm"), (B, "ICP + Chamfer in mm")),
        ("Size-normalized rigid ICP + Chamfer", (B, "ICP + Chamfer (cs)"), (B, "ICP + Chamfer (cs)")),
        ("Size-normalized pairwise NICP", (B, "NICP per coppia (cs)"), (B, "NICP per coppia (cs)")),
        ("Learned correspondence, ULRSSM (retrained), per pair", (PENDING, ""), (PENDING, "")),
        ("Learned correspondence, Hybrid FM (retrained), per pair", (PENDING, ""), (PENDING, "")),
        ("Gromov--Wasserstein (no correspondence)", (PENDING, ""), (PENDING, "")),
    ]),
    ("Per-mesh representation in a common space", [
        ("NICP to a template (mm)", (B, "NICP su template in mm"), (B, "NICP su template in mm")),
        ("ULRSSM (retrained), via a template", (PENDING, ""), (PENDING, "")),
        ("Hybrid FM (retrained), via a template", (PENDING, ""), (PENDING, "")),
        ("Varifold (mm, unit mass)", (B, "varifold in mm (massa unitaria)"), (B, "varifold in mm (massa unitaria)")),
        ("Size only, estimated from the mesh", (B, "taglia stimata"), (B, "taglia stimata")),
        ("Size only, oracle (control)", (B, "taglia oracolo"), (B, "taglia oracolo")),
    ]),
    ("3DMM fit, pre-registered (GNM seen in training, FLAME not)", [
        ("GNM, coefficients", (B, fit("gnm", "orig", "coefficienti")), (B, fit("gnm", "orig", "coefficienti"))),
        ("GNM, identity mesh", (B, fit("gnm", "orig", "mesh d'identita' FR")),
         (B, fit("gnm", "orig", "mesh d'identita' SR"))),
        ("FLAME, coefficients", (B, fit("flame", "orig", "coefficienti")), (B, fit("flame", "orig", "coefficienti"))),
        ("FLAME, identity mesh", (B, fit("flame", "orig", "mesh d'identita' FR")),
         (B, fit("flame", "orig", "mesh d'identita' SR"))),
    ]),
    ("3DMM fit, model in the loop (post hoc)", [
        ("GNM, coefficients", (B, fit("gnm", "vB", "coefficienti")), (B, fit("gnm", "vB", "coefficienti"))),
        ("GNM, identity mesh", (B, fit("gnm", "vB", "mesh d'identita' FR")), (B, fit("gnm", "vB", "mesh d'identita' SR"))),
        ("GNM, size + shape", (B, fit("gnm", "vB", "composizione S_B + k d_P")), None),
        ("FLAME, coefficients", (B, fit("flame", "vB", "coefficienti")), (B, fit("flame", "vB", "coefficienti"))),
        ("FLAME, identity mesh", (B, fit("flame", "vB", "mesh d'identita' FR")),
         (B, fit("flame", "vB", "mesh d'identita' SR"))),
        ("FLAME, size + shape", (B, fit("flame", "vB", "composizione S_B + k d_P")), None),
    ]),
    ("Ours, registration-free (FR: $d_F$; SR: $d_P$)", [
        ("$d_F$ calibrated (post hoc), seed 1", ("factorized_s1234", "form_cal"), ("factorized_s1234", "shape")),
        ("$d_F$ calibrated (post hoc), seed 2", ("factorized_s2345", "form_cal"), ("factorized_s2345", "shape")),
        ("$d_F$ uncalibrated (pre-registered), seed 1", ("factorized_s1234", "form"), None),
        ("$d_F$ uncalibrated (pre-registered), seed 2", ("factorized_s2345", "form"), None),
    ]),
]
MAIN_DOMAINS = ["hifi3d", "facescape", "faceverse", "famos"]


def tab_main() -> str:
    out = [HEADER, "% Tabella principale: Spearman con FR e SR, coppie senza crop fra topologie diverse.\n",
           r"\begin{tabular}{l" + "cc" * len(MAIN_DOMAINS) + "}\n", "\\toprule\n",
           " & " + " & ".join([r"\multicolumn{2}{c}{\textbf{HIFI3D} (primary)}",
                               r"\multicolumn{2}{c}{\textbf{FaceScape} (dev.)}",
                               r"\multicolumn{2}{c}{\textbf{FaceVerse} (expr.)}",
                               r"\multicolumn{2}{c}{\textbf{FaMoS} (15 subj.)}"]) + r" \\" + "\n",
           "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(MAIN_DOMAINS))) + "\n",
           "Method" + " & FR & SR" * len(MAIN_DOMAINS) + r" \\" + "\n"]
    for block, rows in MAIN_ROWS:
        out.append("\\midrule\n" + f"\\multicolumn{{{1 + 2 * len(MAIN_DOMAINS)}}}{{l}}{{\\emph{{{block}}}}} \\\\\n")
        for label, fr, sr in rows:
            cells = []
            for d in MAIN_DOMAINS:
                cells.append(S.e2(d, "fr", *fr) if fr else ("--", "-"))
                cells.append(S.e2(d, "sr", *sr) if sr else ("--", "-"))
            out.append(row(label, cells))
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# tabella della scala dell'allineamento (main): FR, punti; delta appaiati dalla matrice unificata

SCALE_ROWS = [
    # (blocco, etichetta, fonte, chiave, nome nella matrice unificata)
    ("Pairwise", [
        ("Chamfer, centred, no alignment$^\\dagger$", "mm", "mm_chamfer", "mm_chamfer"),
        ("+ rigid ICP (pose)", "e2", "ICP + Chamfer in mm", "mm_rigid_icp_chamfer"),
        ("+ size normalization", "e2", "ICP + Chamfer (cs)", "cs_rigid_icp_chamfer"),
        ("+ non-rigid ICP$^\\dagger$", "mm", "mm_nicp_p2tri", "mm_nicp_p2tri"),
        ("+ size norm. + non-rigid ICP", "e2", "NICP per coppia (cs)", "cs_nicp_p2tri"),
    ]),
    ("Per mesh", [
        ("NICP to a template (mm)", "e2", "NICP su template in mm", "mm_nicp_template"),
        ("FLAME fit (in the loop), identity mesh", "e2", fit("flame", "vB", "mesh d'identita' FR"), "flame2023_vb_fr"),
        ("GNM fit (in the loop), identity mesh", "e2", fit("gnm", "vB", "mesh d'identita' FR"), "gnm_vb_fr"),
        ("Ours, $d_F$ uncalibrated, seed 1", "e2arm", ("factorized_s1234", "form"), "factorized_s1234|form"),
        ("Ours, $d_F$ calibrated, seed 1", "e2arm", ("factorized_s1234", "form_cal"), "factorized_s1234|form_cal"),
    ]),
]


def tab_scale() -> str:
    paired = full_matrix("paired.csv")
    doms = MAIN_DOMAINS
    out = [HEADER, "% Scala dell'allineamento, GT FR, punti senza crop. Delta contro ICP rigido in mm: matrice unificata"
                   " (aau/runs/evidence/full_matrix/paired.csv)" + ("" if paired else ", NON ANCORA DISPONIBILE") + ".\n",
           r"\begin{tabular}{lccccc}" + "\n\\toprule\n",
           r"Distance & HIFI3D & FaceSc. & FaceV. & FaMoS & $\Delta$ HIFI3D \\" + "\n"]
    for block, rows in SCALE_ROWS:
        out.append(f"\\midrule\n\\multicolumn{{6}}{{l}}{{\\emph{{{block}}}}} \\\\\n")
        for label, src, key, fm_name in rows:
            cells = []
            for d in doms:
                if src == "mm":
                    cells.append(S.mm(d, "fr", key, ci=False))
                elif src == "e2":
                    cells.append(S.e2(d, "fr", B, key, ci=False))
                else:
                    cells.append(S.e2(d, "fr", *key, ci=False))
            if fm_name == "mm_rigid_icp_chamfer":
                cells.append(("ref.", "-"))
            elif paired:
                hit = find(paired, domain="hifi3d", group="nocrop_cross", gt="fr", method=fm_name,
                           ref="mm_rigid_icp_chamfer")
                cells.append((ci_cell(float(hit[1]["delta"]), float(hit[1]["ci_low"]), float(hit[1]["ci_high"]), True),
                              f"aau/runs/evidence/full_matrix/paired.csv:{hit[0]}") if hit else ("--", "-"))
            else:
                cells.append((r"\todo{matrice}", "-"))
            out.append(row(label, cells))
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: tabelle complete senza crop

LABELS = {
    "ICP + Chamfer in mm": "Rigid ICP + Chamfer (mm)",
    "ICP + Chamfer (cs)": "Size-normalized rigid ICP + Chamfer",
    "NICP per coppia (cs)": "Size-normalized pairwise NICP",
    "NICP su template in mm": "NICP to a template (mm)",
    "varifold in mm (massa unitaria)": "Varifold (mm, unit mass)",
    "taglia stimata": "Size only, estimated",
    "taglia oracolo": "Size only, oracle",
    "e108": "Legacy size-blind encoder (v1, e108)",
    "taglia stimata + ICP cs cal.": "Estimated size + size-norm.\\ ICP, calibrated (post hoc)",
    "taglia stimata + NICP cs cal.": "Estimated size + size-norm.\\ NICP, calibrated (post hoc)",
    "oracolo taglia + ICP cs cal.": "Oracle size + size-norm.\\ ICP, calibrated (ceiling)",
    "taglia stimata + ICP cs (em. 4, non cal.)": "Estimated size + size-norm.\\ ICP, uncalibrated",
    "oracolo taglia + ICP cs (em. 4, non cal.)": "Oracle size + size-norm.\\ ICP, uncalibrated",
}


def fit_label(name: str) -> str:
    model = "GNM" if name.startswith("GNM") else "FLAME"
    var = ("in the loop, $\\sigma$ sens." if "sigma sens." in name else "in the loop" if " vB" in name
           else "no expression" if " vA" in name else "pre-registered")
    reading = ("coefficients" if "coefficienti" in name else "identity mesh (FR)" if "FR" in name
               else "identity mesh (SR)" if "SR" in name else "size + shape" if "composizione" in name else name)
    return f"{model} fit, {var}, {reading}"


ARM_LABELS = {
    "factorized_s1234": "Ours seed 1", "factorized_s2345": "Ours seed 2",
    "factorized2_s1234": "Two-head variant seed 1", "factorized2_s2345": "Two-head variant seed 2",
    "factorizedc3m_e123": "Ours, $10\\times$ identities, ep.\\ 123", "factorizedc3m_e205": "Ours, $10\\times$ identities, ep.\\ 205",
    "ctrlfr_s1234": "Control: single embedding on FR, seed 1", "ctrlfr_s2345": "Control: single embedding on FR, seed 2",
    "dual_s1234": "Dual-head variant seed 1", "dual_s2345": "Dual-head variant seed 2",
}
DIST_LABELS = {"form_cal": "$d_F$ cal.", "form": "$d_F$ uncal.", "shape": "$d_P$", "form_cal_ls": "$d_F$ cal.\\ (LS)",
               "z": "$\\|z\\|$ dist.", "u": "$u$", "zf": "$z_f$"}


PENDING_SUPP = [
    "ULRSSM (retrained), per pair", "ULRSSM (retrained), via a template",
    "Hybrid FM (retrained), per pair", "Hybrid FM (retrained), via a template",
    "Gromov--Wasserstein (POT)",
    "DINOv2 on renders", "CLIP on renders", "LPIPS on renders", "DreamSim on renders",
]


def supp_full(gt: str) -> str:
    doms = ["hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos"]
    keys = []
    for i, r in S.rows(E2):
        if r["kind"] == "rho" and r["group"] == GROUP.get(r["domain"]) and r["gt"] == gt and r["baseline"] == "-":
            k = (r["arm"], r["distance"])
            if k not in keys:
                keys.append(k)
    out = [HEADER, f"% Supplementary: tutte le righe senza crop di spearman_e2.csv, GT {gt.upper()}.\n",
           r"\begin{tabular}{lccccc}" + "\n\\toprule\n",
           r"Method & HIFI3D & FaceScape & FaceVerse & FaceVerse neutral & FaMoS \\" + "\n\\midrule\n"]
    for arm, dist in keys:
        if arm == B:
            label = LABELS.get(dist) or (fit_label(dist) if ("GNM" in dist or "FLAME" in dist) else tex_escape(dist))
        else:
            label = f"{ARM_LABELS.get(arm, tex_escape(arm))}, {DIST_LABELS.get(dist, tex_escape(dist))}"
        out.append(row(label, [S.e2(d, gt, arm, dist) for d in doms]))
    out.append("\\midrule\n")
    for label in PENDING_SUPP:
        out.append(row(label, [S.e2(d, gt, PENDING, "") for d in doms]))
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: crop

CROP_COLS = {
    "fr": [("factorized_s1234|form_cal", "intero", "Ours seed 1, $d_F$ cal."),
           ("factorized_s2345|form_cal", "intero", "Ours seed 2, $d_F$ cal."),
           ("factorizedc3m_e205|form_cal", "intero", "Ours $10\\times$ id., $d_F$ cal."),
           ("ctrlfr_s1234|z", "intero", "Control (FR embedding) seed 1"),
           ("mm_rigid_icp_chamfer", "-", "Rigid ICP + Chamfer (mm)"),
           ("mm_nicp_template", "-", "NICP to a template (mm)"),
           ("gnm_vb_coef", "-", "GNM fit (in the loop), coefficients"),
           ("gnm_vb_fr", "-", "GNM fit (in the loop), identity mesh (FR)"),
           ("flame2023_vb_coef", "-", "FLAME fit (in the loop), coefficients"),
           ("flame2023_vb_fr", "-", "FLAME fit (in the loop), identity mesh (FR)"),
           ("mm_chamfer_pure", "-", "Uncentred Chamfer (mm), diagnostic only"),
           ("oracle_size", "-", "Size only, oracle")],
    "sr": [("factorized_s1234|shape", "intero", "Ours seed 1, $d_P$"),
           ("factorized_s2345|shape", "intero", "Ours seed 2, $d_P$"),
           ("factorizedc3m_e205|shape", "intero", "Ours $10\\times$ id., $d_P$"),
           ("mm_rigid_icp_chamfer", "-", "Rigid ICP + Chamfer (mm)"),
           ("mm_nicp_template", "-", "NICP to a template (mm)"),
           ("gnm_vb_coef", "-", "GNM fit (in the loop), coefficients"),
           ("gnm_vb_sr", "-", "GNM fit (in the loop), identity mesh (SR)"),
           ("flame2023_vb_coef", "-", "FLAME fit (in the loop), coefficients"),
           ("flame2023_vb_sr", "-", "FLAME fit (in the loop), identity mesh (SR)"),
           ("mm_chamfer_pure", "-", "Uncentred Chamfer (mm), diagnostic only"),
           ("oracle_size", "-", "Size only, oracle")],
}


def supp_crop(gt: str) -> str:
    doms = ["hifi3d", "facescape", "faceverse", "faceverse_neutral"]
    out = [HEADER, f"% Supplementary: crop, GT {gt.upper()} (spearman_e3.csv; righe e semi diversi dalla tabella principale).\n",
           r"\begin{tabular}{l" + "ccc" * len(doms) + "}\n\\toprule\n",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{n}}}}}" for n in
                              ["HIFI3D", "FaceScape", "FaceVerse", "FaceVerse neutral"]) + r" \\" + "\n",
           "".join(f"\\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(doms))) + "\n",
           "Method" + " & no crop & all & crop rows" * len(doms) + r" \\" + "\n\\midrule\n"]
    for col, region, label in CROP_COLS[gt]:
        cells = []
        for d in doms:
            for grp in ("senza crop", "all_cross", "righe col crop"):
                cells.append(S.e3(d, grp, gt, col, region))
        out.append(row(label, cells))
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: per coppia di topologie (nostri bracci)

def supp_topo() -> str:
    rows = S.rows(TOPO)
    out = [HEADER, "% Supplementary: media sulle 20 coppie ordinate di topologie diverse e sulle 5 coppie della stessa"
                   " topologia, coppia peggiore e migliore (topo_pairs_e3.csv).\n",
           r"\begin{tabular}{llcccc}" + "\n\\toprule\n",
           r"Domain & Distance (GT) & Cross-topology mean & Same-topology mean & Worst cross pair & Best cross pair \\"
           + "\n\\midrule\n"]
    methods = [("factorized_s1234|form_cal", "fr", "Ours seed 1, $d_F$ cal.\\ (FR)"),
               ("factorized_s1234|shape", "sr", "Ours seed 1, $d_P$ (SR)"),
               ("factorized_s2345|form_cal", "fr", "Ours seed 2, $d_F$ cal.\\ (FR)"),
               ("factorized_s2345|shape", "sr", "Ours seed 2, $d_P$ (SR)")]
    for dom, name in [("hifi3d", "HIFI3D"), ("facescape", "FaceScape"), ("faceverse", "FaceVerse")]:
        for m, gt, label in methods:
            a = find(rows, domain=dom, gt=gt, method=m, quantity="media 20 senza crop")
            b = find(rows, domain=dom, gt=gt, method=m, quantity="media 5 stessa topologia")
            if a is None or b is None:
                continue
            ra, rb = a[1], b[1]
            arrow = "$\\to$"
            mn = num(float(ra["min"])) + " (" + ra["min_pair"].replace("|", arrow) + ")"
            mx = num(float(ra["max"])) + " (" + ra["max_pair"].replace("|", arrow) + ")"
            cells = [(label, "-"),
                     (ci_cell(float(ra["point"]), float(ra["ci_low"]), float(ra["ci_high"])), f"{rel(TOPO)}:{a[0]}"),
                     (ci_cell(float(rb["point"]), float(rb["ci_low"]), float(rb["ci_high"])), f"{rel(TOPO)}:{b[0]}"),
                     (mn, f"{rel(TOPO)}:{a[0]}"), (mx, f"{rel(TOPO)}:{a[0]}")]
            out.append(row(name, cells))
        out.append("\\midrule\n")
    out[-1] = "\\bottomrule\n\\end{tabular}\n"
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: scala dell'allineamento completa (baselines_mm, tre normalizzazioni)

MM_METHODS = [("mm_chamfer", "Chamfer, centred (mm)"), ("mm_rigid_icp_chamfer", "Rigid ICP + Chamfer (mm)"),
              ("mm_nicp_p2tri", "Pairwise NICP (mm)"), ("mm_nicp_template", "NICP to a template (mm)"),
              ("cs_chamfer", "Chamfer, size-normalized"), ("cs_rigid_icp_chamfer", "Rigid ICP + Chamfer, size-normalized"),
              ("cs_nicp_p2tri", "Pairwise NICP, size-normalized"), ("cs_nicp_template", "NICP to a template, size-normalized"),
              ("maxabs_chamfer", "Chamfer, legacy max-abs normalization"),
              ("maxabs_rigid_icp_chamfer", "Rigid ICP + Chamfer, legacy max-abs"),
              ("maxabs_nicp_p2tri", "Pairwise NICP, legacy max-abs"), ("maxabs_nicp_template", "NICP to a template, legacy max-abs"),
              ("est_cs", "Size only, estimated (centroid size)"), ("oracle_size", "Size only, oracle"),
              ("mm_chamfer_pure", "Uncentred Chamfer (mm), diagnostic only")]


def supp_scale_full() -> str:
    doms = ["hifi3d", "facescape", "faceverse", "famos"]
    out = [HEADER, "% Supplementary: baseline geometriche nelle tre normalizzazioni (baselines_mm/spearman.csv; 99.000"
                   " righe, NICP 98.224 / 88.725 per i fallimenti, senza maschera comune).\n",
           r"\begin{tabular}{l" + "cc" * len(doms) + "}\n\\toprule\n",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{n}}}}}" for n in
                              ["HIFI3D", "FaceScape", "FaceVerse", "FaMoS"]) + r" \\" + "\n",
           "".join(f"\\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(doms))) + "\n",
           "Method" + " & FR & SR" * len(doms) + r" \\" + "\n\\midrule\n"]
    for m, label in MM_METHODS:
        cells = []
        for d in doms:
            cells.append(S.mm(d, "fr", m))
            cells.append(S.mm(d, "sr", m))
        out.append(row(label, cells))
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: D5

def supp_d5() -> str:
    rows = S.rows(D5)
    ks = ["0", "5", "10", "20", "50", "100", "200"]
    # FaceVerse in D5 e' la vista neutra (results_d5.md); FLAME 2023 = flame2023_s1 (suddiviso)
    tnames = [("facescape", "FaceScape"), ("hifi3d", "HIFI3D"), ("faceverse", "FaceVerse neutral"),
              ("flame2023_s1", "FLAME 2023"), ("mean4", "Mean of 4")]
    out = [HEADER, "% Supplementary: D5, guadagno della testa adattata su d_P (SR), variante EQ preregistrata.\n",
           r"\begin{tabular}{ll" + "c" * len(ks) + "}\n\\toprule\n",
           "Seed & Target & " + " & ".join(f"$k={k}$" for k in ks) + r" \\" + "\n\\midrule\n"]
    for arm, sname in [("factorized_s1234", "1"), ("factorized_s2345", "2")]:
        for key, tname in tnames:
            tk = key
            cells = [(tname, "-")]
            for k in ks:
                hit = find(rows, arm=arm, target=tk, k=k, variant="eq", what="vs_dP", gt="sr") if tk else None
                if hit is None:
                    cells.append(("--", "-"))
                else:
                    r = hit[1]
                    cells.append((ci_cell(float(r["delta"]), float(r["ci_low"]), float(r["ci_high"]), True),
                                  f"{rel(D5)}:{hit[0]}"))
            out.append(row(sname, cells))
        out.append("\\midrule\n")
    out[-1] = "\\bottomrule\n\\end{tabular}\n"
    return "".join(out)


# ---------------------------------------------------------------------------------------------------------------
# supplementary: protocolli ed emendamenti

# Stato rispetto ai numeri che il documento regola, letto dalla prima riga di ogni file (verificato a mano il 12
# ottobre 2026): "pre" = scritto prima dei numeri che regola; "post" = scritto dopo (post hoc).
PROTOCOLS = [
    ("trainer_v3/ablation_protocol.md", "Trainer ablations (loss, inputs)", "pre"),
    ("trainer_v3/ablation_protocol_emendamento_2026-10-09.md", "Trainer ablations, amendment", "pre"),
    ("e1/protocol.md", "E1: variety of 3DMMs vs.\\ number of identities", "pre"),
    ("e1/protocol_amendment.md", "E1, amendment 1 (decision rule)", "pre"),
    ("e1/protocol_amendment_2.md", "E1, amendment 2 (calibrated unified reference)", "pre"),
    ("e2/protocol.md", "E2/E3b: rigid canonicalization and uniform remeshing at test", "pre"),
    ("e12/protocol.md", "E12: canonical references (FR, SR); amendment 1 appended", "pre"),
    ("trainer_v3/factorized_protocol.md", "Factorized size-and-shape head", "pre"),
    ("trainer_v3/factorized_protocol_emendamento_1.md", "Factorized, amendment 1 (BFM out of the size loss)", "pre"),
    ("trainer_v3/factorized_protocol_emendamento_2.md", "Factorized, amendment 2 (dropout 0)", "pre"),
    ("trainer_v3/factorized_protocol_emendamento_3.md", "Factorized, amendment 3 (two-head arm)", "pre"),
    ("trainer_v3/factorized_protocol_emendamento_4.md", "Factorized, amendment 4: calibration of $c$, size analysis",
     "post"),
    ("trainer_v3/factorized_protocol_emendamento_5.md", "Factorized, amendment 5: calibrated compositions", "post"),
    ("trainer_v3/ablations/k_ablation/PROTOCOL.md", "Number of Laplacian eigenvectors", "pre"),
    ("trainer_v3/ablations/k_ablation/PROTOCOL_emendamento_1.md", "Eigenvectors, amendment 1 (rule restated)", "post"),
    ("faceverse_neutral/PROTOCOL.md", "FaceVerse neutral view", "pre"),
    ("baselines_param/PROTOCOL.md", "3DMM fits (GNM, FLAME) and varifold", "pre"),
    ("baselines_param/PROTOCOL_emendamento_1.md", "3DMM fits, amendment 1: model in the loop", "post"),
    ("baselines_param/PROTOCOL_emendamento_2.md", "3DMM fits, amendment 2: crops, size + shape, cost, $\\sigma$",
     "post"),
    ("baselines_param/PROTOCOL_emendamento_3.md", "3DMM fits, amendment 3: fit region, crop diagnostics", "post"),
    ("diagnostics/PROTOCOL_D.md", "Diagnostics D1, D2", "pre"),
    ("diagnostics/PROTOCOL_D_emendamento_1.md", "D1, D2, amendment 1", "pre"),
    ("diagnostics/PROTOCOL_D3.md", "Diagnostic D3 (metric head across domains)", "pre"),
    ("diagnostics/PROTOCOL_D3_emendamento_1.md", "D3, amendment 1 (leave one generator out)", "pre"),
    ("diagnostics/PROTOCOL_D3_emendamento_2.md", "D3, amendment 2 (wider $\\lambda$ grid, joint head)", "post"),
    ("diagnostics/PROTOCOL_D4.md", "Diagnostic D4 (canonical remeshing at test)", "pre"),
    ("diagnostics/PROTOCOL_D5.md", "Diagnostic D5 (few-shot adaptation)", "pre"),
    ("diagnostics/PROTOCOL_G0.md", "Gate G0 (ArcFace on renders as a metric)", "pre"),
    ("stream/ablation_2x2/PROTOCOL.md", "Ablation $2\\times2$ (FLAME as 4th generator, partiality)", "pre"),
    ("stream/ablation_2x2/PROTOCOL_emendamento_1.md", "Ablation $2\\times2$, amendment 1", "pre"),
]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def supp_protocols() -> str:
    out = [HEADER, "% Supplementary: protocolli ed emendamenti. sha256 del file attuale (primi 8 caratteri) e confronto"
                   " con l'impronta registrata nel .sha256 accanto, se esiste; data del commit che lo introduce.\n"
                   "% Il commit si stampa solo con \\showcommitstrue (versione finale): in revisione il repository non"
                   " e' anonimo.\n",
           r"\begin{tabular}{llllc}" + "\n\\toprule\n",
           r"Document & Date & sha256 & Commit & Written \\" + "\n\\midrule\n"]
    for f, label, status in PROTOCOLS:
        p = EV / f
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        # impronte registrate: ogni hash a 64 cifre nei .sha256 della cartella sulle righe che nominano il file (E12 ha
        # l'emendamento 1 accodato allo stesso protocol.md, registrato in protocol_amend1.sha256)
        regs = []
        for cand in sorted(p.parent.glob("*.sha256")):
            for line in cand.read_text().splitlines():
                toks = line.split()
                if toks and len(toks[0]) == 64 and (len(toks) == 1 and cand.stem == p.stem or p.name in line):
                    regs.append(toks[0])
        reg = (h if h in regs else regs[-1]) if regs else None
        flag = "" if reg is None else ("" if h in regs else "$^\\ast$")
        log = git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", rel(p)).splitlines()
        commit, date = (log[-1].split() if log else ("?", "?"))
        when = "before results" if status == "pre" else "\\textbf{post hoc}"
        out.append(f"{label} & {date} & \\texttt{{{h[:8]}}}{flag} & \\commit{{{commit}}} & {when} \\\\\n"
                   f"% fonte: {rel(p)} (sha256 {h}; registrato {reg or 'nel messaggio di commit'}; commit {commit})\n")
    out.append("\\bottomrule\n\\end{tabular}\n")
    return "".join(out)


def main() -> None:
    write(TABLES / "tab_main.tex", tab_main())
    write(TABLES / "tab_scale.tex", tab_scale())
    write(TABLES / "supp_full_fr.tex", supp_full("fr"))
    write(TABLES / "supp_full_sr.tex", supp_full("sr"))
    write(TABLES / "supp_crop_fr.tex", supp_crop("fr"))
    write(TABLES / "supp_crop_sr.tex", supp_crop("sr"))
    write(TABLES / "supp_topo.tex", supp_topo())
    write(TABLES / "supp_scale_full.tex", supp_scale_full())
    write(TABLES / "supp_d5.tex", supp_d5())
    write(TABLES / "supp_protocols.tex", supp_protocols())


if __name__ == "__main__":
    main()
