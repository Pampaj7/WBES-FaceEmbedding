#!/usr/bin/env python3
"""Tabella dei frame (OUTLINE_B §5, esperimento 3): Chamfer e D_GT, held-out BFM.

Righe: Chamfer nei frame maxabs / area / rms / global (``frame_matrix.py``); colonne:
stessa topologia (original->original), sola tassellazione (12 coppie), perturbazione noisy
(8), crop (10 coppie con crop da un lato).  Sotto, D_GT ricalcolato nei frame raw /
maxabs / area / rms / global contro il D_GT del repo (grezzo): e' lo Spearman fra le due
matrici, sulle stesse 4950 coppie, con lo stesso bootstrap per soggetto.

Sotto la riga global, il 2x2 che la spezza (``pm_translation``: centroide per mesh e scala
globale; ``pm_scale``: centro globale e scala maxabs per mesh) e il controllo ``bbox_global``
(centro del bbox + diagonale nel frame globale, 4 numeri).  Poi lo stesso su held-out ICT
(maxabs / global / pm_translation / pm_scale / bbox) contro la D_GT ICT GREZZA
(``alignment_table.ict_raw_gt``): quella di train_ready e' maxabs.

Due controlli prima della tabella, stampati e scritti nel json:
  - ``chamfer_maxabs`` deve coincidere con la Chamfer della Tabella 2 estesa
    (``aau/runs/baselines/matrices/chamfer``) e con quella della pipeline faceBench
    (``bfm_heldout/matrices/chamfer``): stessi punti, stesso frame; su ICT con quella della
    pipeline faceBench (``ict_heldout/matrices/chamfer``);
  - ``gt_raw`` deve dare Spearman 1 contro il D_GT del repo (su ICT, contro la grezza).

  aau/run.sh aau/outlineB/frame_table.py --out-root aau/runs/outlineB/bfm_heldout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402
import rank_from_matrix as rank  # noqa: E402

FRAMES = ("maxabs", "area", "rms", "global", "pm_translation", "pm_scale")
GT_FRAMES = ("raw", "maxabs", "area", "rms", "global")
ICT_FRAMES = ("maxabs", "global", "pm_translation", "pm_scale")
ICT_GT_FRAMES = ("raw", "maxabs", "global")
SETTINGS = ("original_to_original", "tessellation_cross_topology",
            "perturbation_cross_topology", "crop_cross_topology")
HEAD = {"original_to_original": "Same topology", "tessellation_cross_topology": "Tessellation",
        "perturbation_cross_topology": "Perturbation (noisy)", "crop_cross_topology": "Crop"}
FRAME_LABEL = {"raw": "raw", "maxabs": "maxabs (per mesh)", "area": "area (per mesh)",
               "rms": "rms (per mesh)", "global": "global (one similarity)",
               "pm_translation": "global scale, per-mesh centroid",
               "pm_scale": "global centre, per-mesh scale (maxabs divisor)"}
BBOX_LABEL = "bbox control (global frame, 4 numbers)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.AAU_DIR / "runs" / "outlineB" / "bfm_heldout")
    p.add_argument("--ict-root", type=Path, default=common.AAU_DIR / "runs" / "outlineB" / "ict_heldout")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def same_matrix_check(out_root: Path, references) -> list[str]:
    """chamfer_maxabs contro le altre Chamfer maxabs dello stesso protocollo."""
    lines = []
    for label, root, metric in references:
        worst, n = 0.0, 0
        for topology_a, topology_b in common.all_topology_pairs(("original_to_original", "all_cross_topology")):
            other = common.matrix_path(metric, topology_a, topology_b, root)
            if not other.exists():
                continue
            A = common.load_matrix(common.matrix_path("chamfer_maxabs", topology_a, topology_b, out_root))[0]
            B = common.load_matrix(other)[0]
            mask = np.isfinite(A) & np.isfinite(B)
            worst = max(worst, float(np.max(np.abs(A[mask] - B[mask]) / B[mask])))
            n += 1
        lines.append(f"  chamfer_maxabs vs {label} ({root}): {n} coppie di topologie, "
                     f"max errore relativo {worst:.2e}")
    return lines


def main() -> None:
    args = parse_args()
    bootstrap_module = rank.load_bootstrap_module()
    ns = argparse.Namespace(out_root=args.out_root, subject_set="heldout",
                            n_bootstrap=args.n_bootstrap, seed=args.seed, strict=True)
    checks = same_matrix_check(args.out_root, (("Tabella 2 estesa", common.OUT_ROOT, "chamfer"),
                                               ("pipeline faceBench", args.out_root, "chamfer")))
    checks += same_matrix_check(args.ict_root, (("pipeline faceBench ICT", args.ict_root, "chamfer"),))
    print("[frame-tab] controlli:\n" + "\n".join(checks), flush=True)

    rows = []
    jobs = [(f"chamfer_{f}", s) for f in FRAMES for s in SETTINGS]
    jobs += [("bbox_global", s) for s in SETTINGS]
    jobs += [(f"gt_{f}", "original_to_original") for f in GT_FRAMES]
    ict_jobs = [(f"chamfer_{f}", s) for f in ICT_FRAMES for s in SETTINGS]
    ict_jobs += [("bbox_global", s) for s in SETTINGS]
    ict_jobs += [(f"gt_{f}", "original_to_original") for f in ICT_GT_FRAMES]
    ns_ict = argparse.Namespace(out_root=args.ict_root, subject_set="ict_heldout",
                                n_bootstrap=args.n_bootstrap, seed=args.seed, strict=True)
    for block_ns, block_jobs in ((ns, jobs), (ns_ict, ict_jobs)):
        if block_ns is ns_ict:
            # Come alignment_table: common legge MESH_ROOT e GT_MATRIX al momento della chiamata.
            from alignment_table import ICT_MESH, ict_raw_gt

            common.MESH_ROOT = ICT_MESH
            common.GT_MATRIX = ict_raw_gt(args.ict_root)
        for metric, setting in block_jobs:
            for row in rank.compute_rows(metric, setting, block_ns, bootstrap_module):
                rows.append(row)
                if row["correlation"] == "spearman":
                    print(f"[frame-tab] {block_ns.subject_set:<11} {metric:<22} {setting:<28} "
                          f"{row['value']:.4f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                          f"n_pairs={row['n_pairs']}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.out_root.parent / "frames.csv", index=False)
    (args.out_root.parent / "frames.json").write_text(json.dumps({"rows": rows, "checks": checks}, indent=2))

    def cell(metric: str, setting: str, subject_set: str = "heldout") -> str:
        r = results[results["metric"].eq(metric) & results["setting"].eq(setting)
                    & results["subject_set"].eq(subject_set) & results["correlation"].eq("spearman")]
        return rank.fmt_interval(r.iloc[0]) if len(r) else "--"

    md = ["# Tabella dei frame, held-out BFM (OUTLINE_B §5, esperimento 3)", "",
          "Spearman vs D_GT del repo (grezzo), CI 95% bootstrap per soggetto (1000 repliche).", "",
          "## Chamfer (variante faceBench, 4096 punti) per frame", "",
          "| frame | " + " | ".join(HEAD[s] for s in SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]
    md += [f"| {FRAME_LABEL[f]} | " + " | ".join(cell(f"chamfer_{f}", s) for s in SETTINGS) + " |"
           for f in FRAMES]
    md += [f"| {BBOX_LABEL} | " + " | ".join(cell("bbox_global", s) for s in SETTINGS) + " |"]
    md += ["", "Le ultime tre righe spezzano global: \"global scale, per-mesh centroid\" toglie a "
           "ogni mesh la propria media dei vertici e divide per lo s0 globale (via la posizione, "
           "resta la taglia); \"global centre, per-mesh scale\" sottrae il c0 globale e divide "
           "per il divisore maxabs della mesh (via la taglia, resta la posizione rispetto a c0, "
           "riscalata); il controllo bbox e' la distanza euclidea fra (centro del bounding box, "
           "diagonale) nel frame globale, senza forma."]
    md += ["", "## D_GT ricalcolato nel frame vs D_GT del repo (original, 4950 coppie)", "",
           "| frame della GT | Spearman con D_GT grezzo |", "|---|---|"]
    md += [f"| {FRAME_LABEL[f]} | {cell(f'gt_{f}', 'original_to_original')} |" for f in GT_FRAMES]
    md += ["", "## Held-out ICT (seed 1234), contro la D_GT ICT grezza", "",
           "Stessi frame e stesso codice, sulle mesh grezze di `datasets/ICT/topo` (vista "
           "`ict_heldout/raw_view`: quelle di eval_view_heldout sono gia' maxabs per mesh); c0, s0 "
           "stimati sulle altre 400 identita' del pool. D_GT: "
           "`datasets/ICT/gt/ict_matrix_distances_raw.npz` (quella di train_ready e' maxabs).", "",
           "| frame | " + " | ".join(HEAD[s] for s in SETTINGS) + " |", "|---|" + "---|" * len(SETTINGS)]
    md += [f"| {FRAME_LABEL[f]} | " + " | ".join(cell(f"chamfer_{f}", s, "ict_heldout") for s in SETTINGS)
           + " |" for f in ICT_FRAMES]
    md += [f"| {BBOX_LABEL} | " + " | ".join(cell("bbox_global", s, "ict_heldout") for s in SETTINGS) + " |"]
    md += ["", "| frame della GT ICT | Spearman con D_GT ICT grezzo |", "|---|---|"]
    md += [f"| {FRAME_LABEL[f]} | {cell(f'gt_{f}', 'original_to_original', 'ict_heldout')} |"
           for f in ICT_GT_FRAMES]
    md += ["", "## Controlli", "", "```", *checks, "```"]
    (args.out_root.parent / "frames.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"% Tabella dei frame, held-out BFM. Spearman vs D_GT (grezzo), CI 95% bootstrap per",
           r"% soggetto. Generato da aau/outlineB/frame_table.py.",
           r"\begin{tabular}{l" + "c" * len(SETTINGS) + "}", r"\toprule",
           "Frame & " + " & ".join(HEAD[s] for s in SETTINGS) + r" \\", r"\midrule",
           r"\multicolumn{" + str(len(SETTINGS) + 1) + r"}{l}{\emph{Chamfer}} \\"]
    tex += [f"{FRAME_LABEL[f]} & " + " & ".join(cell(f"chamfer_{f}", s) for s in SETTINGS) + r" \\"
            for f in FRAMES]
    tex += [f"{BBOX_LABEL} & " + " & ".join(cell("bbox_global", s) for s in SETTINGS) + r" \\"]
    tex += [r"\midrule", r"\multicolumn{" + str(len(SETTINGS) + 1)
            + r"}{l}{\emph{$D_{\mathrm{GT}}$ in the frame vs raw $D_{\mathrm{GT}}$}} \\"]
    tex += [f"{FRAME_LABEL[f]} & {cell(f'gt_{f}', 'original_to_original')} & & & " + r"\\"
            for f in GT_FRAMES]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.out_root.parent / "frames.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"\n[frame-tab] {args.out_root.parent / 'frames.md'}")


if __name__ == "__main__":
    main()
