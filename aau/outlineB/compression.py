#!/usr/bin/env python3
"""Compressione delle distanze fra soggetti diversi da parte della registrazione, con CI.

Tabella di ``tab:distance_compression`` (paper) rifatta nella forma corretta del commit
0281a3f.  Il paper divide gli IQR ASSOLUTI di rigid ICP / NICP per quello di Chamfer raw
("NICP P2P retains only 21.9%"), ma le pipeline registrate passano da ``prealign_by_bbox``,
che riscala la sorgente al raggio del bersaglio, e Chamfer no: le colonne non sono nelle
stesse unita', e una riscalatura non cambia nessun rango.  La statistica invariante e'
l'IQR relativo, IQR/mediana: con quella NICP P2P trattiene il 33.9%, non il 21.9%.
Qui si calcolano entrambe, piu' il CV (std/media, ``ddof=0`` come
``analyze_distance_compression.summarize``), con quantili a interpolazione lineare come
quella funzione.

Per ogni set di soggetti e gruppo di coppie di topologie:

    same          original->original                       4950 coppie
    tessellation  12 coppie ordinate di sola tassellazione    59400
    perturbation  8 coppie ordinate con noisy                39600
    nocrop        le 20 della colonna cross no-crop           99000
    all_cross     le 30 cross, crop compreso                148500  <- l'insieme del paper

CI al 95% dal bootstrap per soggetto dello stesso schema di
``compute_bootstrap_ci.weighted_bootstrap_spearman``: si ricampionano i soggetti, ogni coppia
pesa ``count[a] * count[b]``, e il rapporto con Chamfer e' calcolato nella STESSA replica,
quindi il CI e' quello del rapporto appaiato.

Gate (``bfm_fb100_s2048``, gruppo ``all_cross``): devono tornare i numeri dell'artefatto del paper,
0.537 / 0.219 / 0.236 (IQR assoluto) e 0.681 / 0.339 / 0.408 (IQR relativo).

  aau/run.sh aau/outlineB/compression.py --runs aau/runs/outlineB
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent / "baselines"))

import common  # noqa: E402
import rank_from_matrix as rank  # noqa: E402

# sottodirectory di --runs (quelle di alignment.sbatch) -> set di soggetti.  ``_s2048`` e' il
# campionamento a 2048 punti con cui e' stato prodotto l'artefatto del paper (job 1054872):
# il gate si fa li'; senza suffisso, 4096 punti come original->original e la Tabella 2 estesa.
SETS = {"bfm_fb100": "facebench_first100", "bfm_heldout": "heldout", "ict_heldout": "ict_heldout",
        "bfm_fb100_s2048": "facebench_first100", "bfm_heldout_s2048": "heldout",
        "ict_heldout_s2048": "ict_heldout"}
GATE_SET = "bfm_fb100_s2048"
GROUPS = {
    "same": "original_to_original",
    "tessellation": "tessellation_cross_topology",
    "perturbation": "perturbation_cross_topology",
    "nocrop": "nocrop_cross_topology",
    "all_cross": "all_cross_topology",
}
BASE = "chamfer"
REGISTERED = ("rigid_icp_chamfer", "nicp_p2p", "nicp_p2tri")
LABEL = {"chamfer": "Raw Chamfer", "rigid_icp_chamfer": "Rigid ICP + Chamfer",
         "nicp_p2p": "Rigid ICP + NICP + P2P", "nicp_p2tri": "Rigid ICP + NICP + P2Tri"}

ARTIFACT_DIR = common.REPO_ROOT / "faceBench" / "latentVSpipeline" / "outputs" / "distance_compression_clean_100subj_norm"
SCALE_INVARIANT_CSV = common.REPO_ROOT / "v2_work" / "xdomain" / "gt_matrices" / "compression_scale_invariant.csv"
ARTIFACT_NAME = {"raw_chamfer": "chamfer", "rigid_p2p": "rigid_icp_chamfer",
                 "nicp_p2p": "nicp_p2p", "nicp_p2tri": "nicp_p2tri"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", type=Path, default=common.AAU_DIR / "runs" / "outlineB")
    p.add_argument("--sets", type=str, default=",".join(SETS))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def stats(values: np.ndarray) -> dict[str, float]:
    p25, p50, p75 = np.percentile(values, [25, 50, 75])
    mean = float(values.mean())
    std = float(values.std())
    return {"median": float(p50), "iqr": float(p75 - p25), "rel_iqr": float((p75 - p25) / p50),
            "cv": std / mean}


def load_group(set_dir: Path, setting: str, subjects: list[str]):
    """Indici dei soggetti (a, b) e valori di ogni metrica, concatenati sulle topologie."""
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    a, b, cols = [], [], {m: [] for m in (BASE,) + REGISTERED}
    for topology_a, topology_b in common.setting_topology_pairs(setting):
        for metric in cols:
            D, matrix_subjects, *_ = common.load_matrix(
                common.matrix_path(metric, topology_a, topology_b, set_dir))
            if matrix_subjects != subjects:
                raise ValueError(f"{set_dir}: soggetti diversi fra le matrici")
            cols[metric].append(D[pair_i, pair_j])
        a.append(pair_i)
        b.append(pair_j)
    values = {m: np.concatenate(v) for m, v in cols.items()}
    finite = np.all([np.isfinite(v) for v in values.values()], axis=0)
    return (np.concatenate(a)[finite], np.concatenate(b)[finite],
            {m: v[finite] for m, v in values.items()}, int((~finite).sum()))


def bootstrap(a, b, values, n_subjects, n_bootstrap, rng) -> dict[tuple[str, str], np.ndarray]:
    """Repliche delle tre statistiche e dei rapporti con Chamfer, appaiati per replica."""
    out: dict[tuple[str, str], list[float]] = {}
    for _ in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n_subjects, size=n_subjects), minlength=n_subjects)
        weights = counts[a] * counts[b]
        keep = weights > 0
        rep = {m: stats(np.repeat(v[keep], weights[keep])) for m, v in values.items()}
        for metric, s in rep.items():
            for key, value in s.items():
                out.setdefault((metric, key), []).append(value)
            for key in ("iqr", "rel_iqr", "cv"):
                out.setdefault((metric, f"{key}_ratio"), []).append(s[key] / rep[BASE][key])
    return {k: np.asarray(v) for k, v in out.items()}


def paper_reference() -> dict[str, dict[str, float]]:
    """IQR assoluto e relativo dell'artefatto del paper (fb100, 30 coppie cross)."""
    ref: dict[str, dict[str, float]] = {}
    with open(ARTIFACT_DIR / "distance_distribution_overall.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["group_type"] == "overall" and row["metric"] in ARTIFACT_NAME:
                ref.setdefault(ARTIFACT_NAME[row["metric"]], {})["iqr_ratio"] = float(row["iqr_vs_raw"])
                ref[ARTIFACT_NAME[row["metric"]]]["iqr"] = float(row["iqr"])
    with open(SCALE_INVARIANT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            ref[ARTIFACT_NAME[row["metric"]]]["rel_iqr_ratio"] = float(row["rel_spread_over_raw"])
    return ref


def per_topology_check(set_dir: Path, subjects: list[str]) -> float:
    """Max errore relativo sugli IQR per coppia di topologie contro l'artefatto (solo fb100)."""
    pair_i, pair_j = common.subject_pair_indices(len(subjects))
    worst = 0.0
    with open(ARTIFACT_DIR / "distance_distribution_summary.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["group_type"] != "topology_pair" or row["metric"] not in ARTIFACT_NAME:
                continue
            topology_a, topology_b = row["group"].split("__to__")
            D, *_ = common.load_matrix(common.matrix_path(ARTIFACT_NAME[row["metric"]],
                                                          topology_a, topology_b, set_dir))
            ours = stats(D[pair_i, pair_j])["iqr"]
            worst = max(worst, abs(ours / float(row["iqr"]) - 1.0))
    return worst


def fmt(row: dict, key: str, pct: bool = True) -> str:
    k = 100.0 if pct else 1.0
    v, lo, hi = row[key], row[f"{key}_ci_low"], row[f"{key}_ci_high"]
    if not math.isfinite(v):
        return "--"
    return f"{k * v:.1f} [{k * lo:.1f}, {k * hi:.1f}]" if pct else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    args = parse_args()
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    rows, gate_lines = [], []
    for set_name in sets:
        set_dir = args.runs / set_name
        subjects = [str(s) for s in common.load_matrix(
            common.matrix_path(BASE, "original", "original", set_dir))[1]]
        for group, setting in GROUPS.items():
            a, b, values, n_dropped = load_group(set_dir, setting, subjects)
            rng = np.random.default_rng(rank.stable_seed(args.seed, set_name, group, "compression"))
            boot = bootstrap(a, b, values, len(subjects), args.n_bootstrap, rng)
            point = {m: stats(v) for m, v in values.items()}
            for metric in (BASE,) + REGISTERED:
                row = {"set": set_name, "subject_set": SETS[set_name], "group": group,
                       "setting": setting, "metric": metric, "n_pairs": len(a),
                       "n_dropped_nonfinite": n_dropped, "n_subjects": len(subjects),
                       "n_bootstrap": args.n_bootstrap}
                for key, value in point[metric].items():
                    row[key] = value
                    row[f"{key}_ci_low"], row[f"{key}_ci_high"] = np.percentile(boot[(metric, key)], [2.5, 97.5])
                for key in ("iqr", "rel_iqr", "cv"):
                    row[f"{key}_ratio"] = point[metric][key] / point[BASE][key]
                    row[f"{key}_ratio_ci_low"], row[f"{key}_ratio_ci_high"] = np.percentile(
                        boot[(metric, f"{key}_ratio")], [2.5, 97.5])
                row["median_ratio"] = point[metric]["median"] / point[BASE]["median"]
                rows.append({k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()})
            print(f"[compr] {set_name:<12} {group:<13} n={len(a):>6} " + "  ".join(
                f"{m}: iqr {point[m]['iqr'] / point[BASE]['iqr']:.3f} rel {point[m]['rel_iqr'] / point[BASE]['rel_iqr']:.3f}"
                for m in REGISTERED), flush=True)

        if set_name == GATE_SET and "all_cross" in GROUPS:
            ref = paper_reference()
            for metric in REGISTERED:
                ours = next(r for r in rows if r["set"] == set_name and r["group"] == "all_cross"
                            and r["metric"] == metric)
                for key in ("iqr_ratio", "rel_iqr_ratio"):
                    delta = ours[key] - ref[metric][key]
                    gate_lines.append(f"  [{'OK' if abs(delta) < 5e-3 else 'DIVERSO'}] {metric} {key}: "
                                      f"qui {ours[key]:.4f} vs artefatto {ref[metric][key]:.4f} "
                                      f"(delta {delta:+.4f})")
            gate_lines.append(f"  max errore relativo sugli IQR per coppia di topologie "
                              f"(30 coppie x 4 metriche): {per_topology_check(set_dir, subjects):.2e}")

    out = args.runs
    keys = list(rows[0])
    with open(out / "compression.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out / "compression.json").write_text(json.dumps({"rows": rows, "gate": gate_lines}, indent=2))

    def table(key: str) -> list[dict]:
        body = []
        for set_name in sets:
            for group in GROUPS:
                cells = {m: next(r for r in rows if r["set"] == set_name and r["group"] == group
                                 and r["metric"] == m) for m in REGISTERED}
                body.append((set_name, group, cells[REGISTERED[0]]["n_pairs"],
                             [fmt(cells[m], key) for m in REGISTERED]))
        return body

    md = ["# Compressione delle distanze fra soggetti diversi (OUTLINE_B §4, esperimento 2)", "",
          "Percentuale trattenuta rispetto a Chamfer raw, CI 95% bootstrap per soggetto "
          f"({args.n_bootstrap} repliche, rapporto appaiato nella stessa replica).", "",
          "## IQR relativo (IQR/mediana): invariante per scala, la forma corretta (0281a3f)", "",
          "| set | gruppo | coppie | " + " | ".join(LABEL[m] for m in REGISTERED) + " |",
          "|---|---|---|---|---|---|"]
    md += [f"| {s} | {g} | {n} | " + " | ".join(c) + " |" for s, g, n, c in table("rel_iqr_ratio")]
    md += ["", "## IQR assoluto (la forma del paper, gonfiata dalla riscalatura di prealign_by_bbox)", "",
           "| set | gruppo | coppie | " + " | ".join(LABEL[m] for m in REGISTERED) + " |",
           "|---|---|---|---|---|---|"]
    md += [f"| {s} | {g} | {n} | " + " | ".join(c) + " |" for s, g, n, c in table("iqr_ratio")]
    if gate_lines:
        md += ["", f"## Gate su {GATE_SET} (30 coppie cross a 2048 punti, l'insieme dell'artefatto del paper)", "",
               "```", *gate_lines, "```"]
    (out / "compression.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"% IQR relativo (IQR/mediana) trattenuto rispetto a Chamfer raw, in %, con CI 95%",
           r"% bootstrap per soggetto. Generato da aau/outlineB/compression.py.",
           r"\begin{tabular}{llc" + "c" * len(REGISTERED) + "}", r"\toprule",
           "Set & Pairs & $n$ & " + " & ".join(LABEL[m] for m in REGISTERED) + r" \\", r"\midrule"]
    tex += [f"{s.replace('_', ' ')} & {g.replace('_', ' ')} & {n} & " + " & ".join(c) + r" \\"
            for s, g, n, c in table("rel_iqr_ratio")]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (out / "compression.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    if gate_lines:
        print("\n[compr] gate contro l'artefatto del paper:")
        print("\n".join(gate_lines))
    print(f"\n[compr] {out / 'compression.csv'}\n[compr] {out / 'compression.md'}")


if __name__ == "__main__":
    main()
