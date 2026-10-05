#!/usr/bin/env python3
"""Le 6 celle diagonali (same-topology) della Tabella 1 del paper, per Chamfer e latent.

``tab:clean_xtopo_chamfer_latent_matrices`` (e i CI in ``tab:xtopo_chamfer_ci_app`` /
``tab:xtopo_latent_ci_app``) ha la diagonale prodotta con un protocollo che lo stage
``eval_topology`` **non** copre: Spearman vs D_GT sulle 4950 coppie di soggetti i<j con
entrambe le mesh nella STESSA topologia, sui 100 soggetti ``facebench_first100``.  Le
celle fuori diagonale restano quelle dell'eval del repo.

Qui non si ricalcola niente a mano: le matrici 100x100 arrivano da
``chamfer_matrix.py --variant facebench`` (metrica ``chamfer``, quella che ha prodotto
la diagonale, vedi il suo docstring) e da ``human_study/latent_matrix.py`` (metrica
``latent_v1``, checkpoint v1 e operatori standard), e i 12 valori con CI da
``rank_from_matrix.compute_rows`` su un setting per topologia (``same_<topologia>``),
cioe' dallo stesso bootstrap subject-level di ``scripts/compute_bootstrap_ci.py``.

  aau/run.sh aau/baselines/table1_diagonal.py --subject-set facebench_first100 \
      --out-root aau/runs/baselines_fb100
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import rank_from_matrix as rank  # noqa: E402

# Diagonale pubblicata con i CI dell'appendice (paper/main_short.tex,
# tab:xtopo_chamfer_ci_app e tab:xtopo_latent_ci_app).
PAPER_DIAGONAL = {
    ("chamfer", "crop"): (0.558, 0.480, 0.632),
    ("chamfer", "down8k"): (0.743, 0.687, 0.792),
    ("chamfer", "noisy"): (0.739, 0.674, 0.794),
    ("chamfer", "original"): (0.729, 0.667, 0.790),
    ("chamfer", "remesh"): (0.709, 0.647, 0.763),
    ("chamfer", "up60k"): (0.735, 0.674, 0.791),
    ("latent_v1", "crop"): (0.831, 0.789, 0.866),
    ("latent_v1", "down8k"): (0.883, 0.847, 0.909),
    ("latent_v1", "noisy"): (0.900, 0.874, 0.923),
    ("latent_v1", "original"): (0.902, 0.873, 0.925),
    ("latent_v1", "remesh"): (0.873, 0.838, 0.900),
    ("latent_v1", "up60k"): (0.890, 0.860, 0.917),
}

PAPER_LABEL = {"chamfer": "Raw Chamfer", "latent_v1": "Latent distance"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--metrics", type=str, default="chamfer,latent_v1")
    p.add_argument("--topologies", type=str, default=",".join(common.TOPOLOGIES))
    p.add_argument("--subject-set", type=str, default="facebench_first100",
                   choices=common.SUBJECT_SETS,
                   help="la diagonale del paper e' su facebench_first100, come la Tabella 2")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--strict", action="store_true",
                   help="esce 1 se una matrice same-topology richiesta manca")
    return p.parse_args()


def make_latex_table(results: pd.DataFrame, metrics: list[str], topologies: list[str]) -> str:
    """Topologia x {metrica riprodotta, valore pubblicato} per le celle diagonali."""
    header = " & ".join(f"{PAPER_LABEL.get(m, m)} (qui) & paper" for m in metrics)
    lines = [
        r"% diagonale (same-topology) della Tabella 1: Spearman vs D_GT con CI bootstrap",
        r"% subject-level al 95%, 1000 repliche, soggetti facebench_first100.",
        r"\begin{tabular}{l" + "cc" * len(metrics) + "}",
        r"\toprule",
        f"Topology & {header} " + r"\\",
        r"\midrule",
    ]
    for topology in topologies:
        cells = []
        for metric in metrics:
            row = results[
                results["metric"].eq(metric)
                & results["setting"].eq(f"same_{topology}")
                & results["correlation"].eq("spearman")
            ]
            cells.append(rank.fmt_interval(row.iloc[0]) if len(row) else "--")
            reference = PAPER_DIAGONAL.get((metric, topology))
            cells.append("--" if reference is None else
                         f"{reference[0]:.3f} [{reference[1]:.3f}, {reference[2]:.3f}]")
        lines.append(f"{topology} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    bootstrap_module = rank.load_bootstrap_module()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    unknown = [t for t in topologies if t not in common.TOPOLOGIES]
    if unknown:
        raise SystemExit(f"topologie sconosciute: {unknown} (attese {common.TOPOLOGIES})")
    print(f"[t1diag] metriche={metrics} topologie={topologies} soggetti={args.subject_set} "
          f"bootstrap={args.n_bootstrap} seed={args.seed}", flush=True)

    rows = []
    deltas = []
    for metric in metrics:
        for topology in topologies:
            new_rows = rank.compute_rows(metric, f"same_{topology}", args, bootstrap_module)
            for row in new_rows:
                if row["correlation"] != "spearman":
                    continue
                reference = PAPER_DIAGONAL.get((metric, topology))
                if reference is None:
                    print(f"[t1diag] {metric:<10} {topology:<9} spearman={row['value']:.4f} "
                          f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] (nessun riferimento)",
                          flush=True)
                    continue
                delta = row["value"] - reference[0]
                inside = reference[1] <= row["value"] <= reference[2]
                deltas.append((abs(delta), metric, topology))
                print(f"[t1diag] {metric:<10} {topology:<9} spearman={row['value']:.4f} "
                      f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] vs paper {reference[0]:.3f} "
                      f"[{reference[1]:.3f}, {reference[2]:.3f}] delta {delta:+.4f} "
                      f"[{'OK' if inside else 'FUORI CI'}]", flush=True)
            rows.extend(new_rows)

    if not rows:
        raise SystemExit("[t1diag] nessun risultato: mancano le matrici same-topology")

    # I riferimenti pubblicati finiscono anche nel CSV/JSON, cosi' il delta e' ricontrollabile.
    for row in rows:
        reference = PAPER_DIAGONAL.get((row["metric"], row["setting"][len("same_"):]))
        row["paper_value"] = math.nan if reference is None else reference[0]
        row["paper_ci_low"] = math.nan if reference is None else reference[1]
        row["paper_ci_high"] = math.nan if reference is None else reference[2]
        row["delta_vs_paper"] = (math.nan if reference is None
                                 else row["value"] - reference[0])
    results = pd.DataFrame(rows)

    out_dir = args.out_root / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table1_diagonal_{args.subject_set}"
    results.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.tex").write_text(
        make_latex_table(results, metrics, topologies) + "\n", encoding="utf-8")
    (out_dir / f"{stem}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if deltas:
        worst = max(deltas)
        print(f"\n[t1diag] max |delta| vs paper = {worst[0]:.4f} ({worst[1]} {worst[2]})")
    print(f"[t1diag] CSV   {out_dir / f'{stem}.csv'}")
    print(f"[t1diag] LaTeX {out_dir / f'{stem}.tex'}")


if __name__ == "__main__":
    main()
