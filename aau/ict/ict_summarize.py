#!/usr/bin/env python3
"""Tabelle di WS2 (cross-3DMM zero-shot su ICT) e WS5 (espressioni) dagli output dei job.

Non calcola nessuna distanza: legge quello che hanno scritto
``compare_model_vs_chamfer_rankings.py`` e ``compare_model_vs_chamfer_topology_breakdown.py``
sotto ``aau/runs/eval_*/ict_zeroshot`` e ``aau/runs/eval_*/ict_expr``, e aggiunge la sola
cosa che quegli script non producono: i CI bootstrap subject-level.

Il bootstrap NON e' riscritto: ``weighted_bootstrap_spearman`` viene importata da
``scripts/compute_bootstrap_ci.py``, la stessa funzione che ha prodotto i CI della
Tabella 2 del paper, e mangia direttamente le ``pair_metrics.csv`` del breakdown, che
hanno le stesse colonne delle pair table del paper (subject_a, subject_b, gt_distance,
latent_distance, raw_chamfer).  Stesso idioma di aau/baselines/rank_from_matrix.py.

  aau/ict/ict_summarize.py                      # scopre le run da sole
  aau/ict/ict_summarize.py --n-bootstrap 200    # piu' veloce, CI piu' grossolani

Output (default ``aau/runs/ict_summary/``):
  table_a_scenarios.csv       seed x scenario, Spearman/Pearson latente e Chamfer
  table_a_topology_pairs.csv  seed x coppia di topologie x metrica, Spearman + CI 95%
  table_b_expressions.csv     espressione x intensita' x regime x metrica
  summary.md                  le stesse tabelle in markdown, con le medie sui seed
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
BOOTSTRAP_SRC = REPO_ROOT / "scripts" / "compute_bootstrap_ci.py"

SEED_RE = re.compile(r"seed(\d+)__")
METRICS = (("latent", "latent_distance"), ("chamfer", "raw_chamfer"))
MODES = ("same_expression", "expression_vs_neutral")

# Riferimento storico: modello v1 addestrato su BFM, valutato zero-shot su FLAME, 100
# soggetti, 148.500 coppie cross-topology, aggregazione mesh_pair (v2_work/STATUS.md:398).
FLAME_ZEROSHOT_SPEARMAN = 0.478


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=AAU_DIR / "runs")
    p.add_argument("--out-dir", type=Path, default=AAU_DIR / "runs" / "ict_summary")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    p.add_argument("--skip-bootstrap", action="store_true", help="Solo i punti, senza CI")
    return p.parse_args()


def load_bootstrap_module():
    """Carica scripts/compute_bootstrap_ci.py come modulo (la dir non e' un package)."""
    if not BOOTSTRAP_SRC.exists():
        raise FileNotFoundError(f"bootstrap del repo non trovato: {BOOTSTRAP_SRC}")
    spec = importlib.util.spec_from_file_location("wbes_compute_bootstrap_ci", BOOTSTRAP_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def model_seed_of(stage_dir: Path) -> str:
    """Il seed del modello sta nel nome della run dir del checkpoint, non nella stage dir."""
    match = SEED_RE.search(stage_dir.parent.name)
    return match.group(1) if match else stage_dir.parent.name


def discover_stages(runs_root: Path, stage: str) -> list[Path]:
    """Le stage dir complete (con .done): un job ucciso a meta' non entra nelle tabelle."""
    found = sorted(p for p in runs_root.glob(f"eval_*/{stage}") if (p / ".done").exists())
    if not found:
        print(f"WARNING: nessuna stage '{stage}' completa sotto {runs_root}", file=sys.stderr)
    return found


# ---------------------------------------------------------------- A: scenari aggregati

def table_a_scenarios(stages: list[Path]) -> pd.DataFrame:
    rows = []
    for stage in stages:
        payload = json.loads((stage / "ranking" / "ranking_summary.json").read_text())
        for row in payload["rows"]:
            rows.append({
                "model_seed": model_seed_of(stage),
                "scenario": row["scenario"],
                "latent_spearman": float(row["latent_spearman"]),
                "chamfer_spearman": float(row["chamfer_spearman"]),
                "delta_spearman": float(row["delta_spearman"]),
                "latent_pearson": float(row["latent_pearson"]),
                "chamfer_pearson": float(row["chamfer_pearson"]),
                "n_subjects": int(row["n_subjects"]),
                "n_subject_pairs": int(row["n_subject_pairs"]),
                "n_mesh_pairs": int(row["n_mesh_pairs"]),
                "data_dir": payload["data_dir"],
            })
    return pd.DataFrame(rows).sort_values(["scenario", "model_seed"], ignore_index=True)


# ------------------------------------------------ A: coppie di topologie con CI bootstrap

def read_pair_metrics(stage: Path) -> pd.DataFrame:
    """Tutte le pair_metrics.csv del breakdown di una run, concatenate."""
    paths = sorted((stage / "topology").glob("*/pair_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"nessuna pair_metrics.csv sotto {stage / 'topology'}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def stable_seed(base: int, *tokens: str) -> int:
    """Seme riproducibile fra processi diversi: `hash()` sulle stringhe non lo e'."""
    return int(base + zlib.crc32("|".join(tokens).encode()) % 1_000_000)


def bootstrap_row(df: pd.DataFrame, value_col: str, n_bootstrap: int, rng, bootstrap_module) -> dict:
    point, ci_low, ci_high, n_subjects, n_pairs = bootstrap_module.weighted_bootstrap_spearman(
        df, value_col=value_col, n_bootstrap=n_bootstrap, rng=rng,
    )
    return {
        "spearman": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_subjects": int(n_subjects),
        "n_pairs": int(n_pairs),
        "n_bootstrap": int(n_bootstrap),
    }


def table_a_topology_pairs(stages: list[Path], args, bootstrap_module) -> pd.DataFrame:
    """Una riga per (seed, gruppo di coppie, metrica).

    I gruppi sono le 30 coppie ordinate di topologie piu' due aggregati: `all_cross`
    (tutte) e `nocrop_cross` (senza `crop`, la colonna della Tabella 2 del paper).
    """
    rows = []
    for stage in stages:
        seed = model_seed_of(stage)
        pair_metrics = read_pair_metrics(stage)
        groups: list[tuple[str, pd.DataFrame]] = []
        for (topology_a, topology_b), sub in pair_metrics.groupby(["topology_a", "topology_b"], sort=True):
            groups.append((f"{topology_a}__to__{topology_b}", sub))
        groups.append(("all_cross", pair_metrics))
        nocrop = pair_metrics[
            pair_metrics["topology_a"].ne("crop") & pair_metrics["topology_b"].ne("crop")
        ]
        groups.append(("nocrop_cross", nocrop))

        for group_name, sub in groups:
            for metric_name, value_col in METRICS:
                base = {
                    "model_seed": seed,
                    "topology_pair": group_name,
                    "metric": metric_name,
                    "n_topology_pairs": int(sub.groupby(["topology_a", "topology_b"]).ngroups),
                }
                if args.skip_bootstrap:
                    point = bootstrap_module.finite_spearman(
                        sub["gt_distance"].to_numpy(dtype=np.float64),
                        sub[value_col].to_numpy(dtype=np.float64),
                    )
                    base.update({"spearman": point, "ci_low": np.nan, "ci_high": np.nan,
                                 "n_subjects": int(pd.unique(sub[["subject_a", "subject_b"]].to_numpy().ravel()).size),
                                 "n_pairs": int(len(sub)), "n_bootstrap": 0})
                else:
                    rng = np.random.default_rng(stable_seed(args.seed, seed, group_name, metric_name))
                    base.update(bootstrap_row(sub, value_col, args.n_bootstrap, rng, bootstrap_module))
                rows.append(base)
                print(f"[ict-sum] seed{seed} {group_name:<24} {metric_name:<8} "
                      f"spearman={base['spearman']:.4f} [{base['ci_low']:.3f}, {base['ci_high']:.3f}] "
                      f"n_pairs={base['n_pairs']}", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ B: espressioni

def table_b_expressions(stages: list[Path]) -> pd.DataFrame:
    rows = []
    for stage in stages:
        seed = model_seed_of(stage)
        for cond_dir in sorted(p for p in stage.iterdir() if p.is_dir()):
            if "_" not in cond_dir.name:
                continue
            expression, intensity = cond_dir.name.rsplit("_", 1)
            for mode in MODES:
                summary = cond_dir / mode / "ranking_summary.json"
                if not summary.exists():
                    print(f"WARNING: manca {summary}", file=sys.stderr)
                    continue
                payload = json.loads(summary.read_text())
                row = payload["rows"][0]
                rows.append({
                    "model_seed": seed,
                    "expression": expression,
                    "intensity": intensity,
                    "mode": mode,
                    "latent_spearman": float(row["latent_spearman"]),
                    "chamfer_spearman": float(row["chamfer_spearman"]),
                    "delta_spearman": float(row["delta_spearman"]),
                    "n_subjects": int(row["n_subjects"]),
                    "n_subject_pairs": int(row["n_subject_pairs"]),
                    "n_mesh_pairs": int(row["n_mesh_pairs"]),
                })
    return pd.DataFrame(rows).sort_values(["mode", "expression", "intensity"], ignore_index=True)


# ------------------------------------------------------------------------------ markdown

def md_table(df: pd.DataFrame, columns: list[str], floats: int = 3) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.{floats}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, scenarios: pd.DataFrame, topo: pd.DataFrame, expr: pd.DataFrame) -> None:
    parts = ["# ICT zero-shot (WS2) ed espressioni (WS5)\n"]

    if not scenarios.empty:
        parts.append("## A. Ranking aggregato, per seed\n")
        parts.append(md_table(scenarios, ["model_seed", "scenario", "latent_spearman",
                                          "chamfer_spearman", "delta_spearman", "n_subject_pairs"]))
        mean = (scenarios.groupby("scenario")[["latent_spearman", "chamfer_spearman", "delta_spearman"]]
                .agg(["mean", "std"]).reset_index())
        mean.columns = ["scenario"] + [f"{a}_{b}" for a, b in mean.columns[1:]]
        parts.append("\n### Media sui seed\n")
        parts.append(md_table(mean, list(mean.columns)))
        parts.append(f"\nRiferimento storico BFM->FLAME zero-shot: {FLAME_ZEROSHOT_SPEARMAN:.3f} "
                     "(v2_work/STATUS.md:398, 100 soggetti, 148.500 coppie cross-topology).\n")

    if not topo.empty:
        parts.append("\n## A. Per coppia di topologie, scenario clean, CI 95% bootstrap\n")
        wide = topo.pivot_table(index=["topology_pair", "model_seed"], columns="metric",
                                values=["spearman", "ci_low", "ci_high"])
        wide.columns = [f"{b}_{a}" for a, b in wide.columns]
        wide = wide.reset_index()
        # pivot_table ordina le colonne alfabeticamente e mette i CI davanti al punto:
        # qui si rimettono in ordine leggibile (punto, poi il suo intervallo).
        ordered = ["topology_pair", "model_seed"]
        for metric_name, _ in METRICS:
            ordered += [f"{metric_name}_spearman", f"{metric_name}_ci_low", f"{metric_name}_ci_high"]
        parts.append(md_table(wide, [col for col in ordered if col in wide.columns]))

    if not expr.empty:
        parts.append("\n## B. Espressioni: Spearman vs GT delle identita' neutre\n")
        for mode in MODES:
            sub = expr[expr["mode"].eq(mode)]
            if sub.empty:
                continue
            parts.append(f"\n### {mode}\n")
            for metric in ("latent_spearman", "chamfer_spearman"):
                pivot = sub.pivot_table(index="expression", columns="intensity", values=metric).reset_index()
                parts.append(f"\n{metric}\n")
                parts.append(md_table(pivot, list(pivot.columns)))

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_module = load_bootstrap_module()

    zeroshot = discover_stages(args.runs_root, "ict_zeroshot")
    expr_stages = discover_stages(args.runs_root, "ict_expr")
    print(f"[ict-sum] zero-shot: {len(zeroshot)} run, espressioni: {len(expr_stages)} run", flush=True)

    scenarios = table_a_scenarios(zeroshot) if zeroshot else pd.DataFrame()
    topo = table_a_topology_pairs(zeroshot, args, bootstrap_module) if zeroshot else pd.DataFrame()
    expr = table_b_expressions(expr_stages) if expr_stages else pd.DataFrame()

    if not scenarios.empty:
        scenarios.to_csv(args.out_dir / "table_a_scenarios.csv", index=False)
    if not topo.empty:
        topo.to_csv(args.out_dir / "table_a_topology_pairs.csv", index=False)
    if not expr.empty:
        expr.to_csv(args.out_dir / "table_b_expressions.csv", index=False)
    write_markdown(args.out_dir / "summary.md", scenarios, topo, expr)
    print(f"[ict-sum] scritto in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
