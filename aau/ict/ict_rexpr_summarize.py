#!/usr/bin/env python3
"""Tabella di WS5 rifatto (espressioni casuali per soggetto) dagli output del job.

Gemello di ``aau/ict/ict_summarize.py``, che resta il riepilogo di WS2 e del WS5 vecchio:
quello cerca lo stage ``ict_expr`` e conosce due regimi, qui gli stage sono ``ict_rexpr``
e i regimi sono tre. Tutto il resto -- caricamento del bootstrap del paper, scoperta
delle stage dir, seed del modello dal nome della run dir, seme riproducibile del
ricampionamento, formattazione markdown -- e' IMPORTATO da ict_summarize invece di
essere riscritto, cosi' le due tabelle non possono divergere sul metodo.

Non calcola nessuna distanza: legge quello che hanno scritto
``compare_model_vs_chamfer_rankings.py`` (il punto aggregato) e
``compare_model_vs_chamfer_topology_breakdown.py`` (le ``pair_metrics.csv``, una riga per
coppia di soggetti dentro ogni coppia di etichette) sotto
``aau/runs/eval_*/ict_rexpr/<regime>/<condizione>/``.

  aau/ict/ict_rexpr_summarize.py                   # scopre le run da sole
  aau/ict/ict_rexpr_summarize.py --n-bootstrap 200 # piu' veloce, CI piu' grossolani

Output (default ``aau/runs/ict_rexpr_summary/``):
  table_c_regimes.csv     regime x condizione x metrica, punto + CI 95% bootstrap
  summary.md              la tabella regime x metrica, piu' il dettaglio per k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
import ict_summarize as base  # noqa: E402

STAGE = "ict_rexpr"
# Ordine di lettura, non alfabetico: same-k, poi contro la neutra, poi il caso reale.
REGIMES = ("same_k", "expr_vs_neutral", "mixed")

# Baseline neutra-contro-neutra sugli STESSI 100 soggetti e sulla stessa topologia
# `original`, misurata nel job 1019710: e' la riga da cui leggere ogni cella qui sotto,
# perche' la domanda di WS5 non e' "quanto vale lo Spearman" ma "quanto ne perde".
NEUTRAL_BASELINE = {"latent": 0.8405, "chamfer": 0.9506}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=AAU_DIR / "runs")
    p.add_argument("--out-dir", type=Path, default=AAU_DIR / "runs" / "ict_rexpr_summary")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    p.add_argument("--skip-bootstrap", action="store_true", help="Solo i punti, senza CI")
    return p.parse_args()


def read_pair_metrics(cond_dir: Path) -> pd.DataFrame:
    """Tutte le pair_metrics.csv del breakdown di una condizione, concatenate.

    Le coppie di etichette sono 1 per same_k, 2 per expr_vs_neutral (espressiva x neutra
    e neutra x espressiva) e 20 per mixed (k x k' con k != k'): concatenarle da' lo
    stesso insieme di osservazioni su cui lo script di ranking calcola il suo punto, e
    sono l'unita' che il bootstrap del paper ricampiona per soggetto.
    """
    paths = sorted((cond_dir / "topology").glob("*/pair_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"nessuna pair_metrics.csv sotto {cond_dir / 'topology'}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def ranking_point(cond_dir: Path) -> dict:
    """Lo Spearman aggregato dello script di ranking, se la sua run e' chiusa."""
    summary = cond_dir / "ranking" / "ranking_summary.json"
    if not summary.exists():
        print(f"WARNING: manca {summary}", file=sys.stderr)
        return {}
    row = json.loads(summary.read_text())["rows"][0]
    return {
        "latent": float(row["latent_spearman"]),
        "chamfer": float(row["chamfer_spearman"]),
        "n_subject_pairs": int(row["n_subject_pairs"]),
        "n_mesh_pairs": int(row["n_mesh_pairs"]),
    }


def conditions_of(stage: Path, regime: str) -> list[Path]:
    regime_dir = stage / regime
    if not regime_dir.is_dir():
        print(f"WARNING: regime assente: {regime_dir}", file=sys.stderr)
        return []
    return sorted(p for p in regime_dir.iterdir() if p.is_dir())


def table_c_regimes(stages: list[Path], args, bootstrap_module) -> pd.DataFrame:
    """Una riga per (seed, regime, condizione, metrica), piu' l'aggregato `all_k`."""
    rows = []
    for stage in stages:
        seed = base.model_seed_of(stage)
        for regime in REGIMES:
            cond_dirs = conditions_of(stage, regime)
            if not cond_dirs:
                continue
            groups: list[tuple[str, pd.DataFrame, dict]] = []
            pooled = []
            for cond_dir in cond_dirs:
                pair_metrics = read_pair_metrics(cond_dir)
                pooled.append(pair_metrics)
                groups.append((cond_dir.name, pair_metrics, ranking_point(cond_dir)))
            # L'aggregato ha senso solo dove le condizioni sono piu' di una (same_k e
            # expr_vs_neutral hanno k=1..5, mixed ha una condizione sola).
            if len(groups) > 1:
                groups.append(("all_k", pd.concat(pooled, ignore_index=True), {}))

            for cond_name, sub, point in groups:
                for metric_name, value_col in base.METRICS:
                    row = {
                        "model_seed": seed,
                        "regime": regime,
                        "condition": cond_name,
                        "metric": metric_name,
                        "ranking_spearman": point.get(metric_name, np.nan),
                        "baseline_spearman": NEUTRAL_BASELINE[metric_name],
                    }
                    if args.skip_bootstrap:
                        row.update({
                            "spearman": bootstrap_module.finite_spearman(
                                sub["gt_distance"].to_numpy(dtype=np.float64),
                                sub[value_col].to_numpy(dtype=np.float64),
                            ),
                            "ci_low": np.nan, "ci_high": np.nan,
                            "n_subjects": int(pd.unique(sub[["subject_a", "subject_b"]].to_numpy().ravel()).size),
                            "n_pairs": int(len(sub)), "n_bootstrap": 0,
                        })
                    else:
                        rng = np.random.default_rng(
                            base.stable_seed(args.seed, seed, regime, cond_name, metric_name)
                        )
                        row.update(base.bootstrap_row(sub, value_col, args.n_bootstrap, rng, bootstrap_module))
                    row["delta_vs_baseline"] = float(row["spearman"] - NEUTRAL_BASELINE[metric_name])
                    rows.append(row)
                    print(f"[ict-rexpr-sum] seed{seed} {regime:<16} {cond_name:<6} {metric_name:<8} "
                          f"spearman={row['spearman']:.4f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                          f"n_pairs={row['n_pairs']}", flush=True)
    return pd.DataFrame(rows)


def write_markdown(path: Path, table: pd.DataFrame) -> None:
    parts = ["# WS5 rifatto: espressioni casuali per soggetto su ICT\n"]
    if table.empty:
        path.write_text("\n".join(parts) + "\nNessuna run trovata.\n", encoding="utf-8")
        return

    columns = ["regime", "condition", "model_seed", "metric", "spearman", "ci_low", "ci_high",
               "ranking_spearman", "baseline_spearman", "delta_vs_baseline", "n_pairs"]

    parts.append("## Regime x metrica (coppie di tutte le k), CI 95% bootstrap subject-level\n")
    aggregated = table[table["condition"].isin(("all_k", "all"))]
    parts.append(base.md_table(aggregated, columns))
    parts.append(
        f"\nBaseline neutra-contro-neutra sugli stessi 100 soggetti (job 1019710): "
        f"latent {NEUTRAL_BASELINE['latent']:.4f}, chamfer {NEUTRAL_BASELINE['chamfer']:.4f}. "
        "`delta_vs_baseline` e' la perdita rispetto a quella riga, ed e' il numero che WS5 "
        "vuole.\n\n"
        "`spearman` (con il suo CI) e' il punto del bootstrap sulle pair_metrics del "
        "breakdown: UNA osservazione per coppia (soggetti, coppia di etichette), cioe' una "
        "sola mesh per soggetto, che e' il caso reale. `ranking_spearman` e' il punto di "
        "compare_model_vs_chamfer_rankings.py, che aggrega su TUTTE le mesh pair della "
        "coppia di soggetti: 1 per same_k (e infatti i due numeri coincidono), 2 per "
        "expr_vs_neutral, 20 per mixed. Mediare piu' mesh pair toglie rumore e alza lo "
        "Spearman, quindi lo scarto fra le due colonne cresce col numero di mesh pair "
        "mediate e non e' un disaccordo. Sulle righe aggregate `all_k` la colonna e' vuota: "
        "nessuna singola run di ranking copre tutte le k insieme.\n")

    parts.append("\n## Dettaglio per espressione k\n")
    detail = table[~table["condition"].isin(("all_k",))]
    parts.append(base.md_table(detail.sort_values(["regime", "condition", "metric"]), columns))

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_module = base.load_bootstrap_module()

    stages = base.discover_stages(args.runs_root, STAGE)
    print(f"[ict-rexpr-sum] stage '{STAGE}': {len(stages)} run", flush=True)

    table = table_c_regimes(stages, args, bootstrap_module) if stages else pd.DataFrame()
    if not table.empty:
        table.to_csv(args.out_dir / "table_c_regimes.csv", index=False)
    write_markdown(args.out_dir / "summary.md", table)
    print(f"[ict-rexpr-sum] scritto in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
