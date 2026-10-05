#!/usr/bin/env python3
"""Tabella cross-3DMM di WS2 (3 modelli x 2 domini) e riga espressioni, dagli output dei job.

Non calcola nessuna distanza: legge quello che hanno scritto gli script di eval del repo
(``compare_model_vs_chamfer_rankings.py`` e ``..._topology_breakdown.py``) per le celle
nuove (``aau/cross3dmm/ws2_eval.sbatch``) e per quelle del BFM-only che esistevano gia'
(ranking in-domain 47e4a6be, zero-shot ICT 3c44d0ba, espressioni f3d99929), e aggiunge i
CI bootstrap subject-level. Tutto il metodo -- bootstrap del paper, seme riproducibile,
markdown -- e' importato da ``aau/ict/ict_summarize.py``, come fa ict_rexpr_summarize.py.

Tre protocolli per cella, tutti cross-topology:
  - ``subject_pair_mean`` clean e mixed: il punto dello script di ranking (media delle
    distanze su tutte le mesh pair della coppia di soggetti). Il CI del clean si ricava
    dalle pair_metrics del breakdown mediate per coppia di soggetti sulle 30 coppie di
    topologie, che sono esattamente le mesh pair del clean: il punto ricalcolato deve
    coincidere con quello dello script (colonna ``point_check``). Il mixed applica
    perturbazioni che il breakdown non applica, quindi resta senza CI.
  - ``mesh_pair`` ``all_cross`` / ``nocrop_cross``: una osservazione per (coppia di
    soggetti, coppia ordinata di topologie), lo stesso protocollo dello 0.301 zero-shot
    (ict_summarize, media dei tre seed).

Controllo leak, rifatto qui sugli OUTPUT e non sulle viste: per ogni cella i soggetti
che gli script dichiarano di aver valutato (``selected_subjects`` di ogni json e
``subject_a``/``subject_b`` di ogni pair_metrics) devono essere esattamente la lista di
``splits.json`` e avere intersezione vuota con il training del modello valutato.

  aau/submit.sh cross3dmm/ws2_summarize.sbatch

Output in ``aau/runs/ws2_cross3dmm/``: table_cells.csv, topology_pairs.csv,
expressions.csv, leak_check.csv, summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
RUNS = AAU_DIR / "runs"

sys.path.insert(0, str(AAU_DIR / "ict"))
import ict_summarize as base  # noqa: E402
from ict_rexpr_summarize import NEUTRAL_BASELINE  # noqa: E402

MODELS = ("bfm_only", "ict_only", "joint")
MODEL_LABEL = {"bfm_only": "BFM-only", "ict_only": "ICT-only", "joint": "BFM+ICT"}
DOMAINS = ("bfm", "ict")
CKPT_REL = Path("checkpoints") / "best_by_xtopo_mesh_clean.pth"
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")

# Le celle del BFM-only che esistevano gia' (riusate, non rifatte). bfm_only__bfm prende
# da qui solo il ranking: l'eval in-domain non aveva scritto pair_metrics, quindi il
# breakdown e' rifatto (stage ws2_cell, solo topology) e confrontato con questo.
RUN_NAME = (
    "mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_"
    "sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
)
EXISTING = {
    "bfm_only__bfm": {"ranking": RUNS / f"eval_{RUN_NAME}_47e4a6be" / "ranking_merged",
                      "topology_old": RUNS / f"eval_{RUN_NAME}_47e4a6be" / "topology"},
    "bfm_only__ict": {"ranking": RUNS / f"eval_{RUN_NAME}_3c44d0ba" / "ict_zeroshot" / "ranking",
                      "topology": RUNS / f"eval_{RUN_NAME}_3c44d0ba" / "ict_zeroshot" / "topology"},
    "bfm_only__rexpr": {"mixed": RUNS / f"eval_{RUN_NAME}_f3d99929" / "ict_rexpr" / "mixed" / "all"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=RUNS)
    p.add_argument("--splits", type=Path, default=RUNS / "ws2_cross3dmm" / "splits.json")
    p.add_argument("--out-dir", type=Path, default=RUNS / "ws2_cross3dmm")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    return p.parse_args()


# ------------------------------------------------------------------ dove stanno i risultati

def read_eval_key(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def find_eval_dir(runs_root: Path, ckpt: Path, data_dir: str) -> Path:
    """La out dir di eval_common.sh per (checkpoint, data dir): la chiave e' eval_key.txt."""
    want = (os.path.realpath(ckpt), os.path.realpath(data_dir))
    found = [p.parent for p in sorted(runs_root.glob("eval_*/eval_key.txt"))
             if (lambda k: (k.get("ckpt"), k.get("data_dir")))(read_eval_key(p)) == want]
    if len(found) != 1:
        raise SystemExit(f"{len(found)} out dir per ckpt={want[0]} data_dir={want[1]}")
    return found[0]


def require_done(stage: Path) -> Path:
    if not (stage / ".done").exists():
        raise SystemExit(f"stage incompleta (manca .done): {stage}")
    return stage


def cell_sources(args, splits: dict) -> dict:
    ckpt = {m: Path(splits["models"][m]["run_dir"]) / CKPT_REL for m in MODELS}
    src = {}
    for model in MODELS:
        for domain in DOMAINS:
            cell = f"{model}__{domain}"
            if cell == "bfm_only__ict":
                src[cell] = {"ranking": EXISTING[cell]["ranking"], "topology": EXISTING[cell]["topology"]}
                require_done(EXISTING[cell]["ranking"].parent)
                continue
            stage = require_done(find_eval_dir(args.runs_root, ckpt[model], splits["views"][cell]) / "ws2_cell")
            src[cell] = {"ranking": stage / "ranking", "topology": stage / "topology"}
            if cell == "bfm_only__bfm":
                src[cell]["ranking"] = EXISTING[cell]["ranking"]
                src[cell]["topology_old"] = EXISTING[cell]["topology_old"]
        if model != "bfm_only":
            stage = require_done(find_eval_dir(args.runs_root, ckpt[model], splits["views"][f"{model}__rexpr"])
                                 / "ws2_rexpr")
            src[f"{model}__rexpr"] = {"mixed": stage / "mixed", "neutral": stage / "neutral"}
    src["bfm_only__rexpr"] = {"mixed": require_done(EXISTING["bfm_only__rexpr"]["mixed"].parent.parent)
                              / "mixed" / "all"}
    src["_ckpt"] = {m: str(ckpt[m]) for m in MODELS}
    return src


# ------------------------------------------------------------------------------- leak

def evaluated_subjects(dirs: list[Path]) -> tuple[set[str], list[str]]:
    """Unione dei soggetti dichiarati dagli output: selected_subjects e pair_metrics.

    Ritorna anche i checkpoint dichiarati dai json, per verificare che la cella sia stata
    valutata col modello giusto.
    """
    subjects: set[str] = set()
    ckpts: list[str] = []
    for d in dirs:
        for js in sorted(d.rglob("ranking_summary.json")):
            payload = json.loads(js.read_text())
            subjects |= set(payload.get("selected_subjects", []))
            if "checkpoint" in payload:
                ckpts.append(os.path.realpath(payload["checkpoint"]))
        for csv in sorted(d.rglob("pair_metrics.csv")):
            pm = pd.read_csv(csv, usecols=["subject_a", "subject_b"])
            subjects |= set(pm["subject_a"].astype(str)) | set(pm["subject_b"].astype(str))
    return subjects, ckpts


def leak_check(src: dict, splits: dict) -> pd.DataFrame:
    rows = []
    checks = [(f"{m}__{d}", m, f"{m}__{d}", ["ranking", "topology"]) for m in MODELS for d in DOMAINS]
    checks += [(f"{m}__rexpr", m, "bfm_only__ict" if m == "bfm_only" else f"{m}__ict",
                ["mixed"] if m == "bfm_only" else ["mixed", "neutral"]) for m in MODELS]
    for name, model, list_cell, keys in checks:
        dirs = [src[name][k] for k in keys]
        seen, ckpts = evaluated_subjects(dirs)
        expected = set(splits["cells"][list_cell]["subjects"])
        train = set(splits["models"][model]["train"])
        wrong_ckpt = sorted({c for c in ckpts if c != os.path.realpath(src["_ckpt"][model])})
        row = {
            "cell": name, "model": model,
            "n_evaluated": len(seen), "n_expected": len(expected),
            "evaluated_equals_split_list": seen == expected,
            "n_train_model": len(train), "n_evaluated_in_train": len(seen & train),
            "n_evaluated_in_online_selection": len(seen & set(splits["models"][model]["online_eval"])),
            "n_json_checked": len(ckpts), "wrong_checkpoint": ";".join(wrong_ckpt),
        }
        rows.append(row)
        print(f"[ws2-sum] leak {name:<16} valutati={row['n_evaluated']:>3} attesi={row['n_expected']:>3} "
              f"in training={row['n_evaluated_in_train']} json={row['n_json_checked']}", flush=True)
        if row["n_evaluated_in_train"] or not row["evaluated_equals_split_list"] or wrong_ckpt:
            raise SystemExit(f"LEAK o set di eval inatteso in {name}: {row}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- tabelle

def ranking_rows(ranking_dir: Path) -> dict:
    payload = json.loads((ranking_dir / "ranking_summary.json").read_text())
    return {row["scenario"]: row for row in payload["rows"]}


def read_pair_metrics(topology_dir: Path) -> pd.DataFrame:
    paths = sorted(topology_dir.glob("*/pair_metrics.csv"))
    if not paths:
        raise SystemExit(f"nessuna pair_metrics.csv sotto {topology_dir}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def boot(df: pd.DataFrame, value_col: str, args, bm, *tokens: str) -> dict:
    rng = np.random.default_rng(base.stable_seed(args.seed, *tokens))
    return base.bootstrap_row(df, value_col, args.n_bootstrap, rng, bm)


def cell_tables(src: dict, splits: dict, args, bm) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells, topo = [], []
    for model in MODELS:
        for domain in DOMAINS:
            cell = f"{model}__{domain}"
            pm = read_pair_metrics(src[cell]["topology"])
            ranking = ranking_rows(src[cell]["ranking"])

            # subject_pair_mean: clean con CI dalle pair_metrics mediate, mixed solo punto.
            per_subject_pair = (pm.groupby(["subject_a", "subject_b"], as_index=False)
                                [["gt_distance", "latent_distance", "raw_chamfer"]].mean())
            for scenario in ("clean", "mixed"):
                row = ranking[scenario]
                out = {"model": model, "domain": domain, "protocol": "subject_pair_mean",
                       "scenario": scenario, "n_subjects": int(row["n_subjects"]),
                       "n_subject_pairs": int(row["n_subject_pairs"]), "n_mesh_pairs": int(row["n_mesh_pairs"])}
                for metric, col in base.METRICS:
                    out[f"{metric}_spearman"] = float(row[f"{metric}_spearman"])
                    if scenario == "clean":
                        b = boot(per_subject_pair, col, args, bm, cell, "spm_clean", metric)
                        out.update({f"{metric}_ci_low": b["ci_low"], f"{metric}_ci_high": b["ci_high"],
                                    f"{metric}_point_check": b["spearman"]})
                    else:
                        out.update({f"{metric}_ci_low": np.nan, f"{metric}_ci_high": np.nan,
                                    f"{metric}_point_check": np.nan})
                cells.append(out)

            # mesh_pair: aggregati e coppie ordinate di topologie, clean, CI subject-level.
            groups = [(f"{a}__to__{b}", sub) for (a, b), sub in pm.groupby(["topology_a", "topology_b"], sort=True)]
            groups.append(("all_cross", pm))
            groups.append(("nocrop_cross", pm[pm["topology_a"].ne("crop") & pm["topology_b"].ne("crop")]))
            for group, sub in groups:
                res = {}
                for metric, col in base.METRICS:
                    res[metric] = boot(sub, col, args, bm, cell, group, metric)
                    topo.append({"model": model, "domain": domain, "topology_pair": group, "metric": metric,
                                 **res[metric]})
                if group in ("all_cross", "nocrop_cross"):
                    out = {"model": model, "domain": domain, "protocol": f"mesh_pair_{group}",
                           "scenario": "clean", "n_subjects": res["latent"]["n_subjects"],
                           "n_subject_pairs": int(sub.groupby(["subject_a", "subject_b"]).ngroups),
                           "n_mesh_pairs": res["latent"]["n_pairs"]}
                    for metric in ("latent", "chamfer"):
                        out.update({f"{metric}_spearman": res[metric]["spearman"],
                                    f"{metric}_ci_low": res[metric]["ci_low"],
                                    f"{metric}_ci_high": res[metric]["ci_high"],
                                    f"{metric}_point_check": np.nan})
                    cells.append(out)
                    print(f"[ws2-sum] {cell:<14} {group:<12} latent={res['latent']['spearman']:.4f} "
                          f"[{res['latent']['ci_low']:.3f},{res['latent']['ci_high']:.3f}] "
                          f"chamfer={res['chamfer']['spearman']:.4f} n_pairs={res['latent']['n_pairs']}",
                          flush=True)
    table = pd.DataFrame(cells)
    table["delta_spearman"] = table["latent_spearman"] - table["chamfer_spearman"]
    return table, pd.DataFrame(topo)


def topology_consistency(src: dict, topo: pd.DataFrame) -> float:
    """BFM-only su BFM: breakdown rifatto (con pair_metrics) contro quello esistente."""
    old = pd.read_csv(src["bfm_only__bfm"]["topology_old"] / "topology_breakdown_summary.csv")
    new = topo[(topo["model"] == "bfm_only") & (topo["domain"] == "bfm") & topo["metric"].eq("latent")]
    merged = new.merge(old[["ordered_pair_label", "latent_spearman"]], left_on="topology_pair",
                       right_on="ordered_pair_label")
    if merged.empty:
        print("[ws2-sum] ATTENZIONE: nessuna coppia di topologie in comune col breakdown esistente")
        return float("nan")
    return float((merged["spearman"] - merged["latent_spearman"]).abs().max())


def expression_table(src: dict, args, bm) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        views = ["mixed"] if model == "bfm_only" else ["mixed", "neutral"]
        for view in views:
            d = src[f"{model}__rexpr"][view]
            pm = read_pair_metrics(d / "topology")
            point = ranking_rows(d / "ranking")["clean"]
            for metric, col in base.METRICS:
                b = boot(pm, col, args, bm, model, "rexpr", view, metric)
                rows.append({"model": model, "view": view, "metric": metric, **b,
                             "ranking_spearman": float(point[f"{metric}_spearman"])})
        if model == "bfm_only":
            # Baseline neutra del BFM-only: job 1019710, solo il punto (stessi 100 soggetti).
            for metric, _ in base.METRICS:
                rows.append({"model": model, "view": "neutral", "metric": metric,
                             "spearman": NEUTRAL_BASELINE[metric], "ci_low": np.nan, "ci_high": np.nan,
                             "n_subjects": 100, "n_pairs": 4950, "n_bootstrap": 0,
                             "ranking_spearman": NEUTRAL_BASELINE[metric]})
    table = pd.DataFrame(rows)
    neutral = table[table["view"].eq("neutral")].set_index(["model", "metric"])["spearman"]
    table["delta_vs_neutral"] = [
        float(r.spearman - neutral[(r.model, r.metric)]) if r.view == "mixed" else np.nan
        for r in table.itertuples()
    ]
    return table


# --------------------------------------------------------------------------- markdown

def fmt_ci(point: float, low: float, high: float) -> str:
    if not np.isfinite(low):
        return f"{point:.3f}"
    return f"{point:.3f} [{low:.2f}, {high:.2f}]"


def grid(table: pd.DataFrame, protocol: str, scenario: str) -> str:
    lines = ["| modello | BFM latent | BFM chamfer | ICT latent | ICT chamfer |", "| --- | --- | --- | --- | --- |"]
    for model in MODELS:
        cells = [MODEL_LABEL[model]]
        for domain in DOMAINS:
            r = table[(table["model"] == model) & (table["domain"] == domain)
                      & (table["protocol"] == protocol) & (table["scenario"] == scenario)].iloc[0]
            cells += [fmt_ci(r.latent_spearman, r.latent_ci_low, r.latent_ci_high),
                      fmt_ci(r.chamfer_spearman, r.chamfer_ci_low, r.chamfer_ci_high)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def topology_matrix(topo: pd.DataFrame, model: str, domain: str, metric: str) -> str:
    sub = topo[(topo["model"] == model) & (topo["domain"] == domain) & (topo["metric"] == metric)]
    value = {r.topology_pair: r.spearman for r in sub.itertuples()}
    lines = ["| A \\ B | " + " | ".join(TOPOLOGIES) + " |", "| --- |" + " --- |" * len(TOPOLOGIES)]
    for a in TOPOLOGIES:
        cells = [a] + ["-" if a == b else f"{value.get(f'{a}__to__{b}', np.nan):.3f}" for b in TOPOLOGIES]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, table, topo, expr, leak, consistency: float, args) -> None:
    n = {r.cell: r.n_evaluated for r in leak.itertuples()}
    parts = [
        "# WS2: tabella cross-3DMM (seed 1234, ricetta v1, operatori ad area unitaria)\n",
        "Modelli: BFM-only (job 1019310), ICT-only (1019531), BFM+ICT congiunto (1019532). "
        "Ogni cella e' valutata SOLO su soggetti held-out del modello valutato (vedi controllo leak). "
        f"Soggetti per cella: BFM {n['bfm_only__bfm']}/{n['ict_only__bfm']}/{n['joint__bfm']}, "
        f"ICT {n['bfm_only__ict']}/{n['ict_only__ict']}/{n['joint__ict']} "
        "(BFM-only/ICT-only/congiunto). GT: BFM `normalized_matrix_distances.npz`, "
        "ICT `datasets/ICT/train_ready/gt_matrix.npz` (la GT di `datasets/ICT/gt/` con l'offset id+10000). "
        f"CI 95% bootstrap subject-level, {args.n_bootstrap} ricampionamenti.\n",
        "## Mesh-pair cross-topology, clean (protocollo dello 0.301 zero-shot)\n",
        grid(table, "mesh_pair_all_cross", "clean"),
        "\n### Senza crop (colonna della Tabella 2 del paper)\n",
        grid(table, "mesh_pair_nocrop_cross", "clean"),
        "\n## Subject-pair-mean (script di ranking), clean\n",
        grid(table, "subject_pair_mean", "clean"),
        "\n## Subject-pair-mean, mixed (solo punto: il breakdown non perturba, niente CI)\n",
        grid(table, "subject_pair_mean", "mixed"),
        "\n## Espressioni casuali, regime (c) misto, contro la baseline neutra sugli stessi soggetti\n",
    ]
    lines = ["| modello | metrica | misto | neutro | delta | n soggetti |", "| --- | --- | --- | --- | --- | --- |"]
    for model in MODELS:
        for metric, _ in base.METRICS:
            m = expr[(expr["model"] == model) & (expr["metric"] == metric) & expr["view"].eq("mixed")].iloc[0]
            z = expr[(expr["model"] == model) & (expr["metric"] == metric) & expr["view"].eq("neutral")].iloc[0]
            lines.append(f"| {MODEL_LABEL[model]} | {metric} | {fmt_ci(m.spearman, m.ci_low, m.ci_high)} | "
                         f"{fmt_ci(z.spearman, z.ci_low, z.ci_high)} | {m.delta_vs_neutral:+.3f} | {int(m.n_subjects)} |")
    parts.append("\n".join(lines))
    parts.append("\nBFM-only: misto dalle pair_metrics esistenti (job 1019721), neutro dal job 1019710 "
                 "(solo punto). Il misto e' a una mesh per soggetto (pair_metrics), come in ict_rexpr_summary.\n")
    parts.append("\n## Controllo leak (sugli output, non sulle viste)\n")
    parts.append(base.md_table(leak, ["cell", "n_evaluated", "evaluated_equals_split_list", "n_train_model",
                                      "n_evaluated_in_train", "n_evaluated_in_online_selection",
                                      "n_json_checked"]))
    parts.append(
        "\n`n_evaluated_in_online_selection`: soggetti held-out che erano fra i 16 dell'eval online del "
        "training, cioe' hanno pesato sulla scelta del checkpoint `best_by_xtopo_mesh_clean` (non sul "
        "gradiente). Stesso protocollo di tutte le eval esistenti.\n")
    parts.append(f"\nConsistenza: breakdown BFM-only su BFM rifatto con pair_metrics contro quello esistente, "
                 f"max |delta latent| sulle coppie di topologie = {consistency:.2e}.\n")
    parts.append("\n## Matrici per coppia di topologie (clean, mesh-pair, Spearman; righe A, colonne B)\n")
    for model in MODELS:
        for domain in DOMAINS:
            for metric in ("latent", "chamfer"):
                parts.append(f"\n### {MODEL_LABEL[model]} su {domain.upper()}, {metric}\n")
                parts.append(topology_matrix(topo, model, domain, metric))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads(args.splits.read_text())
    bm = base.load_bootstrap_module()

    src = cell_sources(args, splits)
    leak = leak_check(src, splits)
    table, topo = cell_tables(src, splits, args, bm)
    consistency = topology_consistency(src, topo)
    print(f"[ws2-sum] breakdown BFM-only rifatto vs esistente: max |delta| = {consistency:.2e}", flush=True)
    expr = expression_table(src, args, bm)

    leak.to_csv(args.out_dir / "leak_check.csv", index=False)
    table.to_csv(args.out_dir / "table_cells.csv", index=False)
    topo.to_csv(args.out_dir / "topology_pairs.csv", index=False)
    expr.to_csv(args.out_dir / "expressions.csv", index=False)
    write_markdown(args.out_dir / "summary.md", table, topo, expr, leak, consistency, args)
    print(f"[ws2-sum] scritto in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
