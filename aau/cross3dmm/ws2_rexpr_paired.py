#!/usr/bin/env python3
"""Riga espressioni di WS2 (``<modello>__rexpr``): modello - Chamfer eval APPAIATO, sulla GT d'identita' neutra.

    aau/run.sh aau/cross3dmm/ws2_rexpr_paired.py      # -> aau/runs/ws2_rexpr_paired/

Non rifa' nessuna eval: rilegge le pair_metrics che ``ws2_eval.sbatch`` (WBES_WS2_MODE=rexpr) e
``ict_rexpr_rank.sbatch`` hanno gia' scritto. Ogni riga e' una coppia di mesh (soggetto a con
l'espressione casuale k, soggetto b con k' != k, topologia ``original``) con ``latent_distance`` e
``raw_chamfer`` calcolate sulle STESSE due mesh, e ``gt_distance`` dalla GT delle identita'
NEUTRE (``datasets/ICT/train_ready/gt_matrix.npz``: vertex-mean-L2 maxabs fra le ``original``
neutre, ``build_ict_gt_matrix.py``). La GT quindi non vede l'espressione: e' la GT d'identita'.

``ws2_summarize.py`` riporta modello e Chamfer con CI separati; qui la differenza, sulle stesse
repliche bootstrap per soggetto (``paired_bootstrap`` di ``aau/zs3dmm/zs_summarize.py``,
importata), in due protocolli:
  - ``mesh_pair``: una osservazione per riga (20 coppie ordinate di etichette k -> k');
  - ``subject_pair_mean``: media per coppia di soggetti sulle 20.
Viste: ``mixed`` (espressioni diverse, il caso che conta) e ``neutral`` (neutro contro neutro,
stessi soggetti, riferimento). Per il BFM-only esiste solo il mixed (job 1019721), su 100
soggetti invece che sugli 89 / 95 delle altre due celle: celle diverse non sono appaiate fra loro.
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
REPO_ROOT = AAU_DIR.parent
RUNS = AAU_DIR / "runs"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))

import ws2_summarize as ws2  # noqa: E402  (porta con se' ict_summarize come ws2.base)
from zs_summarize import paired_bootstrap  # noqa: E402

base = ws2.base
GT_NEUTRAL = REPO_ROOT / "datasets" / "ICT" / "train_ready" / "gt_matrix.npz"
PROTOCOLS = ("mesh_pair", "subject_pair_mean")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", type=Path, default=RUNS / "ws2_cross3dmm" / "splits.json")
    p.add_argument("--out-dir", type=Path, default=RUNS / "ws2_rexpr_paired")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="Seme del ricampionamento, non del modello")
    return p.parse_args()


def sources(splits: dict) -> dict:
    """(modello, vista) -> stage dir con topology/*/pair_metrics.csv, e la sua eval_key."""
    src = {("bfm_only", "mixed"): ws2.require_done(ws2.EXISTING["bfm_only__rexpr"]["mixed"].parent.parent)
           / "mixed" / "all"}
    for model in ("ict_only", "joint"):
        ckpt = Path(splits["models"][model]["run_dir"]) / ws2.CKPT_REL
        stage = ws2.require_done(ws2.find_eval_dir(RUNS, ckpt, splits["views"][f"{model}__rexpr"]) / "ws2_rexpr")
        for view in ("mixed", "neutral"):
            src[(model, view)] = stage / view
    return src


def eval_key_of(stage_view: Path) -> dict:
    for parent in stage_view.parents:
        if (parent / "eval_key.txt").exists():
            return ws2.read_eval_key(parent / "eval_key.txt")
    raise SystemExit(f"eval_key.txt non trovato sopra {stage_view}")


def per_subject_pair(pm: pd.DataFrame) -> pd.DataFrame:
    return (pm.groupby(["subject_a", "subject_b"], as_index=False)
            [["gt_distance", "latent_distance", "raw_chamfer"]].mean())


def main() -> None:
    args = parse_args()
    bm = base.load_bootstrap_module()
    splits = json.loads(args.splits.read_text())
    rows = []
    for (model, view), stage in sources(splits).items():
        key = eval_key_of(stage)
        if Path(key["dist_npz"]).resolve() != GT_NEUTRAL.resolve():
            raise SystemExit(f"{model}/{view}: GT {key['dist_npz']}, attesa la GT neutra {GT_NEUTRAL}")
        pm = ws2.read_pair_metrics(stage / "topology")
        if pm.duplicated(["subject_a", "subject_b", "topology_a", "topology_b"]).any():
            raise SystemExit(f"{model}/{view}: righe duplicate in pair_metrics")
        for protocol in PROTOCOLS:
            df = pm if protocol == "mesh_pair" else per_subject_pair(pm)
            rng = np.random.default_rng(base.stable_seed(args.seed, "rexpr_paired", model, view, protocol))
            r = paired_bootstrap(df, "latent_distance", "raw_chamfer", args.n_bootstrap, rng, bm)
            rows.append({"model": model, "view": view, "protocol": protocol,
                         "n_label_pairs": int(pm.groupby(["topology_a", "topology_b"]).ngroups),
                         "ckpt": key["ckpt"], "data_dir": key["data_dir"], **r})
            print(f"[rexpr-paired] {model:<9} {view:<7} {protocol:<17} latent={r['a']:.4f} chamfer={r['b']:.4f} "
                  f"diff={r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                  f"P(<=0)={r['p_boot_le0']:.3f} soggetti={r['n_subjects']} righe={r['n_pairs']}", flush=True)
    table = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "paired.csv", index=False)
    lines = [
        "# Espressioni casuali su ICT (riga rexpr di WS2): modello - Chamfer eval appaiato\n",
        "GT d'identita': vertex-mean-L2 maxabs fra le `original` NEUTRE (`datasets/ICT/train_ready/gt_matrix.npz`). "
        "Input: le due mesh di ogni coppia hanno espressioni casuali diverse (k != k', 3-8 blendshape ICT, "
        "coefficienti U(0.3, 1.0), `aau/ict/ict_expressions_random.py`), topologia `original` per entrambe "
        "(nessuna perturbazione topologica in questa riga). Differenza degli Spearman con la GT, latent - "
        f"Chamfer eval, CI 95% su {args.n_bootstrap} repliche bootstrap per soggetto, le stesse per i due lati.\n",
        "| modello | vista | protocollo | latent | Chamfer eval | latent - Chamfer [CI 95%] | P(boot <= 0) | soggetti | righe |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in table.itertuples():
        lines.append(f"| {ws2.MODEL_LABEL[r.model]} | {r.view} | {r.protocol} | {r.a:.3f} | {r.b:.3f} | "
                     f"{r.diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_boot_le0:.3f} | "
                     f"{r.n_subjects} | {r.n_pairs} |")
    lines.append("\nSorgenti (eval_key.txt): " + "; ".join(
        f"{ws2.MODEL_LABEL[r.model]}/{r.view}: `{r.data_dir}`" for r in table.drop_duplicates(["model", "view"]).itertuples()))
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rexpr-paired] scritto {args.out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
