#!/usr/bin/env python3
"""Seed e soggetti dello split di eval, stampati prima di spendere le ore di GPU.

Serve a rendere visibile nel log l'unica cosa che decide QUALI soggetti vengono valutati:
il seed. ``rebuild_subject_split`` (robustness/data_utils.py:76) estrae gli held-out con
``np.random.default_rng(seed)``, lo stesso seed con cui il training aveva scelto i suoi:
valutare un checkpoint con un seed diverso dal suo significa misurarlo sui soggetti su cui
e' stato addestrato, e nel log non si vedeva (gli script posthoc stampano solo
``Selected subjects: 100``, non chi sono).

La catena e' quella di ``compare_model_vs_chamfer_rankings.py`` e i tre pezzi che contano
sono importati da lui, non riscritti: ``_resolve_run_dir_and_checkpoint`` ->
``merge_run_args`` -> ``_select_subject_subset``. L'unica parte ricopiata sono le tre righe
di precedenza del seed di ``_resolve_runtime_args`` (riga 209), perche' quella funzione
vuole anche i venti argomenti di perturbazione che qui non servono.

  aau/eval_split_info.py --model_path <run_dir|ckpt> --data_dir ... --subject_split eval \
      --eval_fraction 0.2 --max_subjects 500

Lo chiama ``aau_eval_begin`` in aau/eval_common.sh passandogli ``"${common_args[@]}"``
verbatim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PERT_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
sys.path.insert(0, str(PERT_DIR))

import compare_model_vs_chamfer_rankings as ranking  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--checkpoint_selector", type=str, default="best_by_auc")
    p.add_argument("--config_json", type=str, default="")
    p.add_argument("--data_dir", type=str, default="")
    p.add_argument("--dist_npz", type=str, default="")
    p.add_argument("--subject_split", type=str, default="eval", choices=("eval", "train", "all"))
    p.add_argument("--eval_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=-1, help="Negative => use checkpoint/config seed")
    p.add_argument("--max_subjects", type=int, default=16, help="0 = all overlapping subjects")
    p.add_argument("--n_show", type=int, default=5, help="Quanti soggetti stampare")
    # parse_known_args: eval_common.sh passa i common_args interi, cioe' anche i flag di
    # perturbazione e di cache Chamfer che qui non contano. Ignorarli costa meno che tenere
    # due liste di argomenti allineate a mano.
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    cli_args = parse_args()

    run_dir, checkpoint_path = ranking._resolve_run_dir_and_checkpoint(
        cli_args.model_path, selector=str(cli_args.checkpoint_selector)
    )
    base_args = ranking.merge_run_args(checkpoint_path, explicit_config_json=cli_args.config_json)

    if int(cli_args.seed) >= 0:
        seed = int(cli_args.seed)
        source = "--seed da riga di comando (WBES_EVAL_SEED)"
    elif "seed" in base_args:
        seed = int(base_args["seed"])
        source = f"il checkpoint: {run_dir.name}/config.json"
    else:
        seed = 1234
        source = "default di _resolve_runtime_args (ne' CLI ne' config.json)"

    data_dir = cli_args.data_dir or str(base_args.get("data_dir", ranking.DEFAULT_DATA_DIR))
    dist_npz = cli_args.dist_npz or str(base_args.get("dist_npz", ranking.DEFAULT_DIST_NPZ))

    dataset = ranking.GTReadyDataset(data_dir)
    _, gt_name_to_idx = ranking.load_gt_distance_matrix(
        dist_npz, subject_re=ranking.SUBJECT_RE_ANY, dtype=np.float64
    )
    subject_map = ranking.build_subject_map(dataset.files, subject_re=ranking.SUBJECT_RE_ANY)
    subjects = sorted([sid for sid in subject_map.keys() if sid in gt_name_to_idx])
    train_subjects, eval_subjects, target_subjects = ranking._select_subject_subset(
        subjects=subjects,
        subject_split=str(cli_args.subject_split),
        eval_fraction=float(cli_args.eval_fraction),
        seed=seed,
        max_subjects=int(cli_args.max_subjects),
    )

    n_show = max(0, int(cli_args.n_show))
    print(f"[eval-split] seed={seed}, da {source}")
    print(
        f"[eval-split] pool={len(subjects)} train={len(train_subjects)} "
        f"held-out={len(eval_subjects)} -> valutati({cli_args.subject_split})="
        f"{len(target_subjects)}",
    )
    print(f"[eval-split] primi {n_show} valutati: {' '.join(target_subjects[:n_show])}", flush=True)


if __name__ == "__main__":
    main()
