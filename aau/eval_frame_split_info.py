#!/usr/bin/env python3
"""Seed e soggetti valutati da ``eval_by_topology.py``, stampati prima del forward.

Gemello di ``aau/eval_split_info.py`` per lo script dell'autore
(``v2_work/potential/eval_by_topology.py``), che sceglie i soggetti per un'altra strada: il
seed dello split e' ``cfg.get("seed", args.seed)`` con ``cfg`` letto dal checkpoint da
``resolve_checkpoint`` (``v2_work/transfer/eval_transfer.py:42``), e se il checkpoint non ha
il seed ripiega in silenzio su ``--seed`` (default 1234) -- cioe' sul bug dei soggetti di
training valutati come held-out (aau/README.md). Qui quel ripiego e' un ERRORE: esce 1 e
l'sbatch non lancia l'eval.

La catena e' ricopiata riga per riga da ``eval_by_topology.main`` (le funzioni sono importate,
non riscritte): subject map -> intersezione con la matrice GT -> ``rebuild_subject_split``
(eval_fraction 0.2, max_subjects 0) -> sottocampionamento a ``--n-subjects`` con
``default_rng(--seed)``, che con 100 held-out e ``--n-subjects 100`` non scatta.

  aau/run.sh aau/eval_frame_split_info.py --checkpoint <pth> --data-dir ... --dist-npz ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
# Stessi sys.path di eval_by_topology.py, nello stesso ordine.
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/remeshing/intrinsic"))
sys.path.insert(0, str(REPO_ROOT / "face_embedding/gt_encdec/autoencoder"))
sys.path.insert(0, str(REPO_ROOT / "diffusion-net/src"))
sys.path.insert(0, str(REPO_ROOT / "v2_work/transfer"))

from robustness.data_utils import GTReadyDataset, rebuild_subject_split  # noqa: E402
from intrinsic_utils import SUBJECT_RE_ANY, build_subject_map, load_gt_distance_matrix  # noqa: E402
from eval_transfer import resolve_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--dist-npz", type=Path, required=True)
    p.add_argument("--n-subjects", type=int, default=100)
    p.add_argument("--seed", type=int, default=1234, help="il --seed di eval_by_topology.py")
    p.add_argument("--n_show", type=int, default=5, help="Quanti soggetti stampare")
    p.add_argument("--expect-seed", type=int, default=None,
                   help="esce 1 se il seed del checkpoint e' un altro")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt, cfg = resolve_checkpoint(args.checkpoint)
    if "seed" not in cfg:
        print(f"[eval-split] ERRORE: {ckpt} non ha 'seed' negli args: eval_by_topology.py "
              f"userebbe --seed {args.seed} per lo split, cioe' forse i soggetti di training",
              file=sys.stderr)
        raise SystemExit(1)
    seed = int(cfg["seed"])
    if args.expect_seed is not None and seed != args.expect_seed:
        print(f"[eval-split] ERRORE: {ckpt} ha seed {seed}, atteso {args.expect_seed}",
              file=sys.stderr)
        raise SystemExit(1)

    dataset = GTReadyDataset(str(args.data_dir))
    subj_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    _, name_to_idx = load_gt_distance_matrix(str(args.dist_npz), dtype=np.float64)
    pool = sorted(set(subj_map) & set(name_to_idx))
    train, held_out = rebuild_subject_split(subjects=pool, eval_fraction=0.2, seed=seed,
                                            max_subjects=0)
    subjects = held_out
    rng = np.random.default_rng(args.seed)
    if 0 < args.n_subjects < len(subjects):
        pick = np.sort(rng.choice(len(subjects), size=args.n_subjects, replace=False))
        subjects = [subjects[int(i)] for i in pick]

    n_show = max(0, int(args.n_show))
    print(f"[eval-split] seed={seed}, dagli args del checkpoint: {ckpt}")
    print(f"[eval-split] pool={len(pool)} train={len(train)} held-out={len(held_out)} "
          f"-> valutati={len(subjects)}")
    print(f"[eval-split] primi {n_show} valutati: {' '.join(subjects[:n_show])}", flush=True)


if __name__ == "__main__":
    main()
