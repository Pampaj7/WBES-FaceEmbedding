#!/usr/bin/env python3
"""Soggetti valutati di un dominio zero-shot e loro mesh, pronte per gli operatori su /tmp.

    aau/run.sh aau/zs3dmm/zs_stage.py --view-dir datasets/HIFI3D/eval_view/npz \\
        --out-dir /tmp/wbes_zs_hifi_joint/in --seed 1234

I soggetti sono quelli che gli script di ranking estrarrebbero dal pool intero con i
``common_args`` di eval_common.sh (``--subject_split eval --eval_fraction 0.2 --max_subjects
500`` e WBES_EVAL_SEED): ``rebuild_subject_split`` importata, non riscritta, sulla stessa lista
ordinata di soggetti. Sono anche quelli del set ``<dominio>_heldout`` delle baseline
(``zs_bl.py``), che chiama la stessa funzione sullo stesso pool.

Si fa qui e non nello script di ranking perche' gli operatori stanno solo su /tmp e si
calcolano solo per questi soggetti (600 mesh invece di 3000): la data dir dell'eval contiene
gia' e solo i soggetti valutati, e lo sbatch la valuta con ``--subject_split all
--max_subjects 0``, come ws2_eval.sbatch fa con le sue viste.

Scrive in ``--out-dir`` un symlink per ogni mesh dei soggetti scelti (6 topologie ciascuno) e
``subjects.json`` accanto (nella dir madre).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))

from robustness.data_utils import rebuild_subject_split  # noqa: E402

TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")


def select_subjects(view_dir: Path, seed: int) -> list[str]:
    pool = sorted({p.name.split("_GTready_")[0] for p in view_dir.glob("id*_GTready_*.npz")})
    if not pool:
        raise SystemExit(f"nessuna mesh id*_GTready_*.npz in {view_dir}")
    _, eval_subjects = rebuild_subject_split(pool, eval_fraction=0.2, seed=seed, max_subjects=500)
    return sorted(eval_subjects)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()

    subjects = select_subjects(a.view_dir, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in a.out_dir.glob("*.npz"):
        stale.unlink()
    for sid in subjects:
        for topo in TOPOLOGIES:
            src = a.view_dir / f"{sid}_GTready_{topo}.npz"
            if not src.exists():
                raise SystemExit(f"mesh mancante: {src}")
            (a.out_dir / src.name).symlink_to(src.resolve())
    (a.out_dir.parent / "subjects.json").write_text(json.dumps(
        {"seed": a.seed, "view_dir": str(a.view_dir), "subjects": subjects}, indent=1) + "\n")
    print(f"[zs-stage] {len(subjects)} soggetti (primi {subjects[:3]}), "
          f"{len(subjects) * len(TOPOLOGIES)} mesh in {a.out_dir}")


if __name__ == "__main__":
    main()
