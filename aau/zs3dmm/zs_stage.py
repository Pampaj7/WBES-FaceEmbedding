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

``--transform`` (test del frame): invece dei symlink scrive copie con i vertici trasformati da
una matrice di segni sugli assi, p.es. ``x,-y,-z`` (180 gradi attorno a x: dal frame ICT/HIFI3D,
y in alto e naso verso +z, al frame dei dati BFM, y in basso e naso verso -z) o ``-x,y,-z``
(180 gradi attorno a y). Solo segni: rotazioni proprie (determinante +1, verso dei triangoli
invariato) o riflessioni, rifiutate. Il modello usa le coordinate xyz, quindi il frame entra nel
suo ingresso; Chamfer e GT no (distanze invarianti per rotazione, e maxabs prende il massimo dei
valori assoluti, invariante per cambi di segno): Chamfer deve uscire IDENTICO, e' il controllo.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
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


def parse_transform(spec: str) -> np.ndarray:
    """``x,-y,-z`` -> diag(1, -1, -1); solo segni, determinante +1."""
    toks = [t.strip() for t in spec.split(",")]
    if len(toks) != 3 or [t.lstrip("-") for t in toks] != ["x", "y", "z"]:
        raise SystemExit(f"--transform: atteso p.es. 'x,-y,-z', dato '{spec}'")
    signs = np.array([-1.0 if t.startswith("-") else 1.0 for t in toks])
    if np.prod(signs) != 1.0:
        raise SystemExit(f"--transform {spec}: riflessione (det -1), non una rotazione")
    return signs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--view-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--transform", default="", help="p.es. x,-y,-z; vuoto = symlink alle mesh originali")
    a = p.parse_args()
    signs = parse_transform(a.transform) if a.transform else None

    subjects = select_subjects(a.view_dir, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for stale in a.out_dir.glob("*.npz"):
        stale.unlink()
    for sid in subjects:
        for topo in TOPOLOGIES:
            src = a.view_dir / f"{sid}_GTready_{topo}.npz"
            if not src.exists():
                raise SystemExit(f"mesh mancante: {src}")
            if signs is None:
                (a.out_dir / src.name).symlink_to(src.resolve())
            else:
                with np.load(src) as d:
                    V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
                np.savez(a.out_dir / src.name, V=(V * signs).astype(V.dtype), F=F)
    (a.out_dir.parent / "subjects.json").write_text(json.dumps(
        {"seed": a.seed, "view_dir": str(a.view_dir), "transform": a.transform, "subjects": subjects},
        indent=1) + "\n")
    print(f"[zs-stage] {len(subjects)} soggetti (primi {subjects[:3]}), "
          f"{len(subjects) * len(TOPOLOGIES)} mesh in {a.out_dir}"
          + (f", vertici trasformati {a.transform}" if signs is not None else ""))


if __name__ == "__main__":
    main()
