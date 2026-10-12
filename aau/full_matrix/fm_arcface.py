#!/usr/bin/env python3
"""Matrice completa: l'insegnante ArcFace-render di G0 sulle 15 scansioni di galleria di FaMoS TEST.

    aau/baselines/run_bl.sh aau/full_matrix/fm_arcface.py --stage /tmp/<job> --workers 32     (fm.sbatch, famos_arc)

G0 ha FaMoS TRAIN (80 persone rigenerate), non il TEST delle tabelle. Qui ``g0_teacher`` (importato, non modificato)
con un insieme in memoria ``famos_test``: le scansioni ``scan`` / ``gallery`` del manifest (le righe di
``fact_paired.famos_frame``), copiate in ``<stage>/meshes/famos_test/<view_id>_GTready_original.npz`` (etichetta
``original`` perche' ``g0_teacher.frame`` prova le rotazioni sulla original; V nativa: il render normalizza maxabs dopo
la rotazione). Rotazione attesa ``x180`` (quella di FaMoS TRAIN in G0); il frame check di G0
decide: se la scelta (unica, con punteggio > 0) e' un'altra, si usa quella e si dichiara in ``frame_note.json``.
Camera, calibrazione del ritaglio sul detector, normal map, ArcFace ``w600k_r50``, media delle 3 viste: quelli di
``g0_teacher.run``. Uscite ridirette (``g0_teacher.G0``) in ``aau/runs/evidence/full_matrix/arcface/teacher/famos_test``.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
sys.path.insert(0, str(AAU / "diagnostics"))
sys.path.insert(0, str(AAU / "baselines_mm"))

import g0_teacher as gt  # noqa: E402
import blmm  # noqa: E402

OUT = AAU / "runs" / "evidence" / "full_matrix" / "arcface"
NAME = "famos_test"


def stage_meshes(root: Path) -> list[str]:
    rows = [r for r in blmm.famos_manifest() if r["kind"] == "scan" and r["role"] == "gallery"]
    d = root / NAME
    d.mkdir(parents=True, exist_ok=True)
    for r in rows:
        with np.load(blmm.VIEWS["famos"]["dir"] / f"{r['name']}.npz") as z:
            np.savez(d / f"{r['view_id']}_GTready_original.npz", V=np.asarray(z["V"]), F=np.asarray(z["F"]))
    return sorted(r["view_id"] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = OUT / "teacher" / NAME / "teacher.npz"
    if out.exists():
        print(f"[fm-arc] {out}: gia' presente", flush=True)
        return
    mesh_root = a.stage / "meshes"
    subjects = stage_meshes(mesh_root)
    gt.G0 = OUT
    gt.SETS[NAME] = ("gen", NAME, ("original",), "x180")
    base_subjects_of = gt.subjects_of
    gt.subjects_of = lambda name: subjects if name == NAME else base_subjects_of(name)  # noqa: E731
    fc = gt.frame(NAME, mesh_root)
    note = {"expected_first": "x180", "chosen": fc["chosen"], "score": fc["score"]}
    if not fc["agrees"]:
        if fc["chosen"] is None:
            raise SystemExit(f"{NAME}: frame check senza una scelta unica ({fc['score']})")
        gt.SETS[NAME] = ("gen", NAME, ("original",), fc["chosen"])
        fc = gt.frame(NAME, mesh_root)
    note["used"] = gt.SETS[NAME][3]
    (OUT / "teacher" / NAME).mkdir(parents=True, exist_ok=True)
    (OUT / "teacher" / NAME / "frame_note.json").write_text(json.dumps(note, indent=1) + "\n")
    gt.run(NAME, a.stage / "render", mesh_root, a.workers, 512)


if __name__ == "__main__":
    main()
