#!/usr/bin/env python3
"""Errore NoW UFFICIALE: ``external/now_evaluation/compute_error.py``, non modificato.

    aau/submit.sh recon/now_official.sbatch     (venv external/venvs/now)

Chiama ``compute_error.metric_computation`` del clone ufficiale sulle predizioni esportate da
``now_export.py``, una volta per metodo, poi rilegge i ``*_computed_distances.npy`` e scrive
``now_official_<metodo>.csv`` (repo): una riga per immagine con mediana, media e numero di
vertici della scansione ritagliata, piu' il json con le statistiche di challenge (mediana,
media, std di tutte le distanze concatenate, come nel ``.meanmedian`` ufficiale).

Unico intervento: ``psbody.mesh`` (e ``scan2mesh_computations.py``) importano
``psbody.mesh.meshviewer`` in testa, e il viewer carica libGL, che nel container NGC non c'e'.  Il viewer serve solo a
``rigid_scan_2_mesh_alignment(visualize=True)``, che il protocollo non chiama mai: qui
il modulo viene sostituito in ``sys.modules`` da uno stub che solleva se qualcuno lo usa.
Nessuna riga del codice ufficiale e' cambiata.

``--selfcheck``: la scansione come predizione di se stessa (``pred/scan_selfcheck``,
scritta da now_export.py): l'errore deve venire ~0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import now_common as common  # noqa: E402

FIELDS = ("name", "subject", "challenge", "now_median", "now_mean", "n_scan_vertices")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--selfcheck", action="store_true")
    return p.parse_args()


def install_meshviewer_stub() -> None:
    stub = types.ModuleType("psbody.mesh.meshviewer")

    class MeshViewer:  # noqa: D401 - stub
        def __init__(self, *a, **k):
            raise RuntimeError("MeshViewer non disponibile (niente libGL): il protocollo non lo usa")

    stub.MeshViewer = stub.MeshViewers = MeshViewer  # psbody/mesh/__init__.py importa entrambi
    sys.modules["psbody.mesh.meshviewer"] = stub


def run_official(pred: Path, imgs_list: Path, out: Path, tag: str, nproc: int) -> Path:
    import compute_error  # dal clone ufficiale

    t0 = time.time()
    compute_error.metric_computation(
        dataset_folder=str(common.NOW_DIR), predicted_mesh_folder=str(pred),
        gt_mesh_folder=str(common.SCANS_DIR), gt_lmk_folder=str(common.SCAN_LMKS_DIR),
        image_set="val", imgs_list=str(imgs_list), challenge="", error_out_path=str(out),
        method_identifier=tag, nproc=nproc)
    print(f"[now-official] {tag}: {time.time() - t0:.0f}s", flush=True)
    return out / f"{tag}_computed_distances.npy"


def summary(dists: list) -> dict:
    cat = np.concatenate(dists) if dists else np.zeros(0)
    return {"median": float(np.median(cat)), "mean": float(np.mean(cat)), "std": float(np.std(cat)),
            "n_images": len(dists), "n_distances": int(cat.size)}


def main() -> None:
    args = parse_args()
    install_meshviewer_stub()
    sys.path.insert(0, str(common.NOW_EVAL_REPO))
    out_root = common.WORK_ROOT / "official"

    if args.selfcheck:
        pred = common.WORK_ROOT / "pred" / "scan_selfcheck"
        res = np.load(run_official(pred, pred / "imagepaths_selfcheck.txt", out_root, "scan_selfcheck",
                                   args.nproc), allow_pickle=True).item()
        rows = []
        for f, d in zip(res["input_files"], res["computed_distances"]):
            rows.append({"image": str(f), "median_mm": float(np.median(d)), "mean_mm": float(np.mean(d)),
                         "max_mm": float(np.max(d)), "n_vertices": int(len(d))})
            print(f"[now-official] selfcheck {f}: mediana {np.median(d):.4f} mm, media {np.mean(d):.4f} mm, "
                  f"max {np.max(d):.4f} mm su {len(d)} vertici", flush=True)
        (common.OUT_ROOT / "official_selfcheck.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return

    by_image = {it.image: it for it in common.load_items()}
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        res = np.load(run_official(common.pred_dir(method), common.IMAGE_LIST, out_root, method, args.nproc),
                      allow_pickle=True).item()
        rows, per_challenge = [], {}
        for f, d in zip(res["input_files"], res["computed_distances"]):
            it = by_image[str(f)]
            rows.append({"name": it.name, "subject": it.subject, "challenge": it.challenge,
                         "now_median": float(np.median(d)), "now_mean": float(np.mean(d)),
                         "n_scan_vertices": int(len(d))})
            per_challenge.setdefault(it.challenge, []).append(d)
        stats = {"all": summary(res["computed_distances"]),
                 **{c: summary(per_challenge.get(c, [])) for c in common.CHALLENGES}}
        common.write_rows(common.official_csv_path(method), FIELDS, rows, method=method,
                          num_missing_files=int(res["num_missing_files"]), challenge_stats=stats,
                          code="external/now_evaluation compute_error.py (7bd1498), MeshViewer stub")
        a = stats["all"]
        print(f"[now-official] {method}: {a['n_images']} immagini ({res['num_missing_files']} mancanti), "
              f"mediana {a['median']:.3f} mm, media {a['mean']:.3f} mm, std {a['std']:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
