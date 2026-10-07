#!/usr/bin/env python3
"""ArcFace su render di sola geometria per NoW: scansioni e ricostruzioni.

    aau/baselines/run_bl.sh aau/recon/now_arcface.py --mode shaded  --workers 16
    aau/baselines/run_bl.sh aau/recon/now_arcface.py --mode normals --workers 16 \\
        --calibration-from ~/data/now_eval_work/arcface/shaded/renders/arcface_align.json
    (now_arcface.sbatch)

La pipeline di ``aau/zs3dmm/zs_arcface_render.py`` sulle patch canoniche di
``now_prepare_meshes.py``: rotazione fissa x180 dal frame della testa (y in su, z in avanti,
quello di T7 e di Multiface) al frame del renderer, ``normalize_maxabs`` per mesh, camera
unica su tutte le mesh, yaw 0 / -30 / +30, 512 px.  Crop ArcFace fisso: calibrato col detector
sui render ombreggiati (``ws3a_perceptual.calibrate_arcface``), copiato per la normal map,
cosi' la differenza fra le due righe e' solo l'aspetto del render.  Embedding per vista con
``ws3a_perceptual.stage_embed``, media sulle viste e rinormalizzazione
(``view_averaged``), distanza = 1 - coseno, come in WS3a.

Le "topologie" della pipeline sono qui le sorgenti: ``scan`` (le 20 scansioni) e un nome di
metodo per le sue ricostruzioni.  Render ed embedding stanno fuori dal repo
(``<WORK_ROOT>/arcface/<mode>``); nel repo vanno ``gt_arcface_<mode>_<metodo>.csv``, una riga
per ricostruzione contro la scansione del soggetto.  La matrice fra ricostruzioni di un metodo
va in ``pairs/arcface_<mode>_<metodo>.npz``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))

import now_common as common  # noqa: E402
import zs_arcface_render as zs  # noqa: E402  (porta sul path ws3a_perceptual, ws3a_render, arcface_fixed)

perc, render, arcface_fixed = zs.perc, zs.render, zs.arcface_fixed
ROT = zs.BASE_ROTATIONS["x180"]
ITEM_FIELDS = ("name", "subject", "challenge")

_MODE = "shaded"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("shaded", "normals"), default="shaded")
    p.add_argument("--out-root", type=Path, default=None, help="default <WORK_ROOT>/arcface/<mode>")
    p.add_argument("--calibration-from", type=Path, default=None,
                   help="--mode normals: arcface_align.json dei render ombreggiati")
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in render.VIEW_YAWS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    a.out_root = a.out_root or common.WORK_ROOT / "arcface" / a.mode
    a.embed_workers, a.device = a.workers, "cpu"   # letti da calibrate_arcface / stage_embed
    return a


def mesh_path(source: str, name: str) -> Path:
    return (common.scan_face_dir() if source == "scan" else common.recon_face_dir(source)) / f"{name}.npz"


def load_verts_faces(source: str, name: str):
    """Come ``zs_arcface_render.load_verts_faces``: rotazione fissa, poi maxabs."""
    with np.load(mesh_path(source, name)) as z:
        V = np.asarray(z["V"], np.float64) @ ROT.T
        return render.normalize_maxabs(V), np.asarray(z["F"], np.int64)


def compute_camera(meshes, yaws, margin: float = 1.02) -> dict:
    """Come ``zs_arcface_render.compute_camera``: centro medio dei bbox, scala che inquadra tutto."""
    loaded = [load_verts_faces(s, n)[0] for s, n in meshes]
    center = np.mean(np.stack([(V.min(0) + V.max(0)) / 2.0 for V in loaded]), axis=0)
    extent = max(render._required_extent(V, center, yaw) for V in loaded for yaw in yaws)
    return {"center": center.tolist(), "scale": float(extent * margin),
            "yaws": [float(y) for y in yaws], "n_meshes": len(meshes)}


def _init_worker(mode: str) -> None:
    global _MODE
    _MODE = mode


def _render_one(task) -> str:
    source, name, yaw, center, scale, size, out_path = task
    from PIL import Image

    V, F = load_verts_faces(source, name)
    center = np.asarray(center, dtype=np.float64)
    if yaw:
        V = zs._yaw_rotate(V, yaw, center)
    Image.fromarray(zs.draw(V, F, size, scale, center, _MODE)).save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_items()
    subjects = common.subjects_of(items)
    present = {m: [it for it in items if mesh_path(m, it.name).is_file()] for m in methods}
    topologies = {"scan": subjects, **{m: [it.name for it in present[m]] for m in methods}}
    meshes = [(s, n) for s, names in topologies.items() for n in names]
    print(f"[now-arcface] mode={args.mode} {len(meshes)} mesh ({', '.join(f'{k}={len(v)}' for k, v in topologies.items())}), "
          f"yaws={yaws} size={args.size}", flush=True)
    for s in topologies:
        (args.out_root / "renders" / s).mkdir(parents=True, exist_ok=True)

    # Camera: con --calibration-from si riusa quella dei render ombreggiati (stesso riquadro).
    camera_path = args.out_root / "renders" / "camera.json"
    if args.calibration_from is not None:
        camera = json.loads((args.calibration_from.parent / "camera.json").read_text())
    elif camera_path.exists() and not args.overwrite:
        camera = json.loads(camera_path.read_text())
    else:
        t0 = time.time()
        camera = compute_camera(meshes, yaws)
        print(f"[now-arcface] camera su {len(meshes)} mesh in {time.time() - t0:.0f}s", flush=True)
    camera.update(base_rotation="x180", mode=args.mode)
    camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")

    tasks = [(s, n, y, camera["center"], camera["scale"], args.size, str(render.render_path(args.out_root, s, n, y)))
             for s, n in meshes for y in yaws
             if args.overwrite or not render.render_path(args.out_root, s, n, y).exists()]
    print(f"[now-arcface] {len(tasks)}/{len(meshes) * len(yaws)} render da fare", flush=True)
    if tasks:
        t0 = time.time()
        with mp.get_context("fork").Pool(args.workers, initializer=_init_worker, initargs=(args.mode,)) as pool:
            for done, _ in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4), start=1):
                if done % 1000 == 0 or done == len(tasks):
                    print(f"[now-arcface] render {done}/{len(tasks)} ({done / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)

    cal_path = arcface_fixed.calibration_path(args.out_root)
    if args.calibration_from is not None:
        shutil.copyfile(args.calibration_from, cal_path)
        calibration = json.loads(cal_path.read_text())
        print(f"[now-arcface] crop copiato da {args.calibration_from}", flush=True)
    else:
        calibration = perc.calibrate_arcface(topologies, yaws, args)
    for key, v in sorted(calibration["views"].items()):
        print(f"[now-arcface] yaw {key}: {v['n_detected']} detection, IQR max {np.max(v['kps_iqr_px']):.1f} px", flush=True)

    emb = perc.stage_embed("arcface", topologies, yaws, args)
    E = {s: perc.view_averaged(emb, s, names, yaws) for s, names in topologies.items()}
    scan_row = {s: k for k, s in enumerate(subjects)}
    metric = f"arcface_{args.mode}"
    for m in methods:
        Em = E[m]
        Es = E["scan"][[scan_row[it.subject] for it in present[m]]]
        values = 1.0 - (Em * Es).sum(1)
        rows = [{**{k: getattr(it, k) for k in ITEM_FIELDS}, metric: float(v)} for it, v in zip(present[m], values)]
        common.write_rows(common.gt_csv_path(metric, m), ITEM_FIELDS + (metric,), rows, method=m, yaws=yaws,
                          size=args.size, base_rotation="x180", calibration=str(cal_path))
        path = common.pair_matrix_path(metric, m)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, D=np.clip(1.0 - Em @ Em.T, 0.0, None), names=np.asarray(topologies[m]))
        print(f"[now-arcface] {m}: {len(rows)} righe, distanza dalla scansione mediana {np.median(values):.4f}", flush=True)


if __name__ == "__main__":
    main()
