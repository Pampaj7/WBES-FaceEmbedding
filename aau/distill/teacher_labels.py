#!/usr/bin/env python3
"""Etichette dell'insegnante per la distillazione: ArcFace su normal map delle mesh ``original``.

    aau/baselines/run_bl.sh aau/distill/teacher_labels.py --domain ict --stage /tmp/$SLURM_JOB_ID/ict \\
        --out-dir aau/runs/distill_pilot/teacher/ict --workers 32
    (teacher_labels.sbatch; piano in paper/PLAN_DISTILL.md, protocollo in aau/runs/distill_pilot/protocol.md)

E' la pipeline del gate (``aau/zs3dmm/zs_arcface_render.py``, importata e non riscritta: stesso
renderer, ``normalize_maxabs``, camera unica per dominio, yaw 0 / -30 / +30, 512 px, crop fisso
calibrato sui render OMBREGGIATI e copiato sulle normal map, embedding con
``ws3a_perceptual.stage_embed``), applicata ai dati di training del pilota:

- BFM (``datasets/REMESH/npz_data_topo_500``, gia' nel frame del renderer: alto -y, naso -z) e
  ICT-5000 (``datasets/ICT/topo``, alto +y e naso +z: Rx(180)). Rotazioni verificate con
  ``--frame-check`` prima di generare;
- soggetti: il training del congiunto 1019532 (``aau/data_scale/split_scale.json``: 392 BFM e
  4008 ICT-5000) piu' un insieme di VALIDAZIONE held-out per le curve (i 16 BFM dell'eval online
  del congiunto e 84 ICT held-out scelti con ``--seed``); i domini di test (FaceVerse, HIFI3D) non
  entrano;
- l'insegnante vede SOLO la topologia ``original``, e per gli ICT che le hanno anche le 5
  espressioni casuali (``datasets/ICT/expressions_random/id*_rexpr_<k>.npz``, stessa topologia
  della ``original``), etichettate ``rexpr<k>``;
- calibrazione del crop: detector sui render ombreggiati delle ``original`` di ``--calib-subjects``
  soggetti (campione con ``--seed``), non di tutte: e' la mediana dei landmark e il campione
  basta (il gate la faceva su 1800 render).

I render stanno in ``--stage`` (su /tmp, si cancella col job). In ``--out-dir`` vanno solo
``teacher.npz`` (``E`` (n, 3, 512) per vista, ``Z`` (n, 512) media sulle viste rinormalizzata L2,
``subjects``, ``topologies``, ``split``, ``files``, ``yaws``), ``camera.json``,
``arcface_align.json``, ``timing.json`` e qualche png di controllo.

``--limit N``: solo i primi N soggetti di training e i primi N di validazione, per misurare il costo
per mesh prima di generare (``timing.json``: secondi di parete e CPU-secondi per mesh di ogni fase).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))

import zs_arcface_render as zar  # noqa: E402

perc, render, arcface_fixed = zar.perc, zar.render, zar.arcface_fixed

DS = REPO_ROOT / "datasets"
SPLIT_JSON = AAU_DIR / "data_scale" / "split_scale.json"
ICT_OFFSET = 10000
N_REXPR = 5
DOMAINS = {
    # dominio: (rotazione verso il frame del renderer, filtro sugli id dello split)
    "bfm": ("none", lambda n: n < 1000),
    "ict": ("x180", lambda n: ICT_OFFSET <= n < ICT_OFFSET + 5000),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    p.add_argument("--stage", type=Path, required=True, help="radice su /tmp: vista di symlink e render")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--calib-subjects", type=int, default=300)
    p.add_argument("--n-val-ict", type=int, default=84)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--limit", type=int, default=0, help=">0: solo i primi N soggetti (misura del costo)")
    p.add_argument("--frame-check", type=int, default=0, help=">0: solo il controllo del frame, poi esce")
    return p.parse_args()


def id_num(s: str) -> int:
    return int(s[2:])


def subjects_of(domain: str, n_val_ict: int, seed: int) -> tuple[list[str], list[str]]:
    """(training, validazione) del dominio, dallo split esplicito del congiunto."""
    split = json.loads(SPLIT_JSON.read_text())
    keep = DOMAINS[domain][1]
    train = sorted(s for s in split["train"] if keep(id_num(s)))
    if domain == "bfm":
        val = sorted(split["online_eval"])
    else:
        pool = sorted(s for s in split["heldout"] if keep(id_num(s)))
        val = sorted(np.random.default_rng(seed).choice(pool, size=n_val_ict, replace=False).tolist())
    if set(train) & set(val):
        raise SystemExit(f"{domain}: soggetti di validazione nel training")
    return train, val


def source_files(domain: str, subject: str) -> dict[str, Path]:
    """Topologia -> file di sola geometria: ``original`` e, dove ci sono, le espressioni."""
    if domain == "bfm":
        return {"original": DS / "REMESH" / "npz_data_topo_500" / f"{subject}_GTready_original.npz"}
    raw = f"ict{id_num(subject) - ICT_OFFSET:04d}"
    out = {"original": DS / "ICT" / "topo" / f"{raw}_GTready_original.npz"}
    for k in range(1, N_REXPR + 1):
        f = DS / "ICT" / "expressions_random" / f"{subject}_rexpr_{k}.npz"
        if f.exists():
            out[f"rexpr{k}"] = f
    return out


def build_view(view: Path, files: dict[tuple[str, str], Path]) -> None:
    """Symlink ``<soggetto>_GTready_<topologia>.npz``: il nome che ``zar.mesh_file`` si aspetta."""
    view.mkdir(parents=True, exist_ok=True)
    for (s, t), src in files.items():
        if not src.exists():
            raise SystemExit(f"mesh assente: {src}")
        dst = zar.mesh_file(view, s, t)
        if not dst.exists():
            dst.symlink_to(src)


def render_all(view: Path, out_root: Path, meshes, yaws, camera: dict, rot, mode: str, size: int,
               workers: int) -> float:
    """Render di (soggetto, topologia) x yaw in ``out_root/renders``; ritorna i secondi di parete."""
    for t in {t for _, t in meshes}:
        (out_root / "renders" / t).mkdir(parents=True, exist_ok=True)
    tasks = [(s, t, y, camera["center"], camera["scale"], size, str(render.render_path(out_root, t, s, y)))
             for s, t in meshes for y in yaws if not render.render_path(out_root, t, s, y).exists()]
    t0 = time.time()
    with mp.get_context("fork").Pool(workers, initializer=zar._init_render_worker,
                                     initargs=(view, rot, mode)) as pool:
        for done, _ in enumerate(pool.imap_unordered(zar._render_one, tasks, chunksize=4), start=1):
            if done % 3000 == 0 or done == len(tasks):
                print(f"[teacher] render {mode} {done}/{len(tasks)} "
                      f"({done / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
    return time.time() - t0


def main() -> None:
    args = parse_args()
    rot_name = DOMAINS[args.domain][0]
    rot = zar.BASE_ROTATIONS[rot_name]
    yaws = [float(y) for y in render.VIEW_YAWS]
    train, val = subjects_of(args.domain, args.n_val_ict, args.seed)
    if args.limit > 0:
        train, val = train[: args.limit], val[: args.limit]
    files = {}
    split_of = {}
    for subjects, tag in ((train, "train"), (val, "val")):
        for s in subjects:
            for t, f in source_files(args.domain, s).items():
                files[(s, t)] = f
                split_of[(s, t)] = tag
    meshes = sorted(files)
    n_expr = sum(t != "original" for _, t in meshes)
    print(f"[teacher] {args.domain}: {len(train)} soggetti di training, {len(val)} di validazione, "
          f"{len(meshes)} mesh ({n_expr} con espressione), rotazione {rot_name}", flush=True)

    view = args.stage / "view"
    build_view(view, files)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_check > 0:
        # zar.frame_check legge view_dir, out_root, size e frame_check da args
        ns = argparse.Namespace(view_dir=view, out_root=args.out_dir, size=args.size, frame_check=args.frame_check)
        zar.frame_check(ns, train)
        return

    timing = {"domain": args.domain, "n_meshes": len(meshes), "workers": args.workers, "yaws": yaws}
    t0 = time.time()
    camera = zar.compute_camera(view, meshes, yaws, rot)
    camera.update(base_rotation=rot_name, view_dir=str(view))
    timing["camera_s"] = time.time() - t0
    print(f"[teacher] camera su {len(meshes)} mesh in {timing['camera_s']:.0f}s: "
          f"center={np.round(camera['center'], 4).tolist()} scale={camera['scale']:.4f}", flush=True)

    # Crop: detector sui render ombreggiati delle original di un campione di soggetti di training.
    rng = np.random.default_rng(args.seed)
    calib = sorted(rng.choice(train, size=min(args.calib_subjects, len(train)), replace=False).tolist())
    shaded = args.stage / "shaded"
    calib_meshes = [(s, "original") for s in calib]
    timing["render_shaded_s"] = render_all(view, shaded, calib_meshes, yaws, camera, rot, "shaded",
                                           args.size, args.workers)
    pargs = argparse.Namespace(out_root=shaded, embed_workers=args.workers, device="cpu", overwrite=False)
    t1 = time.time()
    calibration = perc.calibrate_arcface({"original": calib}, yaws, pargs)
    timing["calibration_s"] = time.time() - t1
    timing["n_calibration_renders"] = len(calib_meshes) * len(yaws)

    # Normal map di tutte le mesh, stesso crop.
    normals = args.stage / "normals"
    timing["render_normals_s"] = render_all(view, normals, meshes, yaws, camera, rot, "normals",
                                            args.size, args.workers)
    cal_path = arcface_fixed.calibration_path(normals)
    shutil.copyfile(arcface_fixed.calibration_path(shaded), cal_path)
    topologies: dict[str, list[str]] = {}
    for s, t in meshes:
        topologies.setdefault(t, []).append(s)
    pargs.out_root = normals
    t1 = time.time()
    emb = perc.stage_embed("arcface", topologies, yaws, pargs)
    timing["embed_s"] = time.time() - t1

    E = np.stack([np.stack([emb[render.render_name(t, s, y)] for y in yaws]) for s, t in meshes]).astype(np.float32)
    norms = np.linalg.norm(E.astype(np.float64), axis=2)
    if not np.isfinite(norms).all() or np.abs(norms - 1).max() > 1e-3:
        raise SystemExit(f"embedding per vista non finiti o non normalizzati (|norma - 1| max {np.abs(norms - 1).max():.2e})")
    Z = E.astype(np.float64).mean(axis=1)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    np.savez(args.out_dir / "teacher.npz", E=E, Z=Z.astype(np.float32),
             subjects=np.asarray([s for s, _ in meshes]), topologies=np.asarray([t for _, t in meshes]),
             split=np.asarray([split_of[m] for m in meshes]),
             files=np.asarray([str(files[m]) for m in meshes]), yaws=np.asarray(yaws))
    (args.out_dir / "camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    shutil.copyfile(cal_path, args.out_dir / "arcface_align.json")

    # Png di controllo: i primi 3 soggetti di training, render e ritagli 112x112 delle normal map.
    from PIL import Image
    import cv2

    ctrl = args.out_dir / "control"
    ctrl.mkdir(exist_ok=True)
    transforms = arcface_fixed.load_transforms(cal_path)
    for s in train[:3]:
        row = [perc.load_render(normals, "original", s, y) for y in yaws]
        crops = [cv2.warpAffine(img, transforms[float(y)], (112, 112), borderValue=0.0) for img, y in zip(row, yaws)]
        Image.fromarray(np.concatenate(row, axis=1)).save(ctrl / f"{s}__original_normals.png")
        Image.fromarray(np.concatenate(crops, axis=1)).save(ctrl / f"{s}__crops112.png")
        if s in calib:
            shaded_row = [perc.load_render(shaded, "original", s, y) for y in yaws]
            Image.fromarray(np.concatenate(shaded_row, axis=1)).save(ctrl / f"{s}__original_shaded.png")

    n_norm = len(meshes)
    cpu = args.workers
    timing.update(
        total_s=time.time() - t0,
        per_mesh_wall_s={"render_normals": timing["render_normals_s"] / n_norm, "embed": timing["embed_s"] / n_norm},
        per_mesh_cpu_s={"render_normals": timing["render_normals_s"] * cpu / n_norm,
                        "embed": timing["embed_s"] * cpu / n_norm,
                        "calibration_per_calib_mesh": timing["calibration_s"] * cpu / max(len(calib_meshes), 1),
                        "render_shaded_per_calib_mesh": timing["render_shaded_s"] * cpu / max(len(calib_meshes), 1)},
    )
    (args.out_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    print(f"[teacher] {E.shape} -> {args.out_dir / 'teacher.npz'}", flush=True)
    print(f"[teacher] costo per mesh (3 viste), parete con {cpu} processi: "
          f"render {timing['per_mesh_wall_s']['render_normals'] * 1e3:.1f} ms, "
          f"embed {timing['per_mesh_wall_s']['embed'] * 1e3:.1f} ms; CPU-s: "
          f"render {timing['per_mesh_cpu_s']['render_normals']:.2f}, embed {timing['per_mesh_cpu_s']['embed']:.2f}; "
          f"calibrazione {timing['calibration_s']:.0f}s su {len(calib_meshes)} mesh; totale {timing['total_s']:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
