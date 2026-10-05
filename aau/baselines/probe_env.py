#!/usr/bin/env python3
"""Ricognizione preliminare per le baseline estese: dipendenze, rete, dati, tempi.

Non scrive niente: stampa e basta.  Serve a decidere cosa installare e con che --time
sottomettere i job veri.  Da lanciare dentro il container (aau/run.sh o run_baselines.sh).

  aau/baselines/probe_env.sbatch          # partizione cpu
  aau/baselines/probe_env.sbatch --gpu    # su una L40S (variante nel .sbatch)
"""

from __future__ import annotations

import importlib
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

REQUIRED = ["numpy", "scipy", "pandas", "torch", "PIL", "cv2", "tqdm"]
OPTIONAL = ["insightface", "onnxruntime", "open_clip", "lpips"]
NET_HOSTS = [
    ("pypi.org", 443),
    ("github.com", 443),
    ("huggingface.co", 443),
    ("dl.insightface.net", 443),
]


def probe_packages() -> None:
    print("--- pacchetti ---")
    for name in REQUIRED + OPTIONAL:
        tag = "obbligatorio" if name in REQUIRED else "opzionale"
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok]      {name:<14} {version:<12} ({tag})")
        except Exception as exc:  # noqa: BLE001
            print(f"  [MANCA]   {name:<14} {'':<12} ({tag}) {type(exc).__name__}: {exc}")


def probe_network() -> None:
    print("--- rete dal nodo di calcolo ---")
    for host, port in NET_HOSTS:
        t0 = time.time()
        try:
            socket.create_connection((host, port), timeout=8).close()
            print(f"  [ok]      {host}:{port} in {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"  [KO]      {host}:{port} {type(exc).__name__}: {exc}")


def probe_data() -> None:
    print("--- dati e split held-out ---")
    subjects = common.heldout_subjects()
    print(f"  soggetti held-out: {len(subjects)} (primo={subjects[0]}, ultimo={subjects[-1]})")
    G = common.load_gt_submatrix(subjects)
    print(f"  D_GT sottomatrice: {G.shape} range=[{np.nanmin(G):.4f}, {np.nanmax(G):.4f}]")

    # Controllo incrociato: gt_distance dei pair table del paper vs D_GT ricaricata.
    import csv

    csv_path = common.PAIR_TABLE_ROOT / "original__to__remesh" / "pair_metrics.csv"
    idx = {s: i for i, s in enumerate(subjects)}
    diffs = []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            diffs.append(abs(float(row["gt_distance"]) - G[idx[row["subject_a"]], idx[row["subject_b"]]]))
    diffs = np.asarray(diffs)
    print(f"  |gt_distance(csv) - D_GT|: n={len(diffs)} max={diffs.max():.2e} mean={diffs.mean():.2e}")

    for topology in common.NOCROP_TOPOLOGIES:
        V, F = common.load_verts_faces(subjects[0], topology)
        print(f"  {subjects[0]} {topology:<9} V={V.shape} F={F.shape}")


def probe_render_timing(n_meshes: int = 3) -> None:
    print("--- tempi di render (v2_work/phase0/render_mesh.py) ---")
    sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))
    from render_mesh import mesh_frame, render_mesh  # noqa: E402

    subjects = common.heldout_subjects()[:n_meshes]
    for topology in ("original", "down8k", "up60k"):
        for subject in subjects[:1]:
            V, F = common.load_verts_faces(subject, topology)
            centre, extent = mesh_frame(V)
            t0 = time.time()
            img = render_mesh(V, F, size=512, scale=extent, center=centre)
            print(f"  {subject} {topology:<9} tris={len(F):>7} -> {img.shape} in {time.time() - t0:.2f}s")


def probe_varifold_timing(max_tris: int = 2000) -> None:
    print(f"--- tempi varifold/currents (max_tris={max_tris}) ---")
    sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))
    from measure_distances import currents_distance, mesh_measure, varifold_distance  # noqa: E402

    subjects = common.heldout_subjects()[:2]
    t0 = time.time()
    measures = {
        (s, t): mesh_measure(common.mesh_path(s, t), max_tris=max_tris)
        for s in subjects for t in ("original", "up60k")
    }
    print(f"  4 mesh_measure in {time.time() - t0:.2f}s")

    a = measures[(subjects[0], "original")]
    b = measures[(subjects[1], "up60k")]
    for name, fn in (("varifold", varifold_distance), ("currents", currents_distance)):
        fn(a, b)  # scalda la cache dei self-inner
        t0 = time.time()
        for _ in range(3):
            fn(a, b)
        dt = (time.time() - t0) / 3.0
        print(f"  {name:<9} {dt * 1000:.0f} ms/coppia -> {dt * 103950 / 3600:.2f} h per 103950 coppie (1 core)")


def probe_torch() -> None:
    print("--- torch ---")
    import torch

    print(f"  torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu={torch.cuda.get_device_name(0)}")


def main() -> None:
    print(f"host={socket.gethostname()} python={sys.version.split()[0]}")
    print(f"MESH_ROOT={common.MESH_ROOT}")
    print(f"GT_MATRIX={common.GT_MATRIX}")
    print(f"OUT_ROOT={common.OUT_ROOT}")
    probe_torch()
    probe_packages()
    probe_network()
    probe_data()
    probe_render_timing()
    probe_varifold_timing()
    print("probe: fine")


if __name__ == "__main__":
    main()
