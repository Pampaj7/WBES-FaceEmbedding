#!/usr/bin/env python3
"""Render delle mesh held-out in 3 viste, con UNA sola camera e la normalizzazione di Chamfer.

Usa il renderer di ``v2_work/phase0/render_mesh.py`` (numpy+PIL, luce fissa, sfondo nero,
proiezione ortografica).  Qui si aggiungono due cose, e solo due.

**Normalizzazione per mesh.**  Prima del render ogni mesh passa da
``common.maxabs_normalize``: centro sulla media dei vertici, divisione per il massimo
valore assoluto.  E' esattamente la normalizzazione geometrica di ``dataset_gtready``, cioe'
quella con cui il repo misura Chamfer, e per questo va fatta anche qui: altrimenti le
metriche percettive guardano una mesh e Chamfer un'altra, e le due colonne della tabella
non sono confrontabili.  La versione precedente di questo file affermava che
"posizione e dimensione della testa restano informazione di identita' come lo sono per
Chamfer": era falso in tutt'e due i sensi.  Chamfer non le vede affatto, perche' normalizza
ogni mesh; e la posizione/dimensione della mesh grezza non e' informazione di identita' ma
di topologia, perche' media dei vertici e bbox si spostano quando cambia la densita' dei
vertici (misurato su REMESH id0000: original contro down8k, 4.5% di scala e ~0.04 di
centroide in unita' normalizzate, per lo STESSO soggetto).

**Camera condivisa.**  Il default del renderer normalizza sul bbox della singola mesh, e
cosi' una topologia piu' stretta verrebbe disegnata piu' grande della sua stessa original.
``center`` e ``scale`` sono invece calcolati una volta sola sulle mesh gia' normalizzate e
su tutte le viste (centro = media dei centri di bbox, scala = la piu' piccola che non
taglia nessuna mesh in nessuna vista) e salvati in ``renders/camera.json``.  Camera fissa
piu' mesh normalizzata vuol dire che il volto cade sempre nello stesso riquadro
dell'immagine, ed e' quello che permette ad ArcFace di usare un ritaglio geometrico fisso
invece del detector (vedi ``arcface_fixed.py``).

  aau/run_baselines.sh aau/baselines/render_cache.py --workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

sys.path.insert(0, str(common.REPO_ROOT / "v2_work" / "phase0"))

# _to_screen / _yaw_rotate / FACE_SPAN sono interni al renderer: si riusano di proposito,
# perche' la camera condivisa deve seguire esattamente la stessa convenzione di assi e
# di inquadratura di render_mesh(), non una sua copia.
from render_mesh import FACE_SPAN, _to_screen, _yaw_rotate, render_mesh  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-root", type=Path, default=common.OUT_ROOT)
    p.add_argument("--settings", type=str, default=",".join(common.SETTINGS))
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in common.VIEW_YAWS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--subject-set", type=str, default="heldout", choices=common.SUBJECT_SETS,
                   help="heldout = split del repo; facebench_first100 = i soggetti della Tabella 2")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def render_name(subject: str, topology: str, yaw: float) -> str:
    return f"{common.mesh_name(subject, topology)}__yaw{yaw:+07.2f}"


def load_normalized(subject: str, topology: str):
    """(vertici normalizzati come per Chamfer, facce)."""
    V, F = common.load_verts_faces(subject, topology)
    return common.maxabs_normalize(V), F


def _required_extent(V: np.ndarray, center: np.ndarray, yaw: float) -> float:
    """Estensione minima che render_mesh deve ricevere per non tagliare la mesh."""
    S = _to_screen(_yaw_rotate(V, yaw, center) if yaw else np.asarray(V, dtype=np.float64))
    centre_screen = _to_screen(np.asarray(center, dtype=np.float64)[None])[0]
    return float(2.0 * FACE_SPAN * np.abs(S[:, :2] - centre_screen[:2]).max())


def compute_camera(meshes, yaws, margin: float = 1.02) -> dict:
    """Camera unica: centro medio dei bbox, scala che inquadra tutto in ogni vista."""
    centres = []
    for subject, topology in meshes:
        V, _ = load_normalized(subject, topology)
        centres.append((V.min(axis=0) + V.max(axis=0)) / 2.0)
    center = np.mean(np.stack(centres), axis=0)

    extent = 0.0
    for subject, topology in meshes:
        V, _ = load_normalized(subject, topology)
        for yaw in yaws:
            extent = max(extent, _required_extent(V, center, yaw))
    return {"center": center.tolist(), "scale": float(extent * margin),
            "yaws": [float(y) for y in yaws], "n_meshes": len(meshes),
            "normalize": "maxabs"}


def _render_one(task) -> str:
    subject, topology, yaw, center, scale, size, out_path = task
    from PIL import Image

    V, F = load_normalized(subject, topology)
    center = np.asarray(center, dtype=np.float64)
    if yaw:
        V = _yaw_rotate(V, yaw, center)
    Image.fromarray(render_mesh(V, F, size=size, scale=scale, center=center)).save(out_path)
    return out_path


def main() -> None:
    import multiprocessing as mp

    args = parse_args()
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    subjects = common.subject_set(args.subject_set)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    topologies = sorted({t for pair in common.all_topology_pairs(settings) for t in pair})
    meshes = [(s, t) for t in topologies for s in subjects]

    out_dir = args.out_root / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "camera.json"

    t0 = time.time()
    cached = json.loads(camera_path.read_text()) if camera_path.exists() else None
    if cached is not None and cached.get("normalize") != "maxabs":
        # Camera della versione senza normalizzazione per mesh: i suoi render e i nuovi non
        # stanno nello stesso spazio, quindi va rifatta insieme a tutte le immagini.
        print(f"[render] camera in cache senza normalizzazione maxabs: la rifaccio "
              f"(e servono anche tutti i render, usa --overwrite)", flush=True)
        cached = None
    if cached is not None and not args.overwrite:
        camera = cached
        print(f"[render] camera dalla cache: {camera_path}", flush=True)
    else:
        camera = compute_camera(meshes, yaws)
        camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")
        print(f"[render] camera calcolata su {len(meshes)} mesh in {time.time() - t0:.0f}s", flush=True)
    print(f"[render] center={np.round(camera['center'], 4).tolist()} scale={camera['scale']:.4f} "
          f"yaws={yaws} size={args.size}", flush=True)

    tasks = []
    for subject, topology in meshes:
        for yaw in yaws:
            out_path = out_dir / f"{render_name(subject, topology, yaw)}.png"
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append((subject, topology, yaw, camera["center"], camera["scale"],
                          args.size, str(out_path)))
    total = len(meshes) * len(yaws)
    print(f"[render] {len(tasks)}/{total} render da fare con {args.workers} worker", flush=True)
    if not tasks:
        return

    t1 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4), start=1):
            if done % 200 == 0 or done == len(tasks):
                rate = done / max(time.time() - t1, 1e-9)
                print(f"[render] {done}/{len(tasks)} ({rate:.1f}/s)", flush=True)
    print(f"[render] fine in {time.time() - t1:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
