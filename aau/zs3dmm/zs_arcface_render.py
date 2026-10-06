#!/usr/bin/env python3
"""ArcFace su render di sola geometria, sulle 600 mesh valutate di un dominio zero-shot.

    aau/baselines/run_bl.sh aau/zs3dmm/zs_arcface_render.py --view-dir datasets/FACEVERSE_ZS/expr_view/npz \\
        --out-root aau/runs/arcface_render_zs/fv_expr/shaded --base-rotation none --workers 32
    (zs_arcface.sbatch)

E' la pipeline di Multiface (``aau/multiface/ws3a_render.py`` + ``ws3a_perceptual.py``) portata
sulle viste zero-shot: stesso renderer, ``normalize_maxabs`` per mesh, camera unica su tutte le
mesh, yaw 0 / -30 / +30, 512 px; crop fisso di ``arcface_fixed`` ricalibrato su QUESTI render
(``ws3a_perceptual.calibrate_arcface``, importata: detector su tutti i render, mediana per yaw dei
5 landmark, similarita' congelata), embedding per vista con ``ws3a_perceptual.stage_embed``. I
render stanno in ``<out-root>/renders/<topologia>/`` con i nomi di ``ws3a_render.render_path``,
quindi le due funzioni importate li trovano senza adattatori.

Cosa cambia rispetto a Multiface:

1. Le mesh: ``<view-dir>/<soggetto>_GTready_<topologia>.npz``, i soggetti del ``subjects.json``
   scritto da zs_stage.py (``select_subjects``: gli stessi 100 del summarizer, che li ricontrolla),
   le 6 topologie di ``TOPOLOGIES``.
2. La rotazione fissa verso il frame del renderer (alto -y, naso -z) dipende dal dominio:
   FaceVerse ha gia' quel frame (tabella di aau/runs/ws_frame), HIFI3D ha alto +y e naso +z e
   vuole Rx(180). ``--frame-check N`` la verifica PRIMA di renderizzare: disegna i primi N
   soggetti (topologia ``original``, yaw 0) con le quattro rotazioni candidate, fa girare il
   detector e stampa se i landmark sono in ordine (occhi sopra il naso, naso sopra la bocca,
   occhio sinistro dell'immagine a sinistra). Le facce non si invertono: il renderer ombreggia
   a due facce (``render_mesh._normals``), il verso dei triangoli non entra nel render.
3. ``--mode normals``: normal map in spazio camera, RGB = (n + 1) / 2, n girata verso la camera
   come in ``_normals``. Il crop NON si ricalibra: stessa camera e stessa geometria danno lo
   stesso riquadro del volto, quindi si copia ``arcface_align.json`` dai render ombreggiati
   (``--calibration-from``) e la differenza fra le due righe e' solo l'aspetto del render.

Scrive ``<out-root>/arcface_views.npz``: ``E`` (n_mesh, n_yaw, 512) float32 L2-normalizzati per
vista, ``subjects``, ``topologies``, ``yaws``; e in ``<out-root>/control/`` i png di controllo
(render delle 6 topologie e delle 3 viste, e i ritagli 112x112 che vede ArcFace).
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
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(AAU_DIR / "multiface"))

import ws3a_perceptual as perc  # noqa: E402
import ws3a_render as render  # noqa: E402
from render_mesh import BACKGROUND, FACE_SPAN, SSAA, _normals, _to_screen, _yaw_rotate, render_mesh  # noqa: E402

import arcface_fixed  # noqa: E402  (sul path grazie a ws3a_perceptual)

# Le stesse di ``zs_stage.TOPOLOGIES``, che qui non si importa: zs_stage tira dentro igl, assente
# dal venv delle baseline (quello con insightface). Il summarizer le ricontrolla.
TOPOLOGIES = ("crop", "down8k", "noisy", "original", "remesh", "up60k")

# Dal frame del dominio a quello del renderer: solo rotazioni proprie (det = +1).
BASE_ROTATIONS = {"none": np.eye(3), "x180": np.diag([1.0, -1.0, -1.0]),
                  "y180": np.diag([-1.0, 1.0, -1.0]), "z180": np.diag([-1.0, -1.0, 1.0])}

# Stato per-worker del render (fork).
_VIEW_DIR: Path | None = None
_ROT = np.eye(3)
_MODE = "shaded"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--view-dir", type=Path, required=True, help="<vista>/npz")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--base-rotation", choices=sorted(BASE_ROTATIONS), required=True)
    p.add_argument("--mode", choices=("shaded", "normals"), default="shaded")
    p.add_argument("--calibration-from", type=Path, default=None,
                   help="--mode normals: arcface_align.json dei render ombreggiati")
    p.add_argument("--subjects-json", type=Path, required=True,
                   help="subjects.json di zs_stage.py (i soggetti di select_subjects)")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--yaws", type=str, default=",".join(str(y) for y in render.VIEW_YAWS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--frame-check", type=int, default=0,
                   help=">0: solo il controllo del frame sui primi N soggetti, poi esce")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    # Attributi che calibrate_arcface / stage_embed di ws3a_perceptual leggono da args.
    a.embed_workers, a.device = a.workers, "cpu"
    return a


def mesh_file(view_dir: Path, subject: str, topology: str) -> Path:
    return view_dir / f"{subject}_GTready_{topology}.npz"


def load_verts_faces(view_dir: Path, subject: str, topology: str, rot: np.ndarray):
    """Come ``ws3a_render.load_verts_faces``: rotazione attorno all'origine, poi maxabs."""
    with np.load(mesh_file(view_dir, subject, topology), allow_pickle=False) as d:
        V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
        V = np.asarray(V, np.float64) @ rot.T
        return render.normalize_maxabs(V), np.asarray(F, np.int64)


def render_normals(V: np.ndarray, F: np.ndarray, size: int, scale: float, center: np.ndarray) -> np.ndarray:
    """Gemello di ``render_mesh`` con colore = normale in spazio camera invece del grigio.

    Stessa proiezione, stesso painter's, stesso SSAA: cambia solo il riempimento dei triangoli.
    """
    S = _to_screen(np.asarray(V, dtype=np.float64))
    tri = S[np.asarray(F, dtype=np.int64)]
    rgb = np.round(255.0 * (_normals(tri) + 1.0) / 2.0).astype(np.uint8)
    res = size * SSAA
    centre = _to_screen(np.asarray(center, float)[None])[0]
    px_per_unit = FACE_SPAN * res / max(float(scale), 1e-12)
    px = (tri[:, :, 0] - centre[0]) * px_per_unit + res / 2.0
    py = res / 2.0 - (tri[:, :, 1] - centre[1]) * px_per_unit
    order = np.argsort(tri[:, :, 2].mean(axis=1), kind="stable")
    polys = np.stack([px, py], axis=2)[order].reshape(-1, 6).tolist()
    colors = [tuple(c) for c in rgb[order].tolist()]

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (res, res), (BACKGROUND,) * 3)
    draw = ImageDraw.Draw(img)
    for poly, c in zip(polys, colors):
        draw.polygon(poly, fill=c)
    if SSAA != 1:
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def draw(V, F, size, scale, center, mode: str) -> np.ndarray:
    if mode == "normals":
        return render_normals(V, F, size, scale, center)
    return render_mesh(V, F, size=size, scale=scale, center=center)


def compute_camera(view_dir: Path, meshes, yaws, rot: np.ndarray, margin: float = 1.02) -> dict:
    """Come ``ws3a_render.compute_camera``: centro medio dei bbox, scala che inquadra tutto."""
    centres = []
    for subject, topology in meshes:
        V, _ = load_verts_faces(view_dir, subject, topology, rot)
        centres.append((V.min(axis=0) + V.max(axis=0)) / 2.0)
    center = np.mean(np.stack(centres), axis=0)
    extent = 0.0
    for subject, topology in meshes:
        V, _ = load_verts_faces(view_dir, subject, topology, rot)
        for yaw in yaws:
            extent = max(extent, render._required_extent(V, center, yaw))
    return {"center": center.tolist(), "scale": float(extent * margin),
            "yaws": [float(y) for y in yaws], "n_meshes": len(meshes)}


def _init_render_worker(view_dir: Path, rot: np.ndarray, mode: str) -> None:
    global _VIEW_DIR, _ROT, _MODE
    _VIEW_DIR, _ROT, _MODE = view_dir, rot, mode


def _render_one(task) -> str:
    subject, topology, yaw, center, scale, size, out_path = task
    from PIL import Image

    V, F = load_verts_faces(_VIEW_DIR, subject, topology, _ROT)
    center = np.asarray(center, dtype=np.float64)
    if yaw:
        V = _yaw_rotate(V, yaw, center)
    Image.fromarray(draw(V, F, size, scale, center, _MODE)).save(out_path)
    return out_path


# ------------------------------------------------------------------ controllo del frame

def kps_upright(kps: np.ndarray) -> bool:
    """Landmark insightface (occhio sx, occhio dx, naso, bocca sx, bocca dx) in ordine da volto dritto."""
    eyes_y, nose_y, mouth_y = kps[:2, 1].mean(), kps[2, 1], kps[3:, 1].mean()
    return bool(eyes_y < nose_y < mouth_y and kps[0, 0] < kps[1, 0] and kps[3, 0] < kps[4, 0])


def frame_check(args, subjects) -> None:
    """Le quattro rotazioni candidate sui primi N soggetti: png + verdetto del detector."""
    from PIL import Image

    out = args.out_root / "frame_check"
    out.mkdir(parents=True, exist_ok=True)
    probe = arcface_fixed.ArcFaceDetectorProbe()
    for subject in subjects[: args.frame_check]:
        tiles = []
        for name, rot in BASE_ROTATIONS.items():
            V, F = load_verts_faces(args.view_dir, subject, "original", rot)
            img = render_mesh(V, F, size=args.size)
            kps = probe.kps(img)
            verdict = "nessun volto" if kps is None else ("dritto" if kps_upright(kps) else "volto, NON dritto")
            # Il detector scatta anche sul volto CAVO visto da dietro (illusione della maschera):
            # il rilievo si controlla a parte. In un volto convesso il vertice piu' vicino alla
            # camera e' la punta del naso, al centro del bbox; in uno cavo e' sul bordo.
            S = _to_screen(V)
            lo, hi = S[:, :2].min(0), S[:, :2].max(0)
            tip = (S[np.argmax(S[:, 2]), :2] - lo) / (hi - lo)
            verdict += f", vertice piu' vicino alla camera a ({tip[0]:.2f}, {1 - tip[1]:.2f}) del bbox (x, y dall'alto)"
            print(f"[frame-check] {subject} {name:5s}: {verdict}"
                  + ("" if kps is None else f"  kps={np.round(kps).astype(int).tolist()}"), flush=True)
            tiles.append(img)
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out / f"{subject}__none_x180_y180_z180.png")
    print(f"[frame-check] png in {out} (da sinistra: {', '.join(BASE_ROTATIONS)})", flush=True)


# --------------------------------------------------------------------- png di controllo

def control_pngs(args, subjects, yaws, calibration: dict) -> None:
    """Primi 3 soggetti: le 6 topologie a yaw 0, le viste di ``original``, i ritagli 112x112."""
    import cv2
    from PIL import Image

    out = args.out_root / "control"
    out.mkdir(parents=True, exist_ok=True)
    transforms = {float(v["yaw"]): np.asarray(v["transform"]) for v in calibration["views"].values()}
    for subject in subjects[:3]:
        topo_row = [perc.load_render(args.out_root, t, subject, 0.0) for t in TOPOLOGIES]
        Image.fromarray(np.concatenate(topo_row, axis=1)).save(out / f"{subject}__topologies_yaw0.png")
        view_row = [perc.load_render(args.out_root, "original", subject, y) for y in yaws]
        Image.fromarray(np.concatenate(view_row, axis=1)).save(out / f"{subject}__original_views.png")
        crops = []
        for t in TOPOLOGIES:
            for y in yaws:
                img = perc.load_render(args.out_root, t, subject, y)
                crops.append(cv2.warpAffine(img, transforms[float(y)], (112, 112), borderValue=0.0))
        grid = np.concatenate([np.concatenate(crops[i:i + len(yaws)], axis=1)
                               for i in range(0, len(crops), len(yaws))], axis=0)
        Image.fromarray(grid).save(out / f"{subject}__crops112_rows-topologies_cols-yaws.png")
    print(f"[arcface-zs] png di controllo in {out} (topologie: {', '.join(TOPOLOGIES)}; yaw {yaws})", flush=True)


def main() -> None:
    args = parse_args()
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    rot = BASE_ROTATIONS[args.base_rotation]
    staged = json.loads(args.subjects_json.read_text())
    if Path(staged["view_dir"]).resolve() != args.view_dir.resolve():
        raise SystemExit(f"{args.subjects_json}: vista {staged['view_dir']}, non {args.view_dir}")
    subjects = sorted(staged["subjects"])
    print(f"[arcface-zs] {len(subjects)} soggetti (primi {subjects[:3]}), mode={args.mode} "
          f"base_rotation={args.base_rotation} yaws={yaws} size={args.size}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.frame_check > 0:
        frame_check(args, subjects)
        return

    meshes = [(s, t) for s in subjects for t in TOPOLOGIES]
    for t in TOPOLOGIES:
        (args.out_root / "renders" / t).mkdir(parents=True, exist_ok=True)

    # Camera: nessuna cache fra modi diversi, ma la stessa regola; con --mode normals si riusa
    # quella dei render ombreggiati (stesso riquadro, condizione per riusarne il crop).
    camera_path = args.out_root / "renders" / "camera.json"
    t0 = time.time()
    if args.calibration_from is not None:
        src_camera = args.calibration_from.parent / "camera.json"
        camera = json.loads(src_camera.read_text())
        print(f"[arcface-zs] camera dai render ombreggiati: {src_camera}", flush=True)
    elif camera_path.exists() and not args.overwrite:
        camera = json.loads(camera_path.read_text())
        print(f"[arcface-zs] camera dalla cache: {camera_path}", flush=True)
    else:
        camera = compute_camera(args.view_dir, meshes, yaws, rot)
        print(f"[arcface-zs] camera su {len(meshes)} mesh in {time.time() - t0:.0f}s", flush=True)
    if camera.get("base_rotation", args.base_rotation) != args.base_rotation or camera["yaws"] != yaws:
        raise SystemExit(f"camera {camera.get('base_rotation')} {camera['yaws']} != {args.base_rotation} {yaws}")
    camera.update(base_rotation=args.base_rotation, mode=args.mode, view_dir=str(args.view_dir))
    camera_path.write_text(json.dumps(camera, indent=2), encoding="utf-8")
    print(f"[arcface-zs] center={np.round(camera['center'], 4).tolist()} scale={camera['scale']:.4f}", flush=True)

    tasks = [(s, t, y, camera["center"], camera["scale"], args.size, str(render.render_path(args.out_root, t, s, y)))
             for s, t in meshes for y in yaws
             if args.overwrite or not render.render_path(args.out_root, t, s, y).exists()]
    print(f"[arcface-zs] {len(tasks)}/{len(meshes) * len(yaws)} render da fare con {args.workers} worker", flush=True)
    if tasks:
        t1 = time.time()
        with mp.get_context("fork").Pool(args.workers, initializer=_init_render_worker,
                                         initargs=(args.view_dir, rot, args.mode)) as pool:
            for done, _ in enumerate(pool.imap_unordered(_render_one, tasks, chunksize=4), start=1):
                if done % 600 == 0 or done == len(tasks):
                    print(f"[arcface-zs] render {done}/{len(tasks)} ({done / max(time.time() - t1, 1e-9):.1f}/s)",
                          flush=True)

    # Crop fisso: calibrato sui render ombreggiati, copiato per la normal map.
    topologies = {t: list(subjects) for t in TOPOLOGIES}
    cal_path = arcface_fixed.calibration_path(args.out_root)
    if args.calibration_from is not None:
        # Mai ricalibrare qui, nemmeno con --overwrite: il detector sulla normal map darebbe un
        # altro riquadro, e la differenza fra le righe non sarebbe piu' solo l'aspetto del render.
        shutil.copyfile(args.calibration_from, cal_path)
        calibration = json.loads(cal_path.read_text())
        print(f"[arcface-zs] crop copiato da {args.calibration_from} (stessa camera e geometria)", flush=True)
    else:
        calibration = perc.calibrate_arcface(topologies, yaws, args)
    for key, v in sorted(calibration["views"].items()):
        print(f"[arcface-zs] yaw {key}: {v['n_detected']} detection, kps mediani "
              f"{np.round(v['kps_median'], 1).tolist()}, IQR max {np.max(v['kps_iqr_px']):.1f} px", flush=True)
    control_pngs(args, subjects, yaws, calibration)

    emb = perc.stage_embed("arcface", topologies, yaws, args)
    E = np.stack([np.stack([emb[render.render_name(t, s, y)] for y in yaws]) for s, t in meshes]).astype(np.float32)
    np.savez(args.out_root / "arcface_views.npz", E=E, subjects=np.asarray([s for s, _ in meshes]),
             topologies=np.asarray([t for _, t in meshes]), yaws=np.asarray(yaws))
    print(f"[arcface-zs] {E.shape} -> {args.out_root / 'arcface_views.npz'}", flush=True)


if __name__ == "__main__":
    main()
