#!/usr/bin/env python3
"""G0 (``aau/runs/evidence/diagnostics/PROTOCOL_G0.md`` sez. 2 e 7): l'insegnante ArcFace-render sugli insiemi di G0.

    aau/baselines/run_bl.sh aau/diagnostics/g0_teacher.py frame --set hifi3d --stage /tmp/<job>/g0 [--mesh-root ...]
    aau/baselines/run_bl.sh aau/diagnostics/g0_teacher.py run   --set hifi3d --stage /tmp/<job>/g0 --workers 32
    aau/baselines/run_bl.sh aau/diagnostics/g0_teacher.py cost                                  (un thread)
    (g0.sbatch, passi teacher e cost)

La configurazione dell'insegnante del pilota (``aau/distill/teacher_labels.py``), cioe' la normal map del gate, con la
regola di calibrazione del gate (``aau/zs3dmm/zs_arcface_render.py``: crop su TUTTE le mesh dell'insieme). Importati e
non riscritti: ``zs_arcface_render`` (rotazioni, ``load_verts_faces``, ``compute_camera``, render, ``kps_upright``),
``teacher_labels.render_all`` (render in parallelo), ``ws3a_perceptual.calibrate_arcface`` e ``stage_embed``,
``arcface_fixed``. Per insieme: camera unica sulle sue mesh, render ombreggiati di tutte, detector -> crop fisso,
normal map di tutte con lo stesso crop, ArcFace ``w600k_r50`` per vista, media delle 3 viste poi L2.

- ``frame``: le 4 rotazioni candidate sui primi 4 soggetti (``original``, render ombreggiato senza camera, come
  ``zs_arcface_render.frame_check``): verdetto del detector (``kps_upright``) e vertice piu' vicino alla camera nel
  terzo centrale del bbox (volto convesso, non cavo). Scelta = la rotazione che passa entrambi su piu' soggetti;
  ``g0/teacher/<insieme>/frame_check.json``. ``run`` si ferma se la scelta e' diversa dall'attesa (``SETS``).
- ``run``: render e embedding in ``--stage`` (/tmp), uscite in ``g0/teacher/<insieme>``: ``teacher.npz`` (``E`` (n, 3,
  512) per vista, ``Z`` (n, 512), ``names``, ``subjects``, ``topologies``, ``yaws``), ``camera.json``,
  ``arcface_align.json`` (con le statistiche del detector), ``timing.json``, due png di controllo (fuori da git). Su
  HIFI3D anche K2 (``k2.json``): soggetti ed embedding per vista contro il gate.
- ``cost``: costo per mesh, sez. 7: 54 mesh (primi 3 soggetti valutati per id x 6 etichette x 3 viste del benchmark),
  un processo a un thread, in memoria (render -> ritaglio -> ArcFace, senza png); controllo: Z = ``teacher.npz``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402
import d3_pools  # noqa: E402

sys.path.insert(0, str(diag.REPO / "aau/distill"))
import teacher_labels as tl  # noqa: E402  (mette aau/zs3dmm sul path; render_all)

zar = tl.zar
perc, render, arcface_fixed = zar.perc, zar.render, zar.arcface_fixed

G0 = diag.EV / "g0"
DS = diag.REPO / "datasets"
VIEW6 = zar.TOPOLOGIES                                       # crop, down8k, noisy, original, remesh, up60k
# insieme: (tipo, cartella delle mesh o pool, etichette, rotazione attesa verso il frame del renderer)
SETS = {
    "hifi3d": ("view", "hifi3d", VIEW6, "x180"),
    "facescape": ("view", "facescape", VIEW6, "x180"),
    "faceverse_neutral": ("view", "faceverse", VIEW6, "none"),
    "flame2023_s1": ("gen", "flame2023_s1", diag.LABELS, "x180"),
    "famos": ("gen", "famos", diag.LABELS, "x180"),
    "bfm": ("gen", "heldout", diag.LABELS, "none"),
    "ict": ("gen", "heldout", diag.LABELS, "x180"),
    "gnm": ("gen", "heldout", diag.LABELS, "x180"),
}
GATE_HIFI = diag.REPO / "aau/runs/arcface_render_zs/hifi3d/normals/arcface_views.npz"
GATE_HIFI_SUBJ = diag.REPO / "aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip/subjects.json"
N_FRAME = 4
N_COST_SUBJ = 3
COST_VIEWS = ("facescape", "hifi3d", "faceverse_neutral")


def view_dir(name: str, mesh_root: Path | None) -> Path:
    kind, src, _, _ = SETS[name]
    if kind == "view":
        return d3_pools.POOLS[src]["view"] / "npz"
    if mesh_root is None:
        raise SystemExit(f"{name}: serve --mesh-root (mesh rigenerate da g0_meshes.py)")
    return mesh_root / src


def subjects_of(name: str) -> list[str]:
    """I soggetti dell'insieme: i 100 valutati delle viste, 200 FLAME, 80 FaMoS TRAIN, 100 held-out."""
    kind, src, _, _ = SETS[name]
    if kind == "view":
        return sorted(d3_pools.subjects(src)["evaluated"])
    if name == "flame2023_s1":
        names = diag.set_names(name)
    elif name == "famos":
        with np.load(diag.EV / "d3" / "famos" / "scale_table.npz", allow_pickle=True) as z:
            names = sorted(str(n) for n in z["names"])
    else:
        names = diag.set_names(name)
    return sorted({diag.split_name(n)[0] for n in names}, key=lambda s: int(s[2:]))


def meshes_of(name: str, mesh_root: Path | None) -> tuple[Path, list[tuple[str, str]]]:
    view = view_dir(name, mesh_root)
    meshes = [(s, t) for s in subjects_of(name) for t in SETS[name][2]]
    miss = [zar.mesh_file(view, s, t) for s, t in meshes if not zar.mesh_file(view, s, t).exists()]
    if miss:
        raise SystemExit(f"{name}: {len(miss)} mesh assenti (es. {miss[:2]})")
    return view, meshes


# ------------------------------------------------------------------------------------------- frame check

def frame(name: str, mesh_root: Path | None) -> dict:
    from render_mesh import _to_screen, render_mesh
    view, meshes = meshes_of(name, mesh_root)
    subjects = sorted({s for s, _ in meshes}, key=lambda s: int(s[2:]))[:N_FRAME]
    probe = arcface_fixed.ArcFaceDetectorProbe()
    per, score = {}, {r: 0 for r in zar.BASE_ROTATIONS}
    for s in subjects:
        per[s] = {}
        for r, R in zar.BASE_ROTATIONS.items():
            V, F = zar.load_verts_faces(view, s, "original", R)
            kps = probe.kps(render_mesh(V, F, size=512))
            S = _to_screen(V)
            lo, hi = S[:, :2].min(0), S[:, :2].max(0)
            tip = (S[np.argmax(S[:, 2]), :2] - lo) / (hi - lo)
            tip = [float(tip[0]), float(1 - tip[1])]
            up = kps is not None and zar.kps_upright(kps)
            convex = all(1 / 3 <= x <= 2 / 3 for x in tip)
            per[s][r] = {"face": kps is not None, "upright": bool(up), "tip_bbox_xy": tip, "convex": bool(convex)}
            score[r] += int(up and convex)
    best = max(score.values())
    chosen = [r for r, v in score.items() if v == best]
    rec = {"set": name, "expected": SETS[name][3], "score": score,
           "chosen": chosen[0] if len(chosen) == 1 and best > 0 else None, "subjects": per}
    rec["agrees"] = rec["chosen"] == rec["expected"]
    diag.atomic_json(G0 / "teacher" / name / "frame_check.json", rec)
    print(f"[g0-teacher] frame {name}: punteggi {score}, scelta {rec['chosen']}, attesa {rec['expected']}", flush=True)
    return rec


# --------------------------------------------------------------------------------------------------- run

def run(name: str, stage: Path, mesh_root: Path | None, workers: int, size: int) -> None:
    fc = json.loads((G0 / "teacher" / name / "frame_check.json").read_text())
    if not fc["agrees"]:
        raise SystemExit(f"{name}: frame check {fc['chosen']} contro l'attesa {fc['expected']}: fermo (sez. 2)")
    rot_name = SETS[name][3]
    rot = zar.BASE_ROTATIONS[rot_name]
    yaws = [float(y) for y in render.VIEW_YAWS]
    view, meshes = meshes_of(name, mesh_root)
    out = G0 / "teacher" / name
    root = stage / name
    t0 = time.time()
    timing = {"set": name, "n_meshes": len(meshes), "workers": workers, "yaws": yaws, "host": os.uname().nodename}
    camera = zar.compute_camera(view, meshes, yaws, rot)
    camera.update(base_rotation=rot_name, mode="normals", view_dir=str(view))
    timing["camera_s"] = time.time() - t0
    print(f"[g0-teacher] {name}: {len(meshes)} mesh, rotazione {rot_name}, camera in {timing['camera_s']:.0f}s "
          f"center={np.round(camera['center'], 4).tolist()} scale={camera['scale']:.4f}", flush=True)
    shaded, normals = root / "shaded", root / "normals"
    timing["render_shaded_s"] = tl.render_all(view, shaded, meshes, yaws, camera, rot, "shaded", size, workers)
    topologies: dict[str, list[str]] = {}
    for s, t in meshes:
        topologies.setdefault(t, []).append(s)
    pargs = argparse.Namespace(out_root=shaded, embed_workers=workers, device="cpu", overwrite=False)
    t1 = time.time()
    calibration = perc.calibrate_arcface(topologies, yaws, pargs)        # detector su TUTTI i render (gate)
    timing["calibration_s"] = time.time() - t1
    timing["render_normals_s"] = tl.render_all(view, normals, meshes, yaws, camera, rot, "normals", size, workers)
    cal_path = arcface_fixed.calibration_path(normals)
    shutil.copyfile(arcface_fixed.calibration_path(shaded), cal_path)
    pargs.out_root = normals
    t1 = time.time()
    emb = perc.stage_embed("arcface", topologies, yaws, pargs)
    timing["embed_s"] = time.time() - t1
    E = np.stack([np.stack([emb[render.render_name(t, s, y)] for y in yaws]) for s, t in meshes]).astype(np.float32)
    norms = np.linalg.norm(E.astype(np.float64), axis=2)
    if not np.isfinite(E).all() or np.abs(norms - 1).max() > 1e-3:                  # K4
        raise SystemExit(f"{name}: embedding non finiti o non normalizzati (|norma - 1| max {np.abs(norms - 1).max():.2e})")
    Z = E.astype(np.float64).mean(axis=1)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    out.mkdir(parents=True, exist_ok=True)
    diag.atomic_savez(out / "teacher.npz", E=E, Z=Z.astype(np.float32), yaws=np.asarray(yaws),
                      names=np.asarray([zar.mesh_file(view, s, t).name for s, t in meshes]),
                      subjects=np.asarray([s for s, _ in meshes]), topologies=np.asarray([t for _, t in meshes]))
    (out / "camera.json").write_text(json.dumps(camera, indent=1) + "\n")
    shutil.copyfile(cal_path, out / "arcface_align.json")
    control(out, normals, meshes, yaws, cal_path)
    timing["total_s"] = time.time() - t0
    timing["K4"] = {"n": int(len(E)), "max_abs_norm_minus_1": float(np.abs(norms - 1).max()), "finite": True}
    timing["detector"] = calibration["detector"]
    timing["n_detected_by_yaw"] = {k: v["n_detected"] for k, v in calibration["views"].items()}
    (out / "timing.json").write_text(json.dumps(timing, indent=1) + "\n")
    if name == "hifi3d":
        k2(out, E, meshes)
    print(f"[g0-teacher] {name}: {E.shape} -> {out / 'teacher.npz'} in {timing['total_s']:.0f}s "
          f"(camera {timing['camera_s']:.0f}, ombreggiati {timing['render_shaded_s']:.0f}, calibrazione "
          f"{timing['calibration_s']:.0f}, normal map {timing['render_normals_s']:.0f}, embedding {timing['embed_s']:.0f})",
          flush=True)


def control(out: Path, normals: Path, meshes, yaws, cal_path: Path) -> None:
    """Primo soggetto: normal map delle 3 viste della original e i ritagli 112x112 (png fuori da git)."""
    import cv2
    from PIL import Image
    s = meshes[0][0]
    tr = arcface_fixed.load_transforms(cal_path)
    row = [perc.load_render(normals, "original", s, y) for y in yaws]
    crops = [cv2.warpAffine(img, tr[float(y)], (112, 112), borderValue=0.0) for img, y in zip(row, yaws)]
    (out / "control").mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(row, axis=1)).save(out / "control" / f"{s}__original_normals.png")
    Image.fromarray(np.concatenate(crops, axis=1)).save(out / "control" / f"{s}__crops112.png")


def k2(out: Path, E: np.ndarray, meshes) -> None:
    """K2: soggetti = quelli del gate; embedding per vista = arcface_views.npz del gate (normal map)."""
    gate_subj = sorted(json.loads(GATE_HIFI_SUBJ.read_text())["subjects"])
    with np.load(GATE_HIFI, allow_pickle=True) as z:
        pos = {(str(s), str(t)): k for k, (s, t) in enumerate(zip(z["subjects"], z["topologies"]))}
        Eg = np.asarray(z["E"], np.float64)
        same_yaws = [float(y) for y in z["yaws"]] == [float(y) for y in render.VIEW_YAWS]
    mine = sorted({s for s, _ in meshes})
    have = all(m in pos for m in meshes)
    diff = float(np.abs(E.astype(np.float64) - Eg[[pos[m] for m in meshes]]).max()) if have and same_yaws else float("nan")
    rec = {"subjects_equal_gate": mine == gate_subj, "keys_in_gate": bool(have), "yaws_equal": bool(same_yaws),
           "max_abs_diff_E": diff, "pass": bool(mine == gate_subj and have and diff <= 1e-3)}
    diag.atomic_json(out / "k2.json", rec)
    print(f"[g0-teacher] K2 HIFI3D: {rec}", flush=True)


# -------------------------------------------------------------------------------------------------- costo

def cost(size: int) -> None:
    """Sez. 7: 54 mesh, un processo, un thread; render (lettura + 3 normal map) ed embedding (3 ritagli + ArcFace)."""
    from render_mesh import _yaw_rotate
    yaws = [float(y) for y in render.VIEW_YAWS]
    sample = []
    for name in COST_VIEWS:
        view, meshes = meshes_of(name, None)
        subj = sorted({s for s, _ in meshes}, key=lambda s: int(s[2:]))[:N_COST_SUBJ]
        sample += [(name, view, s, t) for s in subj for t in VIEW6]
    ext, cams, ref = {}, {}, {}
    for name in COST_VIEWS:
        out = G0 / "teacher" / name
        ext[name] = arcface_fixed.ArcFaceFixedCrop(arcface_fixed.load_transforms(out / "arcface_align.json"))
        cams[name] = json.loads((out / "camera.json").read_text())
        with np.load(out / "teacher.npz", allow_pickle=True) as z:
            ref[name] = dict(zip(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]),
                                 np.asarray(z["Z"], np.float64)))

    def one(name, view, s, t):
        cam = cams[name]
        center = np.asarray(cam["center"], np.float64)
        t0 = time.perf_counter()
        V0, F = zar.load_verts_faces(view, s, t, zar.BASE_ROTATIONS[cam["base_rotation"]])
        imgs = [zar.draw(_yaw_rotate(V0, y, center) if y else V0, F, size, cam["scale"], center, "normals")
                for y in yaws]
        t1 = time.perf_counter()
        Ev = np.stack([ext[name](img, y) for img, y in zip(imgs, yaws)]).astype(np.float64)
        z = Ev.mean(axis=0)
        z /= max(float(np.linalg.norm(z)), 1e-9)
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1, z

    one(*sample[0])                                           # riscaldamento, escluso
    recs = []
    for name, view, s, t in sample:
        r, e, z = one(name, view, s, t)
        recs.append({"set": name, "subject": s, "label": t, "render_s": r, "embed_s": e, "total_s": r + e,
                     "max_abs_diff_Z": float(np.abs(z - ref[name][(s, t)]).max())})

    def stats(x):
        x = np.asarray(x)
        return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                "n": int(len(x))}

    cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "?")
    rep = {"definition": "PROTOCOL_G0.md sez. 7", "host": os.uname().nodename, "cpu": cpu, "threads": 1,
           "sample": f"primi {N_COST_SUBJ} soggetti valutati per id x 6 etichette x {', '.join(COST_VIEWS)}",
           "size": size, "yaws": yaws, "overall": {k: stats([r[k] for r in recs]) for k in ("render_s", "embed_s", "total_s")},
           "by_label": {t: {k: stats([r[k] for r in recs if r["label"] == t]) for k in ("render_s", "embed_s", "total_s")}
                        for t in VIEW6},
           "by_set": {n: stats([r["total_s"] for r in recs if r["set"] == n]) for n in COST_VIEWS},
           "max_abs_diff_Z": float(max(r["max_abs_diff_Z"] for r in recs)), "meshes": recs}
    diag.atomic_json(G0 / "cost.json", rep)
    o = rep["overall"]
    print(f"[g0-teacher] costo su {len(recs)} mesh ({cpu}, un thread): totale mediana {o['total_s']['median']:.3f}s "
          f"[p10 {o['total_s']['p10']:.3f}, p90 {o['total_s']['p90']:.3f}], render {o['render_s']['median']:.3f}s, "
          f"embedding {o['embed_s']['median']:.3f}s; |Z - teacher.npz| max {rep['max_abs_diff_Z']:.1e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("frame", "run", "cost"))
    ap.add_argument("--set", choices=sorted(SETS))
    ap.add_argument("--stage", type=Path, help="radice su /tmp per i render (run)")
    ap.add_argument("--mesh-root", type=Path, default=None, help="mesh rigenerate (g0_meshes.py --out)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    if a.step == "frame":
        frame(a.set, a.mesh_root)
    elif a.step == "run":
        run(a.set, a.stage, a.mesh_root, a.workers, a.size)
    else:
        cost(a.size)


if __name__ == "__main__":
    main()
