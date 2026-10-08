#!/usr/bin/env python3
"""E2: canonicalizzazione rigida al test. Similarita' (rotazione, scala, traslazione) stimata con un
ICP trimmed verso la faccia media del frame maggioritario del training, e applicata ai vertici.

    aau/run.sh aau/evidence/e2_canon/canon.py reference --out aau/runs/evidence/e2/reference_ict_mean.npz
    aau/run.sh aau/evidence/e2_canon/canon.py calibrate --reference <npz> --out-dir aau/runs/evidence/e2/calibration
    aau/run.sh aau/evidence/e2_canon/canon.py stage --reference <npz> --view-dir <vista>/npz --seed 1234 \\
        --out-dir /tmp/.../in --records <csv> --convention x,-y,-z
    aau/run.sh aau/evidence/e2_canon/canon.py now --reference <npz> --src-work ~/data/now_eval_work \\
        --dst-work <dir> --records <csv>

Riferimento (``reference``): media vertice per vertice delle ``original`` ICT del training del run grande
(gli id10000-14999 di ``aau/data_scale/split_scale_all.json``, ``train``; le mesh di
``datasets/ICT/train_ready/npz_withops``, gia' centrate e ad area unitaria, che e' quello che il trainer
legge). ICT e' 54.008 delle 64.400 identita' di training; GNM (10.000) sta nello stesso frame (+y alto,
naso +z, ``datasets/GNM_DISTILL/README.md``), BFM (392) no. Il controllo ``calibrate`` misura anche
l'angolo fra la faccia media GNM e quella ICT.

Metodo (``canonicalize``), parametri in ``PARAMS``, dichiarati in ``aau/runs/evidence/e2/protocol.md``
prima di qualunque eval sui domini di test:
  - sorgente: punti campionati per area sulla mesh (seme fisso per file), centrati sul loro baricentro e
    scalati a raggio RMS 1; riferimento: ``n_ref`` punti per area sulla faccia media, stessa
    normalizzazione, in un cKDTree;
  - multi-start: le 8 rotazioni di ``STARTS`` (identita', i tre flip di 180 gradi, +-90 attorno a x e y),
    scala 1, traslazione 0;
  - ICP trimmed (Chetverikov et al. 2002) SIMMETRICO e consapevole delle regioni (le regioni dei domini
    sono diverse: ICT arriva ai lati della testa, BFM, FaceVerse e NoW sono patch frontali, HIFI3D e' piu'
    larga): a ogni iterazione due insiemi di coppie, (A) ogni punto della sorgente col piu' vicino della
    faccia media INTERA, tenuta la frazione ``trim`` piu' vicina (la sorgente deve stare sulla superficie
    del riferimento, il di piu' si scarta), e (B) ogni punto del NUCLEO della faccia media (i punti entro
    ``core_radius`` dalla punta del naso: occhi, naso, bocca, guance interne, presenti in tutti i domini
    e in tutte le topologie, crop compreso perche' il crop toglie una banda di bordo) col piu' vicino
    della sorgente, tenuta la frazione ``trim_core`` (il nucleo deve essere coperto); Umeyama pesato
    (similarita' senza riflessione, i due insiemi con peso totale uguale), scala limitata a
    ``scale_bounds`` attorno a quella dei raggi RMS (senza limite una rotazione sbagliata "vince"
    rimpicciolendo la sorgente: visto nella prima calibrazione); distanze oltre ``dist_cap`` contano come
    ``dist_cap`` e non danno coppie; si ferma quando il residuo cambia meno di ``tol`` (relativo) o dopo il
    numero massimo di iterazioni;
  - grossolano -> fine: gli 8 start con ``n_src_coarse`` punti e ``iters_coarse`` iterazioni, i migliori
    ``keep_starts`` rifiniti con ``n_src`` punti e ``iters_fine`` iterazioni;
  - residuo (quello che sceglie lo start e quello della soglia di fallimento): RMS trimmed simmetrico,
    sqrt((media dei quadrati delle distanze tenute in A + idem in B) / 2), in unita' del raggio RMS del
    riferimento;
  - si applica ai vertici SOLO la similarita' trovata: V' = s V R^T + t. Nessuna corrispondenza, nessuna
    deformazione, facce invariate (det R = +1: il verso dei triangoli non cambia).
Scala e traslazione vengono comunque tolte dalla pipeline a valle (operatori ad area unitaria, loader
centrato e maxabs): quello che conta per il modello e' la rotazione.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "aau" / "zs3dmm"))

PARAMS = {"n_src": 4000, "n_src_coarse": 1000, "n_ref": 50_000, "n_core": 3000, "core_radius": 1.1,
          "trim": 0.8, "trim_core": 0.9, "scale_bounds": [0.67, 1.5], "dist_cap": 0.5, "iters_coarse": 30,
          "iters_fine": 60, "tol": 1e-6, "keep_starts": 2, "ref_seed": 1234}


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


STARTS = {
    "I": np.eye(3),
    "Rx180": np.diag([1.0, -1.0, -1.0]),
    "Ry180": np.diag([-1.0, 1.0, -1.0]),
    "Rz180": np.diag([-1.0, -1.0, 1.0]),
    "Rx+90": _rot("x", 90), "Rx-90": _rot("x", -90),
    "Ry+90": _rot("y", 90), "Ry-90": _rot("y", -90),
}
CONVENTIONS = {"I": np.eye(3), "Rx180": np.diag([1.0, -1.0, -1.0])}


# ------------------------------------------------------------------------------ geometria

def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as d:
        if "V" in d:
            return np.asarray(d["V"], np.float64), np.asarray(d["F"], np.int64)
        return np.asarray(d["verts"], np.float64), np.asarray(d["faces"], np.int64)


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def area_sample(V: np.ndarray, F: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` punti uniformi sulla superficie (triangolo scelto per area, baricentriche uniformi)."""
    a = face_areas(V, F)
    f = rng.choice(len(F), size=n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    A, B, C = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
    return A + u[:, None] * (B - A) + v[:, None] * (C - A)


def umeyama(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) con Y ~ s R X + t ai minimi quadrati (pesati), det R = +1 (Umeyama 1991)."""
    w = np.full(len(X), 1.0 / len(X)) if w is None else w / w.sum()
    mx, my = w @ X, w @ Y
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd((Yc * w[:, None]).T @ Xc)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U) * np.linalg.det(Vt)) or 1.0])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (w @ (Xc ** 2).sum(1)))
    return s, R, my - s * R @ mx


def angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1))))


class Reference:
    """La faccia media, campionata per area e normalizzata (baricentro 0, raggio RMS 1), e il suo nucleo."""

    def __init__(self, path: Path, params: dict = PARAMS):
        with np.load(path) as z:
            self.V, self.F = np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)
        rng = np.random.default_rng(params["ref_seed"])
        P = area_sample(self.V, self.F, params["n_ref"], rng)
        self.c = P.mean(0)
        self.r = float(np.sqrt(((P - self.c) ** 2).sum(1).mean()))
        self.Q = (P - self.c) / self.r
        self.tree = cKDTree(self.Q)
        # punta del naso: il punto piu' avanti (z, frame ICT) vicino alla linea mediana
        mid = np.abs(self.Q[:, 0]) < 0.1
        self.nose = self.Q[mid][np.argmax(self.Q[mid, 2])]
        core = self.Q[np.linalg.norm(self.Q - self.nose, axis=1) < params["core_radius"]]
        self.core = core[rng.choice(len(core), size=min(params["n_core"], len(core)), replace=False)]
        self.path = str(path)


def _nn(tree: cKDTree, X: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    d, j = tree.query(X, distance_upper_bound=cap)
    far = ~np.isfinite(d)
    d[far] = cap
    return d, np.where(far, -1, j)


def _pairs(P: np.ndarray, ref: Reference, s: float, R: np.ndarray, t: np.ndarray, p: dict):
    """Le due serie di coppie trimmed (A: sorgente -> faccia intera, B: nucleo -> sorgente) e il residuo."""
    X = s * P @ R.T + t
    dA, jA = _nn(ref.tree, X, p["dist_cap"])
    kA = max(3, int(round(p["trim"] * len(dA))))
    iA = np.argpartition(dA, kA - 1)[:kA]
    dB, jB = _nn(cKDTree(X), ref.core, p["dist_cap"])
    kB = max(3, int(round(p["trim_core"] * len(dB))))
    iB = np.argpartition(dB, kB - 1)[:kB]
    res = float(np.sqrt(0.5 * ((dA[iA] ** 2).mean() + (dB[iB] ** 2).mean())))
    iA, iB = iA[jA[iA] >= 0], iB[jB[iB] >= 0]
    # coppie (punto della sorgente, bersaglio nel frame del riferimento)
    src = np.concatenate([P[iA], P[jB[iB]]])
    dst = np.concatenate([ref.Q[jA[iA]], ref.core[iB]])
    w = np.concatenate([np.full(len(iA), 1.0 / max(len(iA), 1)), np.full(len(iB), 1.0 / max(len(iB), 1))])
    return src, dst, w, res, float(np.sqrt((dA[np.argpartition(dA, kA - 1)[:kA]] ** 2).mean()))


def _icp(P: np.ndarray, ref: Reference, s: float, R: np.ndarray, t: np.ndarray, max_iter: int, p: dict) -> dict:
    lo, hi = p["scale_bounds"]
    prev, it = np.inf, 0
    for it in range(1, max_iter + 1):
        src, dst, w, res, _ = _pairs(P, ref, s, R, t, p)
        if len(src) < 20:
            break
        s, R, t = umeyama(src, dst, w)
        if not lo <= s <= hi:   # scala fuori dai limiti: la si blocca al limite e si ricalcola t
            s = float(np.clip(s, lo, hi))
            t = (w / w.sum()) @ dst - s * R @ ((w / w.sum()) @ src)
        if abs(prev - res) <= p["tol"] * max(res, 1e-12):
            break
        prev = res
    _, _, _, res, res_src = _pairs(P, ref, s, R, t, p)
    return {"s": s, "R": R, "t": t, "res": res, "res_src": res_src, "iters": it}


def canonicalize(V: np.ndarray, F: np.ndarray, ref: Reference, seed: int, params: dict = PARAMS) -> dict:
    """Similarita' (s, R, t) che porta i vertici ORIGINALI nel frame del riferimento: V' = s V R^T + t."""
    t0 = time.perf_counter()
    P = area_sample(V, F, params["n_src"], np.random.default_rng(seed))
    c = P.mean(0)
    r = float(np.sqrt(((P - c) ** 2).sum(1).mean()))
    Pn = (P - c) / r
    Pc = Pn[: params["n_src_coarse"]]          # i campioni sono i.i.d.: i primi n sono un sottocampione
    coarse = {n: _icp(Pc, ref, 1.0, R0, np.zeros(3), params["iters_coarse"], params) for n, R0 in STARTS.items()}
    order = sorted(coarse, key=lambda n: coarse[n]["res"])
    fine = {n: _icp(Pn, ref, coarse[n]["s"], coarse[n]["R"], coarse[n]["t"], params["iters_fine"], params)
            for n in order[: params["keep_starts"]]}
    best = min(fine, key=lambda n: fine[n]["res"])
    b = fine[best]
    # x_ref_norm = s R (x - c)/r + t ; frame del riferimento (non normalizzato): x_ref = r_ref x_ref_norm + c_ref
    s_tot = ref.r * b["s"] / r
    t_tot = ref.r * (b["t"] - b["s"] * b["R"] @ c / r) + ref.c
    second = min([fine[n]["res"] for n in fine if n != best] + [coarse[n]["res"] for n in order[params["keep_starts"]:]])
    return {"start": best, "s": s_tot, "s_norm": b["s"], "R": b["R"], "t": t_tot, "res": b["res"],
            "res_src": b["res_src"], "iters": b["iters"], "res_second": second,
            "res_by_start": {n: v["res"] for n, v in coarse.items()}, "seconds": time.perf_counter() - t0}


def apply(V: np.ndarray, out: dict) -> np.ndarray:
    return out["s"] * V @ out["R"].T + out["t"]


def file_seed(name: str) -> int:
    """Seme del campionamento della sorgente: dipende solo dal nome del file."""
    return zlib.crc32(name.encode()) ^ 1234


def record(out: dict, **extra) -> dict:
    row = {**extra, "start": out["start"], "residual": out["res"], "residual_src": out["res_src"],
           "residual_second_start": out["res_second"], "iters": out["iters"], "scale": out["s"],
           "scale_norm": out["s_norm"], "seconds": out["seconds"]}
    for i in range(3):
        for j in range(3):
            row[f"R{i}{j}"] = float(out["R"][i, j])
    for i, ax in enumerate("xyz"):
        row[f"t{ax}"] = float(out["t"][i])
    for n, v in out["res_by_start"].items():
        row[f"res_coarse_{n}"] = v
    return row


# ------------------------------------------------------------------------- riferimento

def ict_train_ids() -> list[str]:
    split = json.loads((REPO_ROOT / "aau/data_scale/split_scale_all.json").read_text())
    return sorted(s for s in split["train"] if 10000 <= int(s[2:]) <= 14999)


def _load_V(p):
    with np.load(p) as d:
        return np.asarray(d["verts"], np.float64)


def cmd_reference(a) -> None:
    ids = ict_train_ids()
    root = REPO_ROOT / "datasets/ICT/train_ready/npz_withops"
    paths = [root / f"{s}_GTready_original.npz" for s in ids]
    with np.load(paths[0]) as d:
        F = np.asarray(d["faces"], np.int64)
    with mp.get_context("fork").Pool(a.workers) as pool:
        Vs = pool.map(_load_V, paths, chunksize=32)
    shapes = {v.shape for v in Vs}
    if len(shapes) != 1:
        raise SystemExit(f"topologie diverse fra le original ICT: {shapes}")
    M = np.mean(np.stack(Vs), axis=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, V=M.astype(np.float64), F=F, ids=np.asarray(ids), source=str(root))
    print(f"[canon-ref] {len(ids)} identita' ICT di training (primi {ids[:3]}), {M.shape[0]} vertici, "
          f"{len(F)} triangoli, area {face_areas(M, F).sum():.4f} -> {a.out}", flush=True)


# -------------------------------------------------------------------------- calibrazione

TOPOS = ("crop", "down8k", "noisy", "original", "remesh", "up60k")
_REF: Reference | None = None


def _gnm_meshes(ids: set) -> dict:
    tars = sorted((REPO_ROOT / "datasets/GNM_DISTILL/shards").glob("gnm_shard_*.tar"))
    out = {}
    with tarfile.open(tars[-1]) as tf:
        for m in tf.getmembers():
            sid, _, topo = m.name[:-4].partition("_GTready_")
            if sid in ids and topo in TOPOS:
                with np.load(io.BytesIO(tf.extractfile(m).read())) as d:
                    out[(sid, topo)] = (np.asarray(d["V"], np.float64), np.asarray(d["F"], np.int64))
    return out


def _calib_task(task):
    domain, sid, topo, V, F, conv, pert = task
    if pert is not None:
        V = V @ pert.T
    out = canonicalize(V, F, _REF, file_seed(f"{sid}_{topo}"))
    R_conv = CONVENTIONS[conv] @ (pert.T if pert is not None else np.eye(3))
    P = pert if pert is not None else np.eye(3)
    return record(out, domain=domain, subject=sid, topology=topo, perturbed=pert is not None,
                  angle_from_convention=angle_deg(out["R"], R_conv),
                  angle_from_identity=angle_deg(out["R"], np.eye(3)),
                  **{f"P{i}{j}": float(P[i, j]) for i in range(3) for j in range(3)})


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def cmd_calibrate(a) -> None:
    """Domini di TRAINING, soggetti held-out del run grande: ICT (frame I), GNM (I), BFM (Rx180)."""
    global _REF
    _REF = Reference(a.reference)
    held = json.loads((REPO_ROOT / "aau/data_scale/heldout_frozen.json").read_text())
    rng = np.random.default_rng(1234)
    pick = lambda xs: sorted(rng.choice(sorted(xs), size=a.n_subjects, replace=False).tolist())  # noqa: E731
    ict, bfm, gnm = pick(held["ict_view"]), pick(held["bfm"]), pick(held["gnm"])
    meshes = []
    for sid in ict:
        for t in TOPOS:
            meshes.append(("ict", sid, t, *load_mesh(REPO_ROOT / f"datasets/ICT/train_ready/npz_withops/{sid}_GTready_{t}.npz"), "I"))
    for sid in bfm:
        for t in TOPOS:
            meshes.append(("bfm", sid, t, *load_mesh(REPO_ROOT / f"datasets/REMESH/npz_data_topo_500/{sid}_GTready_{t}.npz"), "Rx180"))
    g = _gnm_meshes(set(gnm))
    for sid in gnm:
        for t in TOPOS:
            meshes.append(("gnm", sid, t, *g[(sid, t)], "I"))
    tasks = [(d, s, t, V, F, c, None) for d, s, t, V, F, c in meshes]
    # robustezza: la stessa mesh ruotata a caso (convenzione * fino a 30 gradi, piu' un flip a caso)
    prng = np.random.default_rng(4321)
    flips = list(STARTS.values())[:4]
    for d, s, t, V, F, c in meshes:
        if t in ("original", "down8k", "crop"):
            P = random_rotation(prng, 30.0) @ flips[prng.integers(4)]
            tasks.append((d, s, t, V, F, c, P))
    print(f"[canon-cal] {len(tasks)} canonicalizzazioni ({len(meshes)} mesh + {len(tasks) - len(meshes)} ruotate) "
          f"su {a.workers} processi", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(_calib_task, tasks, chunksize=4)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.out_dir / "calibration.csv", rows)

    # faccia media GNM (held-out, original) contro la faccia media ICT: l'angolo fra i due frame
    Vg = np.mean([g[(s, "original")][0] for s in gnm], axis=0)
    out = canonicalize(Vg, g[(gnm[0], "original")][1], _REF, 1234)
    gnm_info = {"angle_from_identity_deg": angle_deg(out["R"], np.eye(3)), "residual": out["res"],
                "start": out["start"]}
    # area normalizzata (area / raggio RMS per area^2) delle original ICT, per la lunghezza di spigolo di E3b
    an = []
    for d, s, t, V, F, c in meshes:
        if d == "ict" and t == "original":
            fa = face_areas(V, F)
            cen = (V[F].mean(1) * fa[:, None]).sum(0) / fa.sum()
            rr = np.sqrt((((V[F].mean(1) - cen) ** 2).sum(1) * fa).sum() / fa.sum())
            an.append(fa.sum() / rr ** 2)
    (a.out_dir / "calibration_extra.json").write_text(json.dumps(
        {"gnm_mean_vs_ict_mean": gnm_info, "ict_original_area_over_rms2_median": float(np.median(an)),
         "reference_nose_tip_norm": _REF.nose.tolist(), "reference_core_points": len(_REF.core),
         "subjects": {"ict": ict, "bfm": bfm, "gnm": gnm}, "params": PARAMS, "starts": list(STARTS)}, indent=1))
    print(f"[canon-cal] GNM media contro ICT media: {gnm_info}", flush=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- staging

def _stage_task(task):
    src, dst, name, conv = task
    V, F = load_mesh(src)
    out = canonicalize(V, F, _REF, file_seed(name))
    Vc = apply(V, out)
    np.savez(dst, V=Vc, F=F)
    return record(out, file=name, angle_from_convention=angle_deg(out["R"], CONVENTIONS[conv]),
                  angle_from_identity=angle_deg(out["R"], np.eye(3)), n_verts=len(V))


def run_stage(tasks: list, workers: int) -> list[dict]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_stage_task, tasks, chunksize=2)


def cmd_stage(a) -> None:
    """I soggetti di zs_stage.select_subjects, 6 topologie, canonicalizzati in --out-dir (+ subjects.json)."""
    global _REF
    from zs_stage import TOPOLOGIES, select_subjects
    _REF = Reference(a.reference)
    subjects = select_subjects(a.view_dir, a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for sid in subjects:
        for topo in TOPOLOGIES:
            src = a.view_dir / f"{sid}_GTready_{topo}.npz"
            tasks.append((src, a.out_dir / src.name, src.stem, a.convention))
    t0 = time.time()
    rows = run_stage(tasks, a.workers)
    for r in rows:
        r["subject"], r["topology"] = r["file"].split("_GTready_")
    write_csv(a.records, rows)
    (a.out_dir.parent / "subjects.json").write_text(json.dumps(
        {"seed": a.seed, "view_dir": str(a.view_dir), "transform": "e2_canon", "flip_faces": False,
         "subjects": subjects, "reference": str(a.reference), "params": PARAMS}, indent=1) + "\n")
    print(f"[canon-stage] {len(subjects)} soggetti, {len(rows)} mesh in {time.time() - t0:.0f}s -> {a.out_dir}; "
          f"residuo mediano {np.median([r['residual'] for r in rows]):.4f}, secondi/mesh mediani "
          f"{np.median([r['seconds'] for r in rows]):.2f}", flush=True)


def cmd_now(a) -> None:
    """Patch NoW (scan_face, recon_face/<metodo>) canonicalizzate in --dst-work, stessa struttura."""
    global _REF
    _REF = Reference(a.reference)
    tasks = []
    for sub in ["scan_face"] + [f"recon_face/{m}" for m in a.methods.split(",")]:
        (a.dst_work / sub).mkdir(parents=True, exist_ok=True)
        for src in sorted((a.src_work / sub).glob("*.npz")):
            tasks.append((src, a.dst_work / sub / src.name, f"{sub}/{src.stem}", "I"))
    t0 = time.time()
    rows = run_stage(tasks, a.workers)
    write_csv(a.records, rows)
    print(f"[canon-now] {len(rows)} patch in {time.time() - t0:.0f}s -> {a.dst_work}; residuo mediano "
          f"{np.median([r['residual'] for r in rows]):.4f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--workers", type=int, default=16)
    c = sub.add_parser("calibrate")
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--out-dir", type=Path, required=True)
    c.add_argument("--n-subjects", type=int, default=40)
    c.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("stage")
    s.add_argument("--reference", type=Path, required=True)
    s.add_argument("--view-dir", type=Path, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--records", type=Path, required=True)
    s.add_argument("--convention", default="I", choices=sorted(CONVENTIONS),
                   help="frame atteso del dominio, solo per l'angolo riportato (HIFI3D I, FaceVerse Rx180)")
    s.add_argument("--workers", type=int, default=16)
    n = sub.add_parser("now")
    n.add_argument("--reference", type=Path, required=True)
    n.add_argument("--src-work", type=Path, required=True)
    n.add_argument("--dst-work", type=Path, required=True)
    n.add_argument("--records", type=Path, required=True)
    n.add_argument("--methods", default="3ddfa_v2,synergynet,prnet,mica")
    n.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    {"reference": cmd_reference, "calibrate": cmd_calibrate, "stage": cmd_stage, "now": cmd_now}[a.cmd](a)


if __name__ == "__main__":
    main()
