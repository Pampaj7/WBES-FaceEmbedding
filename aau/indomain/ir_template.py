#!/usr/bin/env python3
"""Baseline "NICP su template": ogni mesh si iscrive UNA volta, poi i confronti sono in corrispondenza densa.

    aau/outlineB/run_o3d.sh aau/indomain/ir_template.py --workers 40 --sets bfm,rexpr,ict992
    (ir_template.sbatch; protocollo: aau/runs/indomain_recog/protocol.md, revisione 1, punto B)

Template per dominio: media vertice per vertice delle mesh ``original`` (normalizzate maxabs come in
faceBench, ``mesh_npz_utils.normalize_vertices``) di 100 soggetti di TRAINING del congiunto, scelti con
``rng(1234)`` dal suo split (``splits.json``); la ``original`` e' la topologia del 3DMM, uguale per tutti i
soggetti, quindi la media vertice per vertice ha senso (lo script lo controlla). Del template si tengono
4096 vertici scelti una volta (``rng(0)``): sono i punti in corrispondenza.

Iscrizione di una mesh (``enroll``): ``load_verts`` + ``sample_pts`` di faceBench (4096 punti, seme =
crc32 di soggetto e etichetta, cosi' la stessa mesh si iscrive identica in ogni insieme), ICP di
similarita' del template sulla mesh (``ir_simicp.similarity_icp``), ``nonrigid_icp_align`` di faceBench
del template sulla mesh, poi similarita' (Procrustes con scala, Umeyama) dei 4096 punti registrati verso
il template: un frame canonico comune. Confronto fra due mesh iscritte = distanza L2 media per vertice
(``template_distances``).

Insiemi: bfm, rexpr, ict992. ``ict`` (89 soggetti) e' dentro ict992 con le stesse mesh e gli stessi
semi: il summarizer lo prende da li'. Output in ``aau/runs/indomain_recog/template/``:
``<dominio>_template.npz`` e ``<insieme>.npz`` con ``R`` (n, 4096, 3) float32, ``subjects``, ``labels``,
``seconds`` (tempo di iscrizione per mesh nel worker, lettura compresa), ``failed``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

import ir_simicp  # noqa: E402  (mette faceBench sul path)

RUNS = AAU_DIR / "runs" / "indomain_recog"
JOINT_DIR = AAU_DIR.parent / "datasets" / "JOINT_BFM_ICT" / "npz_withops"
SPLITS = AAU_DIR / "runs" / "ws2_cross3dmm" / "splits.json"
N_POINTS = 4096
N_TEMPLATE_SUBJECTS = 100
TEMPLATE_SETS = ("bfm", "rexpr", "ict992")

_TEMPLATE: np.ndarray | None = None


def domain_of(sid: str) -> str:
    return "ict" if int(sid[2:]) >= 10000 else "bfm"


def mesh_seed(subject: str, label: str) -> int:
    return zlib.crc32(f"{subject}|{label}".encode()) % 2_000_000_000


def build_template(domain: str, out: Path) -> np.ndarray:
    """Template del dominio (cache in ``out``): 4096 vertici della media delle ``original`` di training."""
    if out.exists():
        with np.load(out) as z:
            return z["T"].astype(np.float64)
    from mesh_npz_utils import normalize_vertices

    train = sorted(s for s in json.loads(SPLITS.read_text())["models"]["joint"]["train"] if domain_of(s) == domain)
    chosen = sorted(np.random.default_rng(1234).choice(train, N_TEMPLATE_SUBJECTS, replace=False).tolist())
    Vs = []
    for s in chosen:
        with np.load(JOINT_DIR / f"{s}_GTready_original.npz") as d:
            Vs.append(normalize_vertices(np.asarray(d["verts"])))
    if len({V.shape for V in Vs}) != 1:
        raise SystemExit(f"{domain}: le original dei soggetti di training hanno numeri di vertici diversi")
    mean = normalize_vertices(np.mean(np.stack(Vs), axis=0))
    idx = np.sort(np.random.default_rng(0).choice(len(mean), N_POINTS, replace=False))
    T = mean[idx]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, T=T, idx=idx, subjects=np.asarray(chosen), n_vertices=len(mean))
    print(f"[ir-template] template {domain}: media di {len(chosen)} original di training "
          f"({len(mean)} vertici), {N_POINTS} punti -> {out}", flush=True)
    return T


def procrustes_to(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A portata su B con la similarita' ai minimi quadrati (Umeyama: rotazione, scala, traslazione)."""
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (A0 ** 2).sum())
    return s * A0 @ R + mu_b


def enroll(path: str, seed: int, T: np.ndarray) -> np.ndarray:
    """Iscrizione di una mesh: i 4096 punti del template registrati su di lei, nel frame del template."""
    import facebench as fb
    import run_facebench_remesh as rfr

    Xs = rfr.sample_pts(rfr.load_verts(Path(path)), N_POINTS, seed)
    T_sim = ir_simicp.similarity_icp(T, Xs)
    T_nicp = fb.nonrigid_icp_align(T_sim, Xs)
    return procrustes_to(np.asarray(T_nicp, dtype=np.float64), T)


def template_distances(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Distanza L2 media per vertice fra ogni iscrizione di ``Ra`` (n, K, 3) e ogni di ``Rb`` (m, K, 3)."""
    out = np.empty((len(Ra), len(Rb)))
    for i, a in enumerate(Ra):
        out[i] = np.linalg.norm(Rb - a[None], axis=2).mean(1)
    return out


def _init(T: np.ndarray) -> None:
    global _TEMPLATE
    _TEMPLATE = T


def _enroll_one(task):
    path, seed = task
    t0 = time.perf_counter()
    try:
        R = enroll(path, seed, _TEMPLATE)
        return R.astype(np.float32), time.perf_counter() - t0, ""
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        return np.full((N_POINTS, 3), np.nan, np.float32), time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sets", default=",".join(TEMPLATE_SETS))
    p.add_argument("--out-root", type=Path, default=RUNS / "template")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-subjects", type=int, default=0, help="0 = tutti; >0 per un test rapido")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    sets = json.loads((RUNS / "sets.json").read_text())["sets"]

    import multiprocessing as mp

    for name in a.sets.split(","):
        out = a.out_root / f"{name}.npz"
        if out.exists() and not a.overwrite:
            print(f"[ir-template] {name}: gia' presente, salto", flush=True)
            continue
        spec = sets[name]
        T = build_template(spec["domain"], a.out_root / f"{spec['domain']}_template.npz")
        keys = [(s, t) for s in spec["subjects"][: a.max_subjects or None] for t in spec["labels"]]
        tasks = [(str(Path(spec["view_dir"]) / f"{s}_GTready_{t}.npz"), mesh_seed(s, t)) for s, t in keys]
        t0 = time.time()
        # spawn come alignment_matrix: open3d non e' fork-safe.
        with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(T,)) as pool:
            res = pool.map(_enroll_one, tasks, chunksize=4)
        failed = [(k, e) for k, (_, _, e) in zip(keys, res) if e]
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, R=np.stack([r[0] for r in res]), subjects=np.asarray([s for s, _ in keys], dtype="U16"),
                 labels=np.asarray([t for _, t in keys], dtype="U16"), seconds=np.asarray([r[1] for r in res]),
                 failed=np.asarray([f"{k[0]}|{k[1]}|{e}" for k, e in failed]), seeds=np.asarray([t[1] for t in tasks]))
        os.replace(tmp, out)
        print(f"[ir-template] {name}: {len(keys)} mesh iscritte in {time.time() - t0:.0f}s, mediana "
              f"{np.median([r[1] for r in res]):.2f}s per mesh nel worker, fallite {len(failed)}"
              + (f" (p.es. {failed[:2]})" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
