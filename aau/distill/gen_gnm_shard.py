#!/usr/bin/env python3
"""Identita' GNM Head per il training della distillazione v2: shard di SOLA GEOMETRIA, come gli ICT nuovi.

    aau/run.sh aau/distill/gen_gnm_shard.py --shards 0-40 --out-dir datasets/GNM_DISTILL/shards --n-cores 64
    (gen_gnm.sbatch; protocollo in aau/runs/distill_v2/protocol.md)

Gemello di ``aau/data_scale/gen_ict_shard.py`` (stesso formato di tar, stesse regole), sul modello di
``aau/zs3dmm/gnm_model.py`` (importato: regione ``hockey_mask`` coi quad a ventaglio, ~9.000 vertici e
~17.700 triangoli, la scala della patch ICT; frame +Y/+Z come ICT; metri).

Identita'
---------
Indice ``j`` = 0 .. ``--n-total`` - 1, nome ``id<100000 + j>`` (6 cifre: fuori da BFM 0-499, FLAME
1000-5999, ICT-5000 10000-14999, ICT nuovi 20000-69999 e dai domini zero-shot 900000+). Gli ultimi
``--n-val`` sono di VALIDAZIONE (solo curve), mai nel training. Pesi N(0, 1) non troncati sulle 170
basi ``head_*`` (occhi e denti a zero), come il pool zero-shot di GNM ma con un altro seme. Ogni
numero casuale dipende solo da ``j`` e da un seme di famiglia (``SeedSequence([seme, j])``).

Mesh per identita'
------------------
Le 6 topologie di ``v2_work/genict/make_ict_topologies.py`` (funzioni importate, stessi target di
triangoli, ``mesh_ops.make_crop``) sulla patch neutra, e 1 o 2 espressioni (a caso per identita')
nella topologia ``original``, ``id<g>_GTready_rexpr<k>``. Espressione: le 350 componenti
``lower_face_region`` + ``left/right_eye_region`` della base GNM (pool di ``make_zs_expr_topologies``;
lingua e pupille a zero), tutte attive, coefficienti N(0, sigma^2). ``sigma`` e' tarata UNA volta
(``--calibrate``, prima di generare) perche' lo spostamento medio per vertice dopo maxabs, mediana su
identita', sia quello della ricetta ICT dei rexpr (0.017, ``make_zs_expr_topologies.ICT_REFERENCE``):
ampiezza moderata, quella delle espressioni ICT gia' in training.

Uscita: ``<out-dir>/gnm_shard_<KKKKK>.tar`` (npz compressi V float32 / F int32 + ``manifest.json`` con
pesi, coefficienti e spostamenti) e ``gnm_shard_<KKKKK>.json``. Prefisso ``gnm_``: i nomi
dei tar non devono coincidere con quelli di ``datasets/ICT_SCALE`` (gli indici li confrontano per nome). Shard gia' presenti saltati.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))

import gnm_model  # noqa: E402
import mesh_ops as mo  # noqa: E402
from make_ict_topologies import (  # noqa: E402
    REMESH_DECIMATION, make_down8k, make_noisy, make_remesh, make_up60k, triangle_targets)
from make_zs_expr_topologies import GNM_POOL_GROUPS, ICT_REFERENCE, maxabs_normalize  # noqa: E402

ID_BASE = 100000
SEED_ID = 20261010
SEED_NOISE = 20261011
SEED_EXPR = 20261012
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
SIGMA_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
TARGET_SHIFT = ICT_REFERENCE["recipe_mean_shift"]

_M: dict | None = None
_E: dict | None = None
_POOL = np.zeros(0, dtype=np.int64)
_SIGMA = 0.0


def rng_for(seed: int, j: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, j]))


def _init(model_file: str, sigma: float) -> None:
    global _M, _E, _POOL, _SIGMA
    _M = gnm_model.load_gnm(model_file)
    _E = gnm_model.load_gnm_expressions(model_file)
    _POOL = np.asarray([i for g in GNM_POOL_GROUPS for i in _E["groups"][g]], dtype=np.int64)
    if len(_POOL) != 350:
        raise SystemExit(f"attese 350 componenti {GNM_POOL_GROUPS}, trovate {len(_POOL)}")
    _SIGMA = float(sigma)


def head(w: np.ndarray, coef_pool: np.ndarray | None = None) -> np.ndarray:
    V = gnm_model.shape_mesh(w, _M)
    if coef_pool is not None:
        V = V + _E["exprdirs"][:, :, _POOL] @ coef_pool
    return V


def geom_bytes(V: np.ndarray, F: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, V=np.asarray(V, dtype=np.float32), F=np.asarray(F, dtype=np.int32))
    return buf.getvalue()


def shift(V: np.ndarray, V0: np.ndarray) -> float:
    return float(np.linalg.norm(maxabs_normalize(V) - maxabs_normalize(V0), axis=1).mean())


def calibrate(model_file: str, n: int) -> dict:
    """Mediana su ``n`` identita' dello spostamento medio maxabs, per ogni sigma della griglia."""
    _init(model_file, 0.0)
    k = _M["shapedirs"].shape[2]
    out = {}
    for s in SIGMA_GRID:
        vals = []
        for j in range(n):
            rng = np.random.default_rng([SEED_EXPR + 999, j])
            w = rng.normal(size=k)
            V0, _ = gnm_model.face_patch(head(w), _M)
            V1, _ = gnm_model.face_patch(head(w, rng.normal(0.0, s, size=len(_POOL))), _M)
            vals.append(shift(V1, V0))
        out[s] = float(np.median(vals))
    best = min(out, key=lambda s: abs(out[s] - TARGET_SHIFT))
    return {"grid": out, "target": TARGET_SHIFT, "sigma": best, "n_identities": n}


def process_identity(j: int) -> dict:
    g = ID_BASE + j
    sid = f"id{g}"
    t0 = time.time()
    k = _M["shapedirs"].shape[2]
    w = rng_for(SEED_ID, j).normal(0.0, 1.0, size=k)
    V0, F0 = gnm_model.face_patch(head(w), _M)
    base = mo.prepare_open_surface(*mo.as_arrays(V0, F0))
    if len(base[1]) != len(F0) or len(base[0]) != len(V0):
        raise RuntimeError(f"{sid}: patch non pulita")
    down_t, up_t = triangle_targets(len(base[1]))
    noise_seed = int(np.random.SeedSequence([SEED_NOISE, j]).generate_state(1)[0])
    meshes = {
        "original": base,
        "remesh": make_remesh(*base),
        "crop": mo.make_crop(*base),
        "noisy": make_noisy(*base, seed=noise_seed),
        "down8k": make_down8k(*base, target=down_t),
        "up60k": make_up60k(*base, target=up_t),
    }
    rng = rng_for(SEED_EXPR, j)
    n_expr = int(rng.integers(1, 3))
    coefs, shifts = [], []
    for e in range(1, n_expr + 1):
        c = rng.normal(0.0, _SIGMA, size=len(_POOL))
        V, F = gnm_model.face_patch(head(w, c), _M)
        if not np.array_equal(F, F0):
            raise RuntimeError(f"{sid}: facce dell'espressione diverse dalla neutra")
        meshes[f"rexpr{e}"] = (V, base[1])
        coefs.append(c.astype(np.float32).tolist())
        shifts.append(shift(V, base[0]))
    expect = {"remesh": int(len(F0) * REMESH_DECIMATION), "down8k": down_t, "up60k": up_t}
    errors = []
    for label, (V, F) in meshes.items():
        if not np.isfinite(V).all():
            errors.append(f"{sid}/{label}: vertici non finiti")
        if label in expect and abs(len(F) - expect[label]) > 0.02 * expect[label]:
            errors.append(f"{sid}/{label}: {len(F)} triangoli, attesi ~{expect[label]}")
    if len(meshes["crop"][1]) >= len(F0):
        errors.append(f"{sid}/crop: non ha tolto niente")
    blobs = {f"{sid}_GTready_{lab}.npz": geom_bytes(V, F) for lab, (V, F) in meshes.items()}
    sizes = {lab: [int(len(V)), int(len(F))] for lab, (V, F) in meshes.items()}
    return {"sid": sid, "j": j, "weights": w.astype(np.float32).tolist(), "expr_coefs": coefs,
            "expr_shift": shifts, "sizes": sizes, "errors": errors, "seconds": time.time() - t0, "blobs": blobs}


def parse_range(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shards", default="0-40", help="intervallo di shard, es. 0-40")
    ap.add_argument("--n-per-shard", type=int, default=250)
    ap.add_argument("--n-total", type=int, default=10100)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--model-file", default=os.environ.get("WBES_GNM_NPZ", str(Path.home() / "data/gnm_head/gnm_head.npz")))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--n-cores", type=int, default=1)
    ap.add_argument("--calibrate", type=int, default=50, help="identita' per la taratura di sigma")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.work_dir.mkdir(parents=True, exist_ok=True)

    cal_path = a.out_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
    else:
        t0 = time.time()
        cal = calibrate(a.model_file, a.calibrate)
        cal["seconds"] = time.time() - t0
        cal_path.write_text(json.dumps(cal, indent=1) + "\n")
    print(f"[gnm] sigma {cal['sigma']} (spostamento mediano per sigma {cal['grid']}, bersaglio {cal['target']})",
          flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, a.n_cores), initializer=_init, initargs=(a.model_file, cal["sigma"])) as pool:
        for shard in parse_range(a.shards):
            out = a.out_dir / f"gnm_shard_{shard:05d}.tar"
            js = list(range(shard * a.n_per_shard, min((shard + 1) * a.n_per_shard, a.n_total)))
            if out.exists() or not js:
                continue
            t0 = time.time()
            records = pool.map(process_identity, js, chunksize=1)
            errs = [e for r in records for e in r["errors"]]
            if errs:
                raise SystemExit(f"shard {shard}: GUARDIA FALLITA {errs[:5]}")
            tmp = a.work_dir / f".{out.name}.{os.getpid()}.tmp"
            with tarfile.open(tmp, "w") as tar:
                for r in records:
                    for name, blob in sorted(r.pop("blobs").items()):
                        info = tarfile.TarInfo(name)
                        info.size = len(blob)
                        tar.addfile(info, io.BytesIO(blob))
                manifest = {"shard": shard, "id_base": ID_BASE, "n_val": a.n_val, "n_total": a.n_total,
                            "seeds": {"identity": SEED_ID, "noise": SEED_NOISE, "expression": SEED_EXPR,
                                      "rule": "SeedSequence([seed, j])"},
                            "sigma": cal["sigma"], "expression_pool": list(GNM_POOL_GROUPS),
                            "identities": [{k: r[k] for k in ("sid", "j", "weights", "expr_coefs", "expr_shift")}
                                           for r in records]}
                blob = json.dumps(manifest).encode()
                info = tarfile.TarInfo("manifest.json")
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))
            part = out.with_name(out.name + ".part")
            shutil.move(str(tmp), str(part))
            os.replace(part, out)
            side = {"shard": shard, "ids": [r["sid"] for r in records],
                    "val_ids": [r["sid"] for r in records if r["j"] >= a.n_total - a.n_val],
                    "n_meshes": sum(len(r["sizes"]) for r in records),
                    "n_expr": int(sum(len(r["expr_shift"]) for r in records)),
                    "sizes_first": records[0]["sizes"],
                    "expr_shift_median": float(np.median([s for r in records for s in r["expr_shift"]])),
                    "tar_bytes": out.stat().st_size,
                    "seconds": {"wall": time.time() - t0,
                                "cpu_per_identity": float(np.mean([r["seconds"] for r in records]))}}
            out.with_suffix(".json").write_text(json.dumps(side, indent=1) + "\n")
            print(f"[gnm] shard {shard}: {len(records)} identita', {side['n_meshes']} mesh, "
                  f"{side['tar_bytes'] / 1e6:.0f} MB, {side['seconds']['wall']:.0f}s "
                  f"({side['seconds']['cpu_per_identity']:.1f} CPU-s/identita'), shift mediano {side['expr_shift_median']:.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
