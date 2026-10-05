#!/usr/bin/env python3
"""Shard di identita' ICT NUOVE, SOLA GEOMETRIA (opzione (c) di aau/data_scale/PLAN.md).

    aau/run.sh aau/data_scale/gen_ict_shard.py --shard 0 --out-dir <dir degli shard> \
        --work-dir /tmp/$SLURM_JOB_ID --n-cores 16

Nessun operatore viene salvato: li calcola ``prepass_ops.py`` su /tmp nel job che usa i dati.

Identita'
---------
Indice globale ``g = ID_BASE + shard * n_per_shard + j``, nome ``id<g>`` (5 cifre, come le
viste ICT: BFM id0000-0499, FLAME id1000-5999, ICT-5000 id10000-14999; le nuove stanno in
``heldout_frozen.json::new_id_range``). Ogni numero casuale dipende SOLO da ``g`` e da un seme
di famiglia, via ``SeedSequence([seme, g])``: uno shard e' lo stesso a qualunque
``--n-cores``, e due shard non condividono mai una sorgente. I semi sono diversi dal 1234 di
ICT-5000 (identita') e di WS5 (espressioni).

Pesi ``N(0, 1)`` sui 100 modi PCA, senza troncamento: ``FaceModel.randomize_identity``, la
stessa distribuzione di ICT-5000 (v2_work/genict/generate_identities.py).

Mesh per identita'
------------------
Le 6 topologie di ``v2_work/genict/make_ict_topologies.py``, con le sue funzioni importate senza
modifiche, e ``--n-expr`` espressioni nella topologia ``original`` col campionatore di WS5
(``aau/ict/ict_expressions_random.py::sample_expression``: 3-8 dei 45 blendshape non di sguardo,
coefficienti U(0.3, 1)), nome ``id<g>_GTready_rexpr<k>`` come nelle viste rexpr esistenti. La GT
di un'espressione e' quella della sua identita'. Unica differenza dalla gemella: il seme del
rumore di ``noisy`` e' ``SeedSequence([SEED_NOISE, g])`` invece di ``int(subject[-4:])``, che
per ``id20000`` darebbe 0, lo stesso campo di rumore di ``ict0000``.

Guardia held-out (lo shard NON viene scritto se fallisce)
---------------------------------------------------------
  1. nessun nome dello shard e' in ``heldout_frozen_union15.json`` (BFM, ICT, in nome di vista o
     grezzo) e ogni id sta in ``new_id_range``;
  2. nessuna identita' nuova e' un quasi-duplicato di un held-out ICT: la vertex-mean-L2 fra
     ``original`` normalizzate maxabs (la metrica della GT) verso TUTTE le original di
     ``heldout_ict_originals.npz`` resta sopra ``dup_threshold`` (meta' del vicino piu'
     prossimo piu' vicino di ICT-5000);
  3. mesh: vertici finiti, nessun vertice non referenziato, ``original``/``noisy``/``rexpr*``
     sulla tabella di facce di ICT-5000, triangoli di ``remesh``/``down8k``/``up60k`` entro il
     2% dei target, ``crop`` piu' piccolo di ``original``.

Uscita
------
``<out-dir>/shard_<KKKKK>.tar`` (tar non compresso di npz compressi ``V`` float32 / ``F``
int32, nomi piatti ``id<g>_GTready_<etichetta>.npz`` piu' ``manifest.json`` con semi, pesi e
vettori di espressione) e accanto ``shard_<KKKKK>.json`` (id, byte, tempi, esito della
guardia). Il tar e' scritto in ``--work-dir`` e poi spostato con rename: o c'e' intero o non
c'e'. Uno shard gia' presente viene saltato (riavviabile).
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
sys.path.insert(0, str(REPO_ROOT / "v2_work" / "genict"))
sys.path.insert(0, str(AAU_DIR / "ict"))

import mesh_ops as mo  # noqa: E402
from ict_model import N_SHAPE, ict_shape_mesh, load_ict  # noqa: E402
from make_ict_topologies import (  # noqa: E402
    REMESH_DECIMATION, make_down8k, make_noisy, make_remesh, make_up60k, triangle_targets)

ID_BASE = 20000
SEED_ID = 20261005       # pesi d'identita'      (ICT-5000: 1234)
SEED_NOISE = 20261006    # rumore di `noisy`     (ICT-5000: int(subject[-4:]))
SEED_EXPR = 20261007     # espressioni           (WS5: 1234 + id)
TOPOLOGIES = ("original", "remesh", "crop", "noisy", "down8k", "up60k")
FROZEN = THIS_DIR / "heldout_frozen_union15.json"   # la guardia di generazione usa l'unione
HELDOUT_GEOM = THIS_DIR / "heldout_ict_originals.npz"

_MODEL: dict | None = None
_POOL: tuple[str, ...] = ()
_N_EXPR = 0


def rng_for(seed: int, g: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, g]))


def identity_weights(g: int) -> np.ndarray:
    return rng_for(SEED_ID, g).normal(0.0, 1.0, size=N_SHAPE)


def expression_pool() -> tuple[str, ...]:
    from ict_expressions_random import GAZE_SHAPES, blendshape_names
    from ict_model import MODEL_DIR
    return tuple(n for n in blendshape_names(MODEL_DIR) if n not in GAZE_SHAPES)


def maxabs(V: np.ndarray) -> np.ndarray:
    Vc = V - V.mean(0, keepdims=True)
    return Vc / max(float(np.abs(Vc).max()), 1e-9)


def geom_bytes(V: np.ndarray, F: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, V=np.asarray(V, dtype=np.float32), F=np.asarray(F, dtype=np.int32))
    return buf.getvalue()


def mesh_errors(name: str, label: str, V: np.ndarray, F: np.ndarray, F_ref: np.ndarray,
                expect: dict) -> list[str]:
    err = []
    if not np.isfinite(V).all():
        err.append(f"{name}: vertici non finiti")
    if F.max() != len(V) - 1 or len(np.unique(F)) != len(V):
        err.append(f"{name}: vertici non referenziati")
    if label in ("original", "noisy") or label.startswith("rexpr"):
        if not np.array_equal(F, F_ref):
            err.append(f"{name}: facce diverse dalla topologia fissa di ICT")
    if label in expect and abs(len(F) - expect[label]) > 0.02 * expect[label]:
        err.append(f"{name}: {len(F)} triangoli, attesi ~{expect[label]}")
    if label == "crop" and len(F) >= len(F_ref):
        err.append(f"{name}: crop non ha tolto niente")
    return err


def _init(pool: tuple[str, ...], n_expr: int) -> None:
    global _MODEL, _POOL, _N_EXPR
    _MODEL = load_ict(n_shape=N_SHAPE, expressions=pool)
    _POOL, _N_EXPR = pool, n_expr


def process_identity(g: int) -> dict:
    from ict_expressions_random import sample_expression

    sid = f"id{g:05d}"
    t0 = time.time()
    w = identity_weights(g)
    V0, F0 = ict_shape_mesh(w, _MODEL)
    base = mo.prepare_open_surface(*mo.as_arrays(V0, F0))
    if len(base[1]) != len(F0):
        raise RuntimeError(f"{sid}: superficie non pulita ({len(F0)} -> {len(base[1])} triangoli)")
    down_t, up_t = triangle_targets(len(base[1]))
    noise_seed = int(np.random.SeedSequence([SEED_NOISE, g]).generate_state(1)[0])
    meshes = {
        "original": base,
        "remesh": make_remesh(*base),
        "crop": mo.make_crop(*base),
        "noisy": make_noisy(*base, seed=noise_seed),
        "down8k": make_down8k(*base, target=down_t),
        "up60k": make_up60k(*base, target=up_t),
    }
    rng = rng_for(SEED_EXPR, g)
    vectors = []
    for k in range(1, _N_EXPR + 1):
        idx, coef = sample_expression(rng, len(_POOL))
        vec = {_POOL[int(i)]: float(c) for i, c in zip(idx, coef)}
        V = base[0].copy()
        for name, c in vec.items():
            V += c * _MODEL["exprdirs"][name]
        meshes[f"rexpr{k}"] = (V, base[1])
        vectors.append(vec)

    expect = {"remesh": int(len(F0) * REMESH_DECIMATION), "down8k": down_t, "up60k": up_t}
    blobs, sizes, errors = {}, {}, []
    for label, (V, F) in meshes.items():
        name = f"{sid}_GTready_{label}.npz"
        errors += mesh_errors(name, label, V, F, F0, expect)
        blobs[name] = geom_bytes(V, F)
        sizes[label] = {"n_verts": int(len(V)), "n_faces": int(len(F)), "bytes": len(blobs[name])}
    return {"sid": sid, "g": g, "weights": [float(x) for x in w], "expressions": vectors,
            "sizes": sizes, "errors": errors, "seconds": time.time() - t0,
            "original_maxabs": maxabs(base[0]).astype(np.float32), "blobs": blobs}


def guard_names(ids: list[str]) -> list[str]:
    fz = json.loads(FROZEN.read_text())
    lo, hi = fz["new_id_range"]
    frozen = set(fz["bfm"]) | set(fz["ict_view"]) | set(fz["ict_raw"])
    err = [f"{s}: e' un soggetto di test congelato" for s in ids if s in frozen]
    err += [f"{s}: fuori da new_id_range {lo}-{hi}" for s in ids if not lo <= int(s[2:]) <= hi]
    return err


def guard_duplicates(originals: np.ndarray) -> dict:
    """Vicino piu' prossimo di ogni identita' nuova fra gli held-out ICT, metrica della GT."""
    import torch

    with np.load(HELDOUT_GEOM) as z:
        H = torch.from_numpy(z["V"])
        thr = float(z["dup_threshold"])
    N = torch.from_numpy(originals)
    nn = torch.stack([(H - a).norm(dim=-1).mean(-1).min() for a in N]).double().numpy()
    return {"threshold": thr, "nn_min": float(nn.min()), "nn_median": float(np.median(nn)),
            "n_below": int((nn <= thr).sum()), "ok": bool((nn > thr).all())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-per-shard", type=int, default=250)
    ap.add_argument("--n-expr", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--n-cores", type=int, default=1)
    a = ap.parse_args()

    out = a.out_dir / f"shard_{a.shard:05d}.tar"
    if out.exists():
        print(f"[skip] {out} esiste gia'", flush=True)
        return
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.work_dir.mkdir(parents=True, exist_ok=True)

    gs = [ID_BASE + a.shard * a.n_per_shard + j for j in range(a.n_per_shard)]
    ids = [f"id{g:05d}" for g in gs]
    name_err = guard_names(ids)
    if name_err:
        raise SystemExit("GUARDIA HELD-OUT FALLITA (nomi): " + "; ".join(name_err[:5]))
    pool = expression_pool()
    print(f"shard {a.shard}: {ids[0]}..{ids[-1]}  n_expr={a.n_expr} cores={a.n_cores}", flush=True)

    t0 = time.time()
    records = []
    if a.n_cores > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.n_cores, initializer=_init, initargs=(pool, a.n_expr)) as p:
            records = p.map(process_identity, gs, chunksize=1)
    else:
        _init(pool, a.n_expr)
        records = [process_identity(g) for g in gs]
    t_gen = time.time() - t0

    mesh_err = [e for r in records for e in r["errors"]]
    dup = guard_duplicates(np.stack([r.pop("original_maxabs") for r in records]))
    if mesh_err or not dup["ok"]:
        raise SystemExit(f"GUARDIA FALLITA: mesh={mesh_err[:5]} duplicati={dup}")

    tmp = a.work_dir / f".{out.name}.{os.getpid()}.tmp"
    n_bytes_members = 0
    with tarfile.open(tmp, "w") as tar:
        for r in records:
            for name, blob in sorted(r.pop("blobs").items()):
                info = tarfile.TarInfo(name)
                info.size = len(blob)
                tar.addfile(info, io.BytesIO(blob))
                n_bytes_members += len(blob)
        manifest = {
            "shard": a.shard, "n_per_shard": a.n_per_shard, "id_base": ID_BASE,
            "seeds": {"identity": SEED_ID, "noise": SEED_NOISE, "expression": SEED_EXPR,
                      "rule": "SeedSequence([seed, g])"},
            "n_expr": a.n_expr, "topologies": list(TOPOLOGIES),
            "identities": [{k: r[k] for k in ("sid", "g", "weights", "expressions")} for r in records],
        }
        blob = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(blob)
        tar.addfile(info, io.BytesIO(blob))
    part = out.with_name(out.name + ".part")
    shutil.move(str(tmp), str(part))
    os.replace(part, out)

    side = {
        "shard": a.shard, "ids": ids, "n_identities": len(ids), "n_expr": a.n_expr,
        "n_meshes": sum(len(r["sizes"]) for r in records),
        "tar_bytes": out.stat().st_size, "member_bytes": n_bytes_members,
        "bytes_per_label_mean": {lab: float(np.mean([r["sizes"][lab]["bytes"] for r in records]))
                                 for lab in records[0]["sizes"]},
        "seconds": {"wall_generation": t_gen, "wall_total": time.time() - t0,
                    "cpu_per_identity": float(np.mean([r["seconds"] for r in records]))},
        "n_cores": a.n_cores,
        "guard": {"names": "ok", "meshes": "ok", "duplicates": dup},
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=1) + "\n")
    print(f"scritto {out} ({side['tar_bytes'] / 1e6:.1f} MB, {side['n_meshes']} mesh) in "
          f"{side['seconds']['wall_total'] / 60:.1f} min; nn held-out min={dup['nn_min']:.4f} "
          f"(soglia {dup['threshold']:.4f})", flush=True)


if __name__ == "__main__":
    main()
