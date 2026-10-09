#!/usr/bin/env python
"""Correttezza (a): una vista dello stream ha gli operatori della pipeline attuale per la stessa mesh.

    aau/run.sh v3_work/stream/tests/test_ops_equiv.py --work /tmp/$SLURM_JOB_ID/eq \
        --out aau/runs/evidence/stream/ops_equiv.json [--domains ict,gnm,flame2020,famos,bfm2019]

Per ogni dominio un'identita' dei produttori (sources.py) e le sei discretizzazioni (views.discretize, con
un'espressione su remesh e noisy). La STESSA geometria va per due strade:
  attuale  npz V/F (come gen_ict_shard) -> v3_work/trainer/prepass_v3.py (prepass_ops areanorm + grad_vec, k 128,
           un sottoprocesso) -> GTReadyDatasetNPZ (il loader congelato);
  stream   views.operators -> serve_like_loader -> compact (autovettori fp32 e fp16) -> shard nell'anello ->
           StreamConsumer.materialize (senza rotazione ne' scala, input maxabs).
Confronto: tensore per tensore (vertici, facce, massa, autovalori, autovettori, indici e valori dei gradienti,
layout compreso) e embedding del checkpoint e108 del run su scala (forward congelato v1, CPU fp32, eval).
Il pre-pass attuale gira due volte (``ref2``): lo scarto fra le due e' il pavimento numerico (eigsh di ARPACK
parte da un vettore casuale).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))

CKPT = (REPO / "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411/mixed_xtopo_xyz_dn_rank0.50_"
        "id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__3167d36d/checkpoints/epoch108.pth")
DENSE = ("verts", "mass", "evals", "evecs")


def embed(model, s):
    import torch
    from robustness.model_helpers import forward_model
    with torch.no_grad():
        return forward_model(model, s, s["verts"], False, False)[0].reshape(-1).double()


def compare(ref: dict, got: dict) -> dict:
    import torch
    d = {}
    for k in DENSE:
        d[k] = float((ref[k].double() - got[k].double()).abs().max())
        d[f"{k}_layout_same"] = bool(ref[k].dtype == got[k].dtype and ref[k].stride() == got[k].stride())
    d["faces_equal"] = bool(torch.equal(ref["faces"].long(), got["faces"].long()))
    for k in ("gradX", "gradY"):
        a, b = ref[k].coalesce(), got[k].coalesce()
        d[f"{k}_indices_equal"] = bool(torch.equal(a.indices(), b.indices()))
        d[k] = float((a.values().double() - b.values().double()).abs().max()) if d[f"{k}_indices_equal"] else float("inf")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domains", default="ict,gnm,flame2020,famos,bfm2019")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    import sources as S
    import views as VW
    from consumer import StreamConsumer
    from model_v3 import build_model_v3
    from ring import Ring
    from dataset_gtready import GTReadyDatasetNPZ

    VW.install_grad_vec()
    rng = np.random.default_rng(a.seed)
    uni = S.Unified()
    doms = [d for d in a.domains.split(",") if d]
    srcs = {}
    for d in doms:
        try:
            srcs[d] = S.build_sources([d], uni)[d]
        except FileNotFoundError as exc:          # BFM 2019 senza mappa: si dice e si va avanti
            print(f"[equiv] {d} saltato: {exc}", flush=True)
    geom = a.work / "geom"
    geom.mkdir(parents=True, exist_ok=True)
    meshes = {}
    for di, (d, src) in enumerate(srcs.items()):
        ident = src.identity(rng)
        for lab in VW.LABELS:
            V, F, tag = src.view_mesh(ident, rng, lab in ("remesh", "noisy"))
            Vd, Fd = VW.discretize(V, F, lab, 1234 + di)
            name = f"id{900 + di:04d}_GTready_{lab}.npz"     # nome che prepass_ops sa leggere
            np.savez_compressed(geom / name, V=Vd.astype(np.float32), F=Fd.astype(np.int32))
            meshes[name] = (d, lab, tag, Vd, Fd)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for tag in ("ref", "ref2"):
        subprocess.run([sys.executable, str(REPO / "v3_work/trainer/prepass_v3.py"), "--out-dir", str(a.work / tag),
                        "--geom-dirs", str(geom), "--n-proc", "4", "--convention", "areanorm"], check=True, env=env)
    loaders = {tag: GTReadyDatasetNPZ(str(a.work / tag)) for tag in ("ref", "ref2")}
    # stream: stesse mesh, uno shard per formato degli autovettori
    rings = {}
    t_ops = {}
    for ed in ("fp32", "fp16"):
        ring = Ring(a.work / f"ring_{ed}")
        groups = []
        for name, (d, lab, tag, Vd, Fd) in meshes.items():
            t0 = time.perf_counter()
            data = VW.operators(Vd, Fd, 128)
            t_ops[name] = time.perf_counter() - t0
            arr, ef = VW.compact(VW.serve_like_loader(data), ed)
            meta = {"label": lab, "expr": tag, "n": len(arr["verts"]), "m": len(arr["faces"]), "k": len(arr["evals"]),
                    "nx": len(arr["gxv"]), "ny": len(arr["gyv"]), "evecs_f": ef, "evecs_dtype": ed}
            groups.append({"key": name, "domain": d, "s": np.zeros(3, np.float32), "views": [(arr, meta)]})
        ring.write(groups, 0)
        rings[ed] = StreamConsumer(ring.root, rot_deg=(0, 0, 0), scale=0.0, prefetch=0)
        rings[ed].refresh()
    pack = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = build_model_v3(SimpleNamespace(**pack["args"]), torch.device("cpu"))
    model.load_state_dict(pack["state_dict"])
    model.eval()
    rows, worst = [], {}
    for gi, name in enumerate(meshes):
        d, lab, tag, Vd, Fd = meshes[name]
        ref = loaders["ref"][loaders["ref"].files.index(name)]
        ref2 = loaders["ref2"][loaders["ref2"].files.index(name)]
        z_ref = embed(model, ref)
        row = {"mesh": name, "domain": d, "label": lab, "expr": tag, "n_verts": int(len(Vd)),
               "z_norm": float(z_ref.norm()), "ref2": {**compare(ref, ref2), "z_max_abs": float((embed(model, ref2) - z_ref).abs().max())}}
        for ed, c in rings.items():
            rd = c.readers[0]
            got = c.materialize(rd, gi, 0, [0.0, 0.0, 0.0], 1.0)
            cmp = compare(ref, got)
            dz = (embed(model, got) - z_ref).abs()
            cmp["z_max_abs"] = float(dz.max())
            cmp["z_rel"] = float(dz.norm() / z_ref.norm())
            row[ed] = cmp
        rows.append(row)
        print(f"[equiv] {name} {d}/{lab}/{tag} n={len(Vd)} |z|={row['z_norm']:.3f} dz ref2 {row['ref2']['z_max_abs']:.2e} "
              f"fp32 {row['fp32']['z_max_abs']:.2e} fp16 {row['fp16']['z_max_abs']:.2e} (evecs fp16 "
              f"{row['fp16']['evecs']:.1e}, grad {row['fp32']['gradX']:.1e})", flush=True)
    for key in ("ref2", "fp32", "fp16"):
        worst[key] = {k: (max(r[key][k] for r in rows) if isinstance(rows[0][key][k], float) else all(r[key][k] for r in rows))
                      for k in rows[0][key]}
    tol = 1e-5
    out = {"checkpoint": str(CKPT), "n_meshes": len(rows), "domains": list(srcs), "k_eig": 128, "tolerance_z": tol,
           "pass_fp32": worst["fp32"]["z_max_abs"] <= tol, "pass_fp16": worst["fp16"]["z_max_abs"] <= tol,
           "worst": worst, "stream_ops_seconds_single_process": t_ops, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_meshes", "domains", "pass_fp32", "pass_fp16")}), flush=True)
    print(json.dumps(worst, indent=1), flush=True)


if __name__ == "__main__":
    main()
