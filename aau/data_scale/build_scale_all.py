#!/usr/bin/env python3
"""Dati del run grande BFM + ICT-5000 + ICT_SCALE + GNM: tar co-locati, indice unito, GT congiunta.

    aau/run.sh aau/data_scale/build_scale_all.py

* ``datasets/SCALE_ALL/shards/``: symlink ai 200 tar di ``datasets/ICT_SCALE/shards`` e ai 41 di
  ``datasets/GNM_DISTILL/shards`` (prefisso ``gnm_``, nessuna collisione di nomi), piu'
  ``index.npz``, la concatenazione dei due indici con ``tar_id`` riallineati: ``train_steps.py`` e
  ``prepass_ops.py`` leggono UN indice e cercano i tar nella sua directory.
* ``datasets/SCALE_ALL/gt_joint_bfm_ict_gnm.npz``: la GT congiunta, come ``build_gt.py``: blocco BFM
  e blocco ICT copiati da ``datasets/ICT_SCALE/gt_joint_bfm_ict.npz`` (gia' alla scala della GT in
  uso: BFM max 0.834, ICT diviso per il massimo di ICT-5000, max 1.186), blocco GNM da
  ``datasets/GNM_DISTILL/gt/gnm_matrix_distances_maxabs.npz`` (diviso per il suo massimo, max 1, la
  convenzione di ogni file GT del repo), NaN fra i domini. Accanto ``.json`` con ``global_max``, che
  la guardia ``--gt-keep-scale`` di train_steps.py legge.
Controlli: nessun nome in comune fra i blocchi; il blocco BFM+ICT copiato coincide byte per byte
con la sorgente; il blocco GNM coincide con la sua sorgente.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DS = REPO_ROOT / "datasets"
SRC = {"ict": DS / "ICT_SCALE/shards", "gnm": DS / "GNM_DISTILL/shards"}
OUT = DS / "SCALE_ALL"


def merge_index() -> dict:
    sh = OUT / "shards"
    sh.mkdir(parents=True, exist_ok=True)
    parts, tars, off = [], [], 0
    for key in ("ict", "gnm"):
        with np.load(SRC[key] / "index.npz") as z:
            d = {k: z[k] for k in z.files}
        for t in d["tars"]:
            link = sh / str(t)
            if not link.exists():
                link.symlink_to((SRC[key] / str(t)).resolve())
        d["tar_id"] = d["tar_id"].astype(np.int32) + off
        off += len(d["tars"])
        tars += [str(t) for t in d["tars"]]
        parts.append(d)
    if len(set(tars)) != len(tars):
        raise SystemExit("nomi di tar duplicati fra ICT_SCALE e GNM")
    names = np.concatenate([p["names"] for p in parts])
    if len(set(names.tolist())) != len(names):
        raise SystemExit("nomi di membri duplicati fra gli indici")
    out = sh / "index.npz"
    tmp = sh / ".index.tmp.npz"
    np.savez(tmp, tars=np.array(tars), tar_id=np.concatenate([p["tar_id"] for p in parts]).astype(np.int16),
             names=names, **{k: np.concatenate([p[k] for p in parts]) for k in ("offset", "size", "n", "m", "E")})
    os.replace(tmp, out)
    return {"index": str(out), "n_tars": len(tars), "n_members": int(len(names)),
            "by_source": {k: int(len(p["names"])) for k, p in zip(("ict", "gnm"), parts)}}


def compose_gt() -> dict:
    with np.load(DS / "ICT_SCALE/gt_joint_bfm_ict.npz") as z:
        A, an = z["D_orig"], [str(n) for n in z["names"]]
    with np.load(DS / "GNM_DISTILL/gt/gnm_matrix_distances_maxabs.npz") as z:
        G, gn = z["D_orig"].astype(np.float32), [str(n).split("_GTready")[0] for n in z["names"]]
    if set(an) & set(gn):
        raise SystemExit("nomi in comune fra BFM+ICT e GNM")
    if not all(100000 <= int(n[2:]) < 200000 for n in gn):
        raise SystemExit("la GT GNM contiene id fuori da 100000-199999")
    na, ng = len(an), len(gn)
    D = np.full((na + ng, na + ng), np.nan, dtype=np.float32)
    D[:na, :na] = A
    D[na:, na:] = G
    assert np.array_equal(D[:na, :na], A, equal_nan=True) and np.array_equal(D[na:, na:], G)
    man_a = json.loads((DS / "ICT_SCALE/gt_joint_bfm_ict.json").read_text())
    if man_a.get("scale") != "current":
        raise SystemExit("la GT di ICT_SCALE non e' alla scala attuale (rescale_gt.py)")
    out = OUT / "gt_joint_bfm_ict_gnm.npz"
    tmp = OUT / ".gt.tmp.npz"
    np.savez(tmp, D_orig=D, names=np.array(an + gn))
    os.replace(tmp, out)
    gmax = float(np.nanmax(D))
    man = {"scale": "current", "global_max": gmax, "n_total": na + ng, "n_bfm_ict": na, "n_gnm": ng,
           "blocks_max": {"bfm": man_a["bfm_block_max"], "ict": man_a["ict_block_max"],
                          "gnm": float(np.nanmax(G))},
           "sources": {"bfm_ict": str(DS / "ICT_SCALE/gt_joint_bfm_ict.npz"),
                       "gnm": str(DS / "GNM_DISTILL/gt/gnm_matrix_distances_maxabs.npz")},
           "note": "BFM e ICT alla scala della GT in uso, GNM al suo massimo, NaN fra domini; "
                   "leggere con train_steps.py --gt-keep-scale"}
    out.with_suffix(".json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"index": merge_index(), "gt": compose_gt()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
