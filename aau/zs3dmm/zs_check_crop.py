#!/usr/bin/env python3
"""Controllo: la topologia ``crop`` di ogni identita' e' davvero diversa dalla ``original``.

    aau/run.sh aau/zs3dmm/zs_check_crop.py --topo-dir datasets/HIFI3D/topo --prefix hifi

``mesh_ops.trim_far_boundary_band`` (porting di ``datasets/remesh.py``) restituisce la mesh
INTATTA se la banda da togliere supera il 15% dei vertici (``MIN_KEEP_RATIO = 0.85``), e in
quel caso il ``crop`` di quell'identita' e' una copia della ``original``: le coppie
crop<->original misurerebbero zero perturbazione. Qui si conta per ogni identita' la frazione
di vertici tenuti dal crop; esce 1 se anche una sola identita' ha il crop identico.
``--write`` mette il risultato in ``<topo-dir>/crop_check.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--topo-dir", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--write", action="store_true")
    a = p.parse_args()

    ratios, identical = {}, []
    for orig in sorted(a.topo_dir.glob(f"{a.prefix}[0-9]*_GTready_original.npz")):
        sid = orig.name.split("_GTready_")[0]
        with np.load(orig) as d:
            Vo, Fo = d["V"], d["F"]
        with np.load(a.topo_dir / f"{sid}_GTready_crop.npz") as d:
            Vc, Fc = d["V"], d["F"]
        ratios[sid] = len(Vc) / len(Vo)
        if len(Vc) == len(Vo) and len(Fc) == len(Fo) and np.array_equal(Vc, Vo):
            identical.append(sid)
    r = np.asarray(list(ratios.values()))
    out = {"n_identities": len(r), "n_crop_identical_to_original": len(identical),
           "identical": identical[:20],
           "kept_vertex_fraction": {"min": float(r.min()), "median": float(np.median(r)), "max": float(r.max())}}
    print(json.dumps(out, indent=2))
    if a.write:
        (a.topo_dir / "crop_check.json").write_text(json.dumps(out, indent=2) + "\n")
    if identical:
        raise SystemExit(f"{len(identical)} identita' con crop == original")


if __name__ == "__main__":
    main()
