#!/usr/bin/env python
"""Aree per vertice normali e robuste su mesh VERE: fanno quello che serve a E3?

Per un soggetto BFM e uno ICT, sulle 6 topologie (original, noisy, remesh, crop, down8k, up60k), nel frame del loader:
  * area totale rispetto all'original con i pesi mass, smooth (k 64 e 128) e winsor: la noisy e' gonfiata dal rumore
    con mass, deve esserlo poco con smooth; down8k/remesh/up60k devono stare vicino a 1 con tutti;
  * centro: distanza dal centro dell'original, per la media per vertice (loader v2) e per area (mass, smooth),
    in unita' di sqrt(area) dell'original;
  * invarianza del frame robusto a traslazione e scala uniforme.

    aau/run.sh v3_work/trainer/tests/test_area.py --out aau/runs/evidence/trainer_v3/area.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(TRAINER))
import common  # noqa: E402,F401
import area_v3  # noqa: E402
from dataset_gtready import GTReadyDatasetNPZ  # noqa: E402

DIRS = {"bfm": ("datasets/REMESH/npz_data_topo_500_withops_areanorm", "id0001"),
        "ict": ("datasets/ICT/train_ready/npz_withops", "id10000")}
TOPOS = ("original", "noisy", "remesh", "crop", "down8k", "up60k")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    ok = True
    for dom, (d, sid) in DIRS.items():
        ds = GTReadyDatasetNPZ(str(REPO / d))
        pos = {f: i for i, f in enumerate(ds.files)}
        rows = {}
        base = None
        for t in TOPOS:
            s = ds[pos[f"{sid}_GTready_{t}.npz"]]
            V, F, m, E = s["verts"].double(), s["faces"], s["mass"].double(), s["evecs"].double()
            r = {"n": int(V.shape[0])}
            for mode, k in (("mass", None), ("smooth", 64), ("smooth", 128), ("winsor", None)):
                if k:
                    area_v3.CFG["k"] = k
                w = area_v3.area_weights(mode, V, F, m, E)
                tag = mode + (str(k) if k else "")
                r[f"area_{tag}"] = float(w.sum())
                r[f"center_{tag}"] = ((w.unsqueeze(1) * V).sum(0) / w.sum()).tolist()
            r["center_vertexmean"] = V.mean(0).tolist()
            # invarianza del frame (smooth k 64): stesso risultato da V traslata e scalata
            area_v3.CFG["k"] = 64
            A1 = area_v3.area_frame("smooth", V, F, m, E)
            A2 = area_v3.area_frame("smooth", V * 3.3 + 0.7, F, m, E)
            r["frame_invariance_smooth"] = float((A1 - A2).abs().max())
            rows[t] = r
            if t == "original":
                base = r
        # rapporti e spostamenti rispetto all'original (centro in unita' di sqrt(area mass dell'original))
        unit = base["area_mass"] ** 0.5
        for t, r in rows.items():
            for tag in ("mass", "smooth64", "smooth128", "winsor"):
                r[f"ratio_area_{tag}"] = r[f"area_{tag}"] / base[f"area_{tag}"]
            for tag in ("vertexmean", "mass", "smooth64"):
                c = torch.tensor(r[f"center_{tag}"])
                c0 = torch.tensor(base[f"center_{tag}"])
                r[f"dcenter_{tag}"] = float((c - c0).norm()) / unit
            print(f"{dom} {t:8s} n={r['n']:6d} area/orig: mass {r['ratio_area_mass']:.3f} smooth64 "
                  f"{r['ratio_area_smooth64']:.3f} smooth128 {r['ratio_area_smooth128']:.3f} winsor "
                  f"{r['ratio_area_winsor']:.3f} | spostamento del centro: vertici {r['dcenter_vertexmean']:.4f} "
                  f"mass {r['dcenter_mass']:.4f} smooth {r['dcenter_smooth64']:.4f} | invarianza {r['frame_invariance_smooth']:.1e}",
                  flush=True)
            ok &= r["frame_invariance_smooth"] < 1e-6
        out[dom] = rows
    out["frame_invariance_pass"] = bool(ok)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
