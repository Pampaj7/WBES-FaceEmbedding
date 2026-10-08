#!/usr/bin/env python3
"""Controllo di ``canonical_transforms.json`` come lo userebbe chi lo legge: mesh prese dal disco.

    v3_work/unified_gt/run.sh v3_work/unified_gt/check_canonical.py

Per ogni dominio una mesh di DATI (non il template), trasformata con ``s R x + t`` del json e, se
``flip_faces``, con le facce invertite. Misure, nel frame FLAME in mm:
  - ``normal_z``: normale media (pesata per area) dei triangoli davanti, |z| piu' alta del 50% della
    profondita': deve essere > 0 (normali uscenti, +z fuori dal volto);
  - ``rms_to_flame_mm``: RMS fra la mappa baricentrica della mesh trasformata e la media FLAME sulla
    regione unificata, SENZA altri allineamenti (pochi mm se frame, alto/basso e scala sono giusti;
    decine se una rotazione o un segno sono sbagliati);
  - ``nose_mm``: distanza fra il vertice piu' avanti (+z) della mesh e la punta del naso FLAME.
Scrive ``aau/runs/evidence/e8/canonical_check.json``.
"""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np

import ugt as C
import domains


def shard_original(shard: str, sid: str):
    with tarfile.open(shard) as t:
        with np.load(io.BytesIO(t.extractfile(f"{sid}_GTready_original.npz").read())) as z:
            return np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)


def npz(path):
    with np.load(path) as z:
        return np.asarray(z["V"], np.float64), np.asarray(z["F"], np.int64)


def main() -> None:
    ct = json.loads((C.DATA_ROOT / "canonical_transforms.json").read_text())["domains"]
    sp = np.load(C.DATA_ROOT / "unified_space.npz")
    flame = domains.flame()
    Vf = flame["V"] * 1000.0
    P_flame = Vf[sp["flame_vidx"]]
    nose_flame = Vf[np.argmax(Vf[:, 2])]
    D = domains.DATASETS
    gnm_m = None
    samples = {
        "flame": (flame["V"], flame["F_render"], "v_template di FLAME 2020", None),
        "bfm": (*npz(D / "REMESH/npz_data_topo_500/id0007_GTready_original.npz"), "REMESH id0007 original", "full"),
        "ict": (*npz(D / "ICT/identities/ict0007.npz"), "ICT-5000 ict0007", "full"),
        "gnm": (*shard_original(str(D / "GNM_DISTILL/shards/gnm_shard_00000.tar"), "id100007"), "GNM_DISTILL id100007 (patch)", "gnm_patch"),
        "hifi3d": (*npz(D / "HIFI3D/identities/hifi0007.npz"), "HIFI3D hifi0007 (patch)", "hifi_patch"),
        "faceverse": (*npz(D / "FACEVERSE_ZS/identities/fv0007.npz"), "FaceVerse fv0007", "full"),
        "facescape": (domains.facescape()["V"], domains.facescape()["F_render"], "template FaceScape (nessuna identita' su disco)", "full"),
        "multiface": (domains.multiface()["V"], domains.multiface()["F_render"], "template Multiface (le mesh tracked hanno pose per frame)", "full"),
    }
    out = {}
    for d, (V, F, what, kind) in samples.items():
        c = ct[d]
        X = c["s"] * V @ np.asarray(c["R"]).T + np.asarray(c["t"])
        Fo = F[:, ::-1] if c["flip_faces"] else F
        n = C.face_normals(X, Fo, unit=False)
        zc = X[Fo].mean(1)[:, 2]
        front = zc > np.percentile(zc, 50)
        nz = n[front].sum(0)
        rec = {"mesh": what, "normal_z": float(nz[2] / np.linalg.norm(nz)),
               "nose_mm": float(np.linalg.norm(X[np.argmax(X[:, 2])] - nose_flame))}
        # RMS dalla media FLAME sulla regione: serve la mesh nei vertici del template del dominio
        if kind == "full" or d == "flame":
            Q = C.bary_interp(X, sp[f"vidx_{d}"], sp[f"bary_{d}"])
        elif kind == "hifi_patch":
            import hifi_model
            m = hifi_model.load_hifi(domains.HIFI_MAT)
            Vh = np.zeros_like(m["v_template"])
            Vh[m["face_vertices"]] = X
            used = np.isin(sp[f"vidx_{d}"], m["face_vertices"]).all(1)
            Q = np.where(used[:, None], C.bary_interp(Vh, sp[f"vidx_{d}"], sp[f"bary_{d}"]), np.nan)
        else:  # gnm: la patch e' un sottoinsieme (piu' i centri dei quad) della testa
            import gnm_model
            gnm_m = gnm_m or gnm_model.load_gnm(domains.GNM_NPZ)
            fv = gnm_m["face_vertices"]
            nv = len(domains.gnm()["V"])
            Vh = np.zeros((nv, 3))
            keep = fv < nv
            Vh[fv[keep]] = X[keep]
            used = np.isin(sp[f"vidx_{d}"], fv[keep]).all(1)
            Q = np.where(used[:, None], C.bary_interp(Vh, sp[f"vidx_{d}"], sp[f"bary_{d}"]), np.nan)
        ok = np.isfinite(Q).all(1)
        rec["rms_to_flame_mm"] = float(np.sqrt(((Q[ok] - P_flame[ok]) ** 2).sum(1).mean()))
        rec["n_region_points"] = int(ok.sum())
        out[d] = rec
        print(f"[canon-check] {d}: {rec}", flush=True)
    C.save_json(C.EVID_DIR / "canonical_check.json", out)


if __name__ == "__main__":
    main()
