#!/usr/bin/env python3
"""D3, emendamento 1 (sez. 3): messa in scena dei pool NON valutati di FaceScape, HIFI3D e FaceVerse.

    AAU_NV="" aau/run.sh aau/diagnostics/d3_pools.py          (d3.sbatch, passo pools, prima degli operatori)

Per pool: le 500 identita' della ``eval_view``; valutate = i soggetti dello store ufficiale di factorized s1234 (i 100
di ``zs_stage.select_subjects``, seme 1234); sorgenti = i primi ``N_SOURCE`` per id fra le non valutate; controllo =
i primi ``N_CHECK`` valutati (solo per K7, mai in teste o statistiche). Mesh = gli stessi file di ``eval_view/npz``
senza crop, in ``datasets/DIAG_D3/pools/<pool>/in``: link per FaceScape e HIFI3D, copie con le facce invertite per
FaceVerse (``zs_stage.py --flip-faces``, F[:, ::-1], come gli store neutri). ``d3/pools/<pool>/subjects.json``:
soggetti, vista, tabella di scala ufficiale, GT.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

import diag

D3 = diag.EV / "d3"
STAGE = diag.REPO / "datasets/DIAG_D3/pools"
OPS = diag.REPO / "datasets/V3_OPS_CACHE/diag_d3/pools"
ST = diag.TV3 / "factorized/scale_tables"
EVAL = diag.TV3 / "ablations/c3f_eval"
POOLS = {
    "facescape": {"view": diag.REPO / "datasets/DEV_FACESCAPE/eval_view", "flip": False, "table": ST / "devfs_eval.npz",
                  "store": EVAL / "form_devfs/data_*/scale_v3factorizedfulle072_embed/zs_zeroshot/embeddings.npz",
                  "gt": "facescape"},
    "hifi3d": {"view": diag.REPO / "datasets/HIFI3D/eval_view", "flip": False, "table": ST / "hifi3d_eval.npz",
               "store": EVAL / "form_hifi/data_*/scale_v3factorizedfulle072_embed/zs_zeroshot/embeddings.npz",
               "gt": "hifi3d"},
    "faceverse": {"view": diag.REPO / "datasets/FACEVERSE_ZS/eval_view", "flip": True,
                  "table": diag.REPO / "aau/runs/evidence/faceverse_neutral/scale_tables/fv_eval.npz",
                  "store": diag.REPO / "aau/runs/evidence/faceverse_neutral/embed/data_*/"
                                       "scale_v3factorizedfulle072_flip_embed/zs_zeroshot/embeddings.npz",
                  "gt": "faceverse"},
}
N_SOURCE, N_CHECK = 200, 5


def subjects(pool: str) -> dict:
    """{pool, evaluated, source, check} di un pool."""
    cfg = POOLS[pool]
    ids = sorted({Path(p).name.split("_GTready_")[0] for p in glob.glob(str(cfg["view"] / "npz" / "id*_GTready_*.npz"))})
    hits = sorted(glob.glob(str(cfg["store"])))
    with np.load(hits[0], allow_pickle=True) as z:
        ev = sorted({str(s) for s in z["subjects"]})
    if len(ids) != 500 or len(ev) != 100 or not set(ev) <= set(ids):
        raise SystemExit(f"{pool}: pool {len(ids)}, valutati {len(ev)}")
    rest = [s for s in ids if s not in set(ev)]
    return {"pool": ids, "evaluated": ev, "source": rest[:N_SOURCE], "check": ev[:N_CHECK], "store": hits[0]}


def stage(pool: str, subj: dict) -> int:
    cfg = POOLS[pool]
    out = STAGE / pool / "in"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.npz"):
        stale.unlink()
    n = 0
    for sid in subj["source"] + subj["check"]:
        for lab in diag.LABELS:
            src = cfg["view"] / "npz" / f"{sid}_GTready_{lab}.npz"
            if not src.exists():
                raise SystemExit(f"mesh mancante: {src}")
            dst = out / src.name
            if cfg["flip"]:                                  # zs_stage.py --flip-faces (segni +1)
                with np.load(src) as d:
                    V, F = (d["V"], d["F"]) if "V" in d else (d["verts"], d["faces"])
                np.savez(dst, V=(V * np.ones(3)).astype(V.dtype), F=np.ascontiguousarray(F[:, ::-1]))
            else:
                dst.symlink_to(src.resolve())
            n += 1
    return n


def main() -> None:
    for pool, cfg in POOLS.items():
        subj = subjects(pool)
        if set(subj["source"]) & set(subj["evaluated"]):           # K6: sorgenti disgiunte dai valutati
            raise SystemExit(f"{pool}: sorgenti fra i soggetti valutati")
        with np.load(cfg["table"], allow_pickle=True) as z:
            names = {str(x) for x in z["names"]}
        miss = [f"{s}_GTready_{lab}.npz" for s in subj["source"] + subj["check"] for lab in diag.LABELS
                if f"{s}_GTready_{lab}.npz" not in names]
        if miss:
            raise SystemExit(f"{pool}: {len(miss)} mesh assenti dalla tabella di scala (es. {miss[:2]})")
        n = stage(pool, subj)
        rec = {"definition": "PROTOCOL_D3_emendamento_1.md sez. 3", "pool": pool, "view_dir": str(cfg["view"]),
               "flip_faces": cfg["flip"], "scale_table": str(cfg["table"]), "gt": cfg["gt"], "store": subj["store"],
               "n_pool": len(subj["pool"]), "n_evaluated": len(subj["evaluated"]), "source": subj["source"],
               "check": subj["check"], "n_meshes": n, "stage_dir": str(STAGE / pool / "in"),
               "ops_dir": str(OPS / pool / "ops")}
        diag.atomic_json(D3 / "pools" / pool / "subjects.json", rec)
        print(f"[d3-pools] {pool}: {len(subj['source'])} sorgenti ({subj['source'][0]}-{subj['source'][-1]}), "
              f"{len(subj['check'])} di controllo, {n} mesh in {STAGE / pool / 'in'}", flush=True)


if __name__ == "__main__":
    main()
