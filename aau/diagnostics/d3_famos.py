#!/usr/bin/env python3
"""D3 (PROTOCOL_D3.md sez. 3, testa b): mesh, tabella di scala e GT di FaMoS TRAIN.

    AAU_NV="" aau/run.sh aau/diagnostics/d3_famos.py --workers 32          (d3.sbatch, passo famos)

Le 80 persone TRAIN di ``aau/famos/split.json`` (``sources.FamosSource``, importato, non modificato): neutra di
riferimento registrata FLAME (mm) sulla maschera ``face``, allineata rigidamente senza scala alla patch media FLAME
canonica (``view_mesh(expr=False)``), suddivisa 1-a-4 una volta (``igl.upsample``, come ``flame2023_s1`` di D1), le 5
discretizzazioni con ``views.discretize`` (seme di ``noisy`` = ``SeedSequence([20261120, NNN])``), salvate {V float32,
F int32} in ``datasets/DIAG_D3/famos/in/id<720000 + NNN>_GTready_<etichetta>.npz``. GT al volo dello stream
(``targets.CanonTargets()("famos", neutral_points)``): vettori fr, sr, S e matrici d_FR (mm), d_P. Controllo K3: d_FR
contro il blocco train x train di ``CANONICAL_GT/famos_F_rig_rob.npz`` (E12). Nessun file delle persone di TEST.

Uscite in ``aau/runs/evidence/diagnostics/d3/famos``: ``scale_table.npz`` (formato di fact_calib.stage, dominio
``famos``: u = 1, R = I), ``gt_famos.npz``, ``gt_names.npz`` (D_orig + names: solo per la scelta dei soggetti di
zs_embed), ``gen.json`` (semi, conteggi, K3, tempi). Dati derivati da FaMoS (licenza MPI): fuori da git.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "v3_work/stream"))
import sources as S  # noqa: E402  (mette v3_work/unified_gt su sys.path)
import views as VW  # noqa: E402

OUT = diag.EV / "d3" / "famos"
RAW_IN = diag.REPO / "datasets/DIAG_D3/famos/in"
OPS = diag.REPO / "datasets/V3_OPS_CACHE/diag_d3/famos/ops"
F_RIG_ROB = diag.REPO / "datasets/CANONICAL_GT/famos_F_rig_rob.npz"
SEED_NOISE = 20261120
ID_BASE = 720000
SUBDIV = 1
_C: dict = {}           # sorgente, ereditata dai worker (fork)


def person_num(p: str) -> int:
    """``FaMoS_subject_007`` -> 7."""
    return int(p.rsplit("_", 1)[1])


def sid_of(p: str) -> str:
    return f"id{ID_BASE + person_num(p)}"


def noise_seed(p: str) -> int:
    return int(np.random.SeedSequence([SEED_NOISE, person_num(p)]).generate_state(1)[0])


def work_mesh(src, p: str) -> tuple[np.ndarray, np.ndarray]:
    """Neutra di riferimento nel frame canonico FLAME (mm), suddivisa 1-a-4 ``SUBDIV`` volte."""
    import igl
    V, F, tag = src.view_mesh(p, np.random.default_rng(0), expr=False)
    if tag != "neutral":
        raise RuntimeError(f"{p}: vista {tag}, attesa la neutra")
    return igl.upsample(np.asarray(V, np.float64), np.asarray(F, np.int64), SUBDIV)


def _gen(p: str) -> list:
    V, F = work_mesh(_C["src"], p)
    rows = []
    for lab in diag.LABELS:
        Vd, Fd = VW.discretize(V, F, lab, noise_seed(p))
        fn = f"{sid_of(p)}_GTready_{lab}.npz"
        np.savez(RAW_IN / fn, V=Vd.astype(np.float32), F=Fd.astype(np.int32))
        Vr = Vd.astype(np.float32).astype(np.float64)       # come la rilegge fact_calib.stage
        a = float(0.5 * np.linalg.norm(np.cross(Vr[Fd[:, 1]] - Vr[Fd[:, 0]], Vr[Fd[:, 2]] - Vr[Fd[:, 0]]),
                                       axis=1).sum())
        rows.append((fn, "famos", a, a, float(np.abs(Vr - Vr.mean(0)).max()), int(len(Vd)), int(len(Fd))))
    return rows


def gt_of(src, persons: list[str]) -> dict:
    """Vettori fr, sr, S per persona e matrici d_FR (mm), d_P (``targets.CanonTargets`` sui punti neutri)."""
    from targets import CanonTargets
    tg = CanonTargets()
    t = tg("famos", np.stack([src.neutral_points(p) for p in persons]))
    out = {"names": np.asarray([sid_of(p) for p in persons]), "persons": np.asarray(persons),
           "S": np.asarray(t["S"], np.float64), "converged": np.asarray(t["converged"])}
    for kind in ("fr", "sr"):
        X = np.asarray(t[kind], np.float64)
        G = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None] - 2.0 * X @ X.T
        D = np.sqrt(np.clip(G, 0.0, None) / tg.A)
        np.fill_diagonal(D, 0.0)
        out[kind] = np.asarray(t[kind], np.float32)
        out[f"D_{kind}"] = 0.5 * (D + D.T)
    return out


def k3(g: dict) -> dict:
    """K3: d_FR contro il blocco train x train della GT-F rigida robusta di E12 (mm)."""
    with np.load(F_RIG_ROB, allow_pickle=True) as z:
        pos = {str(n): k for k, n in enumerate(z["names"])}
        ii = np.asarray([pos[str(p)] for p in g["persons"]])
        ref = np.asarray(z["D_orig"], np.float64)[np.ix_(ii, ii)]
    d = float(np.abs(ref - g["D_fr"]).max())
    return {"reference": str(F_RIG_ROB.relative_to(diag.REPO)), "max_abs_mm": d, "pass": bool(d <= 1e-3),
            "median_ref_mm": float(np.median(ref[np.triu_indices(len(ii), 1)]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    src = S.FamosSource(S.Unified())
    split = json.loads(S.FAMOS_SPLIT.read_text())
    persons = list(src.persons)
    if sorted(split["train"]) != persons or set(persons) & set(split["test"]):
        raise SystemExit("FaMoS: persone diverse dallo split TRAIN")
    if src.faces.shape != (3408, 3) or len(src.rv) != 1787:
        raise SystemExit(f"FaMoS: patch inattesa {src.faces.shape}, {len(src.rv)} vertici")
    _C["src"] = src
    RAW_IN.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_gen, persons, chunksize=1)
    rows = sorted((r for block in res for r in block), key=lambda r: r[0])
    t_gen = time.time() - t0
    g = gt_of(src, persons)
    OUT.mkdir(parents=True, exist_ok=True)
    diag.atomic_savez(OUT / "gt_famos.npz", **g)
    diag.atomic_savez(OUT / "gt_names.npz", D_orig=g["D_sr"].astype(np.float32), names=g["names"])
    diag.atomic_savez(OUT / "scale_table.npz", names=np.asarray([r[0] for r in rows]),
                      domain=np.asarray([r[1] for r in rows]), area_mm2=np.asarray([r[2] for r in rows]),
                      area_raw=np.asarray([r[3] for r in rows]), maxabs_raw=np.asarray([r[4] for r in rows]),
                      check=np.full(len(rows), np.nan))
    iu = np.triu_indices(len(persons), 1)
    rep = {"definition": "PROTOCOL_D3.md sez. 3 (b)", "seeds": {"noise": SEED_NOISE}, "id_base": ID_BASE,
           "subdiv": SUBDIV, "n_persons": len(persons), "n_meshes": len(rows), "raw_dir": str(RAW_IN),
           "gt_converged": float(g["converged"].mean()), "S_mean_mm": float(g["S"].mean()),
           "dP_gt_median": float(np.median(g["D_sr"][iu])), "dFR_gt_median_mm": float(np.median(g["D_fr"][iu])),
           "verts_by_label": {lab: [int(min(r[5] for r in rows if r[0].endswith(f"_{lab}.npz"))),
                                    int(max(r[5] for r in rows if r[0].endswith(f"_{lab}.npz")))]
                              for lab in diag.LABELS},
           "K3": k3(g), "seconds_gen": t_gen, "seconds": time.time() - t0}
    diag.atomic_json(OUT / "gen.json", rep)
    print(f"[d3-famos] {rep}", flush=True)
    if not rep["K3"]["pass"]:
        raise SystemExit(f"K3 non passa: {rep['K3']}")


if __name__ == "__main__":
    main()
