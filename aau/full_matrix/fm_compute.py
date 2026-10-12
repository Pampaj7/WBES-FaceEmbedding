#!/usr/bin/env python3
"""Matrice completa: le colonne che mancano (calcolate qui, definizioni importate e non riscritte).

    aau/outlineB/run_o3d.sh aau/full_matrix/fm_compute.py fit-crop --workers 32          (venv open3d)
    aau/run.sh aau/full_matrix/fm_compute.py spectral --workers 32                       (venv del repo)
    aau/outlineB/run_o3d.sh aau/full_matrix/fm_compute.py famos-template --workers 32    (venv open3d)
    aau/outlineB/run_o3d.sh aau/full_matrix/fm_compute.py icp-same --workers 64          (venv open3d)
    (fm.sbatch, passi fit_crop, spectral, famos_tpl, icp_same)

``fit-crop``: il fit PREREGISTRATO (``bp_fit._one``: ``bp.enroll_model`` col seme ``blmm.mesh_seed(s, "crop")`` +
``bp.fit_points``, fallimento con ``bp.FAIL_RMS_MM``) sulle 100 crop di ``bp.CROP_VIEWS``, regione e contesto di
``fit.npz`` (non ricalcolati). Le varianti A e B sulle crop esistono gia' (``fit_e2_crop.npz`` di bp_e2.py). Controllo:
le prime ``N_CHECK`` mesh senza crop di ``fit.npz`` rifatte qui, scarto massimo dei beta. Uscita
``full_matrix/fits/<vista>_<modello>_orig_crop.npz``.

``spectral``: ``comp_spectral.spectral`` (ShapeDNA, HKS, WKS: parametri del protocollo di competitors_hifi3d) sulle
600 mesh di FaceScape dev, FaceVerse con espressioni e neutra (``blmm.mesh_path``, soggetti ``blmm.subjects``) e sulle
15 scansioni di galleria di FaMoS TEST (``bp.meshes("famos")``). HIFI3D esiste (``competitors_hifi3d/desc_spectral.npz``):
controllo sulle sue prime ``N_CHECK`` mesh. Uscita ``full_matrix/spectral/<vista>.npz`` (formato di comp_spectral.py).

``famos-template``: NICP su template in mm sulle 15 scansioni di galleria di FaMoS TEST (blmm lo escludeva: le patch
non hanno una topologia comune). Template = la definizione di ``blmm.build_template`` con i soggetti non valutati
del dominio, cioe' le 80 persone FaMoS TRAIN: neutra di riferimento sulla maschera ``face`` di FLAME 2020
(``sources.FamosSource.view_mesh(expr=False)``, frame canonico FLAME in mm), suddivisa 1-a-4 una volta come
``d3_famos.work_mesh`` (1.787 vertici non bastano per 4.096 punti), media vertice per vertice, centrata, 4.096 vertici
``rng(0)``, nelle coordinate di lavoro del modo ``mm`` (/ L_famos). Iscrizione ``blmm.enroll`` (ICP di similarita' +
NICP, ritorno al template con la rigida) col seme ``blmm.mesh_seed(view_id, "scan_gallery")`` (quello dei fit di bp).
Uscita ``full_matrix/famos_template/mm_template.npz`` (formato di blmm_template.py).

``icp-same``: ICP + Chamfer per coppia (``blmm.pair_metrics``, passo ``fast``, modi ``mm`` e ``cs``) fra soggetti
DIVERSI con la STESSA topologia, che blmm_pairs.py non calcola (solo topologie diverse e stesso soggetto): per ogni
topologia t le 4.950 coppie i < j di ``np.triu_indices`` con X = i in t, Y = j in t e seme = indice della coppia, la
regola di blmm_pairs.py. Controllo: le prime ``N_CHECK_PAIRS`` coppie di original -> remesh rifatte qui contro le
matrici di blmm. Uscita ``full_matrix/icp_same/<vista>_<modo>.npz`` (``values`` (6, 4950) per metrica, mm). Il NICP
per coppia sulle stesse righe (2-4 s a coppia, circa 200 ore-core) non si calcola: non applicabile su quell'insieme.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
OUT = AAU / "runs" / "evidence" / "full_matrix"
N_CHECK = 4
N_CHECK_PAIRS = 40
VIEW6 = ("hifi3d", "facescape", "faceverse", "faceverse_neutral")
SPECTRAL_VIEWS = ("facescape", "faceverse", "faceverse_neutral", "famos")
HIFI_SPECTRAL = AAU / "runs" / "competitors_hifi3d" / "desc_spectral.npz"


# ------------------------------------------------------------------------------------------ fit sulle crop

def fit_crop(workers: int) -> None:
    sys.path.insert(0, str(AAU / "baselines_param"))
    import bp
    import bp_fit
    import blmm
    prm = blmm.params()
    for view in bp.CROP_VIEWS:
        for name in bp.MODELS:
            out = OUT / "fits" / f"{view}_{name}_orig_crop.npz"
            if out.exists():
                print(f"[fm-fit] {out}: gia' presente, salto", flush=True)
                continue
            t0 = time.time()
            with np.load(bp.out_dir(view, name) / "fit.npz") as z:
                reg = np.asarray(z["region_vertices"])
                ref_beta = np.asarray(z["beta"])[:N_CHECK]
                ref_items = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))[:N_CHECK]
            ctx = bp.context(view, name, reg, prm)
            bp_fit._C.update(view=view, prm=prm, ctx=ctx, n_psi=ctx["C_s"].shape[2] - ctx["k_id"])
            items = [(s, "crop", blmm.mesh_path(view, s, "crop")) for s in blmm.subjects(view)]
            items += [(s, t, blmm.mesh_path(view, s, t)) for s, t in ref_items]
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(bp_fit._one, items, chunksize=1)
            n = len(items) - N_CHECK
            check = float(np.nanmax(np.abs(np.stack([r[0]["beta"] for r in res[n:]]) - ref_beta)))
            res, items = res[:n], items[:n]
            failed = [f"{s}|{t}|{r[2]}" for (s, t, _), r in zip(items, res) if r[2]]
            blmm.atomic_savez(
                out, subjects=np.asarray([s for s, _, _ in items]), topologies=np.asarray([t for _, t, _ in items]),
                **{k: np.stack([np.asarray(r[0][k], dtype=np.float64) for r in res]) for k in ("beta", "psi", "theta", "R", "t")},
                rms_mm=np.asarray([r[0]["rms"] for r in res]), nicp_resid_mm=np.asarray([r[0]["nicp"] for r in res]),
                seconds=np.asarray([r[1] for r in res]), failed=np.asarray(failed, dtype="U300"),
                region_vertices=reg, check_beta_vs_fit_npz=check, n_check=N_CHECK)
            print(f"[fm-fit] {view} {name}: {n} crop in {time.time() - t0:.0f}s, RMS mediana "
                  f"{np.nanmedian([r[0]['rms'] for r in res]):.2f} mm, fallite {len(failed)}; controllo |beta - fit.npz| "
                  f"sulle prime {N_CHECK} senza crop {check:.2e}", flush=True)


# ---------------------------------------------------------------------------------------------- spettrali

def _spec(task):
    import comp_spectral as cs
    k, path = task
    return k, cs.spectral(path)


def spectral(workers: int) -> None:
    sys.path.insert(0, str(AAU / "competitors"))
    sys.path.insert(0, str(AAU / "baselines_param"))
    import bp
    import blmm
    from zs_stage import TOPOLOGIES
    jobs = {"hifi3d_check": [(s, t, blmm.mesh_path("hifi3d", s, t)) for s in blmm.subjects("hifi3d")[:1]
                             for t in TOPOLOGIES][:N_CHECK]}
    for view in SPECTRAL_VIEWS:
        if view == "famos":
            jobs[view] = bp.meshes("famos")
        else:
            jobs[view] = [(s, t, blmm.mesh_path(view, s, t)) for s in blmm.subjects(view) for t in TOPOLOGIES]
    for view, items in jobs.items():
        out = OUT / "spectral" / f"{view}.npz"
        if out.exists():
            print(f"[fm-spec] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        with mp.get_context("fork").Pool(workers) as pool:
            res = dict(pool.map(_spec, [((s, t), q) for s, t, q in items], chunksize=2))
        rows = [res[(s, t)] for s, t, _ in items]
        arr = {"subjects": np.asarray([s for s, _, _ in items]), "topologies": np.asarray([t for _, t, _ in items]),
               "paths": np.asarray([blmm.rel(q) for _, _, q in items])}
        for f in ("evals", "hks", "wks"):
            arr[f] = np.stack([r[f] for r in rows])
        for f in ("n_components", "n_dropped", "area_raw", "n_verts", "identity_rel_err"):
            arr[f] = np.asarray([r[f] for r in rows])
        if view == "hifi3d_check":
            with np.load(HIFI_SPECTRAL) as z:
                pos = {(str(s), str(t)): k for k, (s, t) in enumerate(zip(z["subjects"], z["topologies"]))}
                o = [pos[(s, t)] for s, t, _ in items]
                arr["check_rel_max"] = max(float(np.nanmax(np.abs(arr[f] / np.asarray(z[f])[o] - 1)[:, 1:]))
                                           for f in ("evals", "hks", "wks"))
            print(f"[fm-spec] controllo HIFI3D ({len(items)} mesh) contro desc_spectral.npz: scarto relativo max "
                  f"{arr['check_rel_max']:.2e}", flush=True)
        blmm.atomic_savez(out, **arr)
        print(f"[fm-spec] {view}: {len(items)} mesh in {time.time() - t0:.0f}s, componenti > 1 in "
              f"{int((arr['n_components'] > 1).sum())}, identita' max rel {arr['identity_rel_err'].max():.1e}", flush=True)


# ------------------------------------------------------------------------------------- FaMoS, template

_T: dict = {}


def _enroll(task):
    """``blmm.enroll`` (modo mm) e, per la diagnostica, il residuo del NICP dal campione della mesh (mm)."""
    import blmm
    import facebench as fb
    import ir_simicp
    import run_facebench_remesh as rfr
    from scipy.spatial import cKDTree
    key, seed = task
    t0 = time.perf_counter()
    try:
        R = blmm.enroll(_T["X"][key], seed, _T["T"], "mm")
        sec = time.perf_counter() - t0
        Xs = rfr.sample_pts(_T["X"][key], blmm.N_POINTS, seed)
        Tn = np.asarray(fb.nonrigid_icp_align(ir_simicp.similarity_icp(_T["T"], Xs), Xs), dtype=np.float64)
        same = float(np.abs(blmm.rigid_to(Tn, _T["T"]) - R).max())
        return R.astype(np.float32), sec, "", float(cKDTree(Xs).query(Tn)[0].mean()), same
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        return (np.full((blmm.N_POINTS, 3), np.nan, np.float32), time.perf_counter() - t0,
                f"{type(exc).__name__}: {exc}", np.nan, np.nan)


def famos_template(workers: int) -> None:
    import igl
    sys.path.insert(0, str(AAU / "baselines_param"))
    sys.path.insert(0, str(REPO / "v3_work" / "stream"))
    import bp
    import blmm
    import sources as S
    out = OUT / "famos_template" / "mm_template.npz"
    if out.exists():
        print(f"[fm-tpl] {out}: gia' presente, salto", flush=True)
        return
    t0 = time.time()
    src = S.FamosSource(S.Unified())
    split = json.loads(S.FAMOS_SPLIT.read_text())
    items = bp.meshes("famos")
    test_ids = {r["subject"] for r in blmm.famos_manifest()}
    if set(src.persons) & set(split["test"]) or set(src.persons) & test_ids:
        raise SystemExit("FaMoS: persone di TEST fra quelle del template")
    Vs = []
    for p in src.persons:
        V, F, tag = src.view_mesh(p, np.random.default_rng(0), expr=False)
        if tag != "neutral":
            raise SystemExit(f"{p}: vista {tag}, attesa la neutra")
        Vu, _ = igl.upsample(np.asarray(V, np.float64), np.asarray(F, np.int64), 1)
        Vs.append(Vu)
    L = float(blmm.params()["domains"]["famos"]["L"])
    mean = np.mean(np.stack(Vs), axis=0) / L
    mean = mean - mean.mean(0)
    T = mean[np.sort(np.random.default_rng(0).choice(len(mean), blmm.N_POINTS, replace=False))]
    _T.update(T=T, X={(s, t): blmm.work_coords("famos", "mm", q)[0] for s, t, q in items})
    tasks = [((s, t), blmm.mesh_seed(s, t)) for s, t, _ in items]
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        res = pool.map(_enroll, tasks, chunksize=1)
    failed = [f"{k[0]}|{k[1]}|{r[2]}" for (k, _), r in zip(tasks, res) if r[2]]
    resid = np.asarray([r[3] for r in res]) * L
    blmm.atomic_savez(out, R=np.stack([r[0] for r in res]) * np.float32(L), subjects=np.asarray([s for s, _, _ in items]),
                      topologies=np.asarray([t for _, t, _ in items]), seeds=np.asarray([q for _, q in tasks]),
                      seconds=np.asarray([r[1] for r in res]), failed=np.asarray(failed), T=T * L,
                      template_subjects=np.asarray(src.persons), n_template_vertices=len(mean), unit_mm=L, mode="mm",
                      resid_to_mesh_mm=resid, rerun_max_abs=float(np.nanmax([r[4] for r in res])))
    print(f"[fm-tpl] FaMoS: template da {len(src.persons)} persone TRAIN ({len(mean)} vertici), {len(items)} "
          f"iscrizioni in {time.time() - t0:.0f}s, fallite {len(failed)}, residuo mediano {np.nanmedian(resid):.2f} mm",
          flush=True)


# ------------------------------------------------------------------------ ICP fra soggetti, stessa topologia

def icp_same(workers: int) -> None:
    sys.path.insert(0, str(AAU / "baselines_param"))
    import bp  # noqa: F401  (mette blmm su sys.path)
    import blmm
    import blmm_pairs as bpr
    import common
    prm = blmm.params()
    for view in VIEW6:
        subjects = blmm.subjects(view)
        iu, ju = np.triu_indices(len(subjects), 1)
        for mode in ("mm", "cs"):
            out = OUT / "icp_same" / f"{view}_{mode}.npz"
            if out.exists():
                print(f"[fm-icp] {out}: gia' presente, salto", flush=True)
                continue
            t0 = time.time()
            bpr._C.clear()
            for s in subjects:
                for t in blmm.TOPOLOGIES:
                    bpr._C[(s, t)] = blmm.work_coords(view, mode, blmm.mesh_path(view, s, t), prm)
            unit = blmm.unit(view, mode, prm)
            items = [((subjects[i], t), (subjects[j], t), q) for t in blmm.TOPOLOGIES for q, (i, j) in enumerate(zip(iu, ju))]
            items += [((subjects[i], "original"), (subjects[j], "remesh"), q)
                      for q, (i, j) in enumerate(zip(iu[:N_CHECK_PAIRS], ju[:N_CHECK_PAIRS]))]
            with mp.get_context("fork").Pool(workers) as pool:
                vals, sec, errors = bpr.run_items(pool, items, mode, "fast", 64)
            n = len(blmm.TOPOLOGIES) * len(iu)
            arr = {m: (v[:n] * unit).reshape(len(blmm.TOPOLOGIES), len(iu)) for m, v in vals.items()}
            ref, subj, _, _, _ = common.load_matrix(common.matrix_path("rigid_icp_chamfer", "original", "remesh",
                                                                       blmm.view_root(view) / mode))
            if subj != subjects:
                raise SystemExit(f"{view}: soggetti diversi dalle matrici di blmm")
            chk = float(np.nanmax(np.abs(vals["rigid_icp_chamfer"][n:] * unit -
                                         ref[iu[:N_CHECK_PAIRS], ju[:N_CHECK_PAIRS]])))
            blmm.atomic_savez(out, **{f"values_{m}": v for m, v in arr.items()}, topologies=np.asarray(blmm.TOPOLOGIES),
                              subjects=np.asarray(subjects), iu=iu, ju=ju, unit_mm=unit, mode=mode,
                              n_failed=len(errors), check_max_abs_vs_blmm=chk, seconds=sec[:n])
            print(f"[fm-icp] {view} {mode}: {n} coppie in {time.time() - t0:.0f}s ({np.mean(sec):.2f}s/coppia), fallite "
                  f"{len(errors)}; controllo original->remesh ({N_CHECK_PAIRS} coppie) contro blmm {chk:.2e}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("fit-crop", "spectral", "famos-template", "icp-same"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    {"fit-crop": fit_crop, "spectral": spectral, "famos-template": famos_template, "icp-same": icp_same}[a.step](a.workers)


if __name__ == "__main__":
    main()
