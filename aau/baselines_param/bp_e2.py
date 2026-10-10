#!/usr/bin/env python3
"""Emendamento 2 (POST HOC): fit della variante B sul crop, sugli held-out sintetici (k della composizione), pilota
della sensibilita' a sigma, tempi per passo, B con la sigma della sensibilita' (PROTOCOL_emendamento_2.md).

    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py crop --models gnm --workers 48
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py heldout --in-dir <stage>/in --table <stage>/scale_table.npz
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py pilot --models flame2023 --workers 48
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py time --models gnm --workers 48
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_e2.py sens --models gnm --workers 48
    (bp.sbatch, passi e2_*; definizioni: bp.py e bp_fit_e1.py, i delta in bp_paired_e2.py)

Ogni fit e' quello dell'emendamento 1: coordinate di lavoro e NICP col seme ``blmm.mesh_seed`` (``bp.enroll_model``),
variante A (``bp.fit_registered``), variante B (``bp.fit_loop``), fallimenti con ``bp_fit_e1.failure``; tempi per
passo (lettura, NICP, A, B) con ``time.perf_counter`` nel worker (un processo per mesh, un thread BLAS).

``crop`` (sez. 1): le crop dei 100 soggetti valutati delle ``bp.CROP_VIEWS``, regione di ``fit.npz``, B congelata
``bp.LOOP``, piu' l'errore di superficie di B; uscita ``<vista>/<modello>/fit_e2_crop.npz``.
``heldout`` (sez. 2): le 1.500 mesh held-out messe in scena da ``fact_calib.py stage``; per (dominio, modello) la
regione con riferimenti = le 100 original held-out del dominio (``bp.region``), A e B congelate; d_P di B sulle coppie
di ``fact_calib.heldout_pairs`` (regola copiata) contro la GT-SR di training; k = mediana(d_P GT) / mediana(d_P B).
Uscite ``calib_e2/<modello>.npz``, ``calib_e2/<modello>.json`` e ``calib_e2/k.json``.
``pilot`` (sez. 4): ``bp_fit_e1.run_pilot`` con ``bp.SENS_GRID`` -> ``pilot_e2/<modello>.json``, poi la regola di
scelta dell'emendamento 1 sull'unione con ``pilot_e1`` -> ``pilot_e2/selection.json`` (con il controllo dei punteggi
del fit originale e di A).
``time`` (sez. 3): i primi ``TIME_SUBJECTS`` soggetti valutati x 5 topologie (FaMoS le 15 scansioni), B congelata;
uscita ``<vista>/<modello>/time_e2.npz`` con lo scarto dei beta da ``fit_e1.npz``.
``sens`` (sez. 4): solo se la scelta del pilota ha sigma diversa da ``bp.LOOP``; B con quella configurazione su tutte
le mesh di ``bp.meshes``; uscita ``<vista>/<modello>/fit_e2_sens.npz`` con le matrici ``D_vbs_{coef,fr,sr}``.
Ripartibile: le uscite gia' su disco si saltano.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

import bp
import bp_fit_e1
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)

_C: dict = {}           # contesto della (vista, modello), ereditato dai worker (fork)
TIME_SUBJECTS = 20
STAGES = ("load", "nicp", "va", "vb")                # colonne di ``t_stage`` (s)
EV = bp.REPO_ROOT / "aau" / "runs" / "evidence" / "trainer_v3"
GT_SR = EV / "ablations" / "c3f" / "gt_sr.npz"       # fact_calib.GT_SR
BL_PARAMS = EV / "factorized" / "calib_heldout_bl" / "params.json"   # L_d, CS_ref,d delle original held-out
HELDOUT_DOMS = ("bfm", "ict", "gnm")                 # fact_calib.DOMS
CALIB_ROOT = bp.OUT_ROOT / "calib_e2"
PILOT_ROOT = bp.OUT_ROOT / "pilot_e2"


# ---------------------------------------------------------------------------------------------- fit

def _fit(task):
    """NICP, A, B di una mesh con i tempi per passo; per B anche l'errore di superficie se ``_C["surf"]``."""
    i, subject, topology, path = task
    ctx, view, loop, free = _C["ctx"], _C["view"], _C["loop"], _C["free"]
    k = bp.n_coef(ctx, free)
    nan = {"beta": np.full(ctx["k_id"], np.nan), "psi": np.full(k - ctx["k_id"], np.nan), "theta": np.nan,
           "R": np.full((3, 3), np.nan), "t": np.full(3, np.nan), "rms": np.nan}
    out = {"va": dict(nan), "vb": {**nan, "kept": np.full(2, np.nan)}, "fail_va": "", "fail_vb": "", "nicp": np.nan,
           "t": np.full(len(STAGES), np.nan), "surf": np.full(5, np.nan)}
    try:                                             # bp_fit_e1.inputs, con i tempi di lettura e NICP separati
        import run_facebench_remesh as rfr
        t0 = time.perf_counter()
        X, _ = blmm.work_coords(view, "mm", path, _C["prm"])
        _, Fin = blmm.load_raw(path)
        t1 = time.perf_counter()
        seed = blmm.mesh_seed(subject, topology)
        Y, out["nicp"] = bp.enroll_model(X, seed, ctx)
        Vin, Xs = X * ctx["L"], rfr.sample_pts(X, bp.N_POINTS, seed) * ctx["L"]
        out["t"][:2] = (t1 - t0, time.perf_counter() - t1)
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN in entrambe le varianti, contata)
        out["fail_va"] = out["fail_vb"] = f"{type(exc).__name__}: {exc}"
        return out
    try:
        t2 = time.perf_counter()
        fa = bp.fit_registered(Y, ctx, free)
        out["t"][2] = time.perf_counter() - t2
        out["fail_va"] = bp_fit_e1.failure(fa)
        if not out["fail_va"]:
            out["va"] = fa
    except Exception as exc:  # noqa: BLE001
        out["fail_va"] = f"{type(exc).__name__}: {exc}"
    if out["fail_va"]:
        out["fail_vb"] = "variante A fallita"
        return out
    try:
        t3 = time.perf_counter()
        fb = bp.fit_loop(Vin, Fin, Xs, fa, ctx, free, loop["sigma"], loop["tau"], loop["iters"])
        out["t"][3] = time.perf_counter() - t3
        out["fail_vb"] = bp_fit_e1.failure(fb, loop=True)
        out["vb"]["kept"] = fb["kept"]
        if not out["fail_vb"]:
            out["vb"] = fb
            if _C["surf"]:
                out["surf"] = bp_fit_e1.surf_of(fb, Vin, Fin, Xs, ctx, None)
    except Exception as exc:  # noqa: BLE001
        out["fail_vb"] = f"{type(exc).__name__}: {exc}"
    return out


def fit_items(view: str, name: str, items: list, ctx: dict, prm: dict, loop: dict, free: bool, workers: int,
              surf: bool = False) -> dict:
    """Fit delle mesh ``items`` (soggetto, topologia, percorso) -> array come ``fit_e1.npz`` (+ tempi)."""
    _C.update(view=view, prm=prm, ctx=ctx, loop=loop, free=free, surf=surf)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_fit, [(i, *it) for i, it in enumerate(items)], chunksize=1)
    arr = {"subjects": np.asarray([s for s, _, _ in items]), "topologies": np.asarray([t for _, t, _ in items])}
    for v in bp.VARIANTS:
        for key in ("beta", "psi", "theta", "R", "t", "rms"):
            arr[f"{v}_{key}"] = np.stack([np.asarray(r[v][key], dtype=np.float64) for r in res])
        arr[f"failed_{v}"] = np.asarray([f"{s}|{t}|{r['fail_' + v]}" for (s, t, _), r in zip(items, res)
                                         if r["fail_" + v]], dtype="U300")
    arr.update(vb_kept=np.stack([np.asarray(r["vb"]["kept"], dtype=np.float64) for r in res]),
               nicp_resid_mm=np.asarray([r["nicp"] for r in res]), t_stage=np.stack([r["t"] for r in res]),
               stages=np.asarray(STAGES), loop=json.dumps(loop), loop_inner=bp.LOOP_INNER, free_expr=free,
               expr_scale=ctx["expr_scale"], model_sha256=bp.sha256(ctx["file"]), workers=workers)
    if surf:
        arr["surf_vb"] = np.stack([r["surf"] for r in res])
    return arr


def fitted_context(view: str, name: str, prm: dict) -> dict:
    with np.load(bp.out_dir(view, name) / "fit.npz") as z:
        return bp.context(view, name, np.asarray(z["region_vertices"]), prm)


def summary(tag: str, view: str, name: str, arr: dict, t0: float) -> None:
    ts = arr["t_stage"]
    print(f"[bp-e2] {tag} {view} {name}: {len(ts)} mesh in {time.time() - t0:.0f}s; fallite A {len(arr['failed_va'])}, "
          f"B {len(arr['failed_vb'])} {list(arr['failed_vb'][:3])}; s/mesh mediana NICP {np.nanmedian(ts[:, 1]):.1f}, "
          f"A {np.nanmedian(ts[:, 2]):.1f}, B {np.nanmedian(ts[:, 3]):.1f}", flush=True)


# --------------------------------------------------------------------------------------- crop, tempi

def run_crop(name: str, views: list, workers: int) -> None:
    prm = blmm.params()
    for view in views:
        out = bp.out_dir(view, name) / "fit_e2_crop.npz"
        if out.exists():
            print(f"[bp-e2] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        items = [(s, "crop", blmm.mesh_path(view, s, "crop")) for s in blmm.subjects(view)]
        arr = fit_items(view, name, items, fitted_context(view, name, prm), prm, bp.LOOP[name],
                        view not in bp.NEUTRAL_VIEWS, workers, surf=True)
        blmm.atomic_savez(out, **arr)
        summary("crop", view, name, arr, t0)


def run_time(name: str, views: list, workers: int, serial: int = 0) -> None:
    """``serial`` > 0: solo le prime ``serial`` mesh del campione, uscita ``time_e2_serial.npz`` (con ``--workers 1``:
    tempi senza altri processi nostri sul nodo)."""
    prm = blmm.params()
    for view in views:
        out = bp.out_dir(view, name) / ("time_e2_serial.npz" if serial > 0 else "time_e2.npz")
        if out.exists():
            print(f"[bp-e2] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        all_items = bp.meshes(view)
        keep = set(blmm.subjects(view)[:TIME_SUBJECTS]) if view != "famos" else None
        pos = [k for k, (s, _, _) in enumerate(all_items) if keep is None or s in keep]
        pos = pos[:serial] if serial > 0 else pos
        items = [all_items[k] for k in pos]
        arr = fit_items(view, name, items, fitted_context(view, name, prm), prm, bp.LOOP[name],
                        view not in bp.NEUTRAL_VIEWS, workers)
        with np.load(bp.out_dir(view, name) / "fit_e1.npz") as z:
            ref = np.asarray(z["vb_beta"])[pos]
            arr["seconds_e1"] = np.asarray(z["seconds"])
        arr["fit_e1_index"] = np.asarray(pos)
        arr["beta_vs_fit_e1"] = float(np.nanmax(np.abs(arr["vb_beta"] - ref)))
        blmm.atomic_savez(out, **arr)
        summary("time", view, name, arr, t0)
        print(f"[bp-e2] time {view} {name}: max |beta B - fit_e1| = {arr['beta_vs_fit_e1']:.2e}", flush=True)


# ------------------------------------------------------------------------------------------- held-out

def heldout_pairs(names: list, dom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``fact_calib.heldout_pairs`` (copiata): i < j sui nomi ordinati, stesso dominio, soggetti ed etichette
    diversi."""
    sid = np.asarray([n[:-4].split("_GTready_", 1)[0] for n in names])
    lab = np.asarray([n[:-4].split("_GTready_", 1)[1] for n in names])
    i, j = np.triu_indices(len(names), 1)
    keep = (dom[i] == dom[j]) & (sid[i] != sid[j]) & (lab[i] != lab[j])
    return i[keep], j[keep]


def run_heldout(in_dir: Path, table: Path, models: list, workers: int) -> None:
    with np.load(table) as z:
        dmap = dict(zip([str(x) for x in z["names"]], [str(x) for x in z["domain"]]))
    names = sorted(dmap)
    if sorted(p.name for p in in_dir.glob("*.npz")) != names:
        raise SystemExit(f"{in_dir}: mesh diverse dalla tabella {table}")
    dom = np.asarray([dmap[n] for n in names])
    bl = json.loads(BL_PARAMS.read_text())["domains"]
    prm = {"domains": {d: {"L": float(bl[d]["L"]), "cs_ref": float(bl[d]["cs_ref"])} for d in HELDOUT_DOMS}}
    for d in HELDOUT_DOMS:                     # viste in memoria: blmm.work_coords / to_mm col frame del dominio
        blmm.VIEWS[f"heldout_{d}"] = {"dir": in_dir, "domain": d, "gt": None, "template_dir": None, "fb_root": None,
                                      "template_ref": None}
    with np.load(GT_SR, allow_pickle=True) as z:
        G = np.asarray(z["D_orig"], np.float32)
        gpos = {str(n): k for k, n in enumerate(z["names"])}
    dpu_gt = float(json.loads(GT_SR.with_suffix(".json").read_text())["dP_per_unit"])
    i, j = heldout_pairs(names, dom)
    sid = [n[:-4].split("_GTready_", 1)[0] for n in names]
    g = np.asarray([gpos[s] for s in sid])
    t_gt = G[g[i], g[j]].astype(np.float64) * dpu_gt
    for name in models:
        out = CALIB_ROOT / f"{name}.npz"
        if out.exists():
            print(f"[bp-e2] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        m = bp.load_model(name)
        S, betas, fails, regions, tstage, diag = np.full(len(names), np.nan), None, [], {}, None, {}
        dP_pairs = np.full(len(i), np.nan)
        for d in HELDOUT_DOMS:
            view = f"heldout_{d}"
            idx = np.flatnonzero(dom == d)
            refs = []
            for k in idx:
                if names[k].endswith("_GTready_original.npz"):
                    V, F = blmm.load_raw(in_dir / names[k])
                    refs.append((names[k], blmm.to_mm(view, V), F))
            reg = bp.region(view, m, refs)
            regions[d] = reg["vertices"]
            diag[d] = {"n_refs": reg["n_refs"], "n_vertices": int(len(reg["vertices"])),
                       "kept_median": float(np.median([x["kept"] for x in reg["diag"]])),
                       "median_mm_median": float(np.median([x["median_mm"] for x in reg["diag"]]))}
            ctx = bp.context(view, name, reg["vertices"], prm)
            items = [(sid[k], names[k][:-4].split("_GTready_", 1)[1], in_dir / names[k]) for k in idx]
            arr = fit_items(view, name, items, ctx, prm, bp.LOOP[name], False, workers)
            summary("heldout", view, name, arr, t0)
            B = arr["vb_beta"]
            if betas is None:
                betas = np.full((len(names), B.shape[1]), np.nan)
                tstage = np.full((len(names), len(STAGES)), np.nan)
            betas[idx], tstage[idx] = B, arr["t_stage"]
            S[idx] = bp.identity_sizes(B, ctx)
            fails += [f"{d}|{x}" for x in arr["failed_vb"]]
            loc = -np.ones(len(names), dtype=np.int64)
            loc[idx] = np.arange(len(idx))
            Dsr = bp.mesh_distances(B, ctx)["sr"] / bp.identity_sizes(np.zeros((1, ctx["k_id"])), ctx)[0]
            q = dom[i] == d
            dP_pairs[q] = Dsr[loc[i[q]], loc[j[q]]]
        ok = np.isfinite(dP_pairs)
        t, mB, dd = t_gt[ok], dP_pairs[ok], dom[i][ok]
        res = {"model": name, "n_meshes": len(names), "n_failed_vb": len(fails), "failed_vb": fails[:20],
               "n_pairs": int(ok.sum()), "n_pairs_dropped": int((~ok).sum()), "median_dP_gt": float(np.median(t)),
               "median_dP_B": float(np.median(mB)), "k_median": float(np.median(t) / np.median(mB)),
               "k_ls": float((t * mB).sum() / (mB * mB).sum()), "loop": bp.LOOP[name], "regions": diag,
               "definition": "PROTOCOL_emendamento_2.md sez. 2: k = mediana(d_P GT-SR x dP_per_unit) / mediana(SR di "
                             "B / S(mu)) sulle coppie di fact_calib.heldout_pairs", "wall_s": time.time() - t0}
        for d in HELDOUT_DOMS:
            q = dd == d
            res[f"k_median_{d}"] = float(np.median(t[q]) / np.median(mB[q])) if q.any() else float("nan")
        blmm.atomic_savez(out, names=np.asarray(names), domain=dom, beta=betas, S=S, t_stage=tstage,
                          stages=np.asarray(STAGES), pair_i=i, pair_j=j, dP_B=dP_pairs, dP_gt=t_gt,
                          **{f"region_{d}": regions[d] for d in HELDOUT_DOMS})
        bp.atomic_json(CALIB_ROOT / f"{name}.json", res)
        print(f"[bp-e2] heldout {name}: k {res['k_median']:.4f} (LS {res['k_ls']:.4f}; " +
              " ".join(f"{d} {res[f'k_median_{d}']:.3f}" for d in HELDOUT_DOMS) +
              f"), {res['n_pairs']} coppie, B fallite {len(fails)}", flush=True)
    ks = {}
    for name in bp.MODELS:
        p = CALIB_ROOT / f"{name}.json"
        if p.exists():
            r = json.loads(p.read_text())
            ks[name] = {key: r[key] for key in ("k_median", "k_ls", "n_pairs", "n_failed_vb", "n_meshes")}
    bp.atomic_json(CALIB_ROOT / "k.json", ks)


# --------------------------------------------------------------------------------------- pilota, sens

def run_pilot(name: str, workers: int) -> None:
    if not (PILOT_ROOT / f"{name}.json").exists():
        bp_fit_e1.run_pilot(name, workers, bp.SENS_GRID, PILOT_ROOT)
    selection()


def selection() -> dict:
    """Regola dell'emendamento 1 sull'unione dei punteggi di ``pilot_e1`` e ``pilot_e2``, per modello."""
    out = {}
    for name in bp.MODELS:
        p1, p2 = bp.OUT_ROOT / "pilot_e1" / f"{name}.json", PILOT_ROOT / f"{name}.json"
        if not (p1.exists() and p2.exists()):
            continue
        a, b = json.loads(p1.read_text()), json.loads(p2.read_text())
        union = {k: v for k, v in {**a["score"], **b["score"]}.items() if k.startswith("vb")}
        key = bp_fit_e1.select(union)
        _, s, tau, it = key.split("|")
        best = min(v for v in union.values() if np.isfinite(v))
        out[name] = {"key": key, "sigma": float(s), "tau": float(tau), "iters": int(it), "score": union[key],
                     "best": best, "tied": sorted(k for k, v in union.items() if v <= 1.01 * best),
                     "frozen": bp.LOOP[name], "differs": float(s) != bp.LOOP[name]["sigma"],
                     "control_orig": abs(a["score"]["orig"] - b["score"]["orig"]),
                     "control_va": abs(a["score"]["va"] - b["score"]["va"]),
                     "score_e2": {k: v for k, v in b["score"].items() if k.startswith("vb")},
                     "failed_e2": {k: len(v) for k, v in b["failed"].items()}}
        print(f"[bp-e2] pilota {name}: scelta sull'unione {key} (S {union[key]:.4f}; congelata {bp.LOOP[name]}); "
              f"controllo |S orig| {out[name]['control_orig']:.1e}, |S A| {out[name]['control_va']:.1e}", flush=True)
    if out:
        bp.atomic_json(PILOT_ROOT / "selection.json", out)
    return out


def run_sens(name: str, views: list, workers: int) -> None:
    sel = json.loads((PILOT_ROOT / "selection.json").read_text())[name]
    if not sel["differs"]:
        print(f"[bp-e2] sens {name}: la scelta del pilota ha sigma {sel['sigma']} = bp.LOOP, niente da fare", flush=True)
        return
    loop = {"sigma": sel["sigma"], "tau": sel["tau"], "iters": sel["iters"]}
    prm = blmm.params()
    for view in views:
        out = bp.out_dir(view, name) / "fit_e2_sens.npz"
        if out.exists():
            print(f"[bp-e2] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        items = bp.meshes(view)
        with np.load(bp.out_dir(view, name) / "fit.npz") as z:
            if [str(s) for s in z["subjects"]] != [s for s, _, _ in items] or \
                    [str(t) for t in z["topologies"]] != [t for _, t, _ in items]:
                raise SystemExit(f"{view} {name}: mesh in ordine diverso da fit.npz")
        ctx = fitted_context(view, name, prm)
        arr = fit_items(view, name, items, ctx, prm, loop, view not in bp.NEUTRAL_VIEWS, workers)
        arr.update({f"D_vbs_{k}": M for k, M in bp.mesh_distances(arr["vb_beta"], ctx).items()})
        blmm.atomic_savez(out, **arr)
        summary("sens", view, name, arr, t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("crop", "heldout", "pilot", "time", "sens"))
    p.add_argument("--views", default=None)
    p.add_argument("--models", default=",".join(bp.MODELS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--in-dir", type=Path)
    p.add_argument("--table", type=Path)
    p.add_argument("--serial", type=int, default=0, help="time: solo le prime N mesh, uscita time_e2_serial.npz")
    a = p.parse_args()
    models = a.models.split(",")
    if a.step == "heldout":
        run_heldout(a.in_dir, a.table, models, a.workers)
        return
    for name in models:
        if a.step == "crop":
            run_crop(name, (a.views or ",".join(bp.CROP_VIEWS)).split(","), a.workers)
        elif a.step == "time":
            run_time(name, (a.views or ",".join(bp.VIEWS)).split(","), a.workers, a.serial)
        elif a.step == "pilot":
            run_pilot(name, a.workers)
        else:
            run_sens(name, (a.views or ",".join(bp.VIEWS)).split(","), a.workers)


if __name__ == "__main__":
    main()
