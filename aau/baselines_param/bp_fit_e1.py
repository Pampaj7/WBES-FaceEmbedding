#!/usr/bin/env python3
"""Emendamento 1 (POST HOC): varianti A e B del fit dei 3DMM ed errore di superficie (PROTOCOL_emendamento_1.md).

    aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit_e1.py --pilot --models flame2023 --workers 48
    aau/outlineB/run_o3d.sh aau/baselines_param/bp_fit_e1.py --views hifi3d,famos --models gnm --workers 48
    (bp.sbatch, passi pilot_e1 e fit_e1; definizioni: bp.py, sezione "emendamento 1")

``--pilot``: i primi ``bp.PILOT_SUBJECTS`` soggetti NON valutati di ``blmm.template_subjects`` x 5 topologie sulle
viste ``bp.PILOT_VIEWS`` (regione: quella di ``fit.npz`` della vista); per mesh il fit originale, la variante A e la
variante B per ogni (sigma, tau) di ``bp.LOOP_GRID`` con lo stato a ogni ``iters`` della griglia. Punteggio senza GT
(sez. 2): S = media delle distanze ``fr`` stesso soggetto / soggetti diversi, media sulle viste; scelta con la regola
di parita' dell'emendamento. Uscite ``<OUT_ROOT>/pilot_e1/<modello>.json`` (punteggi, scelta, errori di superficie,
tempi) e ``<modello>.npz`` (beta).

Senza ``--pilot``: per (vista, modello) le mesh di ``bp.meshes`` nell'ordine di ``fit.npz`` (controllato), NICP
ricalcolato (``nicp_check`` = scarto dal residuo salvato), errore di superficie del fit originale (parametri di
``fit.npz``), variante A (``bp.fit_registered``) e B (``bp.fit_loop`` coi valori congelati ``bp.LOOP``), coi loro
errori di superficie. Uscita ``fit_e1.npz`` accanto a ``fit.npz``: ``{va,vb}_{beta,psi,theta,R,t,rms}``, ``vb_kept``,
``surf`` (n, 3, 5: originale / A / B x ``bp.surface_error``), ``failed_va``, ``failed_vb``, ``nicp_check``,
``seconds`` e le matrici ``D_{va,vb}_{coef,fr,sr}``. Ripartibile: una (vista, modello) gia' su disco si salta.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import bp
import blmm  # noqa: E402  (aau/baselines_mm, messo su sys.path da bp)

_C: dict = {}           # contesti e dati, ereditati dai worker (fork)
TOPOLOGIES = bp.TOPOLOGIES


def inputs(view: str, subject: str, topology: str, path, ctx: dict):
    """(Vin, Fin, Xs, Y, residuo NICP): mesh di ingresso e 4096 punti del NICP in mm nel frame di lavoro, punti
    registrati del template."""
    import run_facebench_remesh as rfr
    X, _ = blmm.work_coords(view, "mm", path, _C["prm"])
    _, Fin = blmm.load_raw(path)
    seed = blmm.mesh_seed(subject, topology)
    Y, nd = bp.enroll_model(X, seed, ctx)
    return X * ctx["L"], Fin, rfr.sample_pts(X, bp.N_POINTS, seed) * ctx["L"], Y, nd


def failure(f: dict, loop: bool = False) -> str:
    """Motivo di fallimento (sez. 6 del protocollo; per B anche le corrispondenze tenute), "" se il fit e' valido."""
    if not all(np.isfinite(np.asarray(f[k], dtype=np.float64)).all() for k in ("beta", "psi", "theta", "R", "t")):
        return "valori non finiti"
    if not f["rms"] <= bp.FAIL_RMS_MM:
        return f"RMS {f['rms']:.2f} mm"
    if loop and not (f["kept"] >= bp.LOOP_MIN_KEPT).all():
        return f"corrispondenze tenute {f['kept'].round(3).tolist()}"
    return ""


def surf_of(f: dict, Vin, Fin, Xs, ctx, surf_in) -> np.ndarray:
    c = np.r_[f["beta"], f["psi"]]
    return bp.surface_error(Vin, Fin, Xs, bp.posed_model(c, f["theta"], f["R"], f["t"], ctx), ctx, surf_in)


# ------------------------------------------------------------------------------------------- pilota

def _pilot(task):
    view, subject, topology, path = task
    ctx = _C["ctx"][view]
    free = view not in bp.NEUTRAL_VIEWS
    t0 = time.perf_counter()
    out = {"fail": {}, "beta": {}, "surf": {}, "sec": {}}
    try:
        Vin, Fin, Xs, Y, _ = inputs(view, subject, topology, path, ctx)
        surf_in = bp.Surface(Vin, Fin)
        fits = {"orig": bp.fit_points(Y, ctx), "va": bp.fit_registered(Y, ctx, free)}
        out["sec"]["va"] = time.perf_counter() - t0
        for s in bp.LOOP_GRID["sigma"]:
            for tau in bp.LOOP_GRID["tau"]:
                t1 = time.perf_counter()
                try:
                    fb = bp.fit_loop(Vin, Fin, Xs, fits["va"], ctx, free, s, tau, max(bp.LOOP_GRID["iters"]),
                                     record=bp.LOOP_GRID["iters"])
                    for it in bp.LOOP_GRID["iters"]:
                        fits[f"vb|{s}|{tau}|{it}"] = fb["states"][it]
                except Exception as exc:  # noqa: BLE001
                    for it in bp.LOOP_GRID["iters"]:
                        out["fail"][f"vb|{s}|{tau}|{it}"] = f"{type(exc).__name__}: {exc}"
                out["sec"][f"vb|{s}|{tau}"] = time.perf_counter() - t1
        for key, f in fits.items():
            why = failure(f, key.startswith("vb"))
            if why:
                out["fail"][key] = why
                continue
            out["beta"][key] = np.asarray(f["beta"])
            out["surf"][key] = surf_of(f, Vin, Fin, Xs, ctx, surf_in)
    except Exception as exc:  # noqa: BLE001
        out["fail"]["mesh"] = f"{type(exc).__name__}: {exc}"
    out["sec"]["total"] = time.perf_counter() - t0
    return out


def separation(D: np.ndarray, subjects: np.ndarray) -> float:
    """S = media delle distanze stesso soggetto (topologie diverse) / media fra soggetti diversi (coppie i < j)."""
    iu = np.triu_indices(len(subjects), 1)
    same = subjects[iu[0]] == subjects[iu[1]]
    d = D[iu]
    return float(d[same].mean() / d[~same].mean())


def select(scores: dict) -> str:
    """Regola della sez. 2: minimo del punteggio; a parita' (entro l'1%) iters minore, sigma maggiore, tau maggiore."""
    ok = {k: v for k, v in scores.items() if np.isfinite(v)}
    best = min(ok.values())
    tied = [k for k, v in ok.items() if v <= 1.01 * best]

    def key(k):
        _, s, tau, it = k.split("|")
        return (int(it), -float(s), -float(tau))

    return sorted(tied, key=key)[0]


def run_pilot(name: str, workers: int) -> None:
    prm = blmm.params()
    _C.update(prm=prm, ctx={})
    tasks, subj = [], {}
    for view in bp.PILOT_VIEWS:
        with np.load(bp.out_dir(view, name) / "fit.npz") as z:
            reg = np.asarray(z["region_vertices"])
        _C["ctx"][view] = bp.context(view, name, reg, prm)
        ss = blmm.template_subjects(view)[:bp.PILOT_SUBJECTS]
        evaluated = set(blmm.subjects(view))
        if evaluated & set(ss):
            raise SystemExit(f"{view}: soggetti del pilota fra i valutati")
        subj[view] = ss
        tasks += [(view, s, t, blmm.mesh_path(view, s, t)) for s in ss for t in TOPOLOGIES]
    t0 = time.time()
    print(f"[bp-e1-pilot] {name}: {len(tasks)} mesh, viste {bp.PILOT_VIEWS}", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_pilot, tasks, chunksize=1)
    keys = ["orig", "va"] + [f"vb|{s}|{tau}|{it}" for s in bp.LOOP_GRID["sigma"] for tau in bp.LOOP_GRID["tau"]
                             for it in bp.LOOP_GRID["iters"]]
    per_view, surf, fails, betas = {}, {}, {}, {}
    for view in bp.PILOT_VIEWS:
        idx = [i for i, t in enumerate(tasks) if t[0] == view]
        subjects = np.asarray([tasks[i][1] for i in idx])
        per_view[view] = {}
        for key in keys:
            bad = [f"{tasks[i][1]}|{tasks[i][2]}: {res[i]['fail'].get(key) or res[i]['fail'].get('mesh')}"
                   for i in idx if key not in res[i]["beta"]]
            if bad:
                fails.setdefault(key, []).extend(f"{view}|{b}" for b in bad)
                per_view[view][key] = float("nan")
                continue
            B = np.stack([res[i]["beta"][key] for i in idx])
            betas[f"{view}|{key}"] = B
            per_view[view][key] = separation(bp.mesh_distances(B, _C["ctx"][view])["fr"], subjects)
            sv = np.stack([res[i]["surf"][key] for i in idx])
            surf.setdefault(key, {})[view] = {"median_mm": float(np.median(sv[:, 0])),
                                              "p95_mm": float(np.median(sv[:, 1])), "p95_max_mm": float(sv[:, 1].max()),
                                              "kept_in_to_model": float(np.median(sv[:, 4]))}
    scores = {k: float(np.mean([per_view[v][k] for v in bp.PILOT_VIEWS])) for k in keys}
    choice = select({k: v for k, v in scores.items() if k.startswith("vb")})
    _, s, tau, it = choice.split("|")
    sec = {k: float(np.median([r["sec"][k] for r in res if k in r["sec"]])) for k in res[0]["sec"]}
    out = {"model": name, "views": list(bp.PILOT_VIEWS), "subjects": subj, "topologies": list(TOPOLOGIES),
           "grid": bp.LOOP_GRID, "score": scores, "separation_by_view": per_view, "surface": surf,
           "failed": fails, "choice": {"key": choice, "sigma": float(s), "tau": float(tau), "iters": int(it)},
           "seconds_median": sec, "wall_s": time.time() - t0}
    root = bp.OUT_ROOT / "pilot_e1"
    bp.atomic_json(root / f"{name}.json", out)
    blmm.atomic_savez(root / f"{name}.npz", **{k.replace("|", "__"): v for k, v in betas.items()})
    print(f"[bp-e1-pilot] {name}: scelta {out['choice']} (S {scores[choice]:.4f}; originale {scores['orig']:.4f}, "
          f"A {scores['va']:.4f}), {time.time() - t0:.0f}s, fallimenti {sum(len(v) for v in fails.values())}",
          flush=True)


# ------------------------------------------------------------------------------------- viste valutate

def _one(task):
    i, subject, topology, path = task
    ctx, view = _C["ctx"], _C["view"]
    free = view not in bp.NEUTRAL_VIEWS
    cfg = bp.LOOP[ctx["model"]]
    t0 = time.perf_counter()
    k = bp.n_coef(ctx, free)
    nan = {"beta": np.full(ctx["k_id"], np.nan), "psi": np.full(k - ctx["k_id"], np.nan), "theta": np.nan,
           "R": np.full((3, 3), np.nan), "t": np.full(3, np.nan), "rms": np.nan}
    out = {"va": dict(nan), "vb": {**nan, "kept": np.full(2, np.nan)}, "surf": np.full((3, 5), np.nan),
           "fail_va": "", "fail_vb": "", "nicp": np.nan}
    try:
        Vin, Fin, Xs, Y, nd = inputs(view, subject, topology, path, ctx)
        out["nicp"] = nd
        surf_in = bp.Surface(Vin, Fin)
        o = {key: _C["orig"][key][i] for key in ("beta", "psi", "theta", "R", "t")}
        out["surf"][0] = surf_of(o, Vin, Fin, Xs, ctx, surf_in)
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN in entrambe le varianti, contata)
        out["fail_va"] = out["fail_vb"] = f"{type(exc).__name__}: {exc}"
        return out, time.perf_counter() - t0
    try:
        fa = bp.fit_registered(Y, ctx, free)
        out["fail_va"] = failure(fa)
        if not out["fail_va"]:
            out["va"] = fa
            out["surf"][1] = surf_of(fa, Vin, Fin, Xs, ctx, surf_in)
    except Exception as exc:  # noqa: BLE001
        out["fail_va"] = f"{type(exc).__name__}: {exc}"
    if out["fail_va"]:
        out["fail_vb"] = "variante A fallita"
        return out, time.perf_counter() - t0
    try:
        fb = bp.fit_loop(Vin, Fin, Xs, fa, ctx, free, cfg["sigma"], cfg["tau"], cfg["iters"])
        out["fail_vb"] = failure(fb, loop=True)
        out["vb"]["kept"] = fb["kept"]
        if not out["fail_vb"]:
            out["vb"] = fb
            out["surf"][2] = surf_of(fb, Vin, Fin, Xs, ctx, surf_in)
    except Exception as exc:  # noqa: BLE001
        out["fail_vb"] = f"{type(exc).__name__}: {exc}"
    return out, time.perf_counter() - t0


def run_view(view: str, name: str, workers: int, overwrite: bool) -> None:
    root = bp.out_dir(view, name)
    out = root / "fit_e1.npz"
    if out.exists() and not overwrite:
        print(f"[bp-e1] {out}: gia' presente, salto", flush=True)
        return
    if bp.LOOP[name] is None:
        raise SystemExit(f"bp.LOOP[{name!r}] non congelato: prima il pilota (sez. 8 dell'emendamento)")
    t0 = time.time()
    items = bp.meshes(view)
    prm = blmm.params()
    with np.load(root / "fit.npz") as z:
        orig = {k: np.asarray(z[k]) for k in ("beta", "psi", "theta", "R", "t", "nicp_resid_mm", "region_vertices")}
        if [str(s) for s in z["subjects"]] != [s for s, _, _ in items] or \
                [str(t) for t in z["topologies"]] != [t for _, t, _ in items]:
            raise SystemExit(f"{view} {name}: mesh in ordine diverso da fit.npz")
    ctx = bp.context(view, name, orig["region_vertices"], prm)
    _C.update(view=view, prm=prm, ctx=ctx, orig=orig)
    print(f"[bp-e1] {view} {name}: {len(items)} mesh, espressione {'libera' if view not in bp.NEUTRAL_VIEWS else 'no'}"
          f", B {bp.LOOP[name]}", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_one, [(i, *it) for i, it in enumerate(items)], chunksize=1)
    arr = {}
    for v in bp.VARIANTS:
        for k in ("beta", "psi", "theta", "R", "t", "rms"):
            arr[f"{v}_{k}"] = np.stack([np.asarray(r[0][v][k], dtype=np.float64) for r in res])
        D = bp.mesh_distances(arr[f"{v}_beta"], ctx)
        arr.update({f"D_{v}_{k}": M for k, M in D.items()})
    failed = {v: [f"{s}|{t}|{r[0]['fail_' + v]}" for (s, t, _), r in zip(items, res) if r[0]["fail_" + v]]
              for v in bp.VARIANTS}
    nicp = np.asarray([r[0]["nicp"] for r in res])
    blmm.atomic_savez(
        out, subjects=np.asarray([s for s, _, _ in items]), topologies=np.asarray([t for _, t, _ in items]),
        vb_kept=np.stack([np.asarray(r[0]["vb"]["kept"], dtype=np.float64) for r in res]),
        surf=np.stack([r[0]["surf"] for r in res]), nicp_resid_mm=nicp, nicp_check=np.abs(nicp - orig["nicp_resid_mm"]),
        seconds=np.asarray([r[1] for r in res]), failed_va=np.asarray(failed["va"], dtype="U300"),
        failed_vb=np.asarray(failed["vb"], dtype="U300"), free_expr=view not in bp.NEUTRAL_VIEWS,
        expr_scale=ctx["expr_scale"], loop=json.dumps(bp.LOOP[name]), loop_inner=bp.LOOP_INNER,
        model_sha256=bp.sha256(ctx["file"]), **arr)
    s = np.stack([r[0]["surf"] for r in res])
    print(f"[bp-e1] {view} {name}: {time.time() - t0:.0f}s, mediana {np.median([r[1] for r in res]):.1f}s/mesh; "
          f"fallite A {len(failed['va'])}, B {len(failed['vb'])} {(failed['va'] + failed['vb'])[:3]}; NICP scarto max "
          f"{np.nanmax(np.abs(nicp - orig['nicp_resid_mm'])):.2e} mm; superficie mediana/p95 originale "
          f"{np.nanmedian(s[:, 0, 0]):.2f}/{np.nanmedian(s[:, 0, 1]):.2f}, A {np.nanmedian(s[:, 1, 0]):.2f}/"
          f"{np.nanmedian(s[:, 1, 1]):.2f}, B {np.nanmedian(s[:, 2, 0]):.2f}/{np.nanmedian(s[:, 2, 1]):.2f} mm",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default=",".join(bp.VIEWS))
    p.add_argument("--models", default=",".join(bp.MODELS))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    for name in a.models.split(","):
        if a.pilot:
            run_pilot(name, a.workers)
            continue
        for view in a.views.split(","):
            run_view(view, name, a.workers, a.overwrite)


if __name__ == "__main__":
    main()
