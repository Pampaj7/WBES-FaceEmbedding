#!/usr/bin/env python3
"""Ava-256, prova confermativa: protocolli della comunita' (NoW originale, NoW metrical, REALY, Stirling) con i landmark
del rilevatore su un dominio da spec (PROTOCOL_CONFERMATIVO, sez. 9.1). Collaudata su HIFI3D con lo STESSO codice
(spec ``aau/ava256/spec/hifi3d_det.json``). Nessun passo legge GT.

    v3_work/unified_gt/run.sh aau/ava256/conf_protocols.py <passo> --spec <json> [--workers 64]
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --dir <cartella dei render>
    (conf.sbatch, CONF_STEPS)

Funzioni di ``pr_noise.py`` / ``pr_noise_detect.py`` / ``pr_landmarks.py`` / ``pr_run.py`` importate INVARIATE (render,
camera, raggi, retroproiezione, riferimento di REALY, port dei protocolli); qui solo i cicli sulle mesh di una vista da
spec, le coppie e le cartelle (``pr_common.use_roots``: ``<work>/protocolli``, ``<evidence>/protocolli``). Passi:

  define        ``pr_landmarks.define`` sullo spec (68 iBUG, 85 di REALY ed etichette di regione dall'oracolo FLAME,
                aggancio sulle original del template dello spec). Mappa assente o controllo del frame fallito (mediana
                media -> template > ``FRAME_MAX_MM``): REALY NON APPLICABILE (``realy_na.json`` col motivo); NoW e Stirling
                proseguono (usano solo i 7 landmark del rilevatore).
  prep          ``pr_landmarks.prep``: landmark della mappa su ogni mesh (servono i 17 keypoint di REALY oltre il 68 e le
                etichette di regione; i 68 iBUG li sostituisce il rilevatore). REALY non applicabile: landmark NaN,
                nessuna regione.
  gate-render   controllo del rilevatore PRIMA del dominio (sez. 9.1): ``GATE_N`` = 24 mesh di HIFI3D (vista di
                sviluppo, ``rng(1234)``), render rifatto in ``<work>/protocolli/noise/det/gate_hifi3d``; poi il rilevatore
                (``pr_noise_detect.py --dir``, venv di MICA);
  gate-compare  contro i png e i punti 2D di ``aau/runs/protocolli/noise/det/hifi3d`` (job 1071153). TOLLERANZA 0:
                immagini uguali pixel per pixel, camere uguali, punti 2D con scarto massimo 0 px, stessi ripieghi; uno
                scarto -> SystemExit (il job si ferma, nessun render del dominio).
  render        ``pr_noise._render_task`` su ogni mesh della vista (crop compreso) -> png e ``cams.npz``; poi il rilevatore.
  detprep       ``pr_noise._prep_task`` (ramo ``det``): retroproiezione dei 68 punti, ritagli e riferimento di REALY;
                ripieghi del detector per topologia, contro la soglia ``FALLBACK_MAX`` (la applica ``conf_eval.py``
                per gruppo).
  run           coppie ORDINATE per mesh A con ``pr_noise._task`` (``pr_run.one_pair``, ``--now-opt`` 0, i 4 metodi);
                coppie = tutte quelle di ``pr_common.pairs`` (Ava-256: le 597.000 di all_cross) o, con ``pairs`` nello
                spec (solo collaudo), il sottocampione di ``pr_noise.subsample`` (stessa funzione, stesso seme); ``--shard
                k/n`` (A % n == k), ``--reverse``; ogni A salvata, ripartibile.
  merge         plug-in ``<work>/protocolli/plugin/<metodo>/lmk_det/<vista>.npz`` (``D_mm`` simmetrica, media dei due
                versi, NaN fuori dalle coppie) e json con versi mancanti ed errori.

Variante: ``lmk_det`` con ``declared: false`` (su sviluppo e' una sensibilita'); su Ava-256 si calcola SOLO questa per
deroga esplicita del protocollo (sez. 9.1), letta per nome dal valutatore, non dal campo ``declared``.
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
sys.path.insert(0, str(THIS.parent / "competitors" / "protocolli"))
sys.path.insert(0, str(THIS))
import conf_common as cc  # noqa: E402
import pr_common as pc  # noqa: E402
import pr_noise as pn  # noqa: E402
import pr_run as prr  # noqa: E402

blmm = pc.blmm
COND = "det"
GATE_N, GATE_SEED = pn.CHECK_N, 1234                 # 24 mesh di HIFI3D, scelta fissa
GATE_REF = pn.NWORK / "det" / "hifi3d"               # render e rilevatore di sviluppo (pr_noise, job 1071153)
FALLBACK_MAX = 0.02                                  # sez. 9.1: oltre il 2% di mesh del gruppo col ripiego -> non risolto


def setup(spec: Path) -> tuple[dict, dict]:
    """(spec, vista) con le cartelle del dominio."""
    pc.use_roots(spec)
    s = cc.load(spec)
    return s, pc.view(s["name"], spec)


def det_dir(name: str) -> Path:
    return pc.WORK / "noise" / "det" / name


def detprep_path(name: str) -> Path:
    return pc.WORK / "noise" / "prep" / f"{name}_{COND}.npz"


def na_path() -> Path:
    return pc.EVID / "realy_na.json"


def pool(workers: int):
    return mp.get_context("fork").Pool(workers)


# ------------------------------------------------------------------------------------------ define, prep

def step_define(spec: Path) -> None:
    import pr_landmarks as pl
    s, v = setup(spec)
    L = s.get("landmarks") or {}
    reason = None
    if not L.get("corr") or not cc.rp(L["corr"]).exists():
        reason = f"mappa regione FLAME -> template assente ({L.get('corr')})"
    else:
        try:
            pl.define(v["name"], spec)
        except SystemExit as e:
            if "frame diversi" not in str(e):
                raise
            reason = f"controllo del frame fallito: {e}"
    if reason:
        cc.atomic_json(na_path(), {"realy": "non applicabile", "motivo": reason, **cc.stamp()})
        print(f"[conf-pr] {v['name']}: REALY NON APPLICABILE ({reason})", flush=True)
    elif na_path().exists():
        na_path().unlink()


def step_prep(spec: Path, workers: int) -> None:
    import pr_landmarks as pl
    s, v = setup(spec)
    if not na_path().exists():
        pl.prep(v["name"], spec, workers, pl.REALY_CAP)
        return
    # REALY non applicabile: nessuna definizione dei landmark; la prep ha solo mesh e chiavi (landmark NaN, nessuna
    # regione), il rilevatore fornisce i 68 iBUG (detprep)
    n = len(v["items"])
    obj = lambda xs: np.asarray(xs + [None], dtype=object)[:-1]       # noqa: E731
    out = {"subjects": np.asarray([x for x, _, _ in v["items"]]), "topologies": np.asarray([t for _, t, _ in v["items"]]),
           "paths": np.asarray([blmm.rel(p) for _, _, p in v["items"]]), "L68": np.full((n, 68, 3), np.nan),
           "K85": np.full((n, 85, 3), np.nan), "d68": np.full((n, 68), np.nan), "d85": np.full((n, 85), np.nan),
           "labels": obj([np.zeros(len(np.unique(blmm.load_raw(p)[1])), np.int8) for _, _, p in v["items"]]),
           "realy": obj([None] * n), "n_vertices": np.zeros(n, np.int64)}
    pc.prep_path(v["name"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez(pc.prep_path(v["name"]), **out)
    print(f"[conf-pr] prep {v['name']}: REALY non applicabile, {n} mesh senza landmark della mappa", flush=True)


# ------------------------------------------------------------------------------------------ gate del rilevatore

def gate_items() -> tuple[dict, np.ndarray]:
    vd = pc.view("hifi3d")                                       # vista di sviluppo, senza spec
    with np.load(GATE_REF / "cams.npz") as z:
        if list(z["paths"].astype(str)) != [blmm.rel(p) for _, _, p in vd["items"]]:
            raise SystemExit(f"{GATE_REF}: render di sviluppo con mesh diverse dalla vista hifi3d")
    sel = np.sort(np.random.default_rng(GATE_SEED).choice(len(vd["items"]), GATE_N, replace=False))
    return vd, sel


def save_cams(out: Path, res: list, items: list, **extra) -> None:
    blmm.atomic_savez(out / "cams.npz", o=np.stack([r[1] for r in res]), f=np.asarray([r[2] for r in res]),
                      box=np.stack([r[3] for r in res]), paths=np.asarray([blmm.rel(p) for _, _, p in items]),
                      img=pn.IMG, **extra)


def step_gate_render(spec: Path, workers: int) -> None:
    setup(spec)
    vd, sel = gate_items()
    out = det_dir("gate_hifi3d")
    out.mkdir(parents=True, exist_ok=True)
    items = [vd["items"][k] for k in sel]
    pn._ST.update(view=vd, items=items, out=out)
    with pool(min(workers, GATE_N)) as p:
        res = sorted(p.map(pn._render_task, range(len(items))), key=lambda x: x[0])
    save_cams(out, res, items, sel=sel)
    print(f"[conf-pr] gate: {len(items)} render di HIFI3D in {out}", flush=True)


def step_gate_compare(spec: Path) -> None:
    from PIL import Image
    setup(spec)
    vd, sel = gate_items()
    out = det_dir("gate_hifi3d")
    with np.load(out / "cams.npz") as z, np.load(GATE_REF / "cams.npz") as r:
        if not np.array_equal(z["sel"], sel):
            raise SystemExit("gate: render di mesh diverse dalla scelta")
        cams = max(float(np.abs(z[k] - r[k][sel]).max()) for k in ("o", "f", "box"))
    with np.load(out / "lmk2d.npz") as z, np.load(GATE_REF / "lmk2d.npz") as r:
        dp = np.abs(z["P"][:, :, :2] - r["P"][sel][:, :, :2])
        fb_same = bool(np.array_equal(z["fallback"], r["fallback"][sel]))
        score = float(np.abs(z["det_score"] - r["det_score"][sel]).max())
    px = [int((np.asarray(Image.open(out / f"{k:04d}.png")) != np.asarray(Image.open(GATE_REF / f"{m:04d}.png"))).sum())
          for k, m in enumerate(sel)]
    rep = {"meshes": [blmm.rel(vd["items"][k][2]) for k in sel], "reference": blmm.rel(GATE_REF),
           "pixels_different": int(sum(px)), "camera_max_abs_diff": cams, "points2d_max_abs_diff_px": float(dp.max()),
           "points2d_n_different": int((dp > 0).any(axis=2).sum()), "fallback_equal": fb_same,
           "det_score_max_abs_diff": score, "tolerance": "0 (pixel, camere, punti 2D, ripieghi)", **cc.stamp()}
    rep["pass"] = rep["pixels_different"] == 0 and cams == 0.0 and rep["points2d_max_abs_diff_px"] == 0.0 and fb_same
    cc.atomic_json(pc.EVID / "gate_detector.json", rep)
    print(f"[conf-pr] gate: {json.dumps(rep)[:600]}", flush=True)
    if not rep["pass"]:
        raise SystemExit("gate del rilevatore FALLITO: il rilevatore non riproduce i 24 render di HIFI3D (tolleranza 0)")


# ------------------------------------------------------------------------------------------ render, detprep

def step_render(spec: Path, workers: int) -> None:
    t0 = time.time()
    _, v = setup(spec)
    out = det_dir(v["name"])
    out.mkdir(parents=True, exist_ok=True)
    pn._ST.update(view=v, items=v["items"], out=out)
    with pool(workers) as p:
        res = sorted(p.map(pn._render_task, range(len(v["items"])), chunksize=4), key=lambda x: x[0])
    save_cams(out, res, v["items"])
    print(f"[conf-pr] render {v['name']}: {len(res)} immagini in {out}, {time.time() - t0:.0f}s", flush=True)


def stats(x: np.ndarray) -> dict:
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x))}


def step_detprep(spec: Path, workers: int) -> None:
    """Come ``pr_noise.prep`` con ``--cond det`` (stesso compito per mesh, stesse uscite), sulla vista dello spec."""
    t0 = time.time()
    _, v = setup(spec)
    pr = pc.read_prep(v["name"])
    dd = det_dir(v["name"])
    with np.load(dd / "cams.npz") as z, np.load(dd / "lmk2d.npz") as y:
        if list(z["paths"].astype(str)) != [blmm.rel(p) for _, _, p in v["items"]]:
            raise SystemExit(f"{v['name']}: render con mesh diverse dalla vista")
        det = {"o": z["o"], "f": z["f"], "P": y["P"], "fallback": y["fallback"], "det_score": y["det_score"]}
    pn._ST.update(view=v, prep=pr, cond=COND, det=det)
    with pool(workers) as p:
        res = sorted(p.map(pn._prep_task, range(len(v["items"])), chunksize=2), key=lambda r: r["k"])
    obj = lambda xs: np.asarray(xs + [None], dtype=object)[:-1]       # noqa: E731
    out = {k: pr[k] for k in ("subjects", "topologies", "paths", "labels", "n_vertices")}
    out.update(L68=np.stack([r["L68"] for r in res]), K85=np.stack([r["K85"] for r in res]),
               realy=obj([r["realy"] for r in res]), d68=np.stack([r["d68"] for r in res]),
               d85=np.stack([r["d85"] for r in res]), fallback=np.asarray([r["fallback"] for r in res]),
               det_score=np.asarray([r["det_score"] for r in res]))
    detprep_path(v["name"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez(detprep_path(v["name"]), **out)
    topo = pr["topologies"].astype(str)
    fb = out["fallback"]
    d7 = out["d68"][:, list(pc.LMK7)]
    summ = {"view": v["name"], "cond": COND, "meshes": len(res),
            "det_fallback": {t: int(fb[topo == t].sum()) for t in sorted(set(topo))},
            "det_fallback_total": int(fb.sum()), "det_fallback_fraction": float(fb.mean()),
            "fallback_max_fraction": FALLBACK_MAX,
            "det_score": stats(out["det_score"][~fb]) if (~fb).any() else None,
            "rays_missed_ibug68": int(sum(r["miss"].sum() for r in res)),
            "rays_missed_lmk7": {n: int(sum(r["miss"][i] for r in res)) for i, n in zip(pc.LMK7, pc.LMK7_NAMES)},
            "shift_lmk7_mm_from_map": stats(d7) if np.isfinite(d7).all() else None,
            "shift_lmk7_mm_from_map_note": "distanza dai landmark della mappa FLAME (oracolo sui domini di sviluppo, "
                                           "solo indicativa su Ava-256); None se REALY non applicabile",
            "render": {"img": pn.IMG, "fill": pn.FILL, "cam_dist_x_extent": pn.CAM_DIST, "light": pn.LIGHT,
                       "ambient": pn.AMB, "albedo": pn.ALBEDO},
            "seconds": time.time() - t0, **cc.stamp()}
    cc.atomic_json(pc.EVID / f"detprep_{v['name']}.json", summ)
    print(f"[conf-pr] detprep {v['name']}: ripieghi {summ['det_fallback']}, raggi mancati "
          f"{summ['rays_missed_ibug68']}, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ run, merge

def pairs_of(s: dict, items: list) -> tuple[np.ndarray, np.ndarray]:
    """Coppie non ordinate (i < j): tutte (``pr_common.pairs``) o il sottocampione dello spec (``pr_noise.subsample``,
    con la frazione e il seme di sviluppo: solo collaudo)."""
    p = s.get("pairs")
    if not p:
        return pc.pairs(items)
    if not s.get("collaudo") or (p["frac"], p["seed"]) != (pn.FRAC, pn.PAIR_SEED):
        raise SystemExit(f"sottocampione {p}: solo nei collaudi, con FRAC {pn.FRAC} e seme {pn.PAIR_SEED}")
    return pn.subsample(p["key"], items)


def ckdir_of(name: str) -> Path:
    return pc.WORK / "noise" / "parts" / name / f"{COND}.d"


def step_run(spec: Path, workers: int, shard: str, reverse: bool) -> None:
    t0 = time.time()
    s, v = setup(spec)
    with np.load(detprep_path(v["name"]), allow_pickle=True) as z:
        pr = {k: z[k] for k in z.files}
    D = prr.load(v["name"], spec, 0, prep=pr)
    i, j = pairs_of(s, D["items"])
    P = pn.partners_of(i, j, len(D["items"]))
    ck = ckdir_of(v["name"])
    ck.mkdir(parents=True, exist_ok=True)
    pn._ST.update(D=D, partners=P, ckdir=ck)
    k, nsh = (int(x) for x in shard.split("/"))
    todo = [a for a in range(len(P)) if a % nsh == k and len(P[a]) and not (ck / f"A{a}.npz").exists()]
    if reverse:
        todo = todo[::-1]
    print(f"[conf-pr] run {v['name']} {shard}: {len(i)} coppie, {len(todo)} mesh A da fare, {time.time() - t0:.0f}s "
          "di caricamento", flush=True)
    tl = time.time()
    with pool(workers) as p:
        for n, _ in enumerate(p.imap_unordered(pn._task, todo, chunksize=1), 1):
            if n % max(1, len(todo) // 10) == 0:
                print(f"[conf-pr] run {v['name']}: {n}/{len(todo)} mesh A, {time.time() - tl:.0f}s", flush=True)
    print(f"[conf-pr] run {v['name']} {shard}: fatto in {time.time() - t0:.0f}s", flush=True)


def plugin_dir(method: str) -> Path:
    return pc.PLUGIN_ROOT / method / f"lmk_{COND}"


def step_merge(spec: Path) -> None:
    """Come ``pr_noise.merge`` sulle coppie dello spec."""
    s, v = setup(spec)
    items, n = v["items"], len(v["items"])
    i, j = pairs_of(s, items)
    P = pn.partners_of(i, j, n)
    ck = ckdir_of(v["name"])
    need = [a for a in range(n) if len(P[a])]
    miss = [a for a in need if not (ck / f"A{a}.npz").exists()]
    if miss:
        raise SystemExit(f"{v['name']}: {len(miss)} mesh A mancanti (es. {miss[:5]})")
    E = {m: np.full((n, n), np.nan) for m in pc.METHODS}
    errs = {m: [] for m in pc.METHODS}
    secs = {m: [] for m in pc.METHODS}
    core_s, jobs = 0.0, set()
    for a in need:
        with np.load(ck / f"A{a}.npz") as z:
            if not np.array_equal(np.sort(z["b"]), P[a]):
                raise SystemExit(f"{v['name']} A{a}: partner diversi dalle coppie dello spec")
            for m in pc.METHODS:
                E[m][z["a"], z["b"]] = z[f"val:{m}"]
                secs[m].append(z[f"val:{m}|seconds"])
                e = z[f"err:{m}"]
                errs[m] += [str(x) for x in e[e != ""]]
            core_s += float(z["task_s"])
            jobs.add(str(z["job"]))
    subj = np.asarray([x for x, _, _ in items])
    topo = np.asarray([t for _, t, _ in items])
    for m in pc.METHODS:
        Dm = np.full((n, n), np.nan)
        Dm[i, j] = Dm[j, i] = (E[m][i, j] + E[m][j, i]) / 2
        np.fill_diagonal(Dm, 0.0)
        sx = np.concatenate(secs[m])
        info = {"method": m, "cond": COND, "label": f"{pc.LABEL[m]}, {pn.CLABEL[COND]}", "view": v["name"],
                "spec": s["_path"], "pairs_unordered": int(len(i)), "pairs": s.get("pairs") or "tutte (pr_common.pairs)",
                "ordered_missing": int(np.isnan(E[m][i, j]).sum() + np.isnan(E[m][j, i]).sum()),
                "rows_nonfinite": int((~np.isfinite(Dm[i, j])).sum()),
                "errors": len(errs[m]), "error_examples": sorted(set(errs[m]))[:10],
                "seconds_per_ordered_pair": {"mean": float(sx.mean()), "median": float(np.median(sx))},
                "core_hours_all_methods": core_s / 3600, "jobs": sorted(jobs), **cc.stamp()}
        d = plugin_dir(m)
        blmm.atomic_savez(d / f"{v['name']}.npz", subjects=subj, topologies=topo,
                          paths=np.asarray([blmm.rel(p) for _, _, p in items]), D_mm=Dm)
        pc.atomic_json(d / f"{v['name']}.json", info)
        lab = f"{pc.LABEL[m]}, {pn.CLABEL[COND]}"
        pc.atomic_json(d / "fm_plugin.json", {
            "label": lab, "family": pc.FAMILY, "declared": False, "method": m, "variant": f"lmk_{COND}",
            "note": "Ava-256: unica variante calcolata, per deroga della sez. 9.1 del protocollo confermativo",
            "readings": {"mm": {"label": f"{lab}, mm (media dei due versi)", "gt": ["fr", "sr"]}}})
        print(f"[conf-pr] merge {v['name']} {m}: {len(i)} coppie, versi mancanti {info['ordered_missing']}, errori "
              f"{len(errs[m])}", flush=True)
    print(f"[conf-pr] merge {v['name']}: {core_s / 3600:.1f} ore-core, job {sorted(jobs)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("define", "prep", "gate-render", "gate-compare", "render", "detprep", "run", "merge"))
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--reverse", action="store_true")
    a = p.parse_args()
    {"define": lambda: step_define(a.spec), "prep": lambda: step_prep(a.spec, a.workers),
     "gate-render": lambda: step_gate_render(a.spec, a.workers), "gate-compare": lambda: step_gate_compare(a.spec),
     "render": lambda: step_render(a.spec, a.workers), "detprep": lambda: step_detprep(a.spec, a.workers),
     "run": lambda: step_run(a.spec, a.workers, a.shard, a.reverse), "merge": lambda: step_merge(a.spec)}[a.step]()


if __name__ == "__main__":
    main()
