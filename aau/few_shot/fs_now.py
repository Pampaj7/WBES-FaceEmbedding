#!/usr/bin/env python3
"""NoW col rilevatore (metrical e originale) su TUTTE le coppie senza crop delle 4 viste della matrice: il concorrente
"NoW metrical col rilevatore" del protocollo adattabile (REGOLE.md, sez. 5). Nessuna GT letta.

    v3_work/unified_gt/run.sh aau/few_shot/fs_now.py render --view faceverse --workers 32
    aau/recon/run_recon.sh mica aau/competitors/protocolli/pr_noise_detect.py --view faceverse
    v3_work/unified_gt/run.sh aau/few_shot/fs_now.py prep --view faceverse --workers 32
    v3_work/unified_gt/run.sh aau/few_shot/fs_now.py run --view faceverse --workers 64 [--shard 0 --nshards 2]
    v3_work/unified_gt/run.sh aau/few_shot/fs_now.py merge --view faceverse
    (fs.sbatch, passi now-prep, now-run, now-merge)

La catena e' quella della condizione ``det`` di ``pr_noise.py`` (la ``lmk_det`` della matrice), importata e non
modificata: render frontale (``pr_noise.render``), 68 landmark di insightface ``1k3d68`` (``pr_noise_detect.py``),
retroproiezione col raggio della camera e ritagli di NoW (``pr_noise.prep --cond det``, ``pr_noise.load_cond``),
``pr_run.one_pair`` con ``--now-opt`` 0. HIFI3D e FaceScape hanno gia' render, landmark e prep (job 1071153): si
riusano. FaceVerse (con espressioni e neutra): render, rilevatore e prep qui; il riepilogo della prep va in
``EVID/now_det/prep/`` (``pr_noise.NEVID`` rediretto: l'evidenza di ``rumore_landmark`` non si tocca).

Coppie: tutte le i < j di soggetti diversi e topologie diverse, nessuna crop (le 99.000 righe ``nocrop`` per vista),
i due versi (A = i, B = j e A = j, B = i), distanza = media dei due versi (``pr_common``). ``run`` salva per mesh A
(``NOW_WORK/parts/<vista>/A<k>.npz``) e una ripartenza salta le A salvate. ``merge`` scrive
``NOW_WORK/<vista>.npz`` (``subjects``, ``topologies``, ``paths``, ``D_now_metrical``, ``D_now_original``: simmetriche,
NaN fuori dalle coppie) e ``EVID/now_det/<vista>.json`` (errori, versi mancanti, secondi, controllo). Controllo (HIFI3D,
FaceScape): sulle coppie senza crop del sottocampione dell'8% gia' calcolato (``competitors_new/<metodo>/lmk_det``)
scarto massimo <= ``NOW_TOL_MM``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import fs_common as fc
import pr_common as pc  # noqa: E402
import pr_metrics as pm  # noqa: E402
import pr_noise as pn  # noqa: E402
import pr_run as prr  # noqa: E402

METHODS = ("now_original", "now_metrical")
REUSED = ("hifi3d", "facescape")                 # render, landmark e prep det gia' calcolati (pr_noise, job 1071153)
_ST: dict = {}


def nocrop_pairs(items: list) -> tuple[np.ndarray, np.ndarray]:
    """Coppie i < j di soggetti diversi e topologie diverse, nessuna crop."""
    s = np.asarray([x for x, _, _ in items])
    t = np.asarray([y for _, y, _ in items])
    i, j = np.triu_indices(len(items), 1)
    keep = (s[i] != s[j]) & (t[i] != t[j]) & (t[i] != "crop") & (t[j] != "crop")
    return i[keep], j[keep]


def parts_dir(view: str):
    return fc.NOW_WORK / "parts" / view


# ------------------------------------------------------------------------------------------ prep

def render(view: str, workers: int) -> None:
    if view in REUSED:
        raise SystemExit(f"{view}: render gia' presenti (pr_noise), non si rifanno")
    if (pn.NWORK / "det" / view / "cams.npz").exists():
        raise SystemExit(f"{view}: {pn.NWORK / 'det' / view / 'cams.npz'} esiste gia'")
    pn.render(view, workers)


def prep(view: str, workers: int) -> None:
    if view in REUSED:
        raise SystemExit(f"{view}: prep det gia' presente (pr_noise), non si rifa'")
    pn.NEVID = fc.EVID / "now_det"                    # riepilogo fuori dall'evidenza di rumore_landmark
    pn.prep(view, "det", workers)


# ------------------------------------------------------------------------------------------ run

def _task(a: int) -> int:
    ck = _ST["ckdir"] / f"A{a}.npz"
    if ck.exists():
        return a
    t0 = time.perf_counter()
    D, bs = _ST["D"], _ST["partners"][a]
    V, F = D["mesh"][a]
    surf = pm.Surf(V, F)
    rows = [prr.one_pair(surf, a, int(b), METHODS, D) for b in bs]
    fc.atomic_savez(ck, task_s=time.perf_counter() - t0, job=fc.stamp()["job"],
                    **prr.pack([{"a": a, "b": bs, "rows": rows}]))
    return a


def run(view: str, workers: int, shard: int, nshards: int, reverse: bool) -> None:
    t0 = time.time()
    D = pn.load_cond(view, "det")
    i, j = nocrop_pairs(D["items"])
    P = pn.partners_of(i, j, len(D["items"]))
    ck = parts_dir(view)
    ck.mkdir(parents=True, exist_ok=True)
    _ST.update(D=D, partners=P, ckdir=ck)
    todo = [a for a in range(len(P)) if len(P[a]) and a % nshards == shard and not (ck / f"A{a}.npz").exists()]
    if reverse:
        todo = todo[::-1]
    print(f"[fs-now] run {view} parte {shard}/{nshards}: {len(i)} coppie, {len(todo)} mesh A da fare, "
          f"{time.time() - t0:.0f}s di caricamento", flush=True)
    tl = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task, todo, chunksize=1)):
            if (k + 1) % max(1, len(todo) // 20) == 0:
                print(f"[fs-now] run {view}: {k + 1}/{len(todo)} mesh A, {time.time() - tl:.0f}s", flush=True)
    print(f"[fs-now] run {view} parte {shard}/{nshards}: fatto in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ merge

def merge(view: str) -> None:
    v = pc.view(view)
    items, n = v["items"], len(v["items"])
    i, j = nocrop_pairs(items)
    P = pn.partners_of(i, j, n)
    ck = parts_dir(view)
    need = [a for a in range(n) if len(P[a])]
    miss = [a for a in need if not (ck / f"A{a}.npz").exists()]
    if miss:
        raise SystemExit(f"{view}: {len(miss)} mesh A mancanti (es. {miss[:5]})")
    E = {m: np.full((n, n), np.nan) for m in METHODS}
    errs = {m: [] for m in METHODS}
    secs = {m: [] for m in METHODS}
    core_s, jobs = 0.0, set()
    for a in need:
        with np.load(ck / f"A{a}.npz") as z:
            if not np.array_equal(np.sort(z["b"]), P[a]):
                raise SystemExit(f"{view} A{a}: partner diversi dalle coppie senza crop")
            for m in METHODS:
                E[m][z["a"], z["b"]] = z[f"val:{m}"]
                secs[m].append(z[f"val:{m}|seconds"])
                e = z[f"err:{m}"]
                errs[m] += [str(x) for x in e[e != ""]]
            core_s += float(z["task_s"])
            jobs.add(str(z["job"]))
    out, info = {}, {"view": view, "pairs_unordered": int(len(i)), "methods": {}}
    for m in METHODS:
        Dm = np.full((n, n), np.nan)
        Dm[i, j] = Dm[j, i] = (E[m][i, j] + E[m][j, i]) / 2
        np.fill_diagonal(Dm, 0.0)
        out[f"D_{m}"] = Dm
        s = np.concatenate(secs[m])
        info["methods"][m] = {"ordered_missing": int(np.isnan(E[m][i, j]).sum() + np.isnan(E[m][j, i]).sum()),
                              "pairs_nan": int(np.isnan(Dm[i, j]).sum()), "errors": len(errs[m]),
                              "error_examples": sorted(set(errs[m]))[:10],
                              "seconds_per_ordered_pair": {"mean": float(s.mean()), "median": float(np.median(s))}}
        if view in REUSED:                            # riproduzione: il sottocampione dell'8% di pr_noise
            with np.load(pc.PLUGIN_ROOT / m / "lmk_det" / f"{view}.npz") as z:
                if [(str(a), str(b)) for a, b in zip(z["subjects"], z["topologies"])] != \
                        [(x, t) for x, t, _ in items]:
                    raise SystemExit(f"{view} {m}: lmk_det con mesh diverse dalla vista")
                R = np.asarray(z["D_mm"], np.float64)
            ok = np.isfinite(R[i, j]) & np.isfinite(Dm[i, j])
            d = np.abs(R[i, j][ok] - Dm[i, j][ok])
            info["methods"][m]["check_vs_lmk_det"] = {
                "pairs": int(ok.sum()), "max_abs_mm": float(d.max()), "median_abs_mm": float(np.median(d)),
                "tol_mm": fc.NOW_TOL_MM, "pass": bool(d.max() <= fc.NOW_TOL_MM),
                "lmk_det_finite_not_here": int((np.isfinite(R[i, j]) & ~np.isfinite(Dm[i, j])).sum())}
    info.update(core_hours=core_s / 3600, jobs=sorted(jobs), **fc.stamp())
    fc.atomic_savez(fc.NOW_WORK / f"{view}.npz", subjects=np.asarray([x for x, _, _ in items]),
                    topologies=np.asarray([t for _, t, _ in items]),
                    paths=np.asarray([pc.blmm.rel(p) for _, _, p in items]), **out)
    fc.atomic_json(fc.EVID / "now_det" / f"{view}.json", info)
    print(f"[fs-now] merge {view}: {len(i)} coppie, {core_s / 3600:.1f} ore-core, "
          f"{ {m: (x['pairs_nan'], x.get('check_vs_lmk_det', {}).get('max_abs_mm')) for m, x in info['methods'].items()} }",
          flush=True)
    bad = [m for m, x in info["methods"].items() if not x.get("check_vs_lmk_det", {"pass": True})["pass"]]
    if bad:
        raise SystemExit(f"{view}: riproduzione fuori tolleranza per {bad}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("render", "prep", "run", "merge"))
    p.add_argument("--view", required=True, choices=fc.DOMAINS)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--reverse", action="store_true", help="run: dalle ultime mesh A (job di riserva)")
    a = p.parse_args()
    if a.step == "render":
        render(a.view, a.workers)
    elif a.step == "prep":
        prep(a.view, a.workers)
    elif a.step == "run":
        run(a.view, a.workers, a.shard, a.nshards, a.reverse)
    else:
        merge(a.view)


if __name__ == "__main__":
    main()
