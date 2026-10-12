#!/usr/bin/env python3
"""Distanze dei protocolli sulle coppie della matrice unificata: coppie ORDINATE per parte, poi unione simmetrica e
plug-in. Nessuna GT.

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_run.py run --view hifi3d --methods now_original,now_metrical,stirling \
        --shard 0 --nshards 4 --workers 64
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_run.py merge --view hifi3d
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_run.py sample --view hifi3d --n 200 --workers 32
    (pr.sbatch, passi run, merge, sample; dominio nuovo: --spec <json>, pr_common.py)

``run``: per ogni mesh A della parte (A = predetta; parti per indice di A, ``A % nshards``), tutte le mesh B di
soggetto diverso e topologia diversa (FaMoS: soggetto diverso) come riferimento: e(A, B) per i metodi chiesti
(``pr_metrics``). La superficie di A si costruisce una volta; i dati di B (ritagli, riferimento di REALY) sono
calcolati prima nel processo padre e condivisi. Riferimento: mesh in mm (``pr_common.mm_mesh``), 7 landmark NoW =
iBUG 36, 39, 42, 45, 33, 48, 54 dell'embedding FLAME di DECA/MICA, 85 keypoint di REALY (``pr_landmarks prep``).
``--now-opt N``: le iterazioni dell'ICP di NoW (originale e metrical) su al piu' N punti del riferimento ritagliato,
estratti una volta per mesh (seme dal percorso); l'errore finale su tutti i punti (0 = tutti, come [N]). Uscita
``aau/runs/protocolli/parts/<vista>/<metodi>_<k>of<n>.npz``.

``merge``: matrice ordinata E (n, n) per metodo; distanza della coppia = (E[a, b] + E[b, a]) / 2 sulle coppie di
``pr_common.pairs`` (NaN altrove, diagonale 0); ``competitors_new/<metodo>/<vista>.npz`` (``D_mm``; REALY anche le 4
regioni come ``R_<regione>``, NoW anche la mediana come ``Q_median``: chiavi che il loader non legge), json con
costi, iterazioni, riflessioni, fallimenti, e ``fm_plugin.json``.

``sample``: coppie ordinate estratte a caso (seme 1234, soggetti diversi e topologie diverse) per i controlli delle
approssimazioni: ICP di NoW su tutti i punti contro ``--now-opt``, REALY con il tetto di vertici contro un tetto piu'
alto (``--realy-cap-hi``), e i tempi per metodo. Uscita ``aau/runs/evidence/protocolli/checks/sample_<vista>.json``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as pc  # noqa: E402
import pr_metrics as pm  # noqa: E402

blmm = pc.blmm
NOW_R, STIRLING_R = 1.4, 1.2          # [N] :71, [F] :46
_ST: dict = {}


# ------------------------------------------------------------------------------------------ dati

def load(vname: str, spec: Path | None, now_opt: int) -> dict:
    """Mesh in mm, landmark, ritagli e riferimenti di REALY di tutte le mesh della vista (processo padre)."""
    v = pc.view(vname, spec)
    pr = pc.read_prep(v["name"])
    keys = [(s, t) for s, t, _ in v["items"]]
    if list(zip(pr["subjects"].astype(str), pr["topologies"].astype(str))) != keys:
        raise SystemExit(f"{v['name']}: prep con mesh diverse dalla vista")
    mesh, now_crop, now_opt_pts, st_crop = [], [], [], []
    for k, (_, _, p) in enumerate(v["items"]):
        V, F, _ = pc.mm_mesh(v, p)
        L7 = pr["L68"][k][list(pc.LMK7)]
        mesh.append((V, F))
        c = pm.crop(V, L7, NOW_R)
        now_crop.append(c)
        if now_opt and len(c) > now_opt:
            rng = np.random.default_rng(zlib.crc32(blmm.rel(p).encode()))
            now_opt_pts.append(c[np.sort(rng.choice(len(c), now_opt, replace=False))])
        else:
            now_opt_pts.append(None)
        st_crop.append(pm.crop(V, L7, STIRLING_R))
    return {"view": v, "items": v["items"], "mesh": mesh, "L7": pr["L68"][:, list(pc.LMK7)], "K85": pr["K85"],
            "now_crop": now_crop, "now_opt": now_opt_pts, "st_crop": st_crop, "realy": list(pr["realy"]),
            "labels": list(pr["labels"])}


def partners(items: list, a: int) -> np.ndarray:
    s = np.asarray([x for x, _, _ in items])
    t = np.asarray([y for _, y, _ in items])
    keep = s != s[a]
    if len(set(t)) > 1:
        keep &= t != t[a]
    return np.flatnonzero(keep)


def one_pair(surf: pm.Surf, a: int, b: int, methods: tuple, D: dict) -> dict:
    """e(A = a, B = b) per i metodi; un errore lascia NaN e il messaggio."""
    out = {}
    for m in methods:
        t0 = time.perf_counter()
        try:
            if m in ("now_original", "now_metrical"):
                r = pm.now_pair(surf, D["L7"][a], D["now_crop"][b], D["L7"][b], metrical=m == "now_metrical",
                                P_opt=D["now_opt"][b])
                out[m] = r["mean"]
                out.update({f"{m}|median": r["median"], f"{m}|scale": r["scale"], f"{m}|iter": r["iterations"],
                            f"{m}|fevals": r["fevals"], f"{m}|reflection": float(r["reflection"])})
            elif m == "stirling":
                r = pm.stirling_pair(surf, D["L7"][a], D["st_crop"][b], D["L7"][b])
                out[m] = r["rmse"]
                out.update({f"{m}|mean": r["mean"], f"{m}|reflection": float(r["reflection"])})
            elif m == "realy":
                r = pm.realy_pair(surf, D["K85"][a], D["realy"][b])
                out[m] = r["total"]
                out.update({f"{m}|{x}": r[f"err_{x}"] for x in pc.REGIONS})
                out[f"{m}|hit_cap"] = float(sum(r.get(f"{x}_hit_cap", 0) for x in pc.REGIONS))
                out[f"{m}|solves"] = float(sum(r.get(f"{x}_solves", 0) for x in pc.REGIONS))
            out[f"{m}|error"] = ""
        except Exception as exc:  # noqa: BLE001  (la coppia resta NaN)
            out[m] = np.nan
            out[f"{m}|error"] = f"{type(exc).__name__}: {exc}"
        out[f"{m}|seconds"] = time.perf_counter() - t0
    return out


def _task(a: int) -> dict:
    D, methods = _ST["D"], _ST["methods"]
    V, F = D["mesh"][a]
    surf = pm.Surf(V, F)
    bs = partners(D["items"], a)
    rows = [one_pair(surf, a, int(b), methods, D) for b in bs]
    return {"a": a, "b": bs, "rows": rows}


def pack(results: list) -> dict:
    """Righe (a, b, colonne) in array; gli errori come stringhe."""
    A = np.concatenate([np.full(len(r["b"]), r["a"]) for r in results])
    B = np.concatenate([r["b"] for r in results])
    rows = [x for r in results for x in r["rows"]]
    cols = sorted({k for x in rows for k in x})
    out = {"a": A, "b": B}
    for c in cols:
        if c.endswith("|error"):
            out["err:" + c[:-6]] = np.asarray([x.get(c, "") for x in rows])
        else:
            out["val:" + c] = np.asarray([x.get(c, np.nan) for x in rows], np.float64)
    return out


# ------------------------------------------------------------------------------------------- run

def run(vname: str, spec: Path | None, methods: tuple, shard: int, nshards: int, workers: int, now_opt: int) -> None:
    t0 = time.time()
    D = load(vname, spec, now_opt)
    _ST.update(D=D, methods=methods)
    todo = [a for a in range(len(D["items"])) if a % nshards == shard]
    tl = time.time()
    res = []
    with mp.get_context("fork").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_task, todo, chunksize=1)):
            res.append(r)
            if (k + 1) % max(1, len(todo) // 20) == 0:
                print(f"[pr-run] {vname} {shard}/{nshards}: {k + 1}/{len(todo)} mesh A, {time.time() - tl:.0f}s",
                      flush=True)
    out = pack(res)
    tag = "+".join(methods)
    path = pc.WORK / "parts" / D["view"]["name"] / f"{tag}_{shard}of{nshards}.npz"
    blmm.atomic_savez(path, methods=np.asarray(methods), now_opt=now_opt, load_s=tl - t0, wall_s=time.time() - tl,
                      workers=workers, job=pc.stamp()["job"], **out)
    nerr = {m: int((out.get(f"err:{m}", np.asarray([""])) != "").sum()) for m in methods}
    print(f"[pr-run] {vname} parte {shard}/{nshards}: {len(out['a'])} coppie ordinate, errori {nerr}, "
          f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------- merge

def merge(vname: str, spec: Path | None) -> None:
    v = pc.view(vname, spec)
    items, n = v["items"], len(v["items"])
    root = pc.WORK / "parts" / v["name"]
    parts = sorted(root.glob("*of*.npz"))
    by_m: dict = {}
    for p in parts:
        with np.load(p) as z:
            for m in z["methods"]:
                by_m.setdefault(str(m), []).append(p)
    i, j = pc.pairs(items)
    for m, ps in by_m.items():
        E = np.full((n, n), np.nan)
        extra = {}
        info = {"method": m, "label": pc.LABEL[m], "parts": [p.name for p in ps], "view": v["name"],
                "pairs_unordered": int(len(i)), "symmetrization": "media dei due versi: (e(a, b) + e(b, a)) / 2",
                "clones": pc.CLONE_COMMITS, **pc.stamp()}
        diag: dict = {}
        cost = []
        errs: list = []
        for p in ps:
            with np.load(p) as z:
                a, b = z["a"], z["b"]
                if (~np.isnan(E[a, b])).any():
                    raise SystemExit(f"{p}: coppie ordinate gia' viste in un'altra parte")
                E[a, b] = z[f"val:{m}"]
                for k in z.files:
                    if k.startswith(f"val:{m}|"):
                        diag.setdefault(k[len(f"val:{m}|"):], []).append((a, b, z[k]))
                cost.append({"part": p.name, "wall_s": float(z["wall_s"]), "workers": int(z["workers"]),
                             "now_opt": int(z["now_opt"]), "job": str(z["job"]), "pairs": int(len(a))})
                e = z[f"err:{m}"]
                errs += [str(x) for x in e[e != ""]]
        missing = int(np.isnan(E[i, j]).sum() + np.isnan(E[j, i]).sum())
        Dm = np.full((n, n), np.nan)
        Dm[i, j] = Dm[j, i] = (E[i, j] + E[j, i]) / 2
        np.fill_diagonal(Dm, 0.0)
        for key, lst in diag.items():
            M = np.full((n, n), np.nan)
            for a, b, x in lst:
                M[a, b] = x
            if key in pc.REGIONS:
                R = np.full((n, n), np.nan)
                R[i, j] = R[j, i] = (M[i, j] + M[j, i]) / 2
                np.fill_diagonal(R, 0.0)
                extra[f"R_{key}"] = R
            elif key == "median":
                R = np.full((n, n), np.nan)
                R[i, j] = R[j, i] = (M[i, j] + M[j, i]) / 2
                np.fill_diagonal(R, 0.0)
                extra["Q_median"] = R
            vals = np.concatenate([x for _, _, x in lst])
            vals = vals[np.isfinite(vals)]
            if len(vals):
                diag_s = {"mean": float(vals.mean()), "median": float(np.median(vals)),
                          "p05": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95)),
                          "max": float(vals.max())}
                if key == "reflection":
                    diag_s = {"fraction": float(vals.mean()), "count": int(vals.sum())}
                info.setdefault("diagnostics", {})[key] = diag_s
        asym = np.abs(E[i, j] - E[j, i]) / np.maximum((E[i, j] + E[j, i]) / 2, 1e-12)
        info.update(ordered_missing=missing, errors=len(errs), error_examples=sorted(set(errs))[:10], cost=cost,
                    core_hours_parts=float(sum(c["wall_s"] * c["workers"] for c in cost) / 3600),
                    core_hours_note="ore-core delle parti che contengono il metodo (condivise con gli altri metodi della "
                                    "stessa parte); il costo del solo metodo: diagnostics.seconds",
                    asymmetry_relative={"median": float(np.nanmedian(asym)), "p95": float(np.nanpercentile(asym, 95))})
        pc.write_view(m, v["name"], items, Dm, extra, info)
        print(f"[pr-merge] {v['name']} {m}: {len(i)} coppie, versi mancanti {missing}, errori {len(errs)}, "
              f"{info['core_hours_parts']:.1f} ore-core delle parti", flush=True)


# ------------------------------------------------------------------------------------------ sample

def _sample_task(ab: tuple) -> dict:
    """Una coppia ordinata con le varianti dei controlli: NoW su tutti i punti e con ``now_opt``, REALY con il tetto
    della prep e con ``cap_hi``."""
    import pr_landmarks as pl
    a, b = ab
    D, now_opt, cap_hi = _ST["D"], _ST["now_opt"], _ST["cap_hi"]
    V, F = D["mesh"][a]
    surf = pm.Surf(V, F)
    out = {"a": int(a), "b": int(b), **one_pair(surf, a, b, pc.METHODS, D)}
    for m in ("now_original", "now_metrical"):
        P = D["now_crop"][b]
        if now_opt and len(P) > now_opt:
            rng = np.random.default_rng(zlib.crc32(blmm.rel(D["items"][b][2]).encode()))
            Po = P[np.sort(rng.choice(len(P), now_opt, replace=False))]
            t0 = time.perf_counter()
            r = pm.now_pair(surf, D["L7"][a], P, D["L7"][b], metrical=m == "now_metrical", P_opt=Po)
            out[f"{m}|opt"] = r["mean"]
            out[f"{m}|opt_seconds"] = time.perf_counter() - t0
        else:
            out[f"{m}|opt"] = out[m]
            out[f"{m}|opt_seconds"] = out[f"{m}|seconds"]
    Vb, Fb = D["mesh"][b]
    t0 = time.perf_counter()
    ref = pl.realy_reference(Vb, Fb, D["labels"][b], D["K85"][b], cap_hi)
    r = pm.realy_pair(surf, D["K85"][a], ref)
    out["realy|cap_hi"] = r["total"]
    out["realy|cap_hi_seconds"] = time.perf_counter() - t0
    return out


def sample(vname: str, spec: Path | None, n: int, workers: int, now_opt: int, cap_hi: int) -> None:
    from scipy.stats import spearmanr
    t0 = time.time()
    D = load(vname, spec, 0)
    _ST.update(D=D, now_opt=now_opt, cap_hi=cap_hi)
    items = D["items"]
    rng = np.random.default_rng(1234)
    ab = []
    while len(ab) < n:
        a = int(rng.integers(len(items)))
        p = partners(items, a)
        ab.append((a, int(rng.choice(p))))
    with mp.get_context("fork").Pool(workers) as pool:
        rows = pool.map(_sample_task, ab, chunksize=1)

    def cmp(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        rel = np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-12)
        return {"n": int(ok.sum()), "spearman": float(spearmanr(x[ok], y[ok]).correlation),
                "rel_diff_median": float(np.median(rel)), "rel_diff_p95": float(np.percentile(rel, 95)),
                "rel_diff_max": float(rel.max())}

    col = lambda k: [r.get(k, np.nan) for r in rows]      # noqa: E731
    out = {"view": D["view"]["name"], "n_pairs": len(rows), "now_opt": now_opt, "realy_cap_hi": cap_hi,
           "seconds_per_ordered_pair": {m: float(np.median(col(f"{m}|seconds"))) for m in pc.METHODS},
           "seconds_mean": {m: float(np.mean(col(f"{m}|seconds"))) for m in pc.METHODS},
           "now_opt_seconds_median": {m: float(np.median(col(f"{m}|opt_seconds"))) for m in ("now_original", "now_metrical")},
           "realy_cap_hi_seconds_median": float(np.median(col("realy|cap_hi_seconds"))),
           "now_opt_vs_full": {m: cmp(col(f"{m}|opt"), col(m)) for m in ("now_original", "now_metrical")},
           "realy_cap_vs_cap_hi": cmp(col("realy"), col("realy|cap_hi")),
           "now_iterations_median": {m: float(np.median(col(f"{m}|iter"))) for m in ("now_original", "now_metrical")},
           "now_fevals_median": {m: float(np.median(col(f"{m}|fevals"))) for m in ("now_original", "now_metrical")},
           "now_scale": {"median": float(np.median(col("now_original|scale"))),
                         "p05": float(np.percentile(col("now_original|scale"), 5)),
                         "p95": float(np.percentile(col("now_original|scale"), 95))},
           "reflections": {m: int(np.nansum(col(f"{m}|reflection"))) for m in ("now_original", "now_metrical", "stirling")},
           "realy_hit_cap": int(np.nansum(col("realy|hit_cap"))), "realy_solves_median": float(np.median(col("realy|solves"))),
           "errors": {m: sorted({r[f"{m}|error"] for r in rows if r.get(f"{m}|error")})[:5] for m in pc.METHODS},
           "values_median": {m: float(np.nanmedian(col(m))) for m in pc.METHODS},
           "wall_s": time.time() - t0, "workers": workers, **pc.stamp()}
    pc.atomic_json(pc.EVID / "checks" / f"sample_{D['view']['name']}.json", out)
    blmm.atomic_savez(pc.WORK / "checks" / f"sample_{D['view']['name']}.npz",
                      **{k.replace("|", "__"): np.asarray(col(k), float) for k in rows[0] if not k.endswith("|error")})
    print(f"[pr-sample] {D['view']['name']}: {out['seconds_per_ordered_pair']} s per coppia ordinata (mediana); "
          f"NoW opt {out['now_opt_vs_full']}; REALY cap {out['realy_cap_vs_cap_hi']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("run", "merge", "sample"))
    p.add_argument("--view", default="")
    p.add_argument("--spec", type=Path, default=None)
    p.add_argument("--methods", default=",".join(pc.METHODS))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--now-opt", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--realy-cap-hi", type=int, default=3000)
    a = p.parse_args()
    vname = a.view or __import__("json").loads(a.spec.read_text())["name"]
    if a.step == "run":
        ms = tuple(m for m in a.methods.split(",") if m)
        if set(ms) - set(pc.METHODS):
            raise SystemExit(f"metodi sconosciuti: {set(ms) - set(pc.METHODS)}")
        run(vname, a.spec, ms, a.shard, a.nshards, a.workers, a.now_opt)
    elif a.step == "merge":
        merge(vname, a.spec)
    else:
        sample(vname, a.spec, a.n, a.workers, a.now_opt, a.realy_cap_hi)


if __name__ == "__main__":
    main()
