#!/usr/bin/env python3
"""Gromov-Wasserstein fra mesh senza corrispondenza ne' allineamento (POT, MIT): FR in mm, SR a centroid size.

    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py cost --workers 1                  (tempi per configurazione)
    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py prep --view pilot_hifi3d --workers 64
    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py pilot --workers 64                 (distanze del pilota)
    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py pilot-gw         (scelta: GT dei NON valutati)
    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py prep --view hifi3d --workers 64
    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py run --view hifi3d --shard 0 --nshards 2 --workers 64
    gp_run.sh aau/competitors/gw_percettivi/gp_gw.py merge --view hifi3d
    (gp.sbatch; un dominio nuovo: ``--view <nome> --spec <json>``, vedi gp_common.py)

Misura di una mesh (``prep``): la componente connessa piu' grande (per area; la frazione d'area tolta e' registrata),
in mm nel frame canonico (``gp_common.mm_mesh``), sottocampionata UNIFORME PER AREA con il farthest point sampling sui
vertici (partenza: il vertice piu' vicino al baricentro pesato; i primi n di un FPS sono un FPS di n, cosi' un solo
campionamento serve a ogni n della griglia), pesi = area della cella di Voronoi di ogni campione (somma delle aree
baricentriche dei vertici piu' vicini a quel campione), normalizzati a 1: la misura e' l'area della superficie spinta
sui campioni, quindi la densita' dei vertici (remesh, down8k, up60k) non entra. Matrici intra-mesh: ``euclid``
(distanze euclidee fra i campioni) o ``geo`` (geodetiche col metodo del calore, ``potpourri3d``, dalla mesh intera,
simmetrizzate). Modo ``fr``: mm; modo ``sr``: x CS_ref / CS_i (centroid size robusta, ``gp_common.size_of``). Le
matrici si dividono per L_d prima del solutore (scala dell'epsilon) e il risultato torna in mm:
d = 0.5 sqrt(GW_2^2) L_d, GW_2^2 = min_T sum (C1_ik - C2_jl)^2 T_ij T_kl (perdita quadratica di POT).

Solutori (``CONFIGS``): ``cg`` = ``ot.gromov.gromov_wasserstein`` (gradiente condizionale con EMD esatto: il GW
"esatto", minimo locale dall'accoppiamento prodotto p q^T); ``ent<eps>`` = ``ot.gromov.entropic_gromov_wasserstein``
(PGD, Sinkhorn di POT coi suoi default), valore = perdita GW del piano entropico, senza il termine d'entropia. La
configurazione (n, solutore, matrice) si sceglie sul pilota (``gp_eval.py pilot-gw``: 12 soggetti NON valutati per
vista, ``PILOT_TOPO_PAIRS`` coppie ordinate di topologie per coppia di soggetti, la loro GT, mai quella dei valutati)
fra quelle con la mediana dei secondi per soluzione entro ``BUDGET_S`` (regola fissata PRIMA dei punteggi: le 1,19
milioni di soluzioni della valutazione, 148.500 coppie x 4 viste + 105 di FaMoS x 2 modi, devono stare in ~330 ore-core),
e si legge da ``gw/pilot_choice.json``. Griglia fissata dai tempi misurati prima del pilota (job 1070585, 1070589, un
thread, coppie del pilota di HIFI3D): CG a n = 512 da 3 a 6 s per soluzione, entropico da 2 a 18 s, CG a n = 256 da 0,2
a 0,7 s: n = 512 e' fuori budget e non entra; l'entropico entra nel pilota come confronto descrittivo (fuori budget
anche a n = 128). Invarianza: GW e' invariante alle isometrie, quindi
anche alla RIFLESSIONE (un volto e il suo specchio hanno distanza 0): ambiguita' dichiarata, non corretta. Sul crop GW
confronta insiemi diversi della superficie (limite atteso, non un errore).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import gp_common as gc  # noqa: E402

GW = gc.OUT / "gw"
N_GRID = (128, 256)                                   # punti del sottocampionamento (pilota)
N_MAX = max(N_GRID)
SOLVERS = ("cg", "ent0.005")                          # entropico: epsilon sulle matrici / L_d
METRICS = ("euclid", "geo")
CONFIGS = [f"{m}|{n}|{s}" for m in METRICS for n in N_GRID for s in SOLVERS]
CG_ITERS, CG_TOL = 200, 1e-7                          # iterazioni esterne e tolleranza relativa del CG
ENT_ITERS, ENT_TOL = 200, 1e-7                        # iterazioni PGD e tolleranza dell'entropico
TIE = 0.01                                            # pilota: entro 0.01 dal migliore vince la piu' economica
BUDGET_S = 1.0                                        # pilota: mediana dei secondi per soluzione ammessa (un thread)
PILOT_TOPO_PAIRS = 6                                  # coppie ordinate di topologie per coppia di soggetti (rng 0)
COST_PAIRS = 12
_C: dict = {}


# ------------------------------------------------------------------------------------------ misura

def largest_component(F: np.ndarray, w: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    """Vertici della componente connessa piu' grande per area e la frazione d'area tolta."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    k, lab = connected_components(A, directed=False)
    if k == 1:
        return np.arange(n), 0.0
    area = np.bincount(lab, weights=w, minlength=k)
    best = int(area.argmax())
    return np.flatnonzero(lab == best), float(1.0 - area[best] / area.sum())


def fps(X: np.ndarray, n: int, start: int) -> np.ndarray:
    """Farthest point sampling (indici in ordine di scelta)."""
    idx = np.empty(min(n, len(X)), dtype=np.int64)
    idx[0] = start
    d = ((X - X[start]) ** 2).sum(1)
    for k in range(1, len(idx)):
        idx[k] = int(d.argmax())
        d = np.minimum(d, ((X - X[idx[k]]) ** 2).sum(1))
    return idx


def voronoi_weights(X: np.ndarray, w: np.ndarray, samples: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    _, a = cKDTree(X[samples]).query(X)
    p = np.bincount(a, weights=w, minlength=len(samples))
    return p / p.sum()


def _prep(task):
    """Campioni, pesi per ogni n, taglia, geodetiche fra i campioni (se ``geo``) di una mesh, con i tempi."""
    k, path, geo = task
    v = _C["view"]
    out = {"k": k, "fail": "", "t": np.full(3, np.nan)}
    try:
        t0 = time.perf_counter()
        V, F, w = gc.mm_mesh(v, path)
        cs, src = gc.size_of(v, path, V, F)
        keep, dropped = largest_component(F, w, len(V))
        if len(keep) < len(V):
            remap = -np.ones(len(V), dtype=np.int64)
            remap[keep] = np.arange(len(keep))
            F = remap[F]
            F = F[(F >= 0).all(1)]
            V, w = V[keep], w[keep]
        t1 = time.perf_counter()
        c = (w @ V) / w.sum()
        S = fps(V, N_MAX, int(((V - c) ** 2).sum(1).argmin()))
        P = {n: voronoi_weights(V, w, S[:n]) for n in N_GRID if n <= len(S)}
        t2 = time.perf_counter()
        G = None
        if geo:
            import potpourri3d as pp3d
            solver = pp3d.MeshHeatMethodDistanceSolver(np.ascontiguousarray(V), np.ascontiguousarray(F))
            G = np.stack([solver.compute_distance(int(s))[S] for s in S])
            G = 0.5 * (G + G.T)
            np.fill_diagonal(G, 0.0)
            if not np.isfinite(G).all():
                raise ValueError("geodetiche non finite")
        out.update(X=V[S].astype(np.float32), P=P, cs=cs, cs_src=src, dropped=dropped, n_vertices=len(V),
                   G=None if G is None else G.astype(np.float32))
        out["t"] = np.array([t1 - t0, t2 - t1, time.perf_counter() - t2])
    except Exception as exc:  # noqa: BLE001  (la mesh resta NaN, contata)
        out["fail"] = f"{type(exc).__name__}: {exc}"
    return out


def prep_path(vname: str) -> Path:
    return GW / "prep" / f"{vname}.npz"


def run_prep(vname: str, spec: Path | None, workers: int, geo: bool) -> None:
    out = prep_path(vname)
    if out.exists():
        print(f"[gp-gw] {out}: gia' presente, salto", flush=True)
        return
    v = gc.pilot_view(vname.removeprefix("pilot_")) if vname.startswith("pilot_") else gc.view(vname, spec)
    _C["view"] = v
    t0 = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_prep, [(k, p, geo) for k, (_, _, p) in enumerate(v["items"])], chunksize=1)
    n = len(res)
    fails = [f"{v['items'][r['k']][0]}|{v['items'][r['k']][1]}|{r['fail']}" for r in res if r["fail"]]
    X = np.full((n, N_MAX, 3), np.nan, np.float32)
    arr = {f"p{m}": np.full((n, m), np.nan) for m in N_GRID}
    G = np.full((n, N_MAX, N_MAX), np.nan, np.float32) if geo else None
    cs, dropped, nv = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, np.int64)
    for r in res:
        if r["fail"]:
            continue
        k = r["k"]
        m = len(r["X"])
        X[k, :m] = r["X"]
        for q, p in r["P"].items():
            arr[f"p{q}"][k] = p
        cs[k], dropped[k], nv[k] = r["cs"], r["dropped"], r["n_vertices"]
        if geo:
            G[k, :m, :m] = r["G"]
    t = np.stack([r["t"] for r in res])
    extra = {"G": G} if geo else {}
    gc.blmm.atomic_savez(out, X=X, cs=cs, dropped=dropped, n_vertices=nv, t_stage=t,
                         stages=np.asarray(["lettura+taglia", "fps+pesi", "geodetiche"]),
                         subjects=np.asarray([x for x, _, _ in v["items"]]),
                         topologies=np.asarray([y for _, y, _ in v["items"]]),
                         cs_src=np.asarray([r.get("cs_src", "") for r in res]), failed=np.asarray(fails, dtype="U300"),
                         L=v["L"], cs_ref=v["cs_ref"], **arr, **extra)
    print(f"[gp-gw] prep {vname}: {n} mesh in {time.time() - t0:.0f}s, fallite {len(fails)} {fails[:3]}; s/mesh mediana "
          f"lettura+taglia {np.nanmedian(t[:, 0]):.2f}, fps {np.nanmedian(t[:, 1]):.2f}, geodetiche "
          f"{np.nanmedian(t[:, 2]) if geo else float('nan'):.2f}; area tolta max {np.nanmax(dropped):.2e}; "
          f"taglia {dict(zip(*np.unique([r.get('cs_src', '') for r in res], return_counts=True)))}", flush=True)


def load_prep(vname: str) -> dict:
    with np.load(prep_path(vname)) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# --------------------------------------------------------------------------------------------- GW

def matrix(P: dict, k: int, metric: str, n: int, mode: str) -> np.ndarray:
    """Matrice intra-mesh (n, n) / L_d nel modo (``sr``: x CS_ref / CS_k)."""
    if metric == "geo":
        C = P["G"][k, :n, :n].astype(np.float64)
    else:
        X = P["X"][k, :n].astype(np.float64)
        C = np.sqrt(np.clip(((X[:, None] - X[None]) ** 2).sum(-1), 0.0, None))
    f = 1.0 / float(P["L"])
    if mode == "sr":
        f *= float(P["cs_ref"]) / float(P["cs"][k])
    return C * f


def gw_value(C1: np.ndarray, C2: np.ndarray, p: np.ndarray, q: np.ndarray, solver: str) -> float:
    """GW_2^2 (perdita quadratica) fra (C1, p) e (C2, q) col solutore."""
    import ot
    if solver == "cg":
        _, log = ot.gromov.gromov_wasserstein(C1, C2, p, q, "square_loss", log=True, max_iter=CG_ITERS,
                                              tol_rel=CG_TOL, tol_abs=0.0)
    else:
        _, log = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, "square_loss", epsilon=float(solver[3:]),
                                                       max_iter=ENT_ITERS, tol=ENT_TOL, log=True)
    return float(log["gw_dist"])


def _pairs(task):
    """Distanze GW (fr, sr) e secondi per le coppie del blocco, nelle configurazioni ``_C["configs"]``."""
    i, j = task
    P, configs = _C["prep"], _C["configs"]
    L = float(P["L"])
    out = np.full((len(i), len(configs), 2), np.nan)
    sec = np.full((len(i), len(configs), 2), np.nan)
    for r, (a, b) in enumerate(zip(i, j)):
        for c, cfg in enumerate(configs):
            metric, n, solver = cfg.split("|")
            n = int(n)
            pa, pb = P[f"p{n}"][a], P[f"p{n}"][b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            for m, mode in enumerate(gc.MODES):
                t0 = time.perf_counter()
                try:
                    g = gw_value(matrix(P, a, metric, n, mode), matrix(P, b, metric, n, mode), pa, pb, solver)
                    out[r, c, m] = 0.5 * np.sqrt(max(g, 0.0)) * L
                except Exception:  # noqa: BLE001  (la coppia resta NaN, contata)
                    pass
                sec[r, c, m] = time.perf_counter() - t0
    return out, sec


def solve_pairs(P: dict, i: np.ndarray, j: np.ndarray, configs: list, workers: int, block: int = 16):
    _C.update(prep=P, configs=configs)
    blocks = [(i[s:s + block], j[s:s + block]) for s in range(0, len(i), block)]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_pairs, blocks, chunksize=1)
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def items_of(P: dict) -> list:
    return [(str(s), str(t), None) for s, t in zip(P["subjects"], P["topologies"])]


# ------------------------------------------------------------------------------------------ passi

def run_cost(workers: int) -> None:
    """Tempi per coppia di ogni configurazione sulle prime ``COST_PAIRS`` coppie del pilota di HIFI3D (un thread)."""
    P = load_prep("pilot_hifi3d")
    i, j = gc.pairs(items_of(P))
    sel = np.random.default_rng(0).choice(len(i), COST_PAIRS, replace=False)
    D, S = solve_pairs(P, i[sel], j[sel], CONFIGS, workers, block=1)
    res = {cfg: {"s_per_solve_median": float(np.nanmedian(S[:, c])), "s_per_solve_max": float(np.nanmax(S[:, c])),
                 "nan": int(np.isnan(D[:, c]).sum())} for c, cfg in enumerate(CONFIGS)}
    res["prep_s_per_mesh_median"] = {k: float(v) for k, v in zip(P["stages"], np.nanmedian(P["t_stage"], axis=0))}
    res.update(n_pairs=COST_PAIRS, workers=workers, **gc.stamp())
    gc.atomic_json(GW / "cost_configs.json", res)
    for cfg in CONFIGS:
        print(f"[gp-gw] costo {cfg}: {res[cfg]['s_per_solve_median']:.3f} s per soluzione (max "
              f"{res[cfg]['s_per_solve_max']:.3f}), NaN {res[cfg]['nan']}", flush=True)


def pilot_pairs(items: list) -> tuple[np.ndarray, np.ndarray]:
    """``PILOT_TOPO_PAIRS`` coppie di ``gp_common.pairs`` a caso (``rng(0)``) per ogni coppia di soggetti."""
    i, j = gc.pairs(items)
    s = np.asarray([x for x, _, _ in items])
    rng = np.random.default_rng(0)
    keep = []
    for key in sorted(set(zip(s[i], s[j]))):
        rows = np.flatnonzero((s[i] == key[0]) & (s[j] == key[1]))
        keep += sorted(rng.choice(rows, min(PILOT_TOPO_PAIRS, len(rows)), replace=False).tolist())
    keep = np.asarray(keep)
    return i[keep], j[keep]


def run_pilot(workers: int, configs: list) -> None:
    """Distanze del pilota (``pilot_pairs`` sui 12 soggetti NON valutati) in ogni configurazione."""
    for view in gc.PILOT_VIEWS:
        out = GW / "pilot" / f"{view}.npz"
        if out.exists():
            print(f"[gp-gw] {out}: gia' presente, salto", flush=True)
            continue
        t0 = time.time()
        P = load_prep(f"pilot_{view}")
        i, j = pilot_pairs(items_of(P))
        D, S = solve_pairs(P, i, j, configs, workers, block=2)
        gc.blmm.atomic_savez(out, pair_i=i, pair_j=j, D=D, seconds=S, configs=np.asarray(configs),
                             subjects=P["subjects"], topologies=P["topologies"], modes=np.asarray(gc.MODES))
        print(f"[gp-gw] pilota {view}: {len(i)} coppie x {len(configs)} configurazioni in {time.time() - t0:.0f}s, "
              f"NaN {int(np.isnan(D).sum())}", flush=True)


def choice() -> dict:
    return json.loads((GW / "pilot_choice.json").read_text())


def run_view(vname: str, shard: int, nshards: int, workers: int) -> None:
    ch = choice()
    out = GW / "parts" / f"{vname}_{shard}of{nshards}.npz"
    if out.exists():
        print(f"[gp-gw] {out}: gia' presente, salto", flush=True)
        return
    t0 = time.time()
    P = load_prep(vname)
    i, j = gc.pairs(items_of(P))
    i, j = i[shard::nshards], j[shard::nshards]
    D, S = solve_pairs(P, i, j, [ch["config"]], workers)
    gc.blmm.atomic_savez(out, pair_i=i, pair_j=j, D=D[:, 0], seconds=S[:, 0], config=ch["config"], **gc.stamp())
    print(f"[gp-gw] {vname} parte {shard}/{nshards}: {len(i)} coppie in {time.time() - t0:.0f}s ({workers} processi), "
          f"s per soluzione mediana {np.nanmedian(S):.3f}, NaN {int(np.isnan(D).sum())}", flush=True)


def run_merge(vname: str, spec: Path | None) -> None:
    ch = choice()
    P = load_prep(vname)
    v = gc.pilot_view(vname.removeprefix("pilot_")) if vname.startswith("pilot_") else gc.view(vname, spec)
    if [(s, t) for s, t, _ in v["items"]] != [(str(s), str(t)) for s, t in zip(P["subjects"], P["topologies"])]:
        raise SystemExit(f"{vname}: mesh della preparazione diverse dalla vista")
    parts = sorted((GW / "parts").glob(f"{vname}_*of*.npz"))
    n = len(v["items"])
    D = {m: np.full((n, n), np.nan) for m in gc.MODES}
    secs, cfgs, seen = [], set(), 0
    for p in parts:
        with np.load(p) as z:
            i, j, d = np.asarray(z["pair_i"]), np.asarray(z["pair_j"]), np.asarray(z["D"])
            secs.append(np.asarray(z["seconds"]))
            cfgs.add(str(z["config"]))
        for m, mode in enumerate(gc.MODES):
            D[mode][i, j] = D[mode][j, i] = d[:, m]
        seen += len(i)
    i, j = gc.pairs(v["items"])
    if cfgs != {ch["config"]} or seen != len(i):
        raise SystemExit(f"{vname}: parti {len(parts)} con configurazioni {cfgs}, {seen} coppie su {len(i)}")
    for mode in gc.MODES:
        np.fill_diagonal(D[mode], 0.0)
    S = np.concatenate(secs)
    info = {"method": "gw", "config": ch["config"], "pilot_choice": ch, "n_meshes": n, "n_pairs": int(len(i)),
            "n_pairs_nan": {m: int(np.isnan(D[m][i, j]).sum()) for m in gc.MODES},
            "failed_meshes": [str(x) for x in P["failed"]],
            "s_per_solve": {m: {"median": float(np.nanmedian(S[:, k])), "p95": float(np.nanpercentile(S[:, k], 95)),
                                "total_h": float(np.nansum(S[:, k]) / 3600)} for k, m in enumerate(gc.MODES)},
            "prep_s_per_mesh_median": {str(k): float(x) for k, x in zip(P["stages"], np.nanmedian(P["t_stage"], 0))},
            "area_dropped_max": float(np.nanmax(P["dropped"])),
            "cs_source": {str(k): int(c) for k, c in zip(*np.unique(P["cs_src"], return_counts=True))},
            "units": "mm (FR) e mm alla taglia di riferimento CS_ref (SR): 0.5 sqrt(GW_2^2)", **gc.stamp()}
    gc.write_view("gw", v, D, {"cs": P["cs"]}, info)
    print(f"[gp-gw] {vname}: {len(i)} coppie unite da {len(parts)} parti, NaN {info['n_pairs_nan']}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("prep", "cost", "pilot", "run", "merge"))
    p.add_argument("--view")
    p.add_argument("--spec", type=Path, help="vista nuova (gp_common.view)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--configs", default=",".join(CONFIGS), help="pilot: configurazioni (default: tutta la griglia)")
    p.add_argument("--geo", choices=("auto", "yes", "no"), default="auto",
                   help="prep: geodetiche (auto: nel pilota sempre, altrove se la scelta e' geo)")
    a = p.parse_args()
    if a.step == "prep":
        geo = a.geo == "yes" or (a.geo == "auto" and (a.view.startswith("pilot_") or
                                                      choice()["config"].startswith("geo")))
        run_prep(a.view, a.spec, a.workers, geo)
    elif a.step == "cost":
        run_cost(a.workers)
    elif a.step == "pilot":
        run_pilot(a.workers, a.configs.split(","))
    elif a.step == "run":
        run_view(a.view, a.shard, a.nshards, a.workers)
    else:
        run_merge(a.view, a.spec)


if __name__ == "__main__":
    main()
