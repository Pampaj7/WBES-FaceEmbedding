#!/usr/bin/env python3
"""NICP per le corrispondenze: la stessa deformazione del NICP per coppia della matrice, letta come CORRISPONDENZA (mesh
rigida non deformata contro il punto corrispondente, la convenzione di REALY) invece che come residuo dopo la
deformazione. Regole in ``aau/runs/evidence/nicp_corr/REGOLE.md`` (committate prima del codice e dei numeri).

    aau/outlineB/run_o3d.sh   aau/nicp_corr/nc.py run --view hifi3d --mode mm --workers 64
    v3_work/unified_gt/run.sh aau/nicp_corr/nc.py merge --view hifi3d --mode mm
    aau/outlineB/run_o3d.sh   aau/nicp_corr/nc.py check-node --view hifi3d --mode mm --topo "noisy>up60k,remesh>up60k"
    v3_work/unified_gt/run.sh aau/nicp_corr/nc.py check --view hifi3d                 (nessuna GT)
    v3_work/unified_gt/run.sh aau/nicp_corr/nc.py eval --workers 64                   (LEGGE LE GT di test)
    (nc.sbatch; ``--limit N``: prova su N coppie, uscite in ``<WORK>/pilot``, mai valutate)

``run``: ``blmm.pair_metrics`` passo ``nicp_corr`` (``nicp_p2tri``, ``nicp_corr``, ``nicp_corr_xy``, ``nicp_corr_yx``,
``nicp_p2tri_yx``; definizioni nel docstring di ``pair_metrics``) sulle coppie senza crop del sottocampione del rumore
landmark (``ri.view_data``: 8%, seme 20261013; X = soggetto di indice minore, seme della coppia di ``blmm_pairs``), nel
modo ``mm`` o ``cs``, x L_d. Parti di ``CHUNK`` coppie in ``<WORK>/parts/<vista>/<modo>/c<k>.npz``; una ripartenza salta
quelle salvate. ``merge``: ``<WORK>/dist/<vista>/<modo>.npz`` (``i``, ``j``, ``D_<lettura>``) e
``<EVID>/controls/run_<vista>_<modo>.json``.

``check``: ``nicp_p2tri`` contro ``<modo>_nicp_p2tri`` della matrice (``TOL`` mm, NaN sulle stesse coppie), per coppia
ordinata di topologie; quelle fuori tolleranza passano solo con un ``check-node`` riuscito (il passo ``nicp`` della
matrice, cioe' ``blmm.pair_metrics`` com'e', ricalcolato su un nodo con la CPU della matrice per TUTTE le coppie del
sottocampione di quelle coppie di topologie; i valori non entrano nella valutazione). Strutturale: ``nicp_corr_xy`` >=
ICP + Chamfer della matrice - 1e-9 mm. ``eval`` rifiuta di partire se un controllo non passa.

``eval``: Spearman pesato con GT FR e SR (``bp_paired_e3.replicas``, 1000 repliche per soggetto col seme della T1,
``pr_eval.t1_seed``) sulle righe di all_cross del sottocampione, soggetti diversi, UNA maschera per vista (tutte le
colonne finite). Colonne per modo: le letture ricalcolate (piu' ``nicp_p2tri_sym`` = media dei due versi del residuo),
ICP + Chamfer e NICP su template della matrice (``fm_eval.geo_columns``); il nostro (``fact_paired.model_columns``, s1234
e s2345, d_F non cal. e d_P); REALY (plug-in ``realy``). Delta appaiati (``DELTAS``), verdetto della sez. 4 delle regole
sulle coppie primarie (FR, mm) e (SR, cs). Uscite in ``aau/runs/evidence/nicp_corr/``: ``spearman.csv``, ``deltas.csv``,
``controls.json``, ``results.md`` (generato; il testo dopo ``MARK`` si conserva).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
AAU = THIS.parent
REPO = AAU.parent
for _p in (AAU / "robustezza_iperparametri", AAU / "full_matrix"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ri  # noqa: E402  (view_data, maxdiff, ci, f3; e i percorsi di pr_common, blmm, baselines_param)

pc, blmm = ri.pc, ri.blmm
VIEWS = ri.SYNTH                                              # hifi3d, facescape
MODES = ("mm", "cs")
STEP = "nicp_corr"
WORK = AAU / "runs" / "nicp_corr"                             # parti e distanze (npz fuori da git)
EVID = AAU / "runs" / "evidence" / "nicp_corr"
N_BOOT = 1000
CHUNK = 32
TOL = 1e-6                                                    # mm, REGOLE.md sez. 5
MARK = "<!-- da qui in giu': scritto a mano, conservato da nc.py results -->"
PRIMARY = (("fr", "mm"), ("sr", "cs"))
OURS = {"fr": ("factorized_s1234|form", "factorized_s2345|form"), "sr": ("factorized_s1234|shape", "factorized_s2345|shape")}
READS = ("nicp_p2tri", "nicp_corr", "nicp_corr_xy", "nicp_p2tri_sym")
LABEL = {"nicp_p2tri": "NICP per coppia, residuo P2Tri (matrice)",
         "nicp_corr": "NICP per le corrispondenze (media dei due versi)",
         "nicp_corr_xy": "NICP per le corrispondenze, un verso (quello della matrice)",
         "nicp_p2tri_sym": "NICP per coppia, residuo P2Tri, media dei due versi",
         "rigid_icp_chamfer": "ICP + Chamfer", "nicp_template": "NICP su template",
         "factorized_s1234|form": "nostro s1234, d_F non cal.", "factorized_s2345|form": "nostro s2345, d_F non cal.",
         "factorized_s1234|shape": "nostro s1234, d_P", "factorized_s2345|shape": "nostro s2345, d_P",
         "realy": "REALY (protocollo, landmark esatti)"}
_ST: dict = {}


def label(c: str) -> str:
    m, _, r = c.partition("_")
    return f"{LABEL[r]} ({m})" if m in MODES else LABEL[c]


def topo_keys(V: dict) -> np.ndarray:
    """Coppia ordinata di topologie della matrice per ogni coppia: (topologia di X) > (topologia di Y)."""
    return np.asarray([f"{V['items'][a][1]}>{V['items'][b][1]}" for a, b in zip(V["i"], V["j"])])


def load_view(vname: str, mode: str) -> dict:
    V = ri.view_data(vname)
    prm = blmm.params()
    V.update(mode=mode, unit=blmm.unit(vname, mode, prm),
             coords=[blmm.work_coords(vname, mode, p, prm) for _, _, p in V["items"]])
    return V


# ------------------------------------------------------------------------------------------ run

def _task(k: int) -> int:
    out_p = _ST["ckdir"] / f"c{k}.npz"
    if out_p.exists():
        return k
    t0 = time.perf_counter()
    V, pos = _ST["V"], _ST["todo_pos"][k]
    mets = blmm.METRICS[STEP]
    vals = {m: np.full(len(pos), np.nan) for m in mets}
    secs, errs = np.zeros(len(pos)), [""] * len(pos)
    for q, r in enumerate(pos):
        a, b, seed = int(V["i"][r]), int(V["j"][r]), int(V["seed"][r])
        t1 = time.perf_counter()
        res = blmm.pair_metrics(V["coords"][a], V["coords"][b], seed, V["mode"], STEP)
        for m in mets:
            vals[m][q] = res[m] * V["unit"]                   # come blmm_pairs: valore di lavoro x L_d
        errs[q], secs[q] = res["error"], time.perf_counter() - t1
    blmm.atomic_savez(out_p, pos=np.asarray(pos), task_s=time.perf_counter() - t0, job=pc.stamp()["job"],
                      host=os.uname().nodename.split(".")[0], sec=secs, err=np.asarray(errs),
                      **{f"val:{m}": vals[m] for m in mets})
    return k


def run(vname: str, mode: str, workers: int, limit: int) -> None:
    t0 = time.time()
    V = load_view(vname, mode)
    ck = WORK / ("pilot" if limit else "parts") / vname / mode
    ck.mkdir(parents=True, exist_ok=True)
    npairs = min(limit, len(V["i"])) if limit else len(V["i"])
    chunk = 1 if limit else CHUNK                         # prova: una coppia per parte, tutti i worker occupati
    _ST.update(V=V, ckdir=ck, todo_pos=[list(range(s, min(s + chunk, npairs))) for s in range(0, npairs, chunk)])
    todo = list(range(len(_ST["todo_pos"])))
    print(f"[nc] run {vname} {mode}: {npairs}/{len(V['i'])} coppie, {len(todo)} parti, "
          f"{time.time() - t0:.0f}s di caricamento", flush=True)
    tl = time.time()
    with mp.get_context("fork").Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_task, todo, chunksize=1)):
            if (k + 1) % max(1, len(todo) // 10) == 0:
                print(f"[nc] run {vname} {mode}: {k + 1}/{len(todo)} parti, {time.time() - tl:.0f}s", flush=True)
    if limit:                                             # prova: solo tempi, NaN e confronto con la matrice
        D = collect(ck, npairs, chunk)
        ref = matrix_ref(vname, mode, V)["nicp_p2tri"][:npairs]
        print(f"[nc] prova {vname} {mode}: s/coppia mediana {np.median(D['_sec']):.2f}, NaN "
              f"{ {m: int(np.isnan(D[m]).sum()) for m in blmm.METRICS[STEP]} }, nicp_p2tri contro la matrice "
              f"{json.dumps(ri.maxdiff(D['nicp_p2tri'], ref))}, host {sorted(set(D['_host']))}", flush=True)
    print(f"[nc] run {vname} {mode}: fatto in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ merge

def collect(ck: Path, npairs: int, chunk: int = CHUNK) -> dict:
    nch = (npairs + chunk - 1) // chunk
    miss = [k for k in range(nch) if not (ck / f"c{k}.npz").exists()]
    if miss:
        raise SystemExit(f"{ck}: {len(miss)} parti mancanti (es. {miss[:5]})")
    D = {m: np.full(npairs, np.nan) for m in blmm.METRICS[STEP]}
    D.update(_sec=np.full(npairs, np.nan), _host=np.full(npairs, "", dtype="U32"), _err=np.full(npairs, "", dtype="U300"))
    D["_task_s"], D["_jobs"] = 0.0, set()
    for k in range(nch):
        with np.load(ck / f"c{k}.npz") as z:
            pos = z["pos"]
            if not np.array_equal(pos, np.arange(k * chunk, min((k + 1) * chunk, npairs))):
                raise SystemExit(f"{ck} c{k}: coppie diverse dalla parte")
            for m in blmm.METRICS[STEP]:
                D[m][pos] = z[f"val:{m}"]
            D["_sec"][pos], D["_host"][pos], D["_err"][pos] = z["sec"], str(z["host"]), z["err"]
            D["_task_s"] += float(z["task_s"])
            D["_jobs"].add(str(z["job"]))
    return D


def dist_path(vname: str, mode: str) -> Path:
    return WORK / "dist" / vname / f"{mode}.npz"


def merge(vname: str, mode: str) -> None:
    V = ri.view_data(vname)
    D = collect(WORK / "parts" / vname / mode, len(V["i"]))
    mets = blmm.METRICS[STEP]
    blmm.atomic_savez(dist_path(vname, mode), i=V["i"], j=V["j"], host=D["_host"],
                      names=np.asarray([f"{s}|{t}" for s, t, _ in V["items"]]), **{f"D_{m}": D[m] for m in mets})
    errs = D["_err"][D["_err"] != ""]
    info = {"view": vname, "mode": mode, "pairs": int(len(V["i"])), "core_hours": D["_task_s"] / 3600,
            "jobs": sorted(D["_jobs"]), "hosts": {h: int((D["_host"] == h).sum()) for h in sorted(set(D["_host"]))},
            "nan": {m: int(np.isnan(D[m]).sum()) for m in mets}, "errors": int(len(errs)),
            "error_examples": sorted(set(errs.tolist()))[:5], "median_mm": {m: float(np.nanmedian(D[m])) for m in mets},
            "seconds_per_pair_median": float(np.median(D["_sec"])), **pc.stamp()}
    pc.atomic_json(EVID / "controls" / f"run_{vname}_{mode}.json", info)
    print(f"[nc] merge {vname} {mode}: {json.dumps({k: info[k] for k in ('pairs', 'nan', 'hosts', 'core_hours')})}",
          flush=True)


def load_dist(vname: str, mode: str, V: dict) -> dict:
    with np.load(dist_path(vname, mode)) as z:
        if not (np.array_equal(z["i"], V["i"]) and np.array_equal(z["j"], V["j"])):
            raise SystemExit(f"{vname} {mode}: coppie diverse dalla vista")
        out = {m: np.asarray(z[f"D_{m}"], np.float64) for m in blmm.METRICS[STEP]}
    out["nicp_p2tri_sym"] = 0.5 * (out["nicp_p2tri"] + out["nicp_p2tri_yx"])
    return out


# ------------------------------------------------------------------------------------------ check

def matrix_ref(vname: str, mode: str, V: dict) -> dict:
    """``nicp_p2tri`` e ``rigid_icp_chamfer`` della matrice (modo ``mode``) sulle coppie della vista."""
    import bp_paired as bpp
    idx = bpp.be.zes.Index(blmm.subjects(vname))
    pi = np.asarray([idx.pos[(s, t)] for s, t, _ in V["items"]])
    root = blmm.view_root(vname) / mode
    return {m: bpp.be.zes.facebench_distances(root, m, idx)[pi[V["i"]], pi[V["j"]]]
            for m in ("nicp_p2tri", "rigid_icp_chamfer")}


def _node_task(r: int) -> float:
    """Il passo ``nicp`` della matrice com'e' (``blmm.pair_metrics``) su una coppia, x L_d."""
    V = _ST["V"]
    a, b, seed = int(V["i"][r]), int(V["j"][r]), int(V["seed"][r])
    return blmm.pair_metrics(V["coords"][a], V["coords"][b], seed, V["mode"], "nicp")["nicp_p2tri"] * V["unit"]


def check_node(vname: str, mode: str, topo: str, workers: int) -> None:
    """``nicp_p2tri`` della matrice ricalcolato su QUESTO nodo per tutte le coppie delle coppie di topologie ``topo``
    contro la matrice (REGOLE.md sez. 5); ``auto`` = quelle in cui il calcolo principale e' fuori tolleranza. I valori
    non entrano nella valutazione. Nessuna GT."""
    t0 = time.time()
    host = os.uname().nodename.split(".")[0]
    V = load_view(vname, mode)
    T = topo_keys(V)
    if topo == "auto":
        x, y = load_dist(vname, mode, V)["nicp_p2tri"], matrix_ref(vname, mode, V)["nicp_p2tri"]
        bad = (np.isfinite(x) != np.isfinite(y)) | (np.abs(x - y) > TOL)
        topo = ",".join(sorted(set(T[bad])))
    want = [t for t in topo.split(",") if t]
    if not want:
        print(f"[nc] check-node {vname} {mode}: nessuna coppia di topologie da controllare", flush=True)
        return
    sel = np.flatnonzero(np.isin(T, want))
    _ST["V"] = V
    with mp.get_context("fork").Pool(workers) as pool:
        here = np.asarray(pool.map(_node_task, sel.tolist(), chunksize=2))
    ref = matrix_ref(vname, mode, V)["nicp_p2tri"][sel]
    by = {}
    for t in want:
        k = T[sel] == t
        d = ri.maxdiff(here[k], ref[k])
        by[t] = {**d, "pairs": int(k.sum()),
                 "ok": bool(k.any() and d["nan_same_pairs"] and (d["max_abs"] is None or d["max_abs"] <= TOL))}
    out = {"view": vname, "mode": mode, "node": host, "tol": TOL, "by_topo": by, "seconds": time.time() - t0,
           **pc.stamp()}
    pc.atomic_json(EVID / "controls" / f"check_node_{vname}_{mode}_{host}.json", out)
    print(f"[nc] check-node {vname} {mode} {host}: { {t: (x['max_abs'], x['ok']) for t, x in by.items()} }", flush=True)


def check(vname: str) -> None:
    """Riproduzione di ``nicp_p2tri`` (per coppia ordinata di topologie, coi controlli di nodo) e controllo strutturale.
    Nessuna GT."""
    V = ri.view_data(vname)
    T = topo_keys(V)
    out = {"view": vname, "pairs": int(len(V["i"])), "tol": TOL, "modes": {}}
    for mode in MODES:
        D, ref = load_dist(vname, mode, V), matrix_ref(vname, mode, V)
        c = {"all": ri.maxdiff(D["nicp_p2tri"], ref["nicp_p2tri"]), "by_topo": {}}
        nodes = [json.loads(p.read_text()) for p in sorted((EVID / "controls").glob(f"check_node_{vname}_{mode}_*.json"))]
        for t in sorted(set(T)):
            k = T == t
            d = ri.maxdiff(D["nicp_p2tri"][k], ref["nicp_p2tri"][k])
            here = bool(d["nan_same_pairs"] and (d["max_abs"] is None or d["max_abs"] <= TOL))
            ok_nodes = [n["node"] for n in nodes if t in n["by_topo"] and n["by_topo"][t]["ok"]
                        and n["by_topo"][t]["pairs"] == int(k.sum())]
            c["by_topo"][t] = {**d, "pairs": int(k.sum()), "ok_here": here, "ok_nodes": ok_nodes,
                               "ok": bool(here or ok_nodes)}
        s = D["nicp_corr_xy"] - ref["rigid_icp_chamfer"]
        f = np.isfinite(s)
        c["structural"] = {"n": int(f.sum()), "min_corr_xy_minus_icp": float(s[f].min()),
                           "ok": bool(f.any() and s[f].min() >= -1e-9)}
        c["ok"] = bool(all(x["ok"] for x in c["by_topo"].values()) and c["structural"]["ok"])
        out["modes"][mode] = c
        bad = {t: x["max_abs"] for t, x in c["by_topo"].items() if not x["ok"]}
        print(f"[nc] check {vname} {mode}: tutte {json.dumps(c['all'])}; coppie di topologie fuori tolleranza senza "
              f"controllo di nodo {bad}; strutturale {json.dumps(c['structural'])}", flush=True)
    out["ok"] = all(c["ok"] for c in out["modes"].values())
    out.update(pc.stamp())
    pc.atomic_json(EVID / "controls" / f"check_{vname}.json", out)


# ------------------------------------------------------------------------------------------ eval

def columns() -> list:
    cols = [f"{m}_{r}" for m in MODES for r in READS + ("rigid_icp_chamfer", "nicp_template")]
    return cols + [c for g in OURS.values() for c in g] + ["realy"]


def deltas_of() -> list:
    """(gt, modo, a, b, tipo, primaria): REGOLE.md sez. 4."""
    out = []
    for gt in ("fr", "sr"):
        for m in MODES:
            prim = (gt, m) in PRIMARY
            c = f"{m}_nicp_corr"
            out += [(gt, m, c, f"{m}_nicp_p2tri", "D1: corrispondenze - residuo", prim),
                    (gt, m, c, f"{m}_rigid_icp_chamfer", "D2: corrispondenze - ICP", prim)]
            out += [(gt, m, c, o, "corrispondenze - nostro", False) for o in OURS[gt]]
            out += [(gt, m, c, f"{m}_nicp_template", "corrispondenze - NICP su template", False),
                    (gt, m, c, "realy", "corrispondenze - REALY", False),
                    (gt, m, f"{m}_nicp_corr_xy", f"{m}_nicp_p2tri", "solo la lettura, un verso", False),
                    (gt, m, c, f"{m}_nicp_p2tri_sym", "solo la lettura, due versi", False)]
    return out


def verdict(d1: tuple, d2: tuple) -> str:
    """REGOLE.md sez. 4, da (lo, hi) di D1 e D2."""
    if d1[0] <= 0:
        return "cancella ancora l'identita'"
    if d2[1] < 0:
        return "recupera in parte, sotto l'ICP"
    if d2[0] > 0:
        return "sopra l'ICP"
    return "recupera, pari all'ICP"


def synth_view(vname: str, workers: int) -> tuple[dict, dict]:
    """{(gt, colonna): Spearman (1 + repliche)}, controlli."""
    import bp_paired as bpp
    import bp_paired_e2 as bpe2
    import bp_paired_e3 as bpe3
    import fm_eval
    import pr_eval as pe
    from scipy.stats import spearmanr
    fp = bpp.fp
    V = ri.view_data(vname)
    df, idx, _ = bpe2.all_cross_rows(vname)
    seed = pe.t1_seed(vname)
    keys = list(idx.keys)
    n = len(keys)
    pi = np.asarray([idx.pos[(s, t)] for s, t, _ in V["items"]])
    a, b = pi[V["i"]], pi[V["j"]]
    D = {}
    for mode in MODES:
        for r, x in load_dist(vname, mode, V).items():
            if r in READS:
                M = np.full((n, n), np.nan)
                M[a, b] = M[b, a] = x
                D[f"{mode}_{r}"] = M
    geo = fm_eval.geo_columns(vname, idx)
    D.update({c: geo[c] for c in columns() if c in geo and c not in D})
    arms = fp.model_columns(vname, idx)
    D.update({c: arms[c] for g in OURS.values() for c in g})
    D["realy"] = pe.plugin_matrix("realy", vname, keys)
    S = np.zeros((n, n))
    S[a, b] = S[b, a] = 1.0
    D["_sub"] = S
    cols = columns()
    if sorted(c for c in D if c != "_sub") != sorted(cols):
        raise SystemExit(f"{vname}: colonne {sorted(D)} diverse da {cols}")
    df = bpe3.be.add_columns(df, D, idx)
    X = {m: df[m].to_numpy(np.float64) for m in cols + ["oracle_size"]}
    X.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in ("fr", "sr")})
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    sub = (df["_sub"].to_numpy() > 0) & (ta != tb) & (ta != "crop") & (tb != "crop")
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: k for k, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(N_BOOT)]
    mask = sub & (sa != sb) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    R = bpe3.replicas(X, sa, sb, mask, cols, counts, workers)
    out = {(g, c): np.asarray(R[(g, "rho", c)], np.float64) for g in ("fr", "sr") for c in cols}
    plain = {f"{m}_nicp_corr ~ {m}_{o}": float(spearmanr(X[f"{m}_nicp_corr"][mask], X[f"{m}_{o}"][mask])[0])
             for m in MODES for o in ("nicp_p2tri", "rigid_icp_chamfer")}
    info = {"seed": seed, "rows_subsample_nocrop": int(sub.sum()), "rows_mask": int(mask.sum()),
            "pairs_subsample_nocrop": int(len(V["i"])), "subjects": int(len(subjects)),
            "rows_nan_in_subsample_by_column": {c: int((~np.isfinite(X[c]) & sub).sum()) for c in cols},
            "spearman_plain_on_mask": plain}
    print(f"[nc] eval {vname}: {int(mask.sum())}/{int(sub.sum())} righe, seme {seed}", flush=True)
    return out, info


def evaluate(workers: int) -> None:
    import pandas as pd
    t0 = time.time()
    for v in VIEWS:
        p = EVID / "controls" / f"check_{v}.json"
        if not p.exists() or not json.loads(p.read_text())["ok"]:
            raise SystemExit(f"{v}: controllo di riproduzione assente o non passato ({p})")
    S, DL, info = [], [], {}
    for v in VIEWS:
        R, info[v] = synth_view(v, workers)
        for (g, c), x in R.items():
            lo, hi = ri.ci(x)
            S.append({"dominio": v, "gt": g, "colonna": c, "etichetta": label(c), "rho": float(x[0]), "ic_lo": lo,
                      "ic_hi": hi, "n_righe": info[v]["rows_mask"], "seme": info[v]["seed"]})
        for g, m, ca, cb, tipo, prim in deltas_of():
            d = R[(g, ca)] - R[(g, cb)]
            lo, hi = ri.ci(d)
            DL.append({"dominio": v, "gt": g, "modo": m, "a": ca, "b": cb, "tipo": tipo, "primaria": prim,
                       "rho_a": float(R[(g, ca)][0]), "rho_b": float(R[(g, cb)][0]), "delta": float(d[0]),
                       "ic_lo": lo, "ic_hi": hi, "p_le0": float((d[1:] <= 0).mean())})
    EVID.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(S).to_csv(EVID / "spearman.csv", index=False)
    pd.DataFrame(DL).to_csv(EVID / "deltas.csv", index=False)
    pc.atomic_json(EVID / "controls.json", {**info, "n_boot": N_BOOT, "wall_s": time.time() - t0,
                                            "checks": {v: json.loads((EVID / "controls" / f"check_{v}.json").read_text())
                                                       for v in VIEWS}, **pc.stamp()})
    results()


# ------------------------------------------------------------------------------------------ results

def results() -> None:
    import pandas as pd
    S = pd.read_csv(EVID / "spearman.csv")
    DL = pd.read_csv(EVID / "deltas.csv")
    ctrl = json.loads((EVID / "controls.json").read_text())
    L = ["# NICP per le corrispondenze contro il residuo della matrice", "",
         "Generato da `aau/nicp_corr/nc.py` (regole in `REGOLE.md`, committate prima del codice e dei numeri). HIFI3D e "
         "FaceScape dev, righe di all_cross senza crop del sottocampione del rumore landmark (8%, seme 20261013), una "
         f"maschera per vista; Spearman pesato con GT FR e SR, {N_BOOT} repliche per soggetto col seme della T1, IC 95% "
         "percentile, delta appaiati. Coppie primarie: FR con mm, SR con cs.", "",
         "Righe: " + "; ".join(f"{v} {ctrl[v]['rows_mask']} / {ctrl[v]['rows_subsample_nocrop']}" for v in VIEWS) + ".",
         "", "Collaudo (`nicp_p2tri` contro la matrice, tolleranza 1e-6 mm):", ""]
    for v in VIEWS:
        for m, c in ctrl["checks"][v]["modes"].items():
            nodes = sorted({h for x in c["by_topo"].values() if not x["ok_here"] for h in x["ok_nodes"]})
            n_here = sum(x["ok_here"] for x in c["by_topo"].values())
            L.append(f"- {v} {m}: scarto massimo {c['all']['max_abs']:.2e} mm, mediano {c['all']['median_abs']:.1e}, "
                     f"NaN {c['all']['nan_new']} sulle stesse coppie: {c['all']['nan_same_pairs']}; coppie di topologie "
                     f"riprodotte qui {n_here}/{len(c['by_topo'])}, le altre sul nodo d'origine ({', '.join(nodes) or '-'}); "
                     f"strutturale min(corr_xy - ICP) = {c['structural']['min_corr_xy_minus_icp']:.3g} mm")
    L.append("")
    for v in VIEWS:
        L += [f"## {v}", "", "| colonna | rho FR [IC] | rho SR [IC] |", "|---|---|---|"]
        X = S[S["dominio"] == v]
        for c in X["colonna"].drop_duplicates():
            r = {g: X[(X["colonna"] == c) & (X["gt"] == g)].iloc[0] for g in ("fr", "sr")}
            L.append(f"| {label(c)} | " + " | ".join(ri.f3(r[g]["rho"], r[g]["ic_lo"], r[g]["ic_hi"]) for g in ("fr", "sr"))
                     + " |")
        L += ["", "Delta appaiati (a - b, rho con la GT; * = primaria):", "", "| GT | modo | tipo | a | b | delta [IC] |",
              "|---|---|---|---|---|---|"]
        Y = DL[DL["dominio"] == v]
        for _, r in Y.iterrows():
            L.append(f"| {r['gt'].upper()} | {r['modo']} | {r['tipo']}{' *' if r['primaria'] else ''} | {label(r['a'])} | "
                     f"{label(r['b'])} | {ri.f3(r['delta'], r['ic_lo'], r['ic_hi'], True)} |")
        L += ["", "Verdetto (REGOLE.md sez. 4):", ""]
        for g, m in PRIMARY:
            q = Y[(Y["gt"] == g) & (Y["modo"] == m) & Y["primaria"]]
            d1, d2 = (q[q["tipo"].str.startswith(k)].iloc[0] for k in ("D1", "D2"))
            L.append(f"- {g.upper()} con {m}: D1 {ri.f3(d1['delta'], d1['ic_lo'], d1['ic_hi'], True)}, D2 "
                     f"{ri.f3(d2['delta'], d2['ic_lo'], d2['ic_hi'], True)}: **{verdict((d1['ic_lo'], d1['ic_hi']), (d2['ic_lo'], d2['ic_hi']))}**")
        L += ["", "Spearman semplice fra colonne sulle righe della maschera (descrittivo): " + "; ".join(
            f"{k} {x:.3f}" for k, x in ctrl[v]["spearman_plain_on_mask"].items()), ""]
    p = EVID / "results.md"
    tail = p.read_text().split(MARK, 1)[1] if p.exists() and MARK in p.read_text() else "\n"
    p.write_text("\n".join(L) + "\n" + MARK + tail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("run", "merge", "check-node", "check", "eval", "results"))
    p.add_argument("--view", default="", choices=("",) + VIEWS)
    p.add_argument("--mode", default="", choices=("",) + MODES)
    p.add_argument("--topo", default="auto", help="check-node: coppie ordinate di topologie 'a>b,c>d', o 'auto' "
                                                  "(quelle fuori tolleranza nel calcolo principale)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="run: prova su N coppie (<WORK>/pilot)")
    a = p.parse_args()
    if a.step == "run":
        run(a.view, a.mode, a.workers, a.limit)
    elif a.step == "merge":
        merge(a.view, a.mode)
    elif a.step == "check-node":
        check_node(a.view, a.mode, a.topo, a.workers)
    elif a.step == "check":
        check(a.view)
    elif a.step == "eval":
        evaluate(a.workers)
    else:
        results()


if __name__ == "__main__":
    main()
