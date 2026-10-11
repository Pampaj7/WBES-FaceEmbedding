#!/usr/bin/env python3
"""Emendamento 3 (POST HOC): delta dei bracci sull'ingresso ritagliato alla regione di B (esperimento 1) e crop sugli
held-out sintetici (esperimento 2), PROTOCOL_emendamento_3.md, sez. 1, 2, 5.

    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e3.py paired --workers 48    (LEGGE LE GT di test)
    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e3.py heldout --workers 48   (GT sintetiche di training)
    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e3.py topo --workers 48      (sez. 6; GT di test)
    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e3.py summary                (solo results.md)
    (dopo bp_e3.sbatch; ``fact_paired.py`` importato in sola lettura attraverso ``bp_paired``)

``paired``: righe, seme e GT di all_cross dell'emendamento 2 (``bp_paired_e2.all_cross_rows``), 1000 repliche per
soggetto. Colonne: bracci sull'ingresso intero (``fact_paired.model_columns``), bracci sull'ingresso ritagliato alla
regione del modello m (``e3/embed/<vista>/<m>/<braccio>``, distanze con ``fact_paired.distances`` e la c di
``fact_paired.calibration``), B (``bp_paired_e2.vb600``), baseline geometriche (``blmm_eval.view_distances``), taglia
oracolo. Gruppi: senza crop, all_cross, righe col crop (``fact_paired._rep``), media dentro le 15 coppie non ordinate
di topologie diverse (crop compreso) e dentro le 5 col crop (``_rep_pairs``). Delta D1 (braccio@m - B di m), D2
(braccio@m e braccio intero - baseline geometriche), D3 (braccio@m - braccio intero). Uscite ``paired_e3.csv``,
``spearman_e3.csv``, ``controls_e3.json``.

``topo`` (sez. 6, descrittiva): bracci sull'ingresso intero, Spearman dentro ognuna delle 20 coppie ORDINATE di topologie
diverse senza crop (righe di all_cross) e delle 5 coppie di stessa topologia (righe costruite sulle stesse coppie di
soggetti), medie, differenza, min e max; repliche col seme di all_cross. Uscita ``topo_pairs_e3.csv`` e la voce
``topo`` di ``controls_e3.json``.

``heldout``: embedding di ``e3/embed_heldout/<braccio>`` (1.500 senza crop della calibrazione + 300 crop), GT-SR
(``c3f/gt_sr.npz``) e GT-FR calibrata (``c3f/gt_frcal.npz``), per dominio: Spearman e AUC di verifica senza crop e col
crop, delta, spostamento di log S. Uscita ``heldout_e3.csv`` e la voce ``heldout`` di ``controls_e3.json``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

import bp_paired as bpp
import bp_paired_e2 as bpe2
import bp_e3

fp, be, blmm, bp = bpp.fp, bpp.be, bpp.blmm, bpp.bp
OUT = bp.OUT_ROOT
E3 = bp_e3.E3
MODELS_R = bp_e3.REGION_MODELS
GEO = ("mm_rigid_icp_chamfer", "mm_chamfer_pure", "mm_nicp_template")
VB = bpe2.VB
# distanze per braccio: (con FR, con SR); C3M descrittivo
DIST = {"factorized": ("form_cal", "shape"), "ctrlfr": ("z", "z")}
DECLARED = {"fr": ("factorized_s1234|form_cal", "factorized_s2345|form_cal", "ctrlfr_s1234|z", "ctrlfr_s2345|z"),
            "sr": ("factorized_s1234|shape", "factorized_s2345|shape", "ctrlfr_s1234|z", "ctrlfr_s2345|z")}
C3M = {"fr": ("factorizedc3m_e123|form_cal", "factorizedc3m_e205|form_cal"),
       "sr": ("factorizedc3m_e123|shape", "factorizedc3m_e205|shape")}
GROUPS = ("senza crop", "all_cross", "righe col crop", "media 15 coppie", "media crop 5 coppie")
_MR = {"gnm": "regione GNM", "flame2023": "regione FLAME"}
_AL = dict(bpp.REF_ARMS)
_AL.update({"factorizedc3m_e123|form_cal": "C3M e123, d_F cal.", "factorizedc3m_e205|form_cal": "C3M e205, d_F cal.",
            "factorizedc3m_e123|shape": "C3M e123, d_P", "factorizedc3m_e205|shape": "C3M e205, d_P"})
_J: dict = {}


def label(c: str) -> str:
    if c in GEO:
        return {"mm_chamfer_pure": "Chamfer pura in mm"}.get(c, fp.BASELINES.get(c, c))
    if c in VB or c == "oracle_size":
        return bpe2.LABEL.get(c, fp.BASELINES.get(c, c))
    pre, _, d = c.partition("|")
    arm, _, m = pre.partition("@")
    return _AL[f"{arm}|{d}"] + (f" @ {_MR[m]}" if m else "")


def region_of(c: str) -> str:
    pre = c.partition("|")[0]
    return pre.partition("@")[2] or ("intero" if "|" in c else "-")


# ------------------------------------------------------------------------------------------ colonne

def arm_columns(view: str, idx) -> tuple[dict, dict]:
    """Bracci sull'ingresso intero e su quello ritagliato (NaN dove manca la mesh ritagliata); controlli."""
    cal = fp.calibration()
    full = fp.model_columns(view, idx)
    out, ctrl = {}, {}
    n = len(idx.keys)
    i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    for pre, v, e in bp_e3.ARMS:
        dists = DIST["ctrlfr" if pre.startswith("ctrlfr") else "factorized"]
        if any(f"{pre}|{d}" not in full for d in dists):  # es. C3M e123 senza store sulla vista neutra: dichiarato
            ctrl[f"{view}|{pre}"] = "store assente, braccio saltato"
            continue
        for d in set(dists):
            out[f"{pre}|{d}"] = full[f"{pre}|{d}"]
        ck_store = Path(str(np.load(bp_e3.store(view, v, e), allow_pickle=True)["checkpoint"]))
        for m in MODELS_R:
            p = E3 / "embed" / view / m / pre / "embeddings.npz"
            with np.load(p, allow_pickle=True) as z:
                Zr = np.asarray(z["Z"], np.float64)
                keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
                ckpt = Path(str(z["checkpoint"]))
            if ckpt.resolve() != ck_store.resolve():
                raise SystemExit(f"{p}: checkpoint {ckpt}, store {ck_store}")
            pos = {k: r for r, k in enumerate(keys)}
            Z = np.full((n, Zr.shape[1]), np.nan)
            have = np.asarray([k in pos for k in idx.keys])
            Z[have] = Zr[[pos[k] for k, h in zip(idx.keys, have) if h]]
            ctrl[f"{view}|{m}|{pre}"] = {"meshes": int(have.sum())}
            for d, M in fp.distances(Z, i, j, ckpt, cal.get(pre)).items():
                if d in dists:
                    out[f"{pre}@{m}|{d}"] = M.reshape(n, n)
    return out, ctrl


def geo_columns(view: str, idx) -> tuple[dict, list]:
    """Baseline geometriche sulle 600 mesh della vista (``blmm_eval.view_distances``); quelle assenti si dichiarano."""
    try:
        D, missing = be.view_distances(view, idx, set())
    except Exception as exc:  # noqa: BLE001  (vista senza baseline geometriche: dichiarato, niente D2)
        return {}, [f"{view}: {type(exc).__name__}: {exc}"]
    return {c: D[c] for c in GEO if c in D}, missing + [f"{view} {c} assente" for c in GEO if c not in D]


# ------------------------------------------------------------------------------------------ repliche

def _rep_pairs(k: int) -> dict:
    """Gruppo (d) della sez. 1: Spearman dentro ogni coppia NON ordinata di topologie diverse (pesi delle repliche per
    soggetto). "media 15 coppie" = media su tutte le 15 coppie, crop COMPRESO (10 senza crop + 5 col crop): NON e' una
    quantita' senza crop; "media crop 5 coppie" = media sulle 5 col crop. La media senza crop (20 coppie ordinate) e la
    stessa topologia sono della sez. 6 (``_rep_topo``)."""
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][k]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    acc: dict = {}
    for g, rows in J["groups"].items():
        w = wt[rows]
        keep = w > 0
        if keep.sum() < 3:
            continue
        sel, w = rows[keep], w[keep]
        rg = {gt: rankdata(np.repeat(J["cols"][f"gt_{gt}"][sel], w)) for gt in fp.GTS}
        for m in J["methods"]:
            r = rankdata(np.repeat(J["cols"][m][sel], w))
            for gt in fp.GTS:
                acc.setdefault((gt, m), []).append((g in J["crop_groups"], fp._pearson(r, rg[gt])))
    out = {}
    for (gt, m), vals in acc.items():
        out[(gt, "media 15 coppie", m)] = float(np.mean([x for _, x in vals]))
        out[(gt, "media crop 5 coppie", m)] = float(np.mean([x for cr, x in vals if cr]))
    return out


def replicas(cols: dict, sa, sb, mask, methods: list, counts: list, workers: int, topo_pair=None,
             crop_pair=None) -> dict:
    """{(gt, kind, metodo): array (1 + repliche)}: ``fact_paired._rep`` sulle righe di ``mask``, o ``_rep_pairs``."""
    global _J
    if topo_pair is None:
        o = cols["oracle_size"]
        fp._J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
                 "methods": methods, "low": o[mask] <= np.quantile(o[mask], 0.2)}
        fn = fp._rep
    else:
        tp = topo_pair[mask]
        _J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
              "methods": methods, "groups": {g: np.flatnonzero(tp == g) for g in np.unique(tp)},
              "crop_groups": {g for g in np.unique(tp) if crop_pair[g]}}
        fn = _rep_pairs
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(fn, range(len(counts)), chunksize=4)
    return {key: np.array([r[key] for r in reps]) for key in reps[0]}


def records(V: dict, view: str, group: str, n_rows: int, seed: int, deltas: list) -> list:
    """Righe del csv: valore di ogni colonna e delta (colonna - riferimento) delle coppie ``deltas``."""
    recs = []

    def ci(x):
        b = x[1:][np.isfinite(x[1:])]
        return (np.percentile(b, [2.5, 97.5]), b) if len(b) else ((np.nan, np.nan), b)

    kinds = sorted({k[1] for k in V})
    for gt in fp.GTS:
        for kind in kinds:
            for (g, kk, m), a in V.items():
                if g != gt or kk != kind:
                    continue
                (lo, hi), _ = ci(a)
                recs.append({"domain": view, "group": group if kind not in GROUPS else kind, "gt": gt,
                             "kind": "rho" if kind in GROUPS else kind, "column": m, "label": label(m),
                             "region": region_of(m), "ref": "-", "ref_label": "-", "point": a[0], "ci_low": lo,
                             "ci_high": hi, "ref_point": np.nan, "delta": np.nan, "d_ci_low": np.nan,
                             "d_ci_high": np.nan, "p_le0": np.nan, "n_rows": n_rows, "seed": seed})
            for m, ref, fam in deltas:
                if (gt, kind, m) not in V or (gt, kind, ref) not in V or kind not in ("rho",) + GROUPS:
                    continue
                a, r = V[(gt, kind, m)], V[(gt, kind, ref)]
                (alo, ahi), _ = ci(a)
                (lo, hi), b = ci(a - r)
                recs.append({"domain": view, "group": group if kind not in GROUPS else kind, "gt": gt, "kind": "rho",
                             "column": m, "label": label(m), "region": region_of(m), "ref": ref,
                             "ref_label": label(ref), "family": fam, "point": a[0], "ci_low": alo, "ci_high": ahi,
                             "ref_point": r[0], "delta": a[0] - r[0], "d_ci_low": lo, "d_ci_high": hi,
                             "p_le0": float((b <= 0).mean()) if len(b) else np.nan, "n_rows": n_rows, "seed": seed})
    return recs


def delta_pairs(cols: dict) -> list:
    """(colonna, riferimento, famiglia): D1, D2 (anche dal braccio intero), D3, per ogni braccio dichiarato e C3M."""
    out = []
    arms = sorted({a for g in fp.GTS for a in DECLARED[g] + C3M[g]})
    for a in arms:
        pre, _, d = a.partition("|")
        for m in MODELS_R:
            am = f"{pre}@{m}|{d}"
            if am not in cols:
                continue
            out += [(am, f"{m}_vb_{x}", "D1") for x in ("coef", "fr", "sr")]
            out += [(am, g, "D2") for g in GEO if g in cols]
            out.append((am, a, "D3"))
        out += [(a, g, "D2 intero") for g in GEO if g in cols]
    return out


# ------------------------------------------------------------------------------------------ esperimento 1

def run_view(view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    df, idx, seed = bpe2.all_cross_rows(view)
    A, ctrl_arms = arm_columns(view, idx)
    D, ctrl_vb = bpe2.vb600(view, idx)
    G, missing = geo_columns(view, idx)
    df = be.add_columns(df, {**A, **D, **G}, idx)
    methods = list(A) + VB + list(G) + ["oracle_size"]
    cols = {m: df[m].to_numpy(np.float64) for m in methods}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    fin = np.all([np.isfinite(v) for v in cols.values()], axis=0)
    mask = (sa != sb) & fin
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    crop = (ta == "crop") | (tb == "crop")
    pairs = sorted({tuple(sorted(p)) for p in zip(ta, tb)})
    pid = {p: k for k, p in enumerate(pairs)}
    topo_pair = np.asarray([pid[tuple(sorted(p))] for p in zip(ta, tb)])
    crop_pair = {pid[p]: "crop" in p for p in pairs}
    dl = delta_pairs(cols)
    recs = []
    for group, gm in (("senza crop", mask & ~crop), ("all_cross", mask), ("righe col crop", mask & crop)):
        V = replicas(cols, sa, sb, gm, methods, counts, workers)
        recs += records(V, view, group, int(gm.sum()), seed, dl)
    V = replicas(cols, sa, sb, mask, methods, counts, workers, topo_pair, crop_pair)
    recs += records(V, view, "media", int(mask.sum()), seed, dl)
    info = {"seed": seed, "rows": int(len(df)), "rows_mask": int(mask.sum()), "rows_mask_crop": int((mask & crop).sum()),
            "topology_pairs": len(pairs), "nan_rows_by_column": {c: int((~np.isfinite(v) & (sa != sb)).sum())
                                                                  for c, v in cols.items() if not np.isfinite(v[sa != sb]).all()},
            "arm_meshes": ctrl_arms, "vb600_vs_fit_e1": ctrl_vb, "geo_missing": missing}
    print(f"[bp-e3-paired] {view}: {int(mask.sum())} righe ({int((mask & crop).sum())} col crop), {len(pairs)} coppie di "
          f"topologie, {len(methods)} colonne, seme {seed}", flush=True)
    return recs, info


def control_e2(P: pd.DataFrame) -> dict:
    """I valori dei bracci interi e di B su all_cross e righe col crop contro ``paired_e2.csv`` (stesse righe e
    repliche se la maschera e' la stessa)."""
    ref = pd.read_csv(OUT / "paired_e2.csv")
    ref = ref[(ref["baseline"] == "-") & ref["group"].isin(["all_cross", "all_cross, righe col crop"])].copy()
    ref["col"] = np.where(ref["arm"] == "baseline", ref["distance"].map({bpe2.LABEL[c]: c for c in VB}),
                          ref["arm"] + "|" + ref["distance"])
    ref["group"] = ref["group"].map({"all_cross": "all_cross", "all_cross, righe col crop": "righe col crop"})
    ref = ref[["domain", "group", "gt", "kind", "col", "arm_point", "arm_ci_low", "arm_ci_high", "n_rows"]].rename(
        columns={"col": "column", "n_rows": "n_rows_ref"})
    mine = P[(P["ref"] == "-") & P["group"].isin(["all_cross", "righe col crop"])]
    j = mine.merge(ref, on=["domain", "group", "gt", "kind", "column"])
    return {"n_matched": int(len(j)), "columns": sorted(set(j["column"])),
            "max_abs_diff_point": float((j["point"] - j["arm_point"]).abs().max()),
            "max_abs_diff_ci": float(max((j["ci_low"] - j["arm_ci_low"]).abs().max(),
                                         (j["ci_high"] - j["arm_ci_high"]).abs().max())),
            "rows_equal": bool((j["n_rows"] == j["n_rows_ref"]).all())}


def counts_e2() -> dict:
    """Sez. 3, conteggi mancanti: a favore / contro / non risolte dei delta dichiarati dell'emendamento 2 su all_cross e
    sulle righe col crop (``bp_paired_e2.count_cells`` su ``paired_e2.csv``)."""
    E = pd.read_csv(OUT / "paired_e2.csv")
    E = E[E["kind"] == "rho"]
    return {f"{v} | {g}": bpe2.count_cells(E[(E["domain"] == v) & (E["group"] == g)], VB, ("fr", "sr"))
            for v in bp_e3.CROP_VIEWS for g in ("all_cross", "all_cross, righe col crop")}


def control_chain() -> dict:
    """factorized s1234 sulla cache dello store di HIFI3D (senza ritaglio) contro lo store."""
    p = E3 / "embed_check" / "hifi3d_store_factorized_s1234" / "embeddings.npz"
    if not p.exists():
        return {"missing": str(p)}
    with np.load(p, allow_pickle=True) as z, np.load(bp_e3.store("hifi3d", "factorized", "072"), allow_pickle=True) as r:
        a = dict(zip(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]), np.asarray(z["Z"], np.float64)))
        b = dict(zip(zip([str(x) for x in r["subjects"]], [str(x) for x in r["topologies"]]), np.asarray(r["Z"], np.float64)))
    return {"n": len(a), "keys_equal": sorted(a) == sorted(b),
            "max_abs_diff": float(max(np.abs(a[k] - b[k]).max() for k in a))}


# ------------------------------------------------------------------------- sez. 6: coppie di topologie

TOPO_VIEWS = ("hifi3d", "facescape", "faceverse")
TOPO_ARMS = ("factorized_s1234|form_cal", "factorized_s1234|shape", "factorized_s2345|form_cal",
             "factorized_s2345|shape", "ctrlfr_s1234|z", "ctrlfr_s2345|z")
NOCROP = tuple(t for t in bp_e3.TOPOLOGIES if t != "crop")
# stime puntuali del critic del paper (sez. 6, dichiarate prima del calcolo): (vista, colonna) -> (media, min, max, stessa)
CRITIC_TOPO = {("hifi3d", "factorized_s1234|form_cal"): (0.749, 0.730, 0.760, 0.760),
               ("facescape", "factorized_s1234|form_cal"): (0.667, 0.628, 0.706, 0.706)}


def _rep_topo(k: int) -> dict:
    """Spearman dentro ogni coppia ordinata di topologie (sez. 6), pesi c_a c_b delle repliche per soggetto."""
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][k]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    out = {}
    for g, rows in J["groups"].items():
        w = wt[rows]
        keep = w > 0
        sel, w = rows[keep], w[keep]
        rg = {gt: rankdata(np.repeat(J["gt"][gt][sel], w)) for gt in fp.GTS}
        for m, x in J["cols"].items():
            r = rankdata(np.repeat(x[sel], w))
            for gt in fp.GTS:
                out[(gt, m, g)] = fp._pearson(r, rg[gt])
    return out


def run_topo(view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    """Sez. 6: righe senza crop di all_cross (20 coppie ordinate) e di stessa topologia (5), bracci interi."""
    global _J
    df, idx, seed = bpe2.all_cross_rows(view)
    key = ["subject_a", "subject_b"]
    cross = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")].reset_index(drop=True)
    g = cross.groupby(key)[["gt_fr", "gt_sr"]]
    spread = float((g.max() - g.min()).to_numpy().max())          # GT costante dentro la coppia di soggetti
    pairs = g.first().reset_index()
    same = pd.concat([pairs.assign(topology_a=t, topology_b=t) for t in NOCROP], ignore_index=True)
    cols4 = key + ["topology_a", "topology_b", "gt_fr", "gt_sr"]
    rows = pd.concat([cross[cols4], same[cols4]], ignore_index=True)
    Mc = fp.model_columns(view, idx)
    rows = be.add_columns(rows, {m: Mc[m] for m in TOPO_ARMS}, idx)
    cols = {m: rows[m].to_numpy(np.float64) for m in TOPO_ARMS}
    gts = {gt: rows[f"gt_{gt}"].to_numpy(np.float64) for gt in fp.GTS}
    fin = np.all([np.isfinite(v) for v in list(cols.values()) + list(gts.values())], axis=0)
    rows, cols, gts = rows[fin].reset_index(drop=True), {m: v[fin] for m, v in cols.items()}, {k: v[fin] for k, v in gts.items()}
    gid = (rows["topology_a"] + "|" + rows["topology_b"]).to_numpy()
    groups = {x: np.flatnonzero(gid == x) for x in sorted(set(gid))}
    subjects = np.array(sorted(set(rows["subject_a"]) | set(rows["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = rows["subject_a"].map(s2i).to_numpy(), rows["subject_b"].map(s2i).to_numpy()
    if (sa == sb).any():
        raise SystemExit(f"{view}: righe con lo stesso soggetto")
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    _J = {"counts": counts, "sa": sa, "sb": sb, "groups": groups, "cols": cols, "gt": gts}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_rep_topo, range(len(counts)), chunksize=4)
    V = {k: np.array([r[k] for r in reps]) for k in reps[0]}
    xg = [x for x in groups if x.split("|")[0] != x.split("|")[1]]
    sg = [x for x in groups if x.split("|")[0] == x.split("|")[1]]
    recs = []

    def add(gt, m, q, a, grp=None, minmax=True):
        grp = grp or [q]
        b = a[1:][np.isfinite(a[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5])
        r = {"domain": view, "gt": gt, "method": m, "label": _AL[m], "quantity": q, "point": a[0], "ci_low": lo,
             "ci_high": hi, "n_pairs": len(grp), "n_rows": int(sum(len(groups[x]) for x in grp)), "seed": seed}
        if minmax and len(grp) > 1:
            pts = {x: V[(gt, m, x)][0] for x in grp}
            lo_p, hi_p = min(pts, key=pts.get), max(pts, key=pts.get)
            r.update(min=pts[lo_p], min_pair=lo_p, max=pts[hi_p], max_pair=hi_p)
        recs.append(r)

    for gt in fp.GTS:
        for m in TOPO_ARMS:
            a20 = np.mean([V[(gt, m, x)] for x in xg], axis=0)
            a5 = np.mean([V[(gt, m, x)] for x in sg], axis=0)
            add(gt, m, "media 20 senza crop", a20, xg)
            add(gt, m, "media 5 stessa topologia", a5, sg)
            add(gt, m, "differenza 20 - 5", a20 - a5, xg + sg, minmax=False)
            for x in xg + sg:
                add(gt, m, x, V[(gt, m, x)])
    info = {"seed": seed, "rows_cross": int(len(cross)), "rows_same": int(len(same)), "rows_kept": int(fin.sum()),
            "groups": {x: int(len(v)) for x, v in groups.items()}, "gt_spread_within_subject_pair": spread}
    print(f"[bp-e3-topo] {view}: {len(xg)} coppie ordinate senza crop, {len(sg)} stessa topologia, {int(fin.sum())} righe, "
          f"seme {seed}, scarto GT dentro la coppia di soggetti {spread:.1e}", flush=True)
    return recs, info


# ------------------------------------------------------------------------------------------ esperimento 2

def _auc(xg, wg, xi, wi) -> float:
    """P(d impostore > d genuino) + 1/2 P(=), pesata."""
    o = np.argsort(xg)
    xg, cw = xg[o], np.concatenate([[0.0], np.cumsum(wg[o])])
    lo, hi = np.searchsorted(xg, xi, "left"), np.searchsorted(xg, xi, "right")
    return float((wi * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum() / (wg.sum() * wi.sum()))


def _rep_heldout(k: int) -> dict:
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][k]
    out = {}
    for s, rows in J["sets"].items():
        w = c[J["sa"][rows]].astype(np.int64) * c[J["sb"][rows]]
        keep = w > 0
        sel, w = rows[keep], w[keep]
        rg = {gt: rankdata(np.repeat(J["gt"][gt][sel], w)) for gt in fp.GTS}
        for m, x in J["cols"].items():
            r = rankdata(np.repeat(x[sel], w))
            for gt in fp.GTS:
                out[(s, f"rho_{gt}", m)] = fp._pearson(r, rg[gt])
    for s, (gen, imp) in J["auc"].items():
        wg = c[J["sa"][gen]].astype(np.float64)
        wi = c[J["sa"][imp]].astype(np.float64) * c[J["sb"][imp]]
        for m, x in J["cols"].items():
            out[(s, "auc", m)] = _auc(x[gen], wg, x[imp], wi) if wg.sum() > 0 and wi.sum() > 0 else np.nan
    for m, (sub, dl) in J["dlogs"].items():
        w = c[sub].astype(np.float64)
        out[("-", "dlogS", m)] = float((w * dl).sum() / w.sum())
    return out


def run_heldout(n_boot: int, workers: int) -> tuple[list, dict]:
    with np.load(bp_e3.HELDOUT_OPS / "scale_table.npz") as z:
        dom_of = {str(n).split("_GTready_")[0]: str(d) for n, d in zip(z["names"], z["domain"])}
    G = {}
    for gt, f in (("sr", "gt_sr.npz"), ("fr", "gt_frcal.npz")):
        with np.load(bp_e3.GT_SR_TRAIN.with_name(f), allow_pickle=True) as z:
            G[gt] = (np.asarray(z["D_orig"], np.float32), {str(n): k for k, n in enumerate(z["names"])})
    cal = fp.calibration()
    recs, info = [], {"embeddings_vs_calib_heldout": {}, "shift": {}}
    emb = {}
    for pre, _, _ in bp_e3.ARMS:
        p = E3 / "embed_heldout" / pre / "embeddings.npz"
        with np.load(p, allow_pickle=True) as z:
            emb[pre] = (np.asarray(z["Z"], np.float64), [str(x) for x in z["subjects"]],
                        np.asarray([str(x) for x in z["topologies"]]), Path(str(z["checkpoint"])))
        ref = fp.EV / "factorized" / "calib_heldout" / pre / "embeddings.npz"
        if ref.exists():
            Z, s, t, _ = emb[pre]
            with np.load(ref, allow_pickle=True) as z:
                rk = dict(zip(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]), np.asarray(z["Z"], np.float64)))
            mine = {(a, b): Z[k] for k, (a, b) in enumerate(zip(s, t)) if b != "crop"}
            info["embeddings_vs_calib_heldout"][pre] = {
                "n": len(mine), "keys_equal": sorted(mine) == sorted(rk),
                "max_abs_diff": float(max(np.abs(mine[k] - rk[k]).max() for k in mine)) if sorted(mine) == sorted(rk) else None}
    for dom in bp_e2_doms():
        recs_d, sh = heldout_domain(dom, emb, dom_of, G, cal, n_boot, workers)
        recs += recs_d
        info["shift"][dom] = sh
    return recs, info


def bp_e2_doms() -> tuple:
    return ("bfm", "ict", "gnm")                    # fact_calib.DOMS


def heldout_domain(dom: str, emb: dict, dom_of: dict, G: dict, cal: dict, n_boot: int, workers: int):
    global _J
    _, subj0, topo0, _ = next(iter(emb.values()))
    sel = np.flatnonzero([dom_of[s] == dom for s in subj0])
    keys0 = [(subj0[k], topo0[k]) for k in sel]
    subjects = sorted({s for s, _ in keys0})
    s2i = {s: i for i, s in enumerate(subjects)}
    sid = np.asarray([s2i[s] for s, _ in keys0])
    lab = np.asarray([t for _, t in keys0])
    i, j = np.triu_indices(len(keys0), 1)
    cr_i, cr_j = lab[i] == "crop", lab[j] == "crop"
    diff = sid[i] != sid[j]
    sets = {"senza crop": np.flatnonzero(diff & ~cr_i & ~cr_j & (lab[i] != lab[j])),
            "col crop": np.flatnonzero(diff & (cr_i ^ cr_j))}
    same = ~diff
    auc = {"senza crop": (np.flatnonzero(same & ~cr_i & ~cr_j), np.flatnonzero(diff & ~cr_i & ~cr_j)),
           "col crop": (np.flatnonzero(same & (cr_i ^ cr_j)), np.flatnonzero(diff & (cr_i ^ cr_j)))}
    gt = {}
    for g, (D, pos) in G.items():
        gi = np.asarray([pos[s] for s in np.asarray(subjects)[sid]])
        gt[g] = D[gi[i], gi[j]].astype(np.float64)
    cols, dlogs, shift = {}, {}, {}
    for pre, (Z, subj, topo, ckpt) in emb.items():
        pos = {(a, b): k for k, (a, b) in enumerate(zip(subj, topo))}
        Zd = Z[[pos[k] for k in keys0]]
        dists = DIST["ctrlfr" if pre.startswith("ctrlfr") else "factorized"]
        for d, M in fp.distances(Zd, i, j, ckpt, cal.get(pre)).items():
            if d in dists:
                cols[f"{pre}|{d}"] = M
        X = Zd if pre.startswith("ctrlfr") else Zd[:, 1:]
        nc = [t for t in bp_e3.TOPOLOGIES if t != "crop"]
        P = {(s, t): Zd_k for s, t, Zd_k in zip(sid, lab, range(len(keys0)))}
        r = {"same_subject_crop_vs_nocrop": float(np.mean([np.linalg.norm(X[P[(s, "crop")]] - X[P[(s, t)]])
                                                           for s in range(len(subjects)) for t in nc])),
             "same_subject_nocrop": float(np.mean([np.linalg.norm(X[P[(s, a)]] - X[P[(s, b)]]) for s in range(len(subjects))
                                                   for k, a in enumerate(nc) for b in nc[k + 1:]])),
             "different_subjects_original_median": float(np.median(
                 np.linalg.norm(X[[P[(s, "original")] for s in range(len(subjects))]][:, None] -
                                X[[P[(s, "original")] for s in range(len(subjects))]][None], axis=-1)[
                     np.triu_indices(len(subjects), 1)]))}
        if not pre.startswith("ctrlfr"):
            ls = np.asarray([[Zd[P[(s, t)], 0] for t in nc] for s in range(len(subjects))]).mean(1)
            lc = np.asarray([Zd[P[(s, "crop")], 0] for s in range(len(subjects))])
            dlogs[pre] = (np.arange(len(subjects)), lc - ls)
            r.update(dlogS_crop_mean=float((lc - ls).mean()), dlogS_crop_sd=float((lc - ls).std()),
                     logS_sd_between_subjects=float(ls.std()))
        shift[pre] = r
    rng = np.random.default_rng(1234)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    _J = {"counts": counts, "sa": sid[i], "sb": sid[j], "sets": sets, "auc": auc, "gt": gt, "cols": cols, "dlogs": dlogs}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_rep_heldout, range(len(counts)), chunksize=4)
    V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
    recs = []

    def add(s, meas, m, x):
        b = x[1:][np.isfinite(x[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
        recs.append({"domain": dom, "set": s, "measure": meas, "method": m, "label": _AL.get(m, m), "point": x[0],
                     "ci_low": lo, "ci_high": hi, "p_le0": float((b <= 0).mean()) if len(b) else np.nan,
                     "n_pairs": int(len(sets[s])) if s in sets else len(subjects)})

    for m in cols:
        for meas in ("rho_fr", "rho_sr", "auc"):
            a, b = V[("senza crop", meas, m)], V[("col crop", meas, m)]
            add("senza crop", meas, m, a)
            add("col crop", meas, m, b)
            add("col crop - senza crop", meas, m, b - a)
    for m in dlogs:
        add("-", "dlogS", m, V[("-", "dlogS", m)])
    print(f"[bp-e3-heldout] {dom}: {len(subjects)} soggetti, {len(sets['senza crop'])} coppie senza crop, "
          f"{len(sets['col crop'])} col crop", flush=True)
    return recs, shift


# ------------------------------------------------------------------------------------------- markdown

def fmt(p, lo, hi, sign=False) -> str:
    return bpp.fmt(p, lo, hi, sign)


def _pair(x) -> str:
    """Coppia ordinata di topologie "a|b" come "a -> b" (il "|" rompe le tabelle markdown)."""
    return str(x).replace("|", " -> ")


def counts_of(P: pd.DataFrame, view: str, group: str, gt: str, fam: str, arms: tuple) -> str:
    x = P[(P["domain"] == view) & (P["group"] == group) & (P["gt"] == gt) & (P["family"] == fam)]
    x = x[x["column"].map(lambda c: c.split("@")[0] + "|" + c.split("|")[1] if "@" in c else c).isin(arms)]
    return f"{int((x.d_ci_low > 0).sum())} / {int((x.d_ci_high < 0).sum())} / " \
           f"{int(((x.d_ci_low <= 0) & (x.d_ci_high >= 0)).sum())}"


def recovery(P: pd.DataFrame, view: str, gt: str, arm: str, m: str) -> dict:
    """Recupero F e lettura della sez. 1 per (vista, braccio dichiarato, regione, GT)."""
    def val(group, col):
        r = P[(P["domain"] == view) & (P["group"] == group) & (P["gt"] == gt) & (P["kind"] == "rho") &
              (P["column"] == col) & (P["ref"] == "-")]
        return r.iloc[0] if len(r) else None
    pre, _, d = arm.partition("|")
    am = f"{pre}@{m}|{d}"
    a_full, c_full, c_reg = val("senza crop", arm), val("righe col crop", arm), val("righe col crop", am)
    d3 = P[(P["domain"] == view) & (P["group"] == "righe col crop") & (P["gt"] == gt) & (P["column"] == am) &
           (P["ref"] == arm)]
    d3c = P[(P["domain"] == view) & (P["group"] == "media crop 5 coppie") & (P["gt"] == gt) & (P["column"] == am) &
            (P["ref"] == arm)]
    if a_full is None or c_full is None or c_reg is None or not len(d3):
        return {}
    gap = a_full.point - c_full.point
    F = (c_reg.point - c_full.point) / gap if gap >= 0.05 else np.nan
    d3 = d3.iloc[0]
    if not np.isfinite(F):
        read = "calo < 0.05"
    elif d3.d_ci_low > 0 and F >= 0.5:
        read = "dipendenza dal supporto"
    elif not d3.d_ci_low > 0 or F < 0.25:
        read = "limite del descrittore"
    else:
        read = "misto"
    return {"a": a_full.point, "c": c_full.point, "c_reg": c_reg.point, "F": F, "d3": d3, "d3c": d3c.iloc[0] if len(d3c) else None,
            "read": read}


def results_e3() -> list:
    """Sezioni dell'emendamento 3 in fondo a results.md."""
    sha = OUT / "PROTOCOL_emendamento_3.sha256"
    sha = sha.read_text().split()[0] if sha.exists() else "?"
    c = json.loads((OUT / "controls_e3.json").read_text())
    md = ["# Emendamento 3 (POST HOC): bracci sulla regione di B, crop sugli held-out sintetici", "",
          f"Protocollo `PROTOCOL_emendamento_3.md` (sha256 `{sha}`), scritto dopo i numeri dell'emendamento 2. Numeri in "
          "`paired_e3.csv`, `spearman_e3.csv`, `heldout_e3.csv`, `controls_e3.json`, `e3/crop_stats.json`.", ""]
    st = json.loads((E3 / "crop_stats.json").read_text())
    md += ["## Ritaglio alla regione di B (diagnostica, senza GT)", "",
           "Copertura della regione = frazione d'area della regione di B (collocata su ogni mesh) presente nella mesh "
           "con la regola di voto di `bp.region`; area tenuta = area del ritaglio / area della mesh; crop / original = "
           "log(sqrt(area crop) / sqrt(area original)) per soggetto, mediana [IQR], prima e dopo il ritaglio.", "",
           "| vista | regione | fallite | copertura: crop mediana (p5) | altre topologie mediana | area tenuta crop / "
           "original | crop / original prima | dopo |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, r in st.items():
        v, m = key.split("|")
        bt = r["by_topology"]
        oth = [bt[t]["coverage_median"] for t in bp_e3.TOPOLOGIES if t != "crop"]
        b, a = r["log_sqrt_area_crop_over_original_before"], r["log_sqrt_area_crop_over_original_after"]
        md.append(f"| {v} | {_MR[m]} ({r['n_region_vertices']} vertici) | {len(r['failed'])} | "
                  f"{bt['crop']['coverage_median']:.3f} ({bt['crop']['coverage_p5']:.3f}) | {min(oth):.3f}-{max(oth):.3f} | "
                  f"{bt['crop']['area_kept_median']:.2f} / {bt['original']['area_kept_median']:.2f} | "
                  f"{b['median']:+.3f} [{b['q25']:+.3f}, {b['q75']:+.3f}] | {a['median']:+.3f} [{a['q25']:+.3f}, {a['q75']:+.3f}] |")
    if (OUT / "topo_pairs_e3.csv").exists():
        T = pd.read_csv(OUT / "topo_pairs_e3.csv")
        md += ["", "## Spearman dentro le coppie di topologie senza crop (sez. 6, descrittiva, non cieca)", "",
               "Bracci sull'ingresso intero. Media sulle 20 coppie ordinate di topologie diverse senza crop (righe di "
               "all_cross, 4.950 per coppia) e sulle 5 coppie di stessa topologia (righe costruite sulle stesse coppie di "
               "soggetti); IC 95% per soggetto col seme di all_cross; min e max = stime puntuali fra le coppie. Lettura "
               "dichiarata in grassetto (d_F cal. e ctrlfr con FR, d_P e ctrlfr con SR). Stime puntuali del critic "
               "dichiarate prima del calcolo: " + "; ".join(f"{v} {_AL[m]} FR {a:.3f} [{lo:.3f}, {hi:.3f}], stessa "
                                                            f"topologia {s:.3f}" for (v, m), (a, lo, hi, s) in CRITIC_TOPO.items()) + ".", "",
               "| vista | braccio | GT | media 20 senza crop | min (coppia) | max (coppia) | media 5 stessa topologia | min | "
               "max | 20 - 5 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v in TOPO_VIEWS:
            for m in TOPO_ARMS:
                for gt in fp.GTS:
                    y = T[(T["domain"] == v) & (T["method"] == m) & (T["gt"] == gt)]
                    if y.empty:
                        continue
                    a, s, d = (y[y["quantity"] == q].iloc[0] for q in ("media 20 senza crop", "media 5 stessa topologia",
                                                                         "differenza 20 - 5"))
                    decl = m in DECLARED[gt]
                    b0, b1 = ("**", "**") if decl else ("", "")
                    md.append(f"| {v} | {_AL[m]} | {gt.upper()} | {b0}{fmt(a.point, a.ci_low, a.ci_high)}{b1} | "
                              f"{a['min']:.3f} ({_pair(a.min_pair)}) | {a['max']:.3f} ({_pair(a.max_pair)}) | "
                              f"{b0}{fmt(s.point, s.ci_low, s.ci_high)}{b1} | {s['min']:.3f} ({_pair(s.min_pair)}) | "
                              f"{s['max']:.3f} ({_pair(s.max_pair)}) | {fmt(d.point, d.ci_low, d.ci_high, True)} |")
    if not (OUT / "paired_e3.csv").exists():
        return md
    P = pd.read_csv(OUT / "paired_e3.csv")
    md += ["", "## Recupero dei bracci sulle righe col crop (criteri della sez. 1)", "",
           "rho senza crop e righe col crop del braccio intero, righe col crop del braccio sulla regione; F = recupero "
           "della frazione del calo; D3 = braccio sulla regione - braccio intero (righe col crop; media dentro le 5 "
           "coppie col crop). Nota (critic dell'emendamento 3): a regione uguale righe col crop e senza crop coincidono "
           "(da -0.070 a +0.045 con FR), quindi F misura solo il costo o il guadagno del ritaglio sulle righe senza crop e "
           "le letture di questa tabella sono meccaniche; la lettura corretta e' in cima, nella sezione dell'emendamento 3.",
           "",
           "| vista | GT | braccio | regione | rho senza crop | rho col crop | col crop sulla regione | F | D3 righe col crop | "
           "D3 media crop 5 | lettura |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in bp_e3.CROP_VIEWS:
        for gt in fp.GTS:
            for arm in DECLARED[gt] + C3M[gt]:
                for m in MODELS_R:
                    r = recovery(P, v, gt, arm, m)
                    if not r:
                        continue
                    d3, d3c = r["d3"], r["d3c"]
                    md.append(f"| {v} | {gt.upper()} | {_AL[arm]} | {_MR[m]} | {r['a']:.3f} | {r['c']:.3f} | {r['c_reg']:.3f} | "
                              f"{r['F']:.2f} | {fmt(d3.delta, d3.d_ci_low, d3.d_ci_high, True)} | " +
                              (fmt(d3c.delta, d3c.d_ci_low, d3c.d_ci_high, True) if d3c is not None else "-") +
                              f" | {r['read'] if arm in DECLARED[gt] else r['read'] + ' (C3M, descrittivo)'} |")
    md += ["", "## Conteggi dei delta dichiarati (a favore / contro / non risolte)", "",
           "D1 = braccio sulla regione di m - B di m (24 per GT); D2 = braccio sulla regione - baseline geometriche (24); "
           "D2 intero = braccio sull'ingresso intero - baseline geometriche (12); D3 = braccio sulla regione - braccio "
           "intero (8).", "",
           "| vista | gruppo | GT | D1 | D2 | D2 intero | D3 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for v in bp_e3.CROP_VIEWS:
        for grp in GROUPS:
            for gt in fp.GTS:
                if P[(P["domain"] == v) & (P["group"] == grp)].empty:
                    continue
                md.append(f"| {v} | {grp} | {gt.upper()} | " + " | ".join(
                    counts_of(P, v, grp, gt, f, DECLARED[gt]) for f in ("D1", "D2", "D2 intero", "D3")) + " |")
    for v in bp_e3.CROP_VIEWS:
        x = P[(P["domain"] == v) & (P["ref"] == "-")]
        if x.empty:
            continue
        rows = [a for gt in fp.GTS for a in DECLARED[gt] + C3M[gt]]
        rows = list(dict.fromkeys(rows))
        cols = [r for a in rows for r in [a] + [f"{a.split('|')[0]}@{m}|{a.split('|')[1]}" for m in MODELS_R]] + VB + list(GEO)
        md += ["", f"## {v}: Spearman con la GT per gruppo (rho, IC 95%; {int(x[x['group'] == 'all_cross']['n_rows'].iloc[0])} "
                   f"righe all_cross, seme {int(x['seed'].iloc[0])})", ""]
        for gt in fp.GTS:
            md += [f"GT {gt.upper()}:", "", "| metodo | " + " | ".join(GROUPS) + " |", "| --- |" + " --- |" * len(GROUPS)]
            for col in cols:
                y = x[(x["column"] == col) & (x["gt"] == gt) & (x["kind"] == "rho")]
                if y.empty:
                    continue
                cells = []
                for grp in GROUPS:
                    r = y[y["group"] == grp]
                    cells.append(fmt(r.iloc[0].point, r.iloc[0].ci_low, r.iloc[0].ci_high) if len(r) else "-")
                md.append(f"| {label(col)} | " + " | ".join(cells) + " |")
            md.append("")
        for grp in ("righe col crop", "media crop 5 coppie", "senza crop"):
            md += [f"Delta appaiati, {grp} (braccio sulla regione di m - concorrente; IC 95%, P(delta <= 0)):", ""]
            for gt in fp.GTS:
                arms = DECLARED[gt]
                md += [f"GT {gt.upper()}:", "", "| concorrente | " + " | ".join(_AL[a] for a in arms) + " |",
                       "| --- |" + " --- |" * len(arms)]
                for m in MODELS_R:
                    for ref in [f"{m}_vb_{d}" for d in ("coef", "fr", "sr")] + list(GEO) + ["intero"]:
                        cells = []
                        for a in arms:
                            pre, _, d = a.partition("|")
                            rr = a if ref == "intero" else ref
                            y = P[(P["domain"] == v) & (P["group"] == grp) & (P["gt"] == gt) &
                                  (P["column"] == f"{pre}@{m}|{d}") & (P["ref"] == rr)]
                            cells.append(fmt(y.iloc[0].delta, y.iloc[0].d_ci_low, y.iloc[0].d_ci_high, True) +
                                         f", P {y.iloc[0].p_le0:.3f}" if len(y) else "-")
                        if any(cl != "-" for cl in cells):
                            name = "braccio intero" if ref == "intero" else label(ref)
                            md.append(f"| {name} (@ {_MR[m]}) | " + " | ".join(cells) + " |")
                md.append("")
    if (OUT / "heldout_e3.csv").exists():
        H = pd.read_csv(OUT / "heldout_e3.csv")
        md += ["", "## Esperimento 2: crop sugli held-out sintetici del training", "",
               "Per dominio, coppie di soggetti diversi: senza crop (etichette diverse, entrambe senza crop) e col crop "
               "(un lato crop); Spearman con GT-FR calibrata (`gt_frcal`) e GT-SR (`gt_sr`); AUC di verifica (genuine = "
               "stesso soggetto). BFM: FR non si legge (taglia delle original REMESH allineate per similarita').", "",
               "| dominio | metodo | misura | senza crop | col crop | col crop - senza crop |",
               "| --- | --- | --- | --- | --- | --- |"]
        for dom in bp_e2_doms():
            for m in sorted(set(H[H["domain"] == dom]["method"])):
                for meas in ("rho_fr", "rho_sr", "auc"):
                    y = H[(H["domain"] == dom) & (H["method"] == m) & (H["measure"] == meas)]
                    if y.empty:
                        continue
                    g = {s: y[y["set"] == s].iloc[0] for s in ("senza crop", "col crop", "col crop - senza crop")}
                    md.append(f"| {dom} | {_AL.get(m, m)} | {meas} | {fmt(g['senza crop'].point, g['senza crop'].ci_low, g['senza crop'].ci_high)} | "
                              f"{fmt(g['col crop'].point, g['col crop'].ci_low, g['col crop'].ci_high)} | "
                              f"{fmt(g['col crop - senza crop'].point, g['col crop - senza crop'].ci_low, g['col crop - senza crop'].ci_high, True)} |")
        sh = c.get("heldout", {}).get("shift", {})
        md += ["", "Spostamento del crop (held-out): d log S = log S(crop) - media delle 5 senza crop (media [IC 95% per "
                   "soggetto], sd), sd di log S fra soggetti; distanze nell'embedding (||u|| o ||z||).", "",
               "| dominio | braccio | d log S | sd | sd fra soggetti | d log S / sd fra soggetti | stesso sogg. crop | "
               "stesso sogg. senza crop | soggetti diversi |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for dom, arms in sh.items():
            for a, r in arms.items():
                y = H[(H["domain"] == dom) & (H["method"] == a) & (H["measure"] == "dlogS")]
                ds = fmt(y.iloc[0].point, y.iloc[0].ci_low, y.iloc[0].ci_high, True) if len(y) else "-"
                if "dlogS_crop_mean" in r:
                    md.append(f"| {dom} | {a} | {ds} | {r['dlogS_crop_sd']:.4f} | {r['logS_sd_between_subjects']:.4f} | "
                              f"{r['dlogS_crop_mean'] / r['logS_sd_between_subjects']:+.2f} | {r['same_subject_crop_vs_nocrop']:.3f} | "
                              f"{r['same_subject_nocrop']:.3f} | {r['different_subjects_original_median']:.3f} |")
                else:
                    md.append(f"| {dom} | {a} | - | - | - | - | {r['same_subject_crop_vs_nocrop']:.3f} | "
                              f"{r['same_subject_nocrop']:.3f} | {r['different_subjects_original_median']:.3f} |")
    md += ["", "## Controlli dell'emendamento 3", "", "```", json.dumps(c, indent=1, default=str), "```", ""]
    return md


# ----------------------------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("paired", "heldout", "topo", "summary"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--views", default=",".join(bp_e3.CROP_VIEWS))
    a = p.parse_args()
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(bpp.NEUTRAL_EMB, fp.EVAL)
    fp.BASELINES.update(bpe2.LABEL)
    cpath = OUT / "controls_e3.json"
    ctrl = json.loads(cpath.read_text()) if cpath.exists() else {}
    if a.step == "paired":
        recs, info = [], {}
        for view in a.views.split(","):
            r, info[view] = run_view(view, a.n_boot, a.workers)
            recs += r
        P = pd.DataFrame(recs)
        P.to_csv(OUT / "paired_e3.csv", index=False)
        P[P["ref"] == "-"].to_csv(OUT / "spearman_e3.csv", index=False)
        ctrl.update(paired=info, paired_e2_values=control_e2(P), chain=control_chain(), counts_e2=counts_e2(),
                    region=json.loads((E3 / "check_region.json").read_text()))
        print(json.dumps({k: ctrl[k] for k in ("paired_e2_values", "chain", "region")}, indent=1), flush=True)
    elif a.step == "topo":
        recs, info = [], {}
        for view in TOPO_VIEWS:
            r, info[view] = run_topo(view, a.n_boot, a.workers)
            recs += r
        pd.DataFrame(recs).to_csv(OUT / "topo_pairs_e3.csv", index=False)
        ctrl["topo"] = info
    elif a.step == "heldout":
        recs, info = run_heldout(a.n_boot, a.workers)
        pd.DataFrame(recs).to_csv(OUT / "heldout_e3.csv", index=False)
        ctrl["heldout"] = info
        print(json.dumps(info["embeddings_vs_calib_heldout"], indent=1), flush=True)
    cpath.write_text(json.dumps(ctrl, indent=1, default=str) + "\n")
    bpp.write_results(pd.read_csv(OUT / "paired.csv"), *json.loads((OUT / "controls.json").read_text()).values())


if __name__ == "__main__":
    main()
