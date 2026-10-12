#!/usr/bin/env python3
"""G0 (``aau/runs/evidence/diagnostics/PROTOCOL_G0.md``): Spearman e delta appaiati dell'insegnante ArcFace-render contro
le GT FR/SR, controlli, lettura preregistrata e ``results_g0.md``.

    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/g0_stats.py --workers 64    (g0.sbatch, passo stats)
    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/g0_stats.py --summary-only  (solo results_g0.md)
    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/g0_stats.py --fake-teacher  (K5, g0_test.py)

Insiemi (sez. 3), righe e repliche importati in sola lettura:
- FaceScape dev, HIFI3D, FaceVerse neutra, ``nocrop_cross`` (primario): ``d3_stats.view_data`` (righe e GT di
  ``fact_paired.rows_for``, maschera di ``fact_paired.main`` con B finito, conteggi coi semi pubblicati, embedding dei
  bracci, B ``fit_e1``); righe col crop (secondario): ``bp_paired_e2.all_cross_rows`` con il crop su un lato, bracci
  dagli store (600 chiavi), B ``bp_paired_e2.vb600``, stessi conteggi, maschera comune contata;
- FLAME suddiviso e FaMoS TRAIN: ``d3_stats.set_data`` (righe di ``diag.rows``, conteggi di D1 / D3); B su FLAME da
  ``d1_stats.bp_columns``.
Colonne: insegnante d_T = 1 - coseno (``g0/teacher/<insieme>/teacher.npz``); bracci factorized s1234 / s2345 d_P
(``shape``) e d_F calibrata (``form_cal``), calcolate come ``d3_stats.columns``; combinazione insegnante + taglia
(``comb``, sez. 4, c_T dagli held-out); ICP mm (``mm_rigid_icp_chamfer``, matrici di ``baselines_mm``); B-GNM e
B-FLAME ``vb_sr`` / ``vb_fr``. Una colonna di confronto non finita su righe del primario va in un blocco a parte
(righe dove e' finita, con l'insegnante sulle stesse righe). Spearman pesato c_a c_b (``diag.wranks`` /
``diag.wpearson``) per replica, IC 95% percentile (``diag.ci``).

Uscite: ``g0_spearman.csv``, ``g0_delta.csv``, ``g0/controls.json``, ``g0/readings.json``, ``g0/reps.npz``,
``results_g0.md``. ``--fake-teacher``: l'insegnante e' sostituito da d_P di factorized s1234 (c_T = c del braccio), uscite
in ``g0/test`` (K5: delta insegnante - d_P s1234 e combinazione - d_F cal. s1234 nulli su ogni replica).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diag  # noqa: E402
import d1_stats  # noqa: E402
import d3_stats as S  # noqa: E402

sys.path.insert(0, str(diag.REPO / "aau/baselines_param"))
import bp_paired_e2 as bpe2  # noqa: E402  (sola lettura, attraverso bp_paired: fact_paired, blmm_eval)

fp, be = bpe2.bpp.fp, bpe2.bpp.be
blmm = be.blmm
EV = diag.EV
G0 = EV / "g0"
VIEWS = ("facescape", "hifi3d", "faceverse_neutral")         # le tre famiglie della regola
EXTRA = ("flame2023_s1", "famos")
SETS = VIEWS + EXTRA
ARMS = diag.RULE_ARMS                                         # factorized_s1234, factorized_s2345
DOM = {"facescape": "devfs", "hifi3d": "hifi", "faceverse_neutral": "fvn"}
T_COL = "teacher|d_T"
ICP = "ICP mm|rigid_icp_chamfer"
B_MODELS = ("gnm", "flame2023")
NOCROP, CROP = "nocrop_cross", "righe col crop"
SR_MIN, N_FAM = 0.05, 2                                       # lettura (sez. 6)
_J: dict = {}
FAKE = False
OUT_DIR = G0


def label(col: str) -> str:
    if col == T_COL:
        return "insegnante (1 - coseno)"
    if col == ICP:
        return "ICP + Chamfer in mm"
    if col.startswith("B-"):
        return col
    arm, _, d = col.partition("|")
    return f"{arm.replace('factorized_', 'fact. ')}, " + {"shape": "d_P", "form_cal": "d_F cal.",
                                                          "comb": "insegnante + taglia (c_T)"}[d]


# ------------------------------------------------------------------------------------------------ insegnante

def teacher_map(name: str) -> dict:
    """{(soggetto, etichetta): Z (512,)} dell'insegnante di un insieme."""
    with np.load(G0 / "teacher" / name / "teacher.npz", allow_pickle=True) as z:
        return dict(zip(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]),
                        np.asarray(z["Z"], np.float64)))


def d_teacher(Zt: dict, a: list, b: list) -> np.ndarray:
    A = np.stack([Zt[k] for k in a])
    B = np.stack([Zt[k] for k in b])
    return 1.0 - np.einsum("ij,ij->i", A, B)


def calib_teacher() -> dict:
    """Sez. 4 e K6: c_T = mediana(d_P GT) / mediana(d_T) sulle coppie degli held-out (regola di fact_calib.calib_one);
    la stessa regola sugli embedding dei bracci deve ridare ``c_median``."""
    cal = diag.calibration()
    t_all, m_all = [], {k: [] for k in (T_COL,) + ARMS}
    per_dom = {}
    for dom in diag.SEEN:
        names = diag.set_names(dom)
        i, j = diag.rows(names)
        t = d1_stats.gt_static(dom, names)["sr"][i, j]
        t_all.append(t)
        ms = {}
        for arm in ARMS:
            Z, ckpt = diag.load_embeddings(diag.CALIB_EMB / arm / "embeddings.npz", names)
            ms[arm] = np.linalg.norm(Z[i, 1:] - Z[j, 1:], axis=1) * diag.dp_per_unit(ckpt)
        if FAKE:
            ms[T_COL] = ms[ARMS[0]]
        else:
            Zt = teacher_map(dom)
            keys = [diag.split_name(n) for n in names]
            ms[T_COL] = d_teacher(Zt, [keys[k] for k in i], [keys[k] for k in j])
        for k, m in ms.items():
            m_all[k].append(m)
        per_dom[dom] = {"n_pairs": int(len(i)), "c_T": float(np.median(t) / np.median(ms[T_COL]))}
    t = np.concatenate(t_all)
    m = {k: np.concatenate(v) for k, v in m_all.items()}
    out = {"c_T": float(np.median(t) / np.median(m[T_COL])), "median_dP_gt": float(np.median(t)),
           "median_d_T": float(np.median(m[T_COL])), "n_pairs": int(len(t)), "by_domain": per_dom, "K6": {}}
    for arm in ARMS:
        c = float(np.median(t) / np.median(m[arm]))
        out["K6"][arm] = {"c": c, "c_median_csv": cal[arm], "abs_diff": abs(c - cal[arm])}
    out["K6"]["pass"] = bool(all(v["abs_diff"] <= 1e-9 for v in out["K6"].values()))
    return out


# ---------------------------------------------------------------------------------------------- colonne

def arm_cols(Z: np.ndarray, dpu: float, c: float, ia: np.ndarray, ib: np.ndarray, arm: str, dT, cT: float) -> dict:
    """d_P e d_F calibrata come ``d3_stats.columns`` (X = u x dp_per_unit), e la combinazione con d_T."""
    Sx = np.exp(Z[:, 0])
    X = Z[:, 1:] * dpu
    Sa, Sb = Sx[ia], Sx[ib]
    dP = np.linalg.norm(X[ia] - X[ib], axis=1)
    out = {f"{arm}|shape": dP, f"{arm}|form_cal": np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (c * dP) ** 2)}
    if dT is not None:
        out[f"{arm}|comb"] = np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (cT * dT) ** 2)
    return out


def icp_matrix(view: str):
    """(D (600, 600), indice) di ICP + Chamfer in mm della vista (``zs_expr_summarize.facebench_distances``)."""
    idx = be.zes.Index(blmm.subjects(view))
    return be.zes.facebench_distances(blmm.view_root(view) / "mm", "rigid_icp_chamfer", idx), idx


def view_entries(view: str, cT: float, info: dict) -> dict:
    """Blocchi nocrop_cross (righe di D3) e righe col crop di una vista."""
    T = S.view_data(view)                                    # patcha fact_paired per la neutra, come bp_paired
    meshes, ma, mb = T["meshes"], T["ma"], T["mb"]
    ka, kb = [meshes[k] for k in ma], [meshes[k] for k in mb]
    cal = diag.calibration()
    Zt = None if FAKE else teacher_map(view)
    cols = {}
    dpu = {arm: diag.dp_per_unit(T["ckpt"][arm]) for arm in ARMS}
    for arm in ARMS:
        cols.update(arm_cols(T["Z"][arm], dpu[arm], cal[arm], ma, mb, arm, None, cT))
    dT = cols[f"{ARMS[0]}|shape"].copy() if FAKE else d_teacher(Zt, ka, kb)
    cols[T_COL] = dT
    for arm in ARMS:
        cols.update({k: v for k, v in arm_cols(T["Z"][arm], dpu[arm], cal[arm], ma, mb, arm, dT, cT).items()
                     if k.endswith("|comb")})
    cols.update(T["B"])
    M, idx = icp_matrix(view)
    cols[ICP] = M[[idx.pos[k] for k in ka], [idx.pos[k] for k in kb]]
    info[view] = {"seed": T["seed"], "rows_nocrop": int(len(ma)), "rows_all_published": T["n_rows_all"],
                  "subjects": len(T["subjects"]), "stores": T["stores"]}
    out = {(view, NOCROP): {"cols": cols, "gt": T["gt"], "sa": T["sa"], "sb": T["sb"], "counts": T["counts"],
                            "seed": T["seed"]}}
    # ------------------------------------------------------------------ righe col crop (secondario)
    df, idx_ac, _ = bpe2.all_cross_rows(view)
    df = df[(df["topology_a"] == "crop") != (df["topology_b"] == "crop")].reset_index(drop=True)
    ka = list(zip(df["subject_a"].astype(str), df["topology_a"].astype(str)))
    kb = list(zip(df["subject_b"].astype(str), df["topology_b"].astype(str)))
    s2i = {s: k for k, s in enumerate(T["subjects"])}
    if sorted({k[0] for k in ka + kb}) != sorted(T["subjects"]):
        raise SystemExit(f"{view}: soggetti delle righe col crop diversi da quelli del primario")
    cc = {}
    for arm in ARMS:
        with np.load(S.arm_store(arm, DOM[view]), allow_pickle=True) as z:
            pos = {k: r for r, k in enumerate(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))}
            Z = np.asarray(z["Z"], np.float64)
            if Path(str(z["checkpoint"])).resolve() != T["ckpt"][arm].resolve():
                raise SystemExit(f"{view} {arm}: checkpoint dello store diverso da quello del primario")
        ia, ib = np.asarray([pos[k] for k in ka]), np.asarray([pos[k] for k in kb])
        cc.update(arm_cols(Z, dpu[arm], cal[arm], ia, ib, arm, None, cT))
    dTc = cc[f"{ARMS[0]}|shape"].copy() if FAKE else d_teacher(Zt, ka, kb)
    cc[T_COL] = dTc
    for arm in ARMS:
        with np.load(S.arm_store(arm, DOM[view]), allow_pickle=True) as z:
            pos = {k: r for r, k in enumerate(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))}
            Z = np.asarray(z["Z"], np.float64)
        ia, ib = np.asarray([pos[k] for k in ka]), np.asarray([pos[k] for k in kb])
        cc[f"{arm}|comb"] = arm_cols(Z, dpu[arm], cal[arm], ia, ib, arm, dTc, cT)[f"{arm}|comb"]
    vb, _ = bpe2.vb600(view, idx_ac)
    pa, pb = [idx_ac.pos[k] for k in ka], [idx_ac.pos[k] for k in kb]
    for m in B_MODELS:
        for g in ("sr", "fr"):
            cc[f"B-{m}|vb_{g}"] = vb[f"{m}_vb_{g}"][pa, pb]
    cc[ICP] = M[[idx.pos[k] for k in ka], [idx.pos[k] for k in kb]]
    gt = {g: df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS}
    sa, sb = np.asarray([s2i[k[0]] for k in ka]), np.asarray([s2i[k[0]] for k in kb])
    fin = np.all([np.isfinite(v) for v in list(cc.values()) + list(gt.values())], axis=0) & (sa != sb)
    info[view].update(rows_crop=int(len(df)), rows_crop_masked=int(fin.sum()),
                      crop_nan_by_column={c: int((~np.isfinite(v)).sum()) for c, v in cc.items() if not np.isfinite(v).all()})
    out[(view, CROP)] = {"cols": {k: v[fin] for k, v in cc.items()}, "gt": {k: v[fin] for k, v in gt.items()},
                         "sa": sa[fin], "sb": sb[fin], "counts": T["counts"], "seed": T["seed"]}
    return out


def set_entries(name: str, cT: float, info: dict) -> dict:
    """FLAME suddiviso o FaMoS TRAIN (righe e conteggi di D1 / D3)."""
    T = S.set_data(name, {})
    meshes, ma, mb = T["meshes"], T["ma"], T["mb"]
    names = [f"{s}_GTready_{t}.npz" for s, t in meshes]
    cal = diag.calibration()
    cols = {}
    dpu = {arm: diag.dp_per_unit(T["ckpt"][arm]) for arm in ARMS}
    for arm in ARMS:
        cols.update(arm_cols(T["Z"][arm], dpu[arm], cal[arm], ma, mb, arm, None, cT))
    dT = cols[f"{ARMS[0]}|shape"].copy() if FAKE else \
        d_teacher(teacher_map(name), [meshes[k] for k in ma], [meshes[k] for k in mb])
    cols[T_COL] = dT
    for arm in ARMS:
        cols[f"{arm}|comb"] = arm_cols(T["Z"][arm], dpu[arm], cal[arm], ma, mb, arm, dT, cT)[f"{arm}|comb"]
    for (m, g), D in d1_stats.bp_columns(name, names).items():
        cols[f"B-{m}|vb_{g}"] = np.asarray(D, np.float64)[ma, mb]
    info[name] = {"seed": T["seed"], "rows_nocrop": int(len(ma)), "subjects": len(T["subjects"]), "stores": T["stores"]}
    return {(name, NOCROP): {"cols": cols, "gt": T["gt"], "sa": T["sa"], "sb": T["sb"], "counts": T["counts"],
                             "seed": T["seed"]}}


def split_nonfinite(entries: dict, info: dict) -> dict:
    """Sez. 3: una colonna di confronto non finita su righe del primario va in un blocco a parte, con l'insegnante sulle
    sue righe finite; nel blocco primario resta tutto il resto."""
    out = {}
    for key, E in entries.items():
        bad = [c for c, v in E["cols"].items() if not np.isfinite(v).all()]
        if T_COL in bad or not all(np.isfinite(v).all() for v in E["gt"].values()):
            raise SystemExit(f"{key}: insegnante o GT non finiti")
        for c in bad:
            k = np.isfinite(E["cols"][c])
            out[(key[0], f"{key[1]} | righe con {c} finita")] = {
                "cols": {T_COL: E["cols"][T_COL][k], c: E["cols"][c][k]}, "gt": {g: v[k] for g, v in E["gt"].items()},
                "sa": E["sa"][k], "sb": E["sb"][k], "counts": E["counts"], "seed": E["seed"]}
            info.setdefault("nonfinite", {})[f"{key[0]}|{key[1]}|{c}"] = int((~k).sum())
        out[key] = dict(E, cols={c: v for c, v in E["cols"].items() if c not in bad})
    return out


# ------------------------------------------------------------------------------------------------ repliche

def _rep(task):
    """Una replica di un blocco: ranghi pesati di ogni colonna, Pearson con i ranghi di ciascuna GT."""
    key, b = task
    J = _J[key]
    c = J["counts"][b]
    w = c[J["sa"]] * c[J["sb"]]
    k = w > 0
    w = w[k]
    rg = {g: diag.wranks(x[k], w) for g, x in J["gt"].items()}
    out = {}
    for m, x in J["cols"].items():
        r = diag.wranks(x[k], w)
        for g in fp.GTS:
            out[(m, g)] = diag.wpearson(r, rg[g], w)
    return out


def contrasts(cols: list) -> list[tuple[str, str, str]]:
    """(gt, a, b) dei delta della sez. 3 presenti nel blocco."""
    out = []
    for arm in ARMS:
        out += [("sr", T_COL, f"{arm}|shape"), ("fr", T_COL, f"{arm}|form_cal"), ("fr", f"{arm}|comb", f"{arm}|form_cal")]
    out += [(g, T_COL, ICP) for g in ("sr", "fr")]
    out += [(g, T_COL, f"B-{m}|vb_{g}") for m in B_MODELS for g in ("sr", "fr")]
    return [x for x in out if x[1] in cols and x[2] in cols]


def run(workers: int) -> None:
    global _J
    t0 = time.time()
    info = {}
    ct = calib_teacher()
    print(f"[g0-stats] c_T = {ct['c_T']:.4f} su {ct['n_pairs']} coppie held-out; K6 {ct['K6']['pass']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    entries = {}
    for v in VIEWS:
        entries.update(view_entries(v, ct["c_T"], info))
        print(f"[g0-stats] {v}: {info[v]} ({time.time() - t0:.0f}s)", flush=True)
    for n in EXTRA:
        entries.update(set_entries(n, ct["c_T"], info))
        print(f"[g0-stats] {n}: {info[n]} ({time.time() - t0:.0f}s)", flush=True)
    entries = split_nonfinite(entries, info)
    _J = entries
    tasks = [(key, b) for key in entries for b in range(diag.N_BOOT + 1)]
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_rep, tasks, chunksize=8)
    V = {key: {} for key in entries}
    for (key, b), r in zip(tasks, reps):
        for m, x in r.items():
            V[key].setdefault(m, np.empty(diag.N_BOOT + 1))[b] = x
    print(f"[g0-stats] {len(tasks)} repliche ({time.time() - t0:.0f}s)", flush=True)
    recs, drecs = [], []
    for key, Vk in V.items():
        E = entries[key]
        for (m, g), x in Vk.items():
            p, lo, hi, _ = diag.ci(x)
            recs.append({"set": key[0], "group": key[1], "gt": g, "column": m, "label": label(m), "rho": p,
                         "ci_low": lo, "ci_high": hi, "n_rows": int(len(E["sa"])), "seed": str(E["seed"])})
        for g, a, b in contrasts(list(E["cols"])):
            d = Vk[(a, g)] - Vk[(b, g)]
            p, lo, hi, ple = diag.ci(d)
            drecs.append({"set": key[0], "group": key[1], "gt": g, "a": a, "b": b, "delta": p, "ci_low": lo,
                          "ci_high": hi, "p_le0": ple, "a_point": float(Vk[(a, g)][0]), "b_point": float(Vk[(b, g)][0]),
                          "n_rows": int(len(E["sa"])), "seed": str(E["seed"])})
    Sx, Dx = pd.DataFrame(recs), pd.DataFrame(drecs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp, dp = (EV / "g0_spearman.csv", EV / "g0_delta.csv") if not FAKE else (OUT_DIR / "spearman.csv", OUT_DIR / "delta.csv")
    Sx.to_csv(sp, index=False)
    Dx.to_csv(dp, index=False)
    controls = {"c_T": ct, "sets": info, "K1": k1(Sx)}
    if FAKE:
        controls["K5"] = k5(V, entries)
    else:
        controls.update(teacher_controls())
    controls["seconds"] = time.time() - t0
    diag.atomic_json(OUT_DIR / "controls.json", controls)
    diag.atomic_json(OUT_DIR / "readings.json", readings(Dx))
    if not FAKE:
        diag.atomic_savez(OUT_DIR / "reps.npz", keys=np.asarray([f"{k[0]}|{k[1]}|{m}|{g}" for k in V for (m, g) in V[k]]),
                          V=np.stack([x for k in V for x in V[k].values()]))
    print(f"[g0-stats] {len(Sx)} valori, {len(Dx)} delta; K1 {controls['K1']['pass']} (scarto {controls['K1']['max_abs_diff']:.1e})"
          + (f"; K5 {controls['K5']['pass']}" if FAKE else "") + f" in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------------ controlli

def k1(Sx: pd.DataFrame) -> dict:
    """K1: bracci e B sulle righe primarie = d3_spearman.csv (B su FLAME = d1_spearman.csv), ICP mm = factorized_paired.csv
    (FaceScape, HIFI3D): punto ed estremi dell'IC, righe."""
    ref = {}
    for r in csv.DictReader(open(EV / "d3_spearman.csv")):
        if r["rows"] == "cross":
            ref[(r["test"], r["column"], r["gt"])] = (float(r["rho"]), float(r["ci_low"]), float(r["ci_high"]),
                                                      int(r["n_rows"]))
    for r in csv.DictReader(open(EV / "d1_spearman.csv")):
        if r["set"] == "flame2023_s1" and r["method"].startswith("B-"):
            ref[("flame2023_s1", r["method"].split(" ")[0], r["gt"])] = (float(r["rho"]), float(r["ci_low"]),
                                                                         float(r["ci_high"]), int(r["n_rows"]))
    for r in csv.DictReader(open(diag.TV3 / "factorized_paired.csv")):
        if r["arm"] == "baseline" and r["distance"] == fp.BASELINES["mm_rigid_icp_chamfer"] and r["kind"] == "rho" \
                and r["baseline"] == "-" and r["domain"] in ("hifi3d", "facescape"):
            ref[(r["domain"], "ICP", r["gt"])] = (float(r["arm_point"]), float(r["arm_ci_low"]), float(r["arm_ci_high"]),
                                                  int(r["n_rows"]))
    cells, worst = {}, 0.0
    x = Sx[Sx["group"] == NOCROP]
    for _, r in x.iterrows():
        col, g = r["column"], r["gt"]
        if col == ICP:
            key = (r["set"], "ICP", g)
        elif col.endswith("|shape") and g == "sr":
            key = (r["set"], col.replace("|shape", "|dP|shape"), g)
        elif col.endswith("|form_cal") and g == "fr":
            key = (r["set"], col.replace("|form_cal", "|dP|form"), g)
        elif col.startswith("B-") and col.endswith(f"vb_{g}"):
            key = (r["set"], col, g)
        else:
            continue
        hit = ref.get(key)
        if hit is None:
            continue
        diff = max(abs(r["rho"] - hit[0]), abs(r["ci_low"] - hit[1]), abs(r["ci_high"] - hit[2]))
        worst = max(worst, diff)
        cells[f"{r['set']}|{col}|{g}"] = {"max_abs_diff": diff, "rows": int(r["n_rows"]), "rows_published": hit[3],
                                          "rows_equal": int(r["n_rows"]) == hit[3]}
    ok = bool(cells) and worst <= 1e-9 and all(c["rows_equal"] for c in cells.values())
    return {"cells": cells, "n_cells": len(cells), "max_abs_diff": worst, "pass": bool(ok)}


def k5(V: dict, entries: dict) -> dict:
    """K5 (insegnante finto = d_P s1234): delta insegnante - d_P s1234 e combinazione s1234 - d_F cal. s1234 nulli."""
    out, worst = {}, 0.0
    for key, Vk in V.items():
        cols = entries[key]["cols"]
        for g, a, b in (("sr", T_COL, f"{ARMS[0]}|shape"), ("fr", f"{ARMS[0]}|comb", f"{ARMS[0]}|form_cal")):
            if (a, g) in Vk and (b, g) in Vk and a in cols and b in cols:
                d = float(np.nanmax(np.abs(Vk[(a, g)] - Vk[(b, g)])))
                worst = max(worst, d)
                out[f"{key[0]}|{key[1]}|{a}-{b}"] = d
    return {"cells": out, "max_abs_diff": worst, "pass": bool(out) and worst <= 1e-12}


def teacher_controls() -> dict:
    """K2 (catena su HIFI3D), K3 (mesh rigenerate), K4 (embedding), frame check: dai json dei passi precedenti."""
    out = {"frame_check": {}, "K4": {}, "K3": {}}
    for name in SETS + diag.SEEN:
        d = G0 / "teacher" / name
        fc = json.loads((d / "frame_check.json").read_text())
        out["frame_check"][name] = {k: fc[k] for k in ("expected", "chosen", "agrees", "score")}
        tm = json.loads((d / "timing.json").read_text())
        out["K4"][name] = tm["K4"]
    out["K2"] = json.loads((G0 / "teacher" / "hifi3d" / "k2.json").read_text())
    for s in ("flame2023_s1", "famos", "heldout"):
        p = G0 / f"meshes_{s}.json"
        out["K3"][s] = json.loads(p.read_text())["K3"] if p.exists() else {"missing": str(p), "pass": False}
    return out


# ------------------------------------------------------------------------------------------------ lettura

def readings(Dx: pd.DataFrame) -> dict:
    """Sez. 6: P(seme), verdetto, SOTTO; FLAME con la stessa regola (descrittivo)."""
    x = Dx[(Dx["group"] == NOCROP) & (Dx["gt"] == "sr") & (Dx["a"] == T_COL)]
    out = {"arms": {}}
    for arm in ARMS:
        y = x[x["b"] == f"{arm}|shape"].set_index("set")
        cell = {s: [float(y.loc[s, c]) for c in ("delta", "ci_low", "ci_high")] for s in y.index}
        up = [s for s in VIEWS if s in cell and cell[s][0] >= SR_MIN and cell[s][1] > 0]
        down = [s for s in VIEWS if s in cell and cell[s][2] < 0]
        famos = "famos" in cell and cell["famos"][0] >= SR_MIN and cell["famos"][1] > 0
        flame = "flame2023_s1" in cell and cell["flame2023_s1"][0] >= SR_MIN and cell["flame2023_s1"][1] > 0
        out["arms"][arm] = {"families_up": up, "famos_up": bool(famos), "P": bool(len(up) >= N_FAM or famos),
                            "families_below": down, "below": bool(len(down) >= N_FAM),
                            "flame_up_descriptive": bool(flame), "delta_sr": cell}
    n_p = sum(out["arms"][a]["P"] for a in ARMS)
    complete = all(all(s in out["arms"][a]["delta_sr"] for s in VIEWS + ("famos",)) for a in ARMS)
    out["verdict"] = ("non valutabile" if not complete else
                      "PROMETTENTE" if n_p == 2 else "NON RISOLTO" if n_p == 1 else "NO")
    out["below"] = bool(all(out["arms"][a]["below"] for a in ARMS))
    return out


# ------------------------------------------------------------------------------------------------ markdown

NOTES = [               # note del coder, scritte DOPO i numeri (commento, non regole)
    "- Verdetto preregistrato **NO**, e **SOTTO** in entrambi i semi: l'insegnante ordina le coppie peggio di d_P su "
    "tutte le famiglie, con tutti gli IC sotto 0. Delta con SR (s1234 / s2345): FaceScape -0.399 / -0.405, HIFI3D -0.401 / "
    "-0.393, FaceVerse neutra -0.167 / -0.206, FLAME -0.461 / -0.461, FaMoS TRAIN (reale) -0.444 / -0.433. Sta anche "
    "sotto ICP mm (-0.129, -0.145, -0.217) e sotto B (da -0.216 a -0.506). E' l'esito atteso dal precedente dell'8 "
    "ottobre (HIFI3D, GT maxabs: 0.242; qui con SR 0.220).",
    "- **Non e' una pipeline rotta** (controllo di sanita' POST HOC, `g0_sanity.py`, `g0/sanity_posthoc.json`): sulle "
    "stesse mesh l'insegnante riconosce le persone su ogni insieme, AUC stessa persona contro persone diverse da 0.969 "
    "(FaceVerse neutra) a 0.9999 (FaMoS), rank-1 contro le original da 0.98 a 1.00, FLAME e FaMoS compresi. Le distanze "
    "fra persone diverse stanno in una banda stretta e lontana da quelle della stessa persona (HIFI3D: mediana 0.595, IQR "
    "0.544-0.643, contro 0.033 per la stessa persona; FaceScape 0.608, IQR 0.548-0.662): ArcFace separa le identita' ma "
    "ordina male le persone diverse per somiglianza di forma, che e' cio' che lo Spearman con FR/SR misura. "
    "Riconoscere non e' graduare (diario, 6 ottobre), ora con le GT FR/SR e su cinque famiglie.",
    "- FR (descrittivo): da solo l'insegnante, cieco alla taglia, fa 0.13-0.34. La combinazione con la taglia del braccio "
    "(c_T = 0.101 dai soli held-out) arriva a 0.61 su HIFI3D, 0.68 su FLAME, 0.63 su FaMoS, ma resta sotto d_F cal. in "
    "ogni insieme e seme (da -0.116 a -0.315, IC sotto 0): quel FR viene in gran parte da S, non da d_T.",
    "- Righe col crop (secondario): l'insegnante resta sotto d_P (da -0.105 a -0.324). L'unico delta positivo di G0 e' "
    "insegnante - ICP mm sulle righe col crop di FaceScape, +0.077 [+0.021, +0.141]: e' ICP mm che crolla col crop, non "
    "l'insegnante che migliora.",
    "- Conseguenza (sez. 6, \"cosa cambierebbe\"): l'insegnante ArcFace-render non e' una leva per il benchmark FR/SR e "
    "la distillazione verso di esso non ha appoggio; resta una baseline di riconoscimento. G0 non dice nulla su una "
    "distillazione con obiettivo di riconoscimento, che e' un'altra domanda.",
    "- Detector della calibrazione: fallisce su tutti i render noisy (come nel gate) e, oltre a noisy, su 497 render BFM "
    "su 1.200, 120 FaceVerse, 6 FLAME: il crop di quegli insiemi e' calibrato su meno detection. Non ho verificato se un "
    "altro crop cambi lo Spearman; con il riconoscimento intatto (BFM AUC 0.974) e margini da -0.17 a -0.46 non e' "
    "plausibile che cambi la lettura.",
    "- Costo: 0.71 s per mesh a un thread (render 0.28 s, ArcFace 0.43 s; up60k 1.06 s), su AMD EPYC 7302; una tantum "
    "per insieme circa 130 s di calibrazione con 48 processi (detector su tutti i render ombreggiati) e 2-51 s di "
    "camera. Per contesto, non rimisurati e da altri nodi: operatori dei bracci 4.1 s per mesh (campione di D4), "
    "\"ArcFace migliore\" 0.57 s (`indomain_recog`).",
    "- Deviazioni dal protocollo. (1) K3: le tabelle di scala non hanno il numero di vertici; ho confrontato per mesh "
    "area grezza e maxabs (scarto relativo 0 su 2.900 mesh) e gli intervalli di vertici per etichetta contro `gen.json` "
    "di D1 e D3 (uguali; per gli held-out non c'e' riferimento). (2) Il frame check usa il verdetto del detector piu' il "
    "vertice piu' vicino alla camera nel terzo centrale del bbox; su HIFI3D x180 passa 2 soggetti su 4 (le altre "
    "rotazioni 0); tutti gli 8 insiemi concordano con l'attesa. (3) Il costo e' misurato su un altro nodo (a256-t4-01) "
    "rispetto al job dell'insegnante (i256-a10-09): |Z - teacher.npz| 4.2e-5. (4) K2 2.5e-4, entro 1e-3. (5) Aggiunto "
    "dopo i numeri: il controllo di sanita'. (6) Restano 2 png di controllo per insieme in `g0/teacher/*/control`, fuori "
    "da git; i render sono stati cancellati con /tmp.",
]

PILOT = [  # distill_pilot/summary.md (7 ottobre), riga di riferimento e insegnante; DESCRITTIVO, copiato com'e'
    "| dominio | studente, conv. BFM (riferimento) | studente, conv. ICT | insegnante, normal map 3 viste | "
    "insegnante, ombreggiato | congiunto BFM+ICT | studente / insegnante (conv. BFM) |",
    "| --- | --- | --- | --- | --- | --- | --- |",
    "| FaceVerse con espressioni | 0.332 [0.299, 0.366] | 0.570 [0.520, 0.618] | 0.867 [0.843, 0.889] | "
    "0.750 [0.724, 0.775] | 0.680 [0.634, 0.723] | 0.383 [0.344, 0.422] |",
    "| HIFI3D neutre | 0.224 [0.207, 0.244] | 0.355 [0.322, 0.392] | 0.998 [0.995, 1.000] | 0.984 [0.974, 0.992] | "
    "0.397 [0.363, 0.434] | 0.224 [0.208, 0.245] |",
]


def f3(p, lo, hi, sign=True) -> str:
    s = "+" if sign else ""
    return f"{p:{s}.3f} [{lo:{s}.3f}, {hi:{s}.3f}]"


def summary() -> None:
    Sx = pd.read_csv(EV / "g0_spearman.csv")
    Dx = pd.read_csv(EV / "g0_delta.csv")
    C = json.loads((G0 / "controls.json").read_text())
    Rd = json.loads((G0 / "readings.json").read_text())
    cost = json.loads((G0 / "cost.json").read_text()) if (G0 / "cost.json").exists() else None
    sha = (EV / "PROTOCOL_G0.sha256").read_text().split()[0]

    def rho(s, g, gt, col):
        r = Sx[(Sx["set"] == s) & (Sx["group"] == g) & (Sx["gt"] == gt) & (Sx["column"] == col)]
        return f3(r.iloc[0].rho, r.iloc[0].ci_low, r.iloc[0].ci_high, False) if len(r) else "n.d."

    def dl(s, g, gt, a, b):
        r = Dx[(Dx["set"] == s) & (Dx["group"].str.startswith(g)) & (Dx["gt"] == gt) & (Dx["a"] == a) & (Dx["b"] == b)]
        return f3(r.iloc[0].delta, r.iloc[0].ci_low, r.iloc[0].ci_high) if len(r) else "n.d."

    def nrows(s, g):
        r = Sx[(Sx["set"] == s) & (Sx["group"] == g)]
        return int(r.iloc[0].n_rows) if len(r) else 0

    seeds = ", ".join(f"{s} {C['sets'][s]['seed']}" for s in SETS)
    md = ["# Gate G0: l'insegnante ArcFace-render come metrica contro le GT FR/SR, risultati", "",
          f"Protocollo `PROTOCOL_G0.md` (sha256 {sha[:8]}..., commit 0b4a426), scritto prima dei numeri. Insegnante: normal "
          "map, camera unica per insieme, yaw 0 / -30 / +30, 512 px, crop fisso calibrato sui render ombreggiati di tutte "
          "le mesh dell'insieme (dipendenza transduttiva dalle sole geometrie, come il gate), ArcFace `buffalo_l` "
          "`w600k_r50`, media delle 3 viste poi L2, d_T = 1 - coseno. Generato da `aau/diagnostics/g0_stats.py`; numeri "
          "in `g0_spearman.csv`, `g0_delta.csv`, `g0/controls.json`, `g0/readings.json`, `g0/cost.json`. Spearman "
          "pesato per coppia di mesh, IC 95% percentile, 1000 repliche per soggetto coi semi pubblicati (" + seeds + ").",
          "", "## Lettura preregistrata (sez. 6)", "",
          f"- **Verdetto: {Rd['verdict']}**" + (" e SOTTO (l'insegnante ordina peggio di d_P in almeno 2 famiglie su 3 "
                                                 "in entrambi i semi)" if Rd["below"] else "") + "."]
    for arm in ARMS:
        a = Rd["arms"][arm]
        md.append(f"  - {arm}: P = {a['P']} (Delta_SR >= +0.05 con IC > 0 in {len(a['families_up'])} famiglie "
                  f"{a['families_up']}; FaMoS TRAIN {a['famos_up']}); IC sotto 0 in {a['families_below']}; FLAME "
                  f"(descrittivo) {a['flame_up_descriptive']}.")
    md += ["", "Note del coder, scritte DOPO i numeri (commento, non regole):", ""] + (NOTES or ["- (da scrivere)"])
    md += ["", "## SR: insegnante contro d_P, ICP mm e B (nocrop_cross; FLAME e FaMoS: righe di D1 / D3)", "",
           "| insieme | righe | insegnante | d_P s1234 | d_P s2345 | ICP mm | B-GNM vb_sr | B-FLAME vb_sr |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SETS:
        md.append(f"| {s} | {nrows(s, NOCROP)} | **{rho(s, NOCROP, 'sr', T_COL)}** | {rho(s, NOCROP, 'sr', ARMS[0] + '|shape')} | "
                  f"{rho(s, NOCROP, 'sr', ARMS[1] + '|shape')} | {rho(s, NOCROP, 'sr', ICP)} | "
                  f"{rho(s, NOCROP, 'sr', 'B-gnm|vb_sr')} | {rho(s, NOCROP, 'sr', 'B-flame2023|vb_sr')} |")
    md += ["", "Delta appaiati, insegnante - confronto (SR):", "",
           "| insieme | - d_P s1234 (regola) | - d_P s2345 (regola) | - ICP mm | - B-GNM | - B-FLAME |",
           "| --- | --- | --- | --- | --- | --- |"]
    for s in SETS:
        md.append(f"| {s} | **{dl(s, NOCROP, 'sr', T_COL, ARMS[0] + '|shape')}** | **{dl(s, NOCROP, 'sr', T_COL, ARMS[1] + '|shape')}** | "
                  f"{dl(s, NOCROP, 'sr', T_COL, ICP)} | {dl(s, NOCROP, 'sr', T_COL, 'B-gnm|vb_sr')} | "
                  f"{dl(s, NOCROP, 'sr', T_COL, 'B-flame2023|vb_sr')} |")
    md += ["", "B su FLAME: B-GNM incrociato, B-FLAME prior esatto (solo riferimento). ICP mm su FLAME e FaMoS e B su "
               "FaMoS: non esistono e non si sono calcolati (sez. 3).", "",
           "## FR (descrittivo): insegnante, d_F calibrata e combinazione insegnante + taglia", "",
           f"c_T = {C['c_T']['c_T']:.4f} (mediana d_P GT {C['c_T']['median_dP_gt']:.4f} / mediana d_T "
           f"{C['c_T']['median_d_T']:.4f}, {C['c_T']['n_pairs']} coppie degli held-out bfm, ict, gnm; per dominio " +
           ", ".join(f"{d} {v['c_T']:.3f}" for d, v in C["c_T"]["by_domain"].items()) + ").", "",
           "| insieme | insegnante | d_F cal. s1234 | d_F cal. s2345 | comb. s1234 | comb. s2345 | ICP mm | B-GNM vb_fr | B-FLAME vb_fr |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SETS:
        md.append(f"| {s} | {rho(s, NOCROP, 'fr', T_COL)} | {rho(s, NOCROP, 'fr', ARMS[0] + '|form_cal')} | "
                  f"{rho(s, NOCROP, 'fr', ARMS[1] + '|form_cal')} | {rho(s, NOCROP, 'fr', ARMS[0] + '|comb')} | "
                  f"{rho(s, NOCROP, 'fr', ARMS[1] + '|comb')} | {rho(s, NOCROP, 'fr', ICP)} | "
                  f"{rho(s, NOCROP, 'fr', 'B-gnm|vb_fr')} | {rho(s, NOCROP, 'fr', 'B-flame2023|vb_fr')} |")
    md += ["", "| insieme | insegnante - d_F cal. s1234 | - d_F cal. s2345 | comb. - d_F cal. s1234 | comb. - d_F cal. s2345 | "
               "insegnante - ICP mm | - B-GNM | - B-FLAME |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in SETS:
        md.append(f"| {s} | {dl(s, NOCROP, 'fr', T_COL, ARMS[0] + '|form_cal')} | {dl(s, NOCROP, 'fr', T_COL, ARMS[1] + '|form_cal')} | "
                  f"{dl(s, NOCROP, 'fr', ARMS[0] + '|comb', ARMS[0] + '|form_cal')} | "
                  f"{dl(s, NOCROP, 'fr', ARMS[1] + '|comb', ARMS[1] + '|form_cal')} | {dl(s, NOCROP, 'fr', T_COL, ICP)} | "
                  f"{dl(s, NOCROP, 'fr', T_COL, 'B-gnm|vb_fr')} | {dl(s, NOCROP, 'fr', T_COL, 'B-flame2023|vb_fr')} |")
    if C["sets"].get("nonfinite"):
        md += ["", "Colonne con valori non finiti sulle righe del primario (delta calcolato a parte sulle righe finite): " +
               ", ".join(f"{k} {v} righe" for k, v in C["sets"]["nonfinite"].items()) + "."]
    md += ["", "## Righe col crop (secondario; maschera comune)", "",
           "| vista | righe | insegnante SR | d_P s1234 SR | d_P s2345 SR | insegnante - d_P s1234 | insegnante - d_P s2345 | "
           "insegnante FR | d_F cal. s1234 FR | insegnante - d_F cal. s1234 | insegnante - ICP mm (SR) |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in VIEWS:
        md.append(f"| {s} | {nrows(s, CROP)} | {rho(s, CROP, 'sr', T_COL)} | {rho(s, CROP, 'sr', ARMS[0] + '|shape')} | "
                  f"{rho(s, CROP, 'sr', ARMS[1] + '|shape')} | {dl(s, CROP, 'sr', T_COL, ARMS[0] + '|shape')} | "
                  f"{dl(s, CROP, 'sr', T_COL, ARMS[1] + '|shape')} | {rho(s, CROP, 'fr', T_COL)} | "
                  f"{rho(s, CROP, 'fr', ARMS[0] + '|form_cal')} | {dl(s, CROP, 'fr', T_COL, ARMS[0] + '|form_cal')} | "
                  f"{dl(s, CROP, 'sr', T_COL, ICP)} |")
    if cost:
        o = cost["overall"]
        md += ["", "## Costo dell'insegnante per mesh (sez. 7)", "",
               f"{cost['sample']}, un processo a un thread, in memoria; nodo {cost['host']} ({cost['cpu']}). Controllo: "
               f"|Z - teacher.npz| massimo {cost['max_abs_diff_Z']:.1e}.", "",
               "| voce | mediana s | p10 s | p90 s |", "| --- | --- | --- | --- |"]
        for k, lab in (("render_s", "lettura + 3 normal map"), ("embed_s", "3 ritagli + 3 forward ArcFace + media"),
                       ("total_s", "totale per mesh")):
            md.append(f"| {lab} | {o[k]['median']:.3f} | {o[k]['p10']:.3f} | {o[k]['p90']:.3f} |")
        md += ["", "| etichetta | totale mediana s | p90 s |", "| --- | --- | --- |"]
        for t, b in cost["by_label"].items():
            md.append(f"| {t} | {b['total_s']['median']:.3f} | {b['total_s']['p90']:.3f} |")
    md += ["", "Una tantum per insieme e throughput dei job (secondi di parete, N processi):", "",
           "| insieme | mesh | processi | camera | render ombreggiati | calibrazione | normal map | embedding | totale | "
           "detector fallito (noisy, totale) |", "| --- |" + " --- |" * 9]
    for name in SETS + diag.SEEN:
        tm = json.loads((G0 / "teacher" / name / "timing.json").read_text())
        det = tm["detector"]
        md.append(f"| {name} | {tm['n_meshes']} | {tm['workers']} | {tm['camera_s']:.0f} | {tm['render_shaded_s']:.0f} | "
                  f"{tm['calibration_s']:.0f} | {tm['render_normals_s']:.0f} | {tm['embed_s']:.0f} | {tm['total_s']:.0f} | "
                  f"{det.get('noisy', {}).get('n_failed', 0)}/{det.get('noisy', {}).get('n', 0)}, "
                  f"{det['total']['n_failed']}/{det['total']['n']} |")
    md += ["", "## Il pilota della distillazione e il suo insegnante (DESCRITTIVO)", "",
           "Riconoscimento d'identita', rank-1 sulle 5 topologie senza crop, da `aau/runs/distill_pilot/summary.md` "
           "(7 ottobre, protocollo `distill_pilot/protocol.md`), copiati come sono: non sono distanze graduate e non "
           "entrano nella lettura di G0.", ""] + PILOT
    md += ["", "## Controlli", "",
           f"- K1 riferimenti (bracci e B = `d3_spearman.csv` / `d1_spearman.csv`, ICP mm = `factorized_paired.csv`): "
           f"{C['K1']['pass']}, {C['K1']['n_cells']} celle, scarto massimo {C['K1']['max_abs_diff']:.1e}.",
           f"- K2 catena dell'insegnante su HIFI3D contro il gate: {C['K2']['pass']} (soggetti uguali "
           f"{C['K2']['subjects_equal_gate']}, |E - gate| massimo {C['K2']['max_abs_diff_E']:.1e}).",
           "- K3 mesh rigenerate: " + "; ".join(f"{k} {v.get('pass')} (scarto relativo massimo {v.get('max_rel_diff', float('nan')):.1e}, "
                                                f"up60k accettate {v.get('n_up60k_accepted', 0)})" for k, v in C["K3"].items()) + ".",
           "- K4 embedding: " + ", ".join(f"{k} {v['n']}" for k, v in C["K4"].items()) + " mesh, norme entro "
           f"{max(v['max_abs_norm_minus_1'] for v in C['K4'].values()):.1e}.",
           "- K5 (insegnante finto = d_P s1234, `g0_test.py`, `g0/test/controls.json`): " + (
               lambda t: f"{t['K5']['pass']}, scarto {t['K5']['max_abs_diff']:.1e} su {len(t['K5']['cells'])} celle; "
                         f"K1 del test {t['K1']['pass']}.")(json.loads((G0 / "test" / "controls.json").read_text())),
           f"- K6 c dei bracci ricalcolata come c_T: {C['c_T']['K6']['pass']} (" +
           ", ".join(f"{a} {C['c_T']['K6'][a]['c']:.4f} contro {C['c_T']['K6'][a]['c_median_csv']:.4f}" for a in ARMS) + ").",
           "- Frame check: " + ", ".join(f"{k} {v['chosen']} (attesa {v['expected']})" for k, v in C["frame_check"].items()) + ".",
           "", "**Licenza.** I pesi di insightface (`buffalo_l`: detector e `w600k_r50`) sono distribuiti per la sola "
               "ricerca non commerciale; uno studente distillato da loro ne erediterebbe probabilmente il vincolo."]
    (EV / "results_g0.md").write_text("\n".join(md) + "\n")
    print(f"[g0-stats] -> {EV / 'results_g0.md'}", flush=True)


def main() -> None:
    global FAKE, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--fake-teacher", action="store_true", help="K5: insegnante = d_P di s1234, uscite in g0/test")
    a = ap.parse_args()
    if a.fake_teacher:
        FAKE, OUT_DIR = True, G0 / "test"
        run(a.workers)
        return
    if not a.summary_only:
        run(a.workers)
    summary()


if __name__ == "__main__":
    main()
