#!/usr/bin/env python3
"""Delta appaiati dei bracci di factorized_protocol.md contro le baseline in mm, sulle STESSE righe e repliche, e
analisi della taglia dell'emendamento 4.

    v3_work/unified_gt/run.sh v3_work/trainer/tools/fact_paired.py [--workers 32]

Righe, GT e seme per (dominio, gruppo): quelli di aau/baselines_mm/blmm_eval.py (importato: righe di E12, colonne
delle baseline, GT FR / SR / maxabs, seme della differenza pubblicata e108 - Chamfer eval), gruppo primario del
dominio (HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse ``mesh_pair_nocrop``); FaMoS TEST: righe di
``blmm_eval.famos``, blocco scan gallery -> scan, repliche ``famos_eval.bootstrap_counts(15, n, 1234)``. Colonne dei
bracci dagli embedding del passo ``form`` (ultimo checkpoint EMA, 21.096 passi; C3M e123 ed e205), FaMoS dai latenti
in cache di eval_famos_v3.py: d_F e d_P per i fattorizzati, d_F calibrata (``form_cal``: c di
factorized_calibration.csv, tools/fact_calib.py; ``form_cal_ls``: c dei minimi quadrati), ||z|| per ctrlfr, z_F e u
per dual. Baseline: ICP + Chamfer in mm e cs, NICP su template in mm, NICP per coppia cs, taglia stimata e oracolo
(|delta log S|), e108, e le composizioni d_F = sqrt((S_i - S_j)^2 + S_i S_j d_P^2) con S oracolo (centroid size di FR)
o stimata (centroid size robusta della mesh): emendamento 5, d_P = k_ICP x ICP cs / CS_ref (oracolo e stimata) e
d_P = k_NICP x NICP per coppia cs / CS_ref (stimata), k degli held-out sintetici (factorized_calibration_bl.csv,
fact_calib.py calib-bl); quelle non calibrate dell'emendamento 4 restano come ``*_raw``. Per ogni replica, con FR e SR: Spearman
(``rho``); con FR anche (a) Spearman parziale dato l'oracolo (``partial``: ranghi del metodo e della GT regrediti su
[1, r(o), r(o)^2], Pearson dei residui) e (b) Spearman nel quintile basso di o (``q20``, soglia fissa sulle righe).
Delta (braccio - riferimento) contro le baseline e, per la regola dual, fra bracci dello stesso seme. Maschera COMUNE
per dominio (righe con tutte le distanze finite). IC 95% percentile, P(delta <= 0); righe con baseline "-" = valore
del metodo. Uscita: aau/runs/evidence/trainer_v3/factorized_paired.csv.

Esplorativa, NON preregistrata (emendamento 5, sez. 3): su HIFI3D d_F(c) per c della griglia ``C_GRID`` (factorized
s1234, s2345, C3M e205), valori e delta contro ICP mm, NICP cs, taglia stimata + ICP cs cal. ->
factorized_explore.csv; c e k "ideali" sul test (mediana d_P GT / mediana d_P del metodo sulle righe della maschera,
HIFI3D e dev FaceScape; informazione, non parametro) -> factorized_calibration_test.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
TRAINER = THIS.parent
REPO = TRAINER.parents[1]
sys.path.insert(0, str(REPO / "aau/baselines_mm"))

import blmm_eval as be  # noqa: E402  (sola lettura)

blmm = be.blmm

# NIENTE moduli del trainer: blmm_eval importa aau/baselines/common.py, che ha lo stesso nome di
# v3_work/trainer/common.py. Distanze e percorsi come factorized_v3.model_distances e fact_summary (copiati).
EV = REPO / "aau/runs/evidence/trainer_v3"
EVAL = EV / "ablations/c3f_eval"
CALIB = EV / "factorized_calibration.csv"              # tools/fact_calib.py
CALIB_BL = EV / "factorized_calibration_bl.csv"        # tools/fact_calib.py calib-bl (emendamento 5)
ARMS, SEEDS = ("factorized", "factorized2", "ctrlfr", "dual"), (1234, 2345)
# (prefisso della colonna, variante del tag, epoca): C3F all'ultimo checkpoint, C3M ai due checkpoint valutati
MODELS = [(f"{a}_s{s}", a + ("" if s == 1234 else f"s{s}"), "072") for a in ARMS for s in SEEDS] + \
         [(f"factorizedc3m_e{e}", "factorizedc3m", e) for e in ("123", "205")]
FORM_DIR = {"hifi": "form_hifi", "devfs": "form_devfs", "fv": "form_fv_expr"}
FAMOS_LAT = REPO / "datasets/FAMOS/eval/embeddings"


def embeddings(dom: str, v: str, e: str):
    hits = sorted((EVAL / FORM_DIR[dom]).glob(f"data_*/scale_v3{v}fulle{e}*/zs_zeroshot/embeddings.npz"))
    return hits[0] if hits else None


def calibration() -> dict:
    """{prefisso: (c, c_LS)} da factorized_calibration.csv (fact_calib.load, copiata)."""
    if not CALIB.exists():
        return {}
    return {r["key"]: (float(r["c_median"]), float(r["c_ls"])) for r in csv.DictReader(open(CALIB))}


def calibration_bl() -> dict:
    """{icp_cs, nicp_cs: k} da factorized_calibration_bl.csv (vuoto se assente)."""
    if not CALIB_BL.exists():
        return {}
    return {r["key"]: float(r["k_median"]) for r in csv.DictReader(open(CALIB_BL))}


def distances(Z, i, j, ckpt: Path, cal=None, grid=()) -> dict:
    """factorized_v3.model_distances: fattorizzati -> d_F, d_P (e d_F calibrata con ``cal`` = (c, c_LS); d_F(c) per i
    c di ``grid``, esplorativa); dual -> z_F e u; altrimenti z."""
    import torch
    args = torch.load(ckpt, map_location="cpu", weights_only=False)["args"]
    head = args.get("head", "embed")
    if head in ("factorized", "factorized2"):
        side = json.loads(Path(args["dist_npz"]).with_suffix(".json").read_text())
        dpu = float(side.get("dp_per_unit", side.get("dP_per_unit"))) / float(args.get("gt_scale", 1.0))
        S = np.exp(Z[:, 0])
        dP = np.linalg.norm(Z[i, 1:] - Z[j, 1:], axis=1) * dpu
        out = {"form": np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * dP ** 2), "shape": dP}
        if cal is not None:
            for k, c in zip(("form_cal", "form_cal_ls"), cal):
                out[k] = np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (c * dP) ** 2)
        for c in grid:
            out[f"form_at_c{c:.3f}"] = np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (c * dP) ** 2)
        return out
    if head == "dual":
        L = Z.shape[1] // 2
        return {"zf": np.linalg.norm(Z[i, :L] - Z[j, :L], axis=1), "u": np.linalg.norm(Z[i, L:] - Z[j, L:], axis=1)}
    return {"z": np.linalg.norm(Z[i] - Z[j], axis=1)}


VIEWS = {"hifi3d": ("hifi", "nocrop_cross"), "faceverse": ("fv", "mesh_pair_nocrop"), "facescape": ("devfs", "nocrop_cross"),
         "famos": ("famos", "scan gallery -> scan")}
BASELINES = {"mm_rigid_icp_chamfer": "ICP + Chamfer in mm", "mm_nicp_template": "NICP su template in mm",
             "est_cs": "taglia stimata", "oracle_size": "taglia oracolo", "cs_nicp_p2tri": "NICP per coppia (cs)",
             "cs_rigid_icp_chamfer": "ICP + Chamfer (cs)", "comp_oracle": "oracolo taglia + ICP cs cal.",
             "comp_est": "taglia stimata + ICP cs cal.", "comp_est_nicp": "taglia stimata + NICP cs cal.",
             "comp_oracle_raw": "oracolo taglia + ICP cs (em. 4, non cal.)",
             "comp_est_raw": "taglia stimata + ICP cs (em. 4, non cal.)", "scale_e108": "e108"}
GTS = ("fr", "sr")
# esplorativa (emendamento 5, sez. 3): griglia di c, modelli, riferimenti dei delta
C_GRID = (0.2, 0.3, 0.405, 0.5, 0.75, 1.0)
EXPLORE = ("factorized_s1234", "factorized_s2345", "factorizedc3m_e205")
EXPLORE_BL = ("mm_rigid_icp_chamfer", "cs_nicp_p2tri", "comp_est")
# regola dual (emendamento 4, sez. 5): (dominio, gt, misura, colonna dual, colonna di riferimento) per seme
DUAL_RULE = (("hifi3d", "fr", "rho", "zf", "ctrlfr_s{s}|z"), ("hifi3d", "sr", "rho", "u", "factorized_s{s}|shape"),
             ("hifi3d", "fr", "partial", "zf", "factorized_s{s}|form_cal"),
             ("facescape", "fr", "rho", "zf", "factorized_s{s}|form_cal"),
             ("facescape", "sr", "rho", "u", "factorized_s{s}|shape"))
_J: dict = {}


def form_of(S: np.ndarray, Sg: np.ndarray, dP: np.ndarray) -> np.ndarray:
    """d_F per una matrice (righe S, colonne Sg) di d_P."""
    return np.sqrt((S[:, None] - Sg[None, :]) ** 2 + S[:, None] * Sg[None, :] * dP ** 2)


def compositions(D: dict, S_or, Sg_or, S_est, Sg_est, cs_ref: float) -> None:
    """Composizioni in D: non calibrate (emendamento 4, ``*_raw``) e calibrate con k di held-out (emendamento 5;
    assenti se manca factorized_calibration_bl.csv). Righe S, colonne Sg."""
    dP = D["cs_rigid_icp_chamfer"] / cs_ref
    D["comp_oracle_raw"], D["comp_est_raw"] = form_of(S_or, Sg_or, dP), form_of(S_est, Sg_est, dP)
    k = calibration_bl()
    if "icp_cs" in k:
        D["comp_oracle"], D["comp_est"] = form_of(S_or, Sg_or, k["icp_cs"] * dP), form_of(S_est, Sg_est, k["icp_cs"] * dP)
    if "nicp_cs" in k and "cs_nicp_p2tri" in D:
        D["comp_est_nicp"] = form_of(S_est, Sg_est, k["nicp_cs"] * D["cs_nicp_p2tri"] / cs_ref)


def oracle_S(gt_set: str) -> dict:
    """{id della GT: S di FR (mm)}: lo stesso file controllato da blmm_eval.oracle_scalars."""
    with np.load(be.EVAL_GT / f"{gt_set}_centroid_size.npz") as z:
        return dict(zip([str(s) for s in z["names"]], z["S"].astype(np.float64)))


def rows_for(view: str):
    """Righe del gruppo primario con le colonne delle baseline, delle composizioni e delle GT (come
    blmm_eval.frames_for, topologie tenute)."""
    if view == "hifi3d":
        fr = be.e12m.hifi_frames()[0]
        base_df = fr["all_cross"][0]
    elif view == "faceverse":
        fr = be.e12m.fv_frames()
        base_df = fr["mesh_pair_nocrop"][0]
    else:
        fr = be.e12m.fs_frames()
        base_df = fr["all_cross"][0]
    group = VIEWS[view][1]
    idx = be.zes.Index(be.blmm.subjects(view))
    pub = [m for m in list(be.PUBLISHED) + list(be.REFS) if m in base_df]
    D, _ = be.view_distances(view, idx, set(pub))
    orc = oracle_S(blmm.VIEWS[view]["gt"])
    sc = blmm.scalars_of(view)
    S_or = np.asarray([orc[s] for s, _ in idx.keys])
    S_est = np.asarray([sc[blmm.rel(blmm.mesh_path(view, s, t))]["cs"] for s, t in idx.keys])
    compositions(D, S_or, S_or, S_est, S_est, blmm.params()["domains"][blmm.VIEWS[view]["domain"]]["cs_ref"])
    df = be.add_gts(be.add_columns(base_df, D, idx), be.GT_SET[view])
    df = df.rename(columns={k: f"{v[0]}_{v[1]}" for k, v in be.PUBLISHED.items() if k in df})
    df = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")].reset_index(drop=True)
    if len(df) != len(fr[group][0]):
        raise SystemExit(f"{view}: {len(df)} righe invece delle {len(fr[group][0])} di E12")
    return df, idx, fr[group][2]


def model_columns(view: str, idx) -> dict:
    """{colonna: D (n, n) sulle chiavi di idx} per ogni braccio e seme (e C3M), ultimo checkpoint."""
    dom = VIEWS[view][0]
    cal = calibration()
    out = {}
    for pre, v, e in MODELS:
        p = embeddings(dom, v, e)
        if p is None:
            continue
        with np.load(p, allow_pickle=True) as z:
            Z = np.asarray(z["Z"], np.float64)
            keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
            ckpt = Path(str(z["checkpoint"]))
        pos = {k: r for r, k in enumerate(keys)}
        Z = Z[[pos[k] for k in idx.keys]]
        n = len(Z)
        i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
        grid = C_GRID if view == "hifi3d" and pre in EXPLORE else ()
        for k, d in distances(Z, i, j, ckpt, cal.get(pre), grid).items():
            out[f"{pre}|{k}"] = d.reshape(n, n)
    return out


# ------------------------------------------------------------------------------------------ FaMoS TEST

def famos_frame(n_boot: int):
    """(colonne {metodo: x}, gt {fr, sr}, sa, sb, counts, controlli) del blocco scan gallery -> scan di
    blmm_eval.famos (righe, galleria e repliche copiate da li'), con le colonne dei bracci dai latenti in cache."""
    import famos_common as fc
    import famos_eval as fe
    rows = blmm.famos_manifest()
    names = [r["name"] for r in rows]
    kind = np.asarray([r["kind"] for r in rows])
    role = np.asarray([r["role"] for r in rows])
    with np.load(fc.VIEW_DIR / "gt_matrix.npz") as z:
        gt_names = [str(x) for x in z["names"]]
    s2i = {s: k for k, s in enumerate(gt_names)}
    subj = np.asarray([s2i[r["view_id"]] for r in rows])
    gal = {}
    for k in ("scan", "reg"):
        g = np.flatnonzero((kind == k) & (role == "gallery"))
        gal[k] = g[np.argsort(subj[g])]
    gal_all = np.concatenate([gal["scan"], gal["reg"]])
    col = {k: {int(i): j for j, i in enumerate(gal_all) if kind[i] == k} for k in ("scan", "reg")}
    gc = np.asarray([col["scan"][int(i)] for i in gal["scan"]])
    D = {}
    for mode, metrics in (("mm", ("rigid_icp_chamfer",)), ("cs", ("rigid_icp_chamfer", "nicp_p2tri"))):
        parts = sorted((blmm.OUT_ROOT / "famos" / mode).glob("*_*of*.npz"))
        for step, mets in blmm.METRICS.items():
            ps = [p for p in parts if p.name.startswith(f"{step}_")]
            for m in mets:
                if m not in metrics:
                    continue
                M = np.full((len(rows), len(gal_all)), np.nan)
                for p in ps:
                    with np.load(p) as z:
                        if [str(x) for x in z["gallery"]] != [names[i] for i in gal_all]:
                            raise SystemExit(f"{p}: galleria diversa")
                        M[z["rows"]] = z[m]
                D[f"{mode}_{m}"] = M
    sc = blmm.scalars_of("famos")
    S_est = np.asarray([sc[blmm.rel(blmm.VIEWS["famos"]["dir"] / f"{n}.npz")]["cs"] for n in names])
    orc = oracle_S("famos_test")
    S_or = np.asarray([orc[gt_names[s]] for s in subj])
    D["est_cs"] = np.abs(np.log(S_est)[:, None] - np.log(S_est)[None, gal_all])
    D["oracle_size"] = np.abs(np.log(S_or)[:, None] - np.log(S_or)[None, gal_all])
    compositions(D, S_or, S_or[gal_all], S_est, S_est[gal_all], blmm.params()["domains"]["famos"]["cs_ref"])
    cal = calibration()
    checks = []
    for pre, v, e in MODELS:
        tag = f"v3{v}e{e}"
        lat, res = FAMOS_LAT / f"latent_{tag}full_test_view.npz", EVAL / "famos" / tag / "results.json"
        if not (lat.exists() and res.exists()):
            continue
        with np.load(lat) as z:
            Z = np.stack([np.asarray(z[n], np.float64) for n in names])
        ii, jj = np.repeat(np.arange(len(Z)), len(gal_all)), np.tile(gal_all, len(Z))
        for k, d in distances(Z, ii, jj, Path(json.loads(res.read_text())["checkpoint"]), cal.get(pre)).items():
            D[f"{pre}|{k}"] = d.reshape(len(Z), len(gal_all))
        checks.append((pre, tag))
    q_idx = np.flatnonzero((kind == "scan") & (role == "gallery"))
    qa = np.repeat(q_idx, len(gt_names))
    gb = np.tile(np.arange(len(gt_names)), len(q_idx))
    sa, sb = subj[qa], gb
    keep = (sa != sb) & (sa < sb)
    qa, gb, sa, sb = qa[keep], gb[keep], sa[keep], sb[keep]
    G = {}
    for g in GTS:
        Dg, pos = be.zsum.load_gt(be.EVAL_GT / f"famos_test_{g}.npz")
        G[g] = Dg[np.ix_([pos[s] for s in gt_names], [pos[s] for s in gt_names])][sa, sb]
    cols = {m: M[qa, gc[gb]] for m, M in D.items()}
    counts = [np.ones(len(gt_names), dtype=np.int64)] + list(fe.bootstrap_counts(len(gt_names), n_boot, 1234))
    # controllo: d_F grezza / z contro graded.csv di eval_famos_v3 (stesse righe, punto)
    from scipy.stats import spearmanr
    ctrl = []
    for pre, tag in checks:
        k = f"{pre}|{'form' if f'{pre}|form' in cols else ('zf' if f'{pre}|zf' in cols else 'z')}"
        ref = {r["method"]: r for r in csv.DictReader(open(EVAL / "famos" / tag / "graded.csv"))
               if r["block"] == "scan gallery -> scan" and r["gt"] == "fr"}
        name = f"{tag}full" + {"form": "_form", "zf": "_zf", "z": ""}[k.split("|")[1]]
        if name in ref:
            ctrl.append((k, abs(spearmanr(cols[k], G["fr"]).correlation - float(ref[name]["spearman"]))))
    return cols, G, sa, sb, counts, ctrl


# ----------------------------------------------------------------------------------------- repliche

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def _rep(k: int) -> dict:
    from scipy.stats import rankdata
    J = _J
    c = J["counts"][k]
    wt = c[J["sa"]].astype(np.int64) * c[J["sb"]]
    keep = wt > 0
    w = wt[keep]

    def rep(x):
        return np.repeat(x[keep], w)

    ro = rankdata(rep(J["cols"]["oracle_size"]))
    Q, _ = np.linalg.qr(np.c_[np.ones_like(ro), ro, ro * ro])

    def resid(r):
        e = r - Q @ (Q.T @ r)
        return e if np.linalg.norm(e) > 1e-9 * np.linalg.norm(r - r.mean()) else np.zeros_like(e) * np.nan

    low = rep(J["low"])
    rg = {g: rankdata(rep(J["cols"][f"gt_{g}"])) for g in GTS}
    eg = resid(rg["fr"])
    gq = rankdata(rep(J["cols"]["gt_fr"])[low])
    out = {}
    for m in J["methods"]:
        x = rep(J["cols"][m])
        r = rankdata(x)
        for g in GTS:
            out[(g, "rho", m)] = _pearson(r, rg[g])
        out[("fr", "partial", m)] = _pearson(resid(r), eg)
        out[("fr", "q20", m)] = _pearson(rankdata(x[low]), gq)
    return out


def summarize(V: dict, view: str, M: list, bl: list, n_rows: int, seed: int) -> list[dict]:
    """Righe del csv: valore di ogni metodo (baseline "-"), delta bracci - baseline, delta della regola dual."""
    recs = []

    def rec(g, kind, m, ref, label):
        a = V[(g, kind, m)]
        d = a - V[(g, kind, ref)] if ref else a
        b = d[1:][np.isfinite(d[1:])]
        if not len(b):
            return
        lo, hi = np.percentile(b, [2.5, 97.5])
        alo, ahi = np.percentile(a[1:][np.isfinite(a[1:])], [2.5, 97.5]) if np.isfinite(a[1:]).any() else (np.nan,) * 2
        arm, _, dist = m.partition("|")
        recs.append({"domain": view, "group": VIEWS[view][1], "gt": g, "kind": kind, "arm": arm if dist else "baseline",
                     "distance": dist or BASELINES[m], "baseline": label, "arm_point": a[0],
                     "arm_ci_low": alo, "arm_ci_high": ahi,
                     "baseline_point": V[(g, kind, ref)][0] if ref else np.nan, "delta": d[0], "ci_low": lo,
                     "ci_high": hi, "p_le0": float((b <= 0).mean()), "n_rows": n_rows, "seed": int(seed)})

    for (g, kind, m) in V:
        rec(g, kind, m, None, "-")
        if m in M:
            for b in bl:
                rec(g, kind, m, b, BASELINES[b])
    for s in SEEDS:
        for dom, g, kind, a, ref in DUAL_RULE:
            m, r = f"dual_s{s}|{a}", ref.format(s=s)
            if dom == view and (g, kind, m) in V and (g, kind, r) in V:
                rec(g, kind, m, r, f"braccio {r}")
    return recs


def test_scale(view: str, cols: dict, mask: np.ndarray, M: list) -> list[dict]:
    """Emendamento 5, sez. 3 (iii): c e k "ideali" sul test, mediana(d_P GT) / mediana(d_P del metodo) sulle righe
    della maschera (d_P GT = GT-SR x dP_per_unit del suo json). Informazione, non parametro."""
    dpu = float(json.loads((be.EVAL_GT / f"{be.GT_SET[view]}_sr.json").read_text())["dP_per_unit"])
    t = np.median(cols["gt_sr"][mask] * dpu)
    cs_ref = blmm.params()["domains"][blmm.VIEWS[view]["domain"]]["cs_ref"]
    dp = {m.replace("|", " "): cols[m][mask] for m in M if m.endswith("|shape") or m.endswith("|u")}
    dp.update({b: cols[b][mask] / cs_ref for b in ("cs_rigid_icp_chamfer", "cs_nicp_p2tri") if b in cols})
    return [{"domain": view, "method": k, "median_dP_gt": float(t), "median_dP_method": float(np.median(v)),
             "scale_test": float(t / np.median(v)), "n_rows": int(mask.sum())} for k, v in dp.items()]


def main() -> None:
    global _J
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    recs, controls, explore, ctest = [], [], [], []
    for view in VIEWS:
        if view == "famos":
            cols, G, sa, sb, counts, ctrl = famos_frame(a.n_boot)
            M = [m for m in cols if "|" in m]
            seed = 1234
            controls += [f"famos {k}: |rho - graded.csv| = {d:.1e}" for k, d in ctrl]
            cols.update({f"gt_{g}": G[g] for g in GTS})
        else:
            df, idx, seed = rows_for(view)
            Mc = model_columns(view, idx)
            df = be.add_columns(df, Mc, idx)
            M = [m for m in Mc if "|form_at_c" not in m]
            cols = {m: df[m].to_numpy(np.float64) for m in list(Mc) + [b for b in BASELINES if b in df]}
            cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in GTS})
            subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
            s2i = {s: i for i, s in enumerate(subjects)}
            sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
            rng = np.random.default_rng(seed)
            counts = [np.ones(len(subjects), dtype=np.int64)] + \
                [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(a.n_boot)]
        if not M:
            print(f"[paired] {view}: nessuna colonna dei bracci", flush=True)
            continue
        bl = [b for b in BASELINES if b in cols]
        X = [m for m in cols if "|form_at_c" in m]          # esplorativa: fuori da factorized_paired.csv
        mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
        o = cols["oracle_size"]
        _J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
              "methods": M + X + bl, "low": o[mask] <= np.quantile(o[mask], 0.2)}
        with mp.get_context("fork").Pool(a.workers) as pool:
            reps = pool.map(_rep, range(len(counts)), chunksize=4)
        V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
        recs += summarize({k: v for k, v in V.items() if k[2] not in X}, view, M, bl, int(mask.sum()), seed)
        if X:
            xb = [b for b in EXPLORE_BL if b in cols]
            explore += [r for r in summarize({k: v for k, v in V.items() if k[2] in X or k[2] in xb}, view, X, xb,
                                             int(mask.sum()), seed) if r["arm"] != "baseline"]
        if view in ("hifi3d", "facescape"):
            ctest += test_scale(view, cols, mask, M)
        print(f"[paired] {view}: {len(M)} colonne dei bracci, {len(bl)} baseline, {int(mask.sum())} righe", flush=True)
    out = EV / "factorized_paired.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    (EV / "factorized_paired_controls.json").write_text(json.dumps(controls, indent=1) + "\n")
    pd.DataFrame(explore).to_csv(EV / "factorized_explore.csv", index=False)
    pd.DataFrame(ctest).to_csv(EV / "factorized_calibration_test.csv", index=False)
    for r in ctest:
        print(f"[paired] scala sul test {r['domain']} {r['method']}: {r['scale_test']:.3f}", flush=True)
    print("\n".join(controls), flush=True)
    print(f"[paired] {len(recs)} righe -> {out}", flush=True)


if __name__ == "__main__":
    main()
