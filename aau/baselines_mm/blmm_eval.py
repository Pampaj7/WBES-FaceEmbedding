#!/usr/bin/env python3
"""Baseline geometriche con la normalizzazione coerente con la GT: Spearman con FR, SR, maxabs e riconoscimento.

    v3_work/unified_gt/run.sh aau/baselines_mm/blmm_eval.py --views hifi3d,faceverse,facescape,famos
    (blmm.sbatch, passo eval; dopo blmm_scalars.py, blmm_pairs.py, blmm_template.py)

Righe, soggetti, gruppi e repliche bootstrap sono quelli di E12 (``v3_work/canonical_gt/methods.py``, importata:
``hifi_frames``, ``fv_frames``, ``fs_frames``; seme per (dominio, gruppo) = quello della differenza pubblicata e108 -
Chamfer eval). Alle righe si aggiungono le colonne nuove, per (soggetto, topologia) di ogni lato:
  - ``<modo>_<metrica>``: le matrici di blmm_pairs.py (``zs_expr_summarize.facebench_distances``) e il NICP su
    template di blmm_template.py (``ir_template.template_distances``), modi ``mm`` (FR), ``cs`` (SR), ``maxabs``
    (dove non c'e' gia' la riga pubblicata: FaceScape; il template anche su HIFI3D, come controllo);
  - banali NON oracolo dalla mesh osservata (``blmm.mesh_scalars``): |log x_a - log x_b| con x = centroid size
    robusta (``est_cs``), radice dell'area robusta (``est_sqrt_area``), altezza (``est_height``);
  - oracolo dalla GT: ``oracle_size`` (log S_i di ``datasets/CANONICAL_GT/eval/<set>_centroid_size.npz``, la S di FR)
    e ``oracle_height`` (estensione in y della regione dopo la rigida robusta, ricalcolata qui con ``cgt``).
GT: ``fr`` e ``sr`` (``datasets/CANONICAL_GT/eval/<set>_{fr,sr}.npz``), ``maxabs`` (quella delle righe). Il
subject_pair_mean si rifa' dalle righe con colonne nuove (stessa ``groupby`` di E12). Delta appaiati sulle stesse
repliche: ogni metodo contro e108 e, per ogni baseline, il modo coerente contro maxabs.

Riconoscimento (rank-1, mAP, AUC) dove c'e' gia' il protocollo: HIFI3D e FaceVerse (``zs_expr_summarize``,
blocchi nocrop e crop, seme ``expr_recognition``), FaceScape dev (vista neutra e con espressioni, seme
``devfs_recognition``), FaMoS TEST (``aau/famos/famos_eval.py``: graduata e riconoscimento verso la galleria di 15).

Controlli: le righe pubblicate ricalcolate qui con la GT maxabs e con F_rig_rob devono coincidere con
``aau/runs/evidence/e12/methods_spearman.csv`` (punto e IC); FR deve dare gli stessi numeri di F_rig_rob; il
riconoscimento delle righe pubblicate con ``competitors_hifi3d/recognition.csv`` e ``ws_faceverse_expr``; il modo
``maxabs`` ricalcolato (check_*.json di blmm_pairs.py, template di HIFI3D) con le matrici pubblicate.

Uscite in ``aau/runs/evidence/baselines_mm/``: ``spearman.csv``, ``paired.csv``, ``recognition.csv``,
``controls.csv``, ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import blmm

sys.path.insert(0, str(blmm.REPO_ROOT / "v3_work" / "canonical_gt"))
sys.path.insert(0, str(blmm.AAU_DIR / "famos"))
import cgt  # noqa: E402
import methods as e12m  # noqa: E402  (E12: righe, gruppi, semi; fa os.chdir alla radice)

zes, zsum, base, comp = e12m.zes, e12m.zsum, e12m.base, e12m.comp
import ir_template as irt  # noqa: E402  (importata da comp_summarize)

EVAL_GT = blmm.DATASETS / "CANONICAL_GT" / "eval"
GTS = ("fr", "sr", "maxabs")
GT_LABEL = {"fr": "FR (form, mm)", "sr": "SR (shape)", "maxabs": "maxabs"}
METRICS = ("chamfer", "chamfer_pure", "rigid_icp_chamfer", "nicp_p2tri", "nicp_template")
METRIC_LABEL = {"chamfer": "Chamfer (4096 pt)",
                "chamfer_pure": "Chamfer non centrata (DIAGNOSTICA: posizione nel frame del generatore)",
                "rigid_icp_chamfer": "ICP + Chamfer", "nicp_p2tri": "ICP + NICP P2Tri (per coppia)",
                "nicp_template": "NICP su template"}
MODE_LABEL = {"maxabs": "maxabs per mesh (pubblicata)", "mm": "mm, senza scala (coerente con FR)",
              "cs": "CS robusta (coerente con SR)"}
# righe pubblicate (E12) -> (modo, metrica)
PUBLISHED = {"fb_chamfer": ("maxabs", "chamfer"), "fb_rigid_icp_chamfer": ("maxabs", "rigid_icp_chamfer"),
             "fb_nicp_p2tri": ("maxabs", "nicp_p2tri"), "nicp_template": ("maxabs", "nicp_template")}
REFS = {"scale_e108": "e108 (BFM+ICT+GNM)", "chamfer_eval": "Chamfer eval"}
TRIVIAL = {"est_cs": "taglia stimata: centroid size robusta", "est_sqrt_area": "taglia stimata: sqrt(area robusta)",
           "est_height": "altezza stimata (y)", "oracle_size": "ORACOLO: solo taglia (S di FR)",
           "oracle_height": "ORACOLO: solo altezza (regione, FR)"}
GT_SET = {"hifi3d": "hifi3d", "faceverse": "faceverse", "facescape": "facescape"}
RECOG_SEED = {"hifi3d": "expr_recognition", "faceverse": "expr_recognition", "facescape": "devfs_recognition",
              "facescape_expr": "devfs_recognition"}


# -------------------------------------------------------------------------------------- distanze

def log_abs(x: np.ndarray) -> np.ndarray:
    lx = np.log(np.asarray(x, dtype=np.float64))
    return np.abs(lx[:, None] - lx[None, :])


def oracle_scalars(gt_set: str) -> dict:
    """{id della vista: (S, H)}: S dal file di FR (controllata contro il ricalcolo), H dopo la rigida robusta."""
    cn = cgt.Canon()
    name = "famos" if gt_set == "famos_test" else gt_set
    nat = cgt.native_points(name, cn)
    a = cn.rigid_robust(cn.to_F(nat["domain"], nat["P"]))["a"]
    S, H = cn.centroid_size(a), cn.height(a)
    ids = list(nat["ids"])
    if name == "famos":
        import famos_common as fc
        ids = [fc.view_id(s) for s in ids]
    with np.load(EVAL_GT / f"{gt_set}_centroid_size.npz") as z:
        ref = dict(zip([str(s) for s in z["names"]], z["S"]))
    err = max(abs(ref[i] / s - 1.0) for i, s in zip(ids, S) if i in ref)
    if err > 1e-9:
        raise SystemExit(f"{gt_set}: centroid size ricalcolata diversa da quella di FR ({err:.2e})")
    return {i: (float(ref[i]), float(h)) for i, h in zip(ids, H) if i in ref}


def view_distances(view: str, idx, have_published: set) -> tuple[dict, list]:
    """{colonna: D (600, 600)} delle colonne nuove di una vista a 6 topologie, e le colonne mancanti."""
    D, missing = {}, []
    for mode in blmm.MODES:
        root = blmm.OUT_ROOT / view / mode
        for m in METRICS:
            col = f"{mode}_{m}"
            if (mode == "maxabs" and any(v == (mode, m) for k, v in PUBLISHED.items() if k in have_published)) \
                    or (m == "chamfer_pure" and mode != "mm"):
                continue
            try:
                if m == "nicp_template":
                    z = np.load(root / "template.npz")
                    R = z["R"][comp.ordered(z, idx)].astype(np.float64)
                    D[col] = irt.template_distances(R, R)
                else:
                    D[col] = zes.facebench_distances(root, m, idx)
            except (FileNotFoundError, OSError) as exc:
                missing.append(f"{view} {col} ({type(exc).__name__})")
    sc = blmm.scalars_of(view)
    for c in ("cs", "sqrt_area", "height"):
        D[f"est_{c}"] = log_abs([sc[blmm.rel(blmm.mesh_path(view, s, t))][c] for s, t in idx.keys])
    orc = oracle_scalars(blmm.VIEWS[view]["gt"])
    for k, c in ((0, "size"), (1, "height")):
        D[f"oracle_{c}"] = log_abs([orc[s][k] for s, _ in idx.keys])
    return D, missing


def add_columns(df: pd.DataFrame, D: dict, idx) -> pd.DataFrame:
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"].astype(str), df["topology_a"].astype(str))])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"].astype(str), df["topology_b"].astype(str))])
    out = df.copy()
    for c, M in D.items():
        out[c] = M[ia, ib]
    return out


def add_gts(df: pd.DataFrame, gt_set: str) -> pd.DataFrame:
    out = df.copy()
    if "gt_maxabs" not in out:
        out["gt_maxabs"] = out["gt_distance"]
    for g in ("fr", "sr"):
        out[f"gt_{g}"] = zsum.with_gt(df, zsum.load_gt(EVAL_GT / f"{gt_set}_{g}.npz"))["gt_distance"].to_numpy()
    return out


# ---------------------------------------------------------------------------------------- righe

def frames_for(view: str) -> tuple[dict, dict, list]:
    """{gruppo: (df, metodi, seme)} con le colonne nuove, {colonna: D} per il riconoscimento, colonne mancanti."""
    if view == "hifi3d":
        fr = e12m.hifi_frames()[0]
        base_df, groups = fr["all_cross"][0], ("nocrop_cross", "all_cross", "subject_pair_mean")
    elif view == "faceverse":
        fr = e12m.fv_frames()
        base_df, groups = fr["mesh_pair_nocrop"][0], ("mesh_pair_nocrop", "subject_pair_mean_nocrop")
    else:
        fr = e12m.fs_frames()
        base_df, groups = fr["all_cross"][0], ("nocrop_cross", "all_cross", "subject_pair_mean")
    idx = zes.Index(blmm.subjects(view))
    pub = [m for m in list(PUBLISHED) + list(REFS) if m in base_df]
    D, missing = view_distances(view, idx, set(pub))
    df = add_gts(add_columns(base_df, D, idx), GT_SET[view])
    df = df.rename(columns={k: f"{v[0]}_{v[1]}" for k, v in PUBLISHED.items() if k in df})
    cols = [c for c in list(REFS) if c in df] + sorted({f"{v[0]}_{v[1]}" for k, v in PUBLISHED.items() if k in pub}
                                                     | {c for c in D if not c.startswith(("est_", "oracle_"))}) \
        + [c for c in TRIVIAL if c in D]
    gcols = [f"gt_{g}" for g in GTS] + [f"gt_{g}" for g in ("F_rig_rob",) if f"gt_{g}" in df]
    out = {}
    for g in groups:
        seed = fr[g][2]
        if g.startswith("subject_pair_mean"):          # come E12 / E8: media per coppia di soggetti delle righe
            d = df.groupby(["subject_a", "subject_b"], as_index=False)[gcols + cols].mean()
        elif g == "nocrop_cross":
            d = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
        else:
            d = df
        if len(d) != len(fr[g][0]):
            raise SystemExit(f"{view} {g}: {len(d)} righe invece delle {len(fr[g][0])} di E12")
        out[g] = (d[["subject_a", "subject_b"] + gcols + cols].reset_index(drop=True), cols, seed)
    if "nocrop_cross" in out:
        # il subject_pair_mean dei bracci (tools/eval_factorized.py, --pairs nocrop_cross): media sulle sole coppie
        # senza crop; seme 1234 come eval_factorized (E12 non ha questo gruppo)
        nc = out["nocrop_cross"][0]
        out["subject_pair_mean_nocrop"] = (nc.groupby(["subject_a", "subject_b"], as_index=False)[gcols + cols].mean(),
                                           cols, 1234)
    Drec = {c: M for c, M in D.items() if not c.startswith("oracle_")}
    return out, Drec, missing


# ------------------------------------------------------------------------------------ bootstrap
# Come methods._replicate / evaluate di E12 (copiate), con le GT e i confronti di qui.

_BM = None
_JOB: dict = {}


def _replicate(k: int) -> dict:
    global _BM
    if _BM is None:
        _BM = base.load_bootstrap_module()
    J = _JOB
    counts = J["counts"][k]
    wt = counts[J["sa"]].astype(np.int64) * counts[J["sb"]]
    out, cache = {}, {}

    def sp(g, m, mask):
        mask = J["masks"]["_canon"][mask]
        key = (g, m, mask)
        if key not in cache:
            keep = J["masks"][mask] & (wt > 0)
            cache[key] = _BM.finite_spearman(np.repeat(J["cols"][f"gt_{g}"][keep], wt[keep]),
                                             np.repeat(J["cols"][m][keep], wt[keep]))
        return cache[key]

    for g in J["gts"]:
        for m in J["methods"]:
            out[("row", g, m)] = sp(g, m, m)
    for g, a, b in J["pairs"]:
        out[("pair", g, a, b)] = sp(g, a, f"{a}&{b}") - sp(g, b, f"{a}&{b}")
    return out


def evaluate(dom: str, group: str, df: pd.DataFrame, methods: list, seed: int, n_boot: int, workers: int,
             gts: tuple = GTS):
    global _JOB
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = df["subject_a"].map(s2i).to_numpy(np.int32)
    sb = df["subject_b"].map(s2i).to_numpy(np.int32)
    cols = {c: df[c].to_numpy(np.float64) for c in [f"gt_{g}" for g in gts] + methods}
    gt_ok = np.all([np.isfinite(cols[f"gt_{g}"]) for g in gts], axis=0) & (sa != sb)
    masks = {m: gt_ok & np.isfinite(cols[m]) for m in methods}
    pairs = []
    for g in GTS:
        if "scale_e108" in methods:
            pairs += [(g, a, "scale_e108") for a in methods if a != "scale_e108"]
        for a in methods:
            mode, _, metric = a.partition("_")
            if mode in ("mm", "cs") and f"maxabs_{metric}" in methods:
                pairs.append((g, a, f"maxabs_{metric}"))
    for g, a, b in pairs:
        masks[f"{a}&{b}"] = masks[a] & masks[b]
    canon, uniq = {}, []
    for name, mk in masks.items():
        hit = next((u for u in uniq if np.array_equal(masks[u], mk)), None)
        if hit is None:
            uniq.append(name)
            hit = name
        canon[name] = hit
    masks = {**masks, "_canon": canon}
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)]
    for _ in range(n_boot):
        counts.append(np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects)))
    _JOB = {"counts": counts, "sa": sa, "sb": sb, "cols": cols, "masks": masks, "methods": methods, "pairs": pairs,
            "gts": gts}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_replicate, range(n_boot + 1), chunksize=8)
    rows, prs = [], []
    for key in reps[0]:
        v = np.array([r[key] for r in reps])
        b = v[1:][np.isfinite(v[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5])
        rec = {"domain": dom, "group": group, "point": float(v[0]), "ci_low": float(lo), "ci_high": float(hi),
               "p_le0": float((b <= 0).mean()), "n_boot": int(len(b)), "n_subjects": len(subjects)}
        if key[0] == "row":
            rows.append({**rec, "gt": key[1], "method": key[2], "n_rows": int(masks[key[2]].sum()),
                         "n_nan": int((~masks[key[2]] & gt_ok).sum())})
        else:
            prs.append({**rec, "gt": key[1], "a": key[2], "b": key[3]})
    return pd.DataFrame(rows), pd.DataFrame(prs)


# --------------------------------------------------------------------------------- riconoscimento

def recognition(view: str, D: dict, n_boot: int, workers: int) -> pd.DataFrame:
    idx = zes.Index(blmm.subjects(view))
    counts = zes.bootstrap_counts(len(idx.subjects), n_boot, base.stable_seed(1234, RECOG_SEED[view]))
    blocks = {"nocrop": ([(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b],
                         [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]),
              "crop": ([(a, b) for a in zes.TOPOLOGIES for b in zes.TOPOLOGIES if a != b and "crop" in (a, b)],
                       [("crop", b) for b in zes.NOCROP])}
    tasks = [((blk, name), M, idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for name, M in D.items()]
    rows = []
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        for (blk, name), vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            r = {"domain": view, "block": blk, "method": name, "n_nan_distances": n_nan}
            for m in ("rank1", "map", "auc"):
                r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
            rows.append(r)
    return pd.DataFrame(rows)


def published_distances(view: str) -> dict:
    """Le matrici pubblicate delle stesse baseline (modo maxabs), per il controllo del riconoscimento."""
    idx = zes.Index(blmm.subjects(view))
    out = {f"maxabs_{m}": zes.facebench_distances(blmm.VIEWS[view]["fb_root"], m, idx)
           for m in ("chamfer", "rigid_icp_chamfer", "nicp_p2tri")}
    if blmm.VIEWS[view]["template_ref"] is not None:
        z = np.load(blmm.VIEWS[view]["template_ref"])
        R = z["R"][comp.ordered(z, idx)].astype(np.float64)
        out["maxabs_nicp_template"] = irt.template_distances(R, R)
    return out


# ------------------------------------------------------------------------------------------- FaMoS

def famos(n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Graduata (GT FR, SR, maxabs di E12) e riconoscimento di ``famos_eval.py`` con le distanze di qui."""
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
    gcols = {k: np.asarray([col[k][int(i)] for i in gal[k]]) for k in ("scan", "reg")}
    D, missing = {}, []
    for mode in blmm.MODES:
        parts = sorted((blmm.OUT_ROOT / "famos" / mode).glob("*_*of*.npz"))
        for step, mets in blmm.METRICS.items():
            ps = [p for p in parts if p.name.startswith(f"{step}_")]
            for m in mets:
                if m == "chamfer_pure" and mode != "mm":
                    continue
                M = np.full((len(rows), len(gal_all)), np.nan)
                got = np.zeros(len(rows), bool)
                for p in ps:
                    with np.load(p) as z:
                        if [str(x) for x in z["gallery"]] != [names[i] for i in gal_all]:
                            raise SystemExit(f"{p}: galleria diversa")
                        M[z["rows"]] = z[m]
                        got[z["rows"]] = True
                if got.all():
                    D[f"{mode}_{m}"] = M
                else:
                    missing.append(f"famos {mode}_{m} ({int(got.sum())}/{len(rows)} righe)")
    sc = blmm.scalars_of("famos")
    for c in ("cs", "sqrt_area", "height"):
        x = np.log([sc[blmm.rel(blmm.VIEWS["famos"]["dir"] / f"{n}.npz")][c] for n in names])
        D[f"est_{c}"] = np.abs(x[:, None] - x[None, gal_all])
    orc = oracle_scalars("famos_test")
    for k, c in ((0, "size"), (1, "height")):
        x = np.log([orc[gt_names[s]][k] for s in subj])
        D[f"oracle_{c}"] = np.abs(x[:, None] - x[None, gal_all])
    G = {}
    for g in ("fr", "sr"):
        Dg, pos = zsum.load_gt(EVAL_GT / f"famos_test_{g}.npz")
        G[g] = Dg[np.ix_([pos[s] for s in gt_names], [pos[s] for s in gt_names])]
    with np.load(cgt.OUT_DIR / "famos_maxabs.npz") as z:
        pos = {fc.view_id(str(s)): k for k, s in enumerate(z["names"])}
        ii = [pos[s] for s in gt_names]
        G["maxabs"] = np.asarray(z["D_orig"], np.float64)[np.ix_(ii, ii)]
    counts = fe.bootstrap_counts(len(gt_names), n_boot, 1234)
    gr = []
    for qk, qr, gk in fe.GRADED_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        qa = np.repeat(q_idx, len(gt_names))
        gb = np.tile(np.arange(len(gt_names)), len(q_idx))
        sa, sb = subj[qa], gb
        keep = sa != sb
        if qr == "gallery" and qk == gk:
            keep &= sa < sb
        qa, gb, sa, sb = qa[keep], gb[keep], sa[keep], sb[keep]
        for g in GTS:
            for m, M in D.items():
                x = M[qa, gcols[gk][gb]]
                ok = np.isfinite(x)          # NICP per coppia non e' definito con una patch reg come sorgente
                if not ok.any():
                    continue
                v = fe.spearman_reps(x[ok], G[g][sa, sb][ok], sa[ok], sb[ok], counts)
                lo, hi = fe.ci(v)
                gr.append({"domain": "famos", "group": f"{qk} {qr} -> {gk}", "gt": g, "method": m, "point": v[0],
                           "ci_low": lo, "ci_high": hi, "n_subjects": len(gt_names), "n_rows": int(ok.sum()),
                           "n_nan": int((~ok).sum())})
    rec = []
    for qk, qr, gk in fe.RECOG_BLOCKS:
        q_idx = np.flatnonzero((kind == qk) & (role == qr))
        for m, M in D.items():
            if m.startswith("oracle_") or not np.isfinite(M[np.ix_(q_idx, gcols[gk])]).any():
                continue
            v = fe.recognition(M[:, gcols[gk]], q_idx, subj[q_idx], np.arange(len(gt_names)), counts)
            r = {"domain": "famos", "block": f"{qk} {qr} -> {gk}", "method": m,
                 "n_nan_distances": int((~np.isfinite(M[np.ix_(q_idx, gcols[gk])])).sum())}
            for k in ("rank1", "auc"):
                r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(v[k][0]), fe.ci(v[k])
            rec.append(r)
    return pd.DataFrame(gr), pd.DataFrame(rec), missing


# ------------------------------------------------------------------------------------------ controlli

def controls(R: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e12 = pd.read_csv(blmm.AAU_DIR / "runs" / "evidence" / "e12" / "methods_spearman.csv")
    name = {**{k: f"{v[0]}_{v[1]}" for k, v in PUBLISHED.items()}, **{k: k for k in REFS}}
    for r in e12[e12["gt"].isin(["maxabs", "F_rig_rob"]) & e12["method"].isin(name)].itertuples():
        for gt in ((r.gt, "fr") if r.gt == "F_rig_rob" else (r.gt,)):
            n = R[(R["domain"] == r.domain) & (R["group"] == r.group) & (R["gt"] == gt) & (R["method"] == name[r.method])]
            if len(n):
                for c in ("point", "ci_low", "ci_high"):
                    src = "e12/methods_spearman.csv" + (", F_rig_rob contro FR (float32)" if gt == "fr" else "")
                    rows.append({"what": f"{r.domain} {r.group} {r.method} GT {r.gt}->{gt} ({c})", "published": getattr(r, c),
                                 "recomputed": float(n[c].iloc[0]), "source": src})
    for dom, path in (("hifi3d", blmm.AAU_DIR / "runs" / "competitors_hifi3d" / "recognition.csv"),
                      ("faceverse", blmm.AAU_DIR / "runs" / "ws_faceverse_expr" / "recognition.csv")):
        if not path.exists() or rec.empty:
            continue
        ref = pd.read_csv(path)
        for r in ref.itertuples():
            if r.method not in ("chamfer", "rigid_icp_chamfer", "nicp_p2tri", "nicp_template"):
                continue
            m = f"maxabs_{r.method}"
            n = rec[(rec["domain"] == dom) & (rec["block"] == r.block) & (rec["method"] == m)]
            if len(n):
                for c in ("rank1", "auc"):
                    rows.append({"what": f"{dom} {r.block} {r.method} ({c})", "published": getattr(r, c),
                                 "recomputed": float(n[c].iloc[0]), "source": f"{path.parent.name}/{path.name}"})
    for p in sorted(blmm.OUT_ROOT.glob("*/maxabs/check_*.json")):
        for rep in json.loads(p.read_text()):
            for m, v in rep.items():
                if isinstance(v, dict) and v.get("max_abs_diff") is not None:
                    rows.append({"what": f"{p.parent.parent.name} {rep['pair']} {m}: {v['n_equal']}/{v['n_both_finite']} "
                                         f"uguali, NaN nuovi {v['nan_new']} vecchi {v['nan_old']}",
                                 "published": 0.0, "recomputed": v["max_abs_diff"],
                                 "source": f"{p.parent.parent.name} {p.name}: modo maxabs ricalcolato contro le matrici"})
    tpl = blmm.OUT_ROOT / "hifi3d" / "maxabs" / "template.npz"
    if tpl.exists():
        with np.load(tpl) as a, np.load(blmm.VIEWS["hifi3d"]["template_ref"]) as b:
            for k in ("T", "R"):
                rows.append({"what": f"hifi3d template maxabs {k}", "published": 0.0,
                             "recomputed": float(np.nanmax(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))),
                             "source": "competitors_hifi3d/template_hifi3d.npz"})
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["published"] - out["recomputed"]).abs()
    return out


# ------------------------------------------------------------------------------------------ markdown

def label(m: str) -> tuple[str, str]:
    if m in REFS:
        return REFS[m], "-"
    if m in TRIVIAL:
        return TRIVIAL[m], "-"
    mode, _, metric = m.partition("_")
    return METRIC_LABEL[metric], MODE_LABEL[mode]


def fmt(r) -> str:
    return f"{r.point:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def order(methods: list) -> list:
    key = {c: k for k, c in enumerate(list(REFS) + [f"{mo}_{me}" for me in METRICS for mo in blmm.MODES] + list(TRIVIAL))}
    return sorted(methods, key=lambda m: key.get(m, 999))


def table(R: pd.DataFrame, dom: str, group: str) -> list[str]:
    x = R[(R["domain"] == dom) & (R["group"] == group)]
    out = ["| metodo | normalizzazione | " + " | ".join(GT_LABEL[g] for g in GTS) + " | righe (NaN) |",
           "| --- | --- | " + " | ".join("---" for _ in GTS) + " | --- |"]
    for m in order(list(dict.fromkeys(x["method"]))):
        cells = []
        for g in GTS:
            r = x[(x["method"] == m) & (x["gt"] == g)]
            cells.append(fmt(r.iloc[0]) if len(r) else "-")
        r0 = x[x["method"] == m].iloc[0]
        lab, mode = label(m)
        out.append(f"| {lab} | {mode} | " + " | ".join(cells) + f" | {int(r0.n_rows)} ({int(r0.n_nan)}) |")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default="hifi3d,faceverse,facescape,famos")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--summary-only", action="store_true", help="solo summary.md dai csv gia' scritti")
    a = p.parse_args()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    out = blmm.OUT_ROOT
    if a.summary_only:
        write_summary(*(pd.read_csv(out / f) for f in ("spearman.csv", "paired.csv", "recognition.csv", "controls.csv")),
                      json.loads((out / "missing.json").read_text()))
        return
    R, P, REC, missing = [], [], [], []
    for view in a.views.split(","):
        if view == "famos":
            gr, rc, miss = famos(a.n_bootstrap)
            R.append(gr)
            REC.append(rc)
            missing += miss
            print(f"[blmm-eval] famos: {gr['method'].nunique()} metodi", flush=True)
            continue
        frames, Drec, miss = frames_for(view)
        missing += miss
        for group, (df, methods, seed) in frames.items():
            gts = GTS + (("F_rig_rob",) if "gt_F_rig_rob" in df else ())
            r, pr = evaluate(view, group, df, methods, seed, a.n_bootstrap, workers, gts)
            R.append(r)
            P.append(pr)
            print(f"[blmm-eval] {view} {group}: {len(df)} righe, {len(methods)} metodi", flush=True)
        if blmm.VIEWS[view]["fb_root"] is not None:
            Drec.update(published_distances(view))       # le righe maxabs pubblicate (anche per il controllo)
        REC.append(recognition(view, Drec, a.n_bootstrap, workers))
        if view == "facescape":
            Dx, _ = view_distances("facescape_expr", zes.Index(blmm.subjects("facescape_expr")), set())
            REC.append(recognition("facescape_expr", {c: M for c, M in Dx.items() if not c.startswith("oracle_")},
                                   a.n_bootstrap, workers))
    R = pd.concat(R, ignore_index=True)
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame()
    REC = pd.concat(REC, ignore_index=True) if REC else pd.DataFrame()
    R.to_csv(out / "spearman.csv", index=False)
    P.to_csv(out / "paired.csv", index=False)
    REC.to_csv(out / "recognition.csv", index=False)
    C = controls(R, REC)
    C.to_csv(out / "controls.csv", index=False)
    print(C.groupby("source")["abs_diff"].agg(["count", "max"]).to_string(), flush=True)
    (out / "missing.json").write_text(json.dumps(missing, indent=1) + "\n")
    write_summary(R, P, REC, C, missing)


def write_summary(R, P, REC, C, missing) -> None:
    prm = blmm.params()
    md = ["# Baseline geometriche con la rimozione dei disturbi coerente con la GT", ""]
    concl = blmm.OUT_ROOT / "conclusions.md"
    if concl.exists():                                   # scritte a mano sui numeri qui sotto
        md += [concl.read_text().strip(), ""]
    md += ["## Definizioni", "",
          "Codice in `aau/baselines_mm/` (definizioni in `blmm.py`), numeri qui. Le baseline pubblicate normalizzano "
          "ogni mesh per maxabs e l'ICP prealinea col bbox, che scala: sono cieche alla taglia, mentre FR la conserva. "
          "Qui la stessa pipeline faceBench in tre normalizzazioni:",
          "- **maxabs** (pubblicata): righe esistenti, o ricalcolate dove mancavano (FaceScape, FaMoS, il template);",
          "- **mm** (coerente con FR): mm nel frame canonico di E12, UNA trasformazione per dominio, nessuna scala per "
          "mesh; ICP rigido senza scala (prealineamento solo per traslazione); template NICP riportato con la rigida;",
          "- **cs** (coerente con SR): come mm, ma ogni mesh a centroid size robusta CS_ref (aree della geometria "
          "passa-basso, `area_v3` modo smooth).",
          "",
          "Unita' di lavoro per dominio (la stessa costante per tutte le mesh, perche' NICP non e' invariante alla "
          "scala): " + ", ".join(f"{d} L = {v['L']:.1f} mm, CS_ref = {v['cs_ref']:.1f} mm" for d, v in prm["domains"].items()) + ".",
          "",
          "Righe, soggetti, gruppi e repliche bootstrap (1000, per soggetto) sono quelli di E12; IC 95%. Colonne = GT: "
          "FR e SR di `datasets/CANONICAL_GT/eval`, maxabs per continuita'. Banali NON oracolo: |log x_a - log x_b| "
          "dalla mesh osservata. Oracolo: dalla GT (S di FR; altezza della regione dopo la rigida robusta).",
          "",
          "Per affiancare i bracci factorized e ctrl-FR (`v3_work/trainer/tools/eval_factorized.py`, `form_spearman.csv`): "
          "il loro `mesh_pair` sono le righe di `nocrop_cross` (FaceVerse: `mesh_pair_nocrop`), il loro "
          "`subject_pair_mean` e' `subject_pair_mean_nocrop` (stesso seme 1234); le colonne gt `fr`, `sr`, `maxabs` "
          "sono le stesse matrici. Tutti i numeri anche in `spearman.csv` (colonne domain, group, method, gt, point, "
          "ci_low, ci_high, n_subjects, n_rows), i delta appaiati in `paired.csv`.", ""]
    if missing:
        md += ["**Mancanti al momento della scrittura** (righe assenti dalle tabelle): " + "; ".join(missing), ""]
    md += ["## Controlli di riproduzione", ""]
    if not C.empty:
        g = C.groupby("source")["abs_diff"].agg(["count", "max"])
        md += ["| sorgente | confronti | max abs diff |", "| --- | --- | --- |"]
        md += [f"| {s} | {int(r['count'])} | {r['max']:.2e} |" for s, r in g.iterrows()]
        md.append("")
    for dom in R["domain"].unique():
        for group in R[R["domain"] == dom]["group"].unique():
            md += [f"## {dom}, {group}", ""] + table(R, dom, group) + [""]
            if not P.empty:
                x = P[(P["domain"] == dom) & (P["group"] == group) & (P["b"].str.startswith("maxabs_"))]
                if len(x):
                    md += ["Delta appaiati, modo coerente - maxabs (stesse repliche):", "",
                           "| baseline | " + " | ".join(GT_LABEL[g] for g in GTS) + " |",
                           "| --- | " + " | ".join("---" for _ in GTS) + " |"]
                    for a_ in order(list(dict.fromkeys(x["a"]))):
                        cells = []
                        for g in GTS:
                            r = x[(x["a"] == a_) & (x["gt"] == g)]
                            cells.append(f"{r.iloc[0].point:+.3f} [{r.iloc[0].ci_low:+.3f}, {r.iloc[0].ci_high:+.3f}]" if len(r) else "-")
                        md.append(f"| {label(a_)[0]}, {a_.split('_')[0]} | " + " | ".join(cells) + " |")
                    md.append("")
    if not REC.empty:
        md += ["## Riconoscimento", ""]
        for dom in REC["domain"].unique():
            x = REC[REC["domain"] == dom]
            for blk in x["block"].unique():
                y = x[x["block"] == blk]
                md += [f"### {dom}, {blk}", "", "| metodo | normalizzazione | rank-1 | AUC |", "| --- | --- | --- | --- |"]
                for m in order(list(dict.fromkeys(y["method"]))):
                    r = y[y["method"] == m].iloc[0]
                    lab, mode = label(m)
                    md.append(f"| {lab} | {mode} | {r.rank1:.3f} [{r.rank1_ci_low:.3f}, {r.rank1_ci_high:.3f}] | "
                              f"{r.auc:.3f} [{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}] |")
                md.append("")
    (blmm.OUT_ROOT / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
