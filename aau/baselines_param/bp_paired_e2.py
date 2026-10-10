#!/usr/bin/env python3
"""Emendamento 2 (POST HOC): delta appaiati di B su all_cross, della composizione forma B + taglia B e della B con la
sigma della sensibilita' (PROTOCOL_emendamento_2.md, sez. 1, 2, 4); tabelle dei costi (sez. 3).

    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired_e2.py --workers 32
    (bp.sbatch, passo paired_e2; dopo bp_e2.py crop / heldout / pilot / time / sens e bp_cost.py; LEGGE LE GT di test)

``fact_paired.py`` e' importato in sola lettura (attraverso ``bp_paired``): repliche ``_rep``, riepilogo ``summarize``,
colonne dei bracci ``model_columns``, taglia oracolo ``oracle_S``.

all_cross (sez. 1): righe ``all_cross`` di E12 per HIFI3D e FaceScape (``hifi_frames`` / ``fs_frames``, seme del
gruppo), per FaceVerse (con espressioni e neutra) le ``pair_metrics`` dello stage di ``fv_frames`` senza il filtro del
crop, seme ``stable_seed(1234, "expr_sec", "scale_e108@bfm", "all_cross", ...)``; GT con ``be.add_gts``. Colonne
``<modello>_vb_{coef,fr,sr}`` sulle 600 mesh (``fit_e1.npz`` + ``fit_e2_crop.npz``), bracci dalle 600 chiavi, taglia
oracolo. Anche il sottoinsieme delle righe col crop su almeno un lato (descrittivo).

Gruppi primari (sez. 2 e 4): ``bp_paired.frame`` con le colonne dell'emendamento 1 e, in piu' (``bp_paired.EXTRA``),
``<modello>_vb_comp`` (composizione con k di ``calib_e2/k.json``) e ``<modello>_vbs_{coef,fr,sr}`` (se esiste
``fit_e2_sens.npz``). Controlli: righe di ``paired_e1.csv`` ridate; 600 matrici di B = ``fit_e1.npz`` senza crop;
chiavi e GT di all_cross senza crop = ``fact_paired.rows_for``; formula con le GT (S oracolo, d_P GT) contro FR;
rho di B sulle coppie original - original. Uscite ``paired_e2.csv``, ``spearman_e2.csv``, ``controls_e2.json`` e le
sezioni dell'emendamento 2 in fondo a ``results.md`` (``results_e2``, chiamata da ``bp_paired.write_results``).
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

fp, be, blmm, bp = bpp.fp, bpp.be, bpp.blmm, bpp.bp
REPO = bpp.REPO
OUT = bp.OUT_ROOT
VB = [f"{m}_vb_{d}" for m in bp.MODELS for d in ("coef", "fr", "sr")]
COMP = [f"{m}_vb_comp" for m in bp.MODELS]
SENS = [f"{m}_vbs_{d}" for m in bp.MODELS for d in ("coef", "fr", "sr")]
FV_STAGE = REPO / "aau/runs/ws_faceverse_expr/data_736f96956a/scale_e108_flip_topology"   # methods.fv_frames
LABEL = dict(bpp.LABEL)
LABEL.update({f"{m}_vb_comp": f"{bpp._MN[m]} vB, composizione S_B + k d_P" for m in bp.MODELS})
LABEL.update({f"{m}_vbs_{d}": f"{bpp._MN[m]} vB sigma sens., {bpp._DN[d]}" for m in bp.MODELS for d in bpp._DN})
FR_ARMS = ("factorized_s1234|form_cal", "factorized_s2345|form_cal", "ctrlfr_s1234|z", "ctrlfr_s2345|z")
SR_ARMS = ("factorized_s1234|shape", "factorized_s2345|shape", "ctrlfr_s1234|z", "ctrlfr_s2345|z")


def k_values() -> dict:
    return {m: float(v["k_median"]) for m, v in json.loads((OUT / "calib_e2" / "k.json").read_text()).items()}


def fit_context(view: str, name: str) -> dict:
    with np.load(bp.out_dir(view, name) / "fit.npz") as z:
        return bp.context(view, name, np.asarray(z["region_vertices"]))


# ------------------------------------------------------------------------------------- colonne nuove

def extra_matrices(view: str, keys: list) -> dict:
    """``bp_paired.EXTRA``: composizione (sez. 2) e B della sensibilita' (sez. 4) sulle chiavi di ``fit.npz``."""
    out, K = {}, k_values()
    for m in bp.MODELS:
        root = bp.out_dir(view, m)
        with np.load(root / "fit_e1.npz") as z:
            if list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]])) != keys:
                raise SystemExit(f"{view} {m}: fit_e1.npz in ordine diverso")
            beta, Dsr = np.asarray(z["vb_beta"]), np.asarray(z["D_vb_sr"])
        ctx = fit_context(view, m)
        if m in K:
            out[f"{m}_vb_comp"] = bp.composition(bp.identity_sizes(beta, ctx), Dsr, ctx, K[m])
        if (root / "fit_e2_sens.npz").exists():
            with np.load(root / "fit_e2_sens.npz") as z:
                if list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]])) != keys:
                    raise SystemExit(f"{view} {m}: fit_e2_sens.npz in ordine diverso")
                out.update({f"{m}_vbs_{d}": np.asarray(z[f"D_vbs_{d}"], np.float64) for d in ("coef", "fr", "sr")})
    return out


def vb600(view: str, idx) -> tuple[dict, dict]:
    """({colonna: D (600, 600)} di B sulle chiavi di ``idx``, scarto dalle D di fit_e1.npz sulle coppie senza crop)."""
    out, ctrl = {}, {}
    for m in bp.MODELS:
        root = bp.out_dir(view, m)
        ctx = fit_context(view, m)
        beta = np.full((len(idx.keys), ctx["k_id"]), np.nan)
        D1 = {}
        for name in ("fit_e1.npz", "fit_e2_crop.npz"):
            with np.load(root / name) as z:
                pos = np.asarray([idx.pos[(str(s), str(t))] for s, t in zip(z["subjects"], z["topologies"])])
                beta[pos] = np.asarray(z["vb_beta"])
                if name == "fit_e1.npz":
                    D1 = {d: (pos, np.asarray(z[f"D_vb_{d}"])) for d in ("coef", "fr", "sr")}
        D = bp.mesh_distances(beta, ctx)
        for d, M in D.items():
            out[f"{m}_vb_{d}"] = M
            pos, ref = D1[d]
            dd = np.abs(M[np.ix_(pos, pos)] - ref)
            ctrl[f"{m}_vb_{d}"] = float(np.nanmax(dd))
    return out, ctrl


# ------------------------------------------------------------------------------------------ all_cross

def all_cross_rows(view: str):
    """(df con GT e taglia oracolo, idx, seme) del gruppo all_cross (sez. 1)."""
    src = "faceverse" if view == "faceverse_neutral" else view
    if src == "hifi3d":
        fr = be.e12m.hifi_frames()[0]
        df, seed = fr["all_cross"][0], int(fr["all_cross"][2])
    elif src == "facescape":
        fr = be.e12m.fs_frames()
        df, seed = fr["all_cross"][0], int(fr["all_cross"][2])
    else:
        pm = be.base.read_pair_metrics(FV_STAGE / be.zsum.STAGE)
        df = pm.assign(subject_a=pm["subject_a"].astype(str), subject_b=pm["subject_b"].astype(str))
        seed = int(be.base.stable_seed(1234, "expr_sec", "scale_e108@bfm", "all_cross", "latent_distance", "raw_chamfer"))
    df = df[be.zsum.PAIR_KEYS + ["gt_distance"]].assign(subject_a=lambda d: d["subject_a"].astype(str),
                                                         subject_b=lambda d: d["subject_b"].astype(str))
    idx = be.zes.Index(blmm.subjects(src))
    orc = fp.oracle_S(blmm.VIEWS[src]["gt"])
    D = {"oracle_size": be.log_abs([orc[s] for s, _ in idx.keys])}
    df = be.add_gts(be.add_columns(df, D, idx), be.GT_SET[src]).reset_index(drop=True)
    return df, idx, seed


def check_rows(view: str, df: pd.DataFrame) -> dict:
    """Controllo 2: tolto il crop, chiavi e GT = ``fact_paired.rows_for`` della vista."""
    ref, _, _ = fp.rows_for("faceverse" if view == "faceverse_neutral" else view)
    nc = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    key = ["subject_a", "topology_a", "subject_b", "topology_b"]
    a = nc[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    b = ref[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    same = len(a) == len(b) and bool((a[key].to_numpy() == b[key].to_numpy()).all())
    return {"rows_nocrop": int(len(a)), "rows_fact_paired": int(len(b)), "keys_equal": same,
            "max_abs_diff_gt": float(max(np.abs(a[g] - b[g]).max() for g in ("gt_fr", "gt_sr"))) if same else None}


def bootstrap(view: str, group: str, cols: dict, sa, sb, mask, M: list, seed: int, n_boot: int, workers: int,
              n_subj: int) -> list:
    rng = np.random.default_rng(seed)
    counts = [np.ones(n_subj, dtype=np.int64)] + \
        [np.bincount(rng.integers(0, n_subj, n_subj), minlength=n_subj) for _ in range(n_boot)]
    o = cols["oracle_size"]
    fp._J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
             "methods": M + ["oracle_size"] + VB, "low": o[mask] <= np.quantile(o[mask], 0.2)}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(fp._rep, range(len(counts)), chunksize=4)
    V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
    recs = fp.summarize(V, view, M, VB, int(mask.sum()), seed)
    for r in recs:
        r["group"] = group
    return recs


def run_all_cross(view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    df, idx, seed = all_cross_rows(view)
    info = {"seed": seed, "rows": int(len(df)), "check_rows": check_rows(view, df)}
    Mc = fp.model_columns(view, idx)
    D, info["vb600_vs_fit_e1"] = vb600(view, idx)
    df = be.add_columns(df, {**Mc, **D}, idx)
    M = [m for m in Mc if "|form_at_c" not in m]
    cols = {m: df[m].to_numpy(np.float64) for m in M + ["oracle_size"] + VB}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
    crop = (df["topology_a"].eq("crop") | df["topology_b"].eq("crop")).to_numpy()
    info.update(rows_mask=int(mask.sum()), rows_mask_crop=int((mask & crop).sum()),
                nan_rows_by_column={c: int((~np.isfinite(cols[c])).sum()) for c in VB})
    recs = bootstrap(view, "all_cross", cols, sa, sb, mask, M, seed, n_boot, workers, len(subjects))
    recs += bootstrap(view, "all_cross, righe col crop", cols, sa, sb, mask & crop, M, seed, n_boot, workers,
                      len(subjects))
    print(f"[bp-e2-paired] {view} all_cross: {int(mask.sum())} righe ({int((mask & crop).sum())} col crop), seme {seed}; "
          f"controllo righe {info['check_rows']}; B 600 contro fit_e1 {max(info['vb600_vs_fit_e1'].values()):.1e}",
          flush=True)
    return recs, info


# ------------------------------------------------------------------------------------------ controlli

def gt_controls() -> dict:
    """Controllo 4 (formula con le GT, coppie di soggetti) e rho di B sulle coppie original - original; per la lettura
    della sez. 5: CV della taglia oracolo S sui soggetti e Spearman fra GT FR e SR sulle coppie di soggetti (sulle
    righe dei gruppi primari e' lo stesso valore: ogni coppia di soggetti vi compare lo stesso numero di volte)."""
    from scipy.stats import spearmanr
    out = {}
    for view in bpp.VIEW_ORDER:
        src = "faceverse" if view == "faceverse_neutral" else view
        gset = blmm.VIEWS[src]["gt"]
        G = {g: be.zsum.load_gt(be.EVAL_GT / f"{gset}_{g}.npz") for g in fp.GTS}
        dpu = float(json.loads((be.EVAL_GT / f"{gset}_sr.json").read_text())["dP_per_unit"])
        orc = fp.oracle_S(gset)
        if view == "famos":
            with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
                subj = [str(x) for x in z["names"]]
        else:
            subj = sorted(blmm.subjects(src))
        i, j = np.triu_indices(len(subj), 1)
        gt = {g: G[g][0][np.ix_([G[g][1][s] for s in subj], [G[g][1][s] for s in subj])][i, j] for g in fp.GTS}
        S = np.asarray([orc[s] for s in subj])
        comp = np.sqrt((S[i] - S[j]) ** 2 + S[i] * S[j] * (gt["sr"] * dpu) ** 2)
        rec = {"formula_gt_rho_fr": float(spearmanr(comp, gt["fr"]).correlation), "n_subject_pairs": int(len(i)),
               "S_cv": float(S.std() / S.mean()), "rho_fr_sr_gt": float(spearmanr(gt["fr"], gt["sr"]).correlation)}
        if view != "famos":
            for m in bp.MODELS:
                with np.load(bp.out_dir(view, m) / "fit_e1.npz") as z:
                    pos = {(str(s), str(t)): k for k, (s, t) in enumerate(zip(z["subjects"], z["topologies"]))}
                    o = np.asarray([pos[(s, "original")] for s in subj])
                    for d in ("coef", "fr", "sr"):
                        x = np.asarray(z[f"D_vb_{d}"])[np.ix_(o, o)][i, j]
                        for g in fp.GTS:
                            rec[f"orig_orig_{m}_vb_{d}_{g}"] = float(spearmanr(x, gt[g]).correlation)
        out[view] = rec
    return out


def controls_e1_rows(P: pd.DataFrame) -> dict:
    """Controllo 3: le righe di ``paired_e1.csv`` (bracci, colonne originali e dell'emendamento 1) ridate."""
    ref = pd.read_csv(OUT / "paired_e1.csv")
    key = ["domain", "gt", "kind", "arm", "distance", "baseline"]
    j = P.merge(ref, on=key, suffixes=("", "_ref"))
    num = ["arm_point", "delta", "ci_low", "ci_high", "p_le0"]
    return {"reference": "paired_e1.csv", "n_reference": int(len(ref)), "n_matched": int(len(j)),
            "max_abs_diff": {c: float((j[c] - j[f"{c}_ref"]).abs().max()) for c in num},
            "rows_equal": bool((j["n_rows"] == j["n_rows_ref"]).all())}


# ------------------------------------------------------------------------------------------- markdown

def fmt(p, lo, hi, sign=False) -> str:
    return bpp.fmt(p, lo, hi, sign)


def table_rho(x: pd.DataFrame, rows: list) -> list:
    md = ["| metodo | FR | SR |", "| --- | --- | --- |"]
    for c, lab, arm, dist in rows:
        cells = []
        for g in fp.GTS:
            r = x[(x["arm"] == arm) & (x["distance"] == dist) & (x["gt"] == g)]
            cells.append(fmt(r.iloc[0].arm_point, r.iloc[0].arm_ci_low, r.iloc[0].arm_ci_high) if len(r) else "-")
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    return md


def table_delta(D: pd.DataFrame, cols: list, gts: tuple, neg: list, tag: str) -> list:
    md = []
    for g, arms in (("fr", FR_ARMS), ("sr", SR_ARMS)):
        if g not in gts:
            continue
        md += [f"GT {g.upper()}:", "", "| concorrente | " + " | ".join(dict(bpp.REF_ARMS)[a] for a in arms) + " |",
               "| --- |" + " --- |" * len(arms)]
        for c in cols:
            cells = []
            for a in arms:
                arm, dist = a.split("|")
                r = D[(D["arm"] == arm) & (D["distance"] == dist) & (D["gt"] == g) & (D["baseline"] == LABEL[c])]
                if not len(r):
                    cells.append("-")
                    continue
                r = r.iloc[0]
                cells.append(fmt(r.delta, r.ci_low, r.ci_high, True) + f", P {r.p_le0:.3f}")
                if r.ci_high < 0:
                    neg.append(f"{tag} {g.upper()}: {dict(bpp.REF_ARMS)[a]} - {LABEL[c]} = "
                               f"{fmt(r.delta, r.ci_low, r.ci_high, True)}")
            md.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
        md.append("")
    return md


def count_cells(D: pd.DataFrame, cols: list, gts: tuple) -> str:
    """A favore / contro / non risolte fra i delta dichiarati."""
    out = []
    for g, arms in (("fr", FR_ARMS), ("sr", SR_ARMS)):
        if g not in gts:
            continue
        x = D[(D["gt"] == g) & D["baseline"].isin([LABEL[c] for c in cols]) &
              (D["arm"] + "|" + D["distance"]).isin(arms)]
        out.append(f"{g.upper()} {int((x.ci_low > 0).sum())} / {int((x.ci_high < 0).sum())} / "
                   f"{int(((x.ci_low <= 0) & (x.ci_high >= 0)).sum())}")
    return ", ".join(out)


def cost_tables() -> list:
    """Sez. 3: tempi di B per vista e modello, costi dei bracci, tabella per iscrizione e coppia."""
    md = ["### B: tempi per mesh (s; CPU, un processo per mesh a un thread)", "",
          "| vista | modello | campione | mesh | NICP mediana | A mediana | B mediana | con NICP mediana / p95 | "
          "senza NICP mediana / p95 | fit_e1 `seconds` (tutte, con 3 errori di superficie) mediana / p95 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    agg = {}
    for v in bpp.VIEW_ORDER:
        for m in bp.MODELS:
            for name, lab in (("time_e2.npz", "20 soggetti x 5 topologie" if v != "famos" else "15 scansioni"),
                              ("fit_e2_crop.npz", "crop, 100"), ("time_e2_serial.npz", "un processo solo")):
                p = bp.out_dir(v, m) / name
                if not p.exists():
                    continue
                with np.load(p) as z:
                    T = np.asarray(z["t_stage"])
                    se = np.asarray(z["seconds_e1"]) if "seconds_e1" in z.files else None
                w, wo = T.sum(1), T[:, 2:].sum(1)
                if name != "time_e2_serial.npz":
                    agg.setdefault(m, []).append(T)
                md.append(f"| {v} | {m} | {lab} | {len(T)} | {np.nanmedian(T[:, 1]):.2f} | {np.nanmedian(T[:, 2]):.2f} | "
                          f"{np.nanmedian(T[:, 3]):.2f} | {np.nanmedian(w):.2f} / {np.nanpercentile(w, 95):.2f} | "
                          f"{np.nanmedian(wo):.2f} / {np.nanpercentile(wo, 95):.2f} | " +
                          (f"{np.median(se):.2f} / {np.percentile(se, 95):.2f}" if se is not None else "-") + " |")
    cdir = OUT / "cost_e2"
    ops = json.loads((cdir / "ops.json").read_text()) if (cdir / "ops.json").exists() else None
    fw = {k: json.loads((cdir / f"forward_{k}.json").read_text()) for k in ("factorized_s1234", "ctrlfr_s1234")
          if (cdir / f"forward_{k}.json").exists()}
    pairs = json.loads((cdir / "pairs.json").read_text()) if (cdir / "pairs.json").exists() else None
    ops1 = [json.loads(q.read_text()) for q in sorted(cdir.glob("ops_*s_1w.json"))]
    if ops:
        md += ["", f"### Bracci: preprocessing degli operatori (CPU, k_eig {ops['k_eig']}, {ops['workers']} processi a un "
                   f"thread, {ops['host']}, {ops['cpu']}, job {ops['job']})", "",
               "| topologia | vertici (mediana) | s/mesh mediana | p95 |", "| --- | --- | --- | --- |"]
        for t, r in ops["by_topology"].items():
            md.append(f"| {t} | {r['n_vertices_median']:.0f} | {r['total']['median']:.2f} | {r['total']['p95']:.2f} |")
        tt = ops["total_s_per_mesh"]
        md += [f"| tutte (600) | - | {tt['median']:.2f} | {tt['p95']:.2f} |", "",
               "Per passo (mediana s): " + ", ".join(f"{s} {r['median']:.3f}" for s, r in ops["by_stage"].items()) +
               f"; {ops['meshes_per_s_node']:.2f} mesh/s sul nodo ({ops['wall_s']:.0f} s per 600)."]
        for r in ops1:
            md += ["", f"Un processo solo ({r['n_meshes']} mesh, {r['host']}, job {r['job']}): s/mesh mediana "
                       f"{r['total_s_per_mesh']['median']:.2f}, p95 {r['total_s_per_mesh']['p95']:.2f}; per topologia " +
                   ", ".join(f"{t} {x['total']['median']:.2f}" for t, x in r["by_topology"].items()) + "."]
    for k, r in fw.items():
        b = r["batch1"]
        md += ["", f"### Bracci: forward {k} ({r['device']}, {r['host']}, job {r['job']}; scarto dagli embedding di "
                   f"fact_paired {r['max_abs_diff_vs_reference']:.1e})", "",
               f"batch 1, mediana / p95 (ms): lettura npz operatori {b['load_ops_npz']['median'] * 1e3:.1f} / "
               f"{b['load_ops_npz']['p95'] * 1e3:.1f}, copia sul device {b['to_device']['median'] * 1e3:.1f} / "
               f"{b['to_device']['p95'] * 1e3:.1f}, forward {b['forward']['median'] * 1e3:.1f} / {b['forward']['p95'] * 1e3:.1f}, "
               f"totale {b['total']['median'] * 1e3:.1f} / {b['total']['p95'] * 1e3:.1f}; forward per topologia (mediana ms): " +
               ", ".join(f"{t} {x['median'] * 1e3:.1f}" for t, x in r["batch1_forward_by_topology"].items()) +
               f". Gruppi di {r['group']} gia' sul device (forward sequential del training): "
               f"{r['group_forward_per_mesh']['median'] * 1e3:.1f} ms/mesh (mediana sui gruppi)."]
    if pairs:
        md += ["", f"### Confronto di una coppia (numpy, un thread, {pairs['host']}, job {pairs['job']})", "",
               "| metodo | per coppia (us) | ammortizzato su tutte le coppie (us) | per mesh, una volta (ms) |",
               "| --- | --- | --- | --- |"]
        for m, r in pairs["B"].items():
            md += [f"| {m} vB coefficienti ({r['k_id']}) | {r['pair_coef_s'] * 1e6:.1f} | - | - |",
                   f"| {m} vB mesh FR / SR ({r['n_region_vertices']} vertici) | {r['pair_mesh_fr_or_sr_s'] * 1e6:.1f} | "
                   f"{r['mesh_distances_per_pair_s'] * 1e6:.2f} (coef + FR + SR) | {r['enroll_identity_mesh_s'] * 1e3:.2f} |",
                   f"| {m} vB composizione | {r['pair_comp_s'] * 1e6:.1f} | - | {r['enroll_identity_mesh_s'] * 1e3:.2f} |"]
        for k, r in pairs["arms"].items():
            md.append(f"| {k} (dimensione {r['dim']}) | {r['pair_s'] * 1e6:.1f} | {r['per_pair_amortized_s'] * 1e6:.3f} | - |")
    # tabella riassuntiva per iscrizione
    if agg and ops and fw:
        md += ["", "### Costo per iscrizione di una mesh (mediana, s)", "",
               "| metodo | passi | hardware | s/mesh |", "| --- | --- | --- | --- |"]
        for m, Ts in agg.items():
            T = np.concatenate(Ts)
            md.append(f"| B {m} | lettura + NICP + A + B (tutte le mesh cronometrate, {len(T)}) | CPU, 1 thread | "
                      f"{np.nanmedian(T.sum(1)):.2f} (p95 {np.nanpercentile(T.sum(1), 95):.2f}) |")
        for k, r in fw.items():
            tot = ops["total_s_per_mesh"]["median"] + r["batch1"]["total"]["median"]
            md.append(f"| {k} | operatori k128 (CPU, 1 thread) + lettura + copia + forward ({r['device']}) | CPU + GPU | "
                      f"{tot:.2f} (operatori {ops['total_s_per_mesh']['median']:.2f} + rete "
                      f"{r['batch1']['total']['median']:.3f}) |")
    return md


def results_e2() -> list:
    """Sezioni dell'emendamento 2 in fondo a results.md."""
    sha = OUT / "PROTOCOL_emendamento_2.sha256"
    sha = sha.read_text().split()[0] if sha.exists() else "?"
    c = json.loads((OUT / "controls_e2.json").read_text())
    P = pd.read_csv(OUT / "paired_e2.csv")
    md = ["# Emendamento 2 (POST HOC): B su crop e all_cross, composizione, costi, sensibilita' a sigma", "",
          f"Protocollo `PROTOCOL_emendamento_2.md` (sha256 `{sha}`), scritto dopo i numeri dell'emendamento 1. Numeri in "
          "`paired_e2.csv`, `spearman_e2.csv`, `controls_e2.json`, `calib_e2/`, `pilot_e2/`, `cost_e2/`.", ""]
    neg: list = []
    body: list = []
    # sez. 1
    R = P[(P["kind"] == "rho") & P["group"].str.startswith("all_cross")]
    for v in bpp.VIEW_ORDER:
        for grp in ("all_cross", "all_cross, righe col crop"):
            x = R[(R["domain"] == v) & (R["group"] == grp)]
            if x.empty:
                continue
            base = x[x["baseline"] == "-"]
            rows = [(cc, LABEL[cc], "baseline", LABEL[cc]) for cc in VB] + \
                [(cc, lab, cc.split("|")[0], cc.split("|")[1]) for cc, lab in bpp.REF_ARMS]
            body += [f"## {v}, {grp} ({int(x['n_rows'].iloc[0])} righe, seme {int(x['seed'].iloc[0])})", "",
                     "Spearman con la GT (rho, IC 95%):", ""] + table_rho(base, rows) + \
                ["", "Delta appaiati, braccio - B (IC 95%, P(delta <= 0)); a favore / contro / non risolte: " +
                 count_cells(x, VB, ("fr", "sr")), ""] + table_delta(x, VB, ("fr", "sr"), neg, f"{v} {grp}")
    # sez. 2 e 4
    Q = P[(P["kind"] == "rho") & ~P["group"].str.startswith("all_cross")]
    sens = [cc for cc in SENS if LABEL[cc] in set(Q["baseline"]) or LABEL[cc] in set(Q["distance"])]
    for v in bpp.VIEW_ORDER:
        x = Q[Q["domain"] == v]
        if x.empty:
            continue
        base = x[x["baseline"] == "-"]
        rows = [(cc, LABEL[cc], "baseline", LABEL[cc]) for cc in COMP + [f"{m}_vb_{d}" for m in bp.MODELS for d in ("fr", "sr")] + sens] + \
            [(cc, lab, cc.split("|")[0], cc.split("|")[1]) for cc, lab in bpp.REF_ARMS]
        body += [f"## {v}, composizione forma B + taglia B" + (" e sensibilita' a sigma" if sens else "") +
                 f" ({int(x['n_rows'].iloc[0])} righe)", "", "Spearman con la GT (rho, IC 95%):", ""] + table_rho(base, rows)
        body += ["", "Delta appaiati, braccio - composizione (lettura dichiarata: FR); a favore / contro / non risolte: " +
                 count_cells(x, COMP, ("fr",)), ""] + table_delta(x, COMP, ("fr",), neg, f"{v} composizione")
        if sens:
            body += ["Delta appaiati, braccio - B della sensibilita' (come l'emendamento 1); a favore / contro / non "
                     "risolte: " + count_cells(x, sens, ("fr", "sr")), ""] + table_delta(x, sens, ("fr", "sr"), neg, f"{v} sens.")
    md += ["**Celle con IC sotto 0 (concorrente davanti al braccio) fra i delta dichiarati dell'emendamento 2:** " +
           (f"{len(neg)}: " + "; ".join(neg) if neg else "nessuna") + ".", ""]
    md += ["## Valori senza GT di test (sez. 9)", "", "```", json.dumps(c.get("frozen", {}), indent=1), "```", ""]
    md += ["## Fit di B sul crop", "",
           "| vista | modello | mesh | B fallite | superficie mm: mediana | p95 (mediana) | p95 max | corrisp. tenute |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in bp.CROP_VIEWS:
        for m in bp.MODELS:
            p = bp.out_dir(v, m) / "fit_e2_crop.npz"
            if p.exists():
                with np.load(p) as z:
                    S, k = z["surf_vb"], np.nanmedian(z["vb_kept"], axis=0)
                    md.append(f"| {v} | {m} | {len(S)} | {len(z['failed_vb'])} | {np.nanmedian(S[:, 0]):.2f} | "
                              f"{np.nanmedian(S[:, 1]):.2f} | {np.nanmax(S[:, 1]):.2f} | {k[0]:.2f} / {k[1]:.2f} |")
    md += ["", "## Costi misurati (sez. 3)", ""] + cost_tables()
    md += ["", "## Controlli dell'emendamento 2", "", "```", json.dumps({k: v for k, v in c.items() if k != "frozen"},
                                                                       indent=1), "```", ""]
    return md + body


# ----------------------------------------------------------------------------------------------- main

def frozen() -> dict:
    out = {}
    if (OUT / "calib_e2" / "k.json").exists():
        out["k"] = json.loads((OUT / "calib_e2" / "k.json").read_text())
    if (OUT / "pilot_e2" / "selection.json").exists():
        out["pilot"] = {m: {k: v for k, v in r.items() if k in ("key", "score", "best", "tied", "differs", "control_orig",
                                                               "control_va")}
                        for m, r in json.loads((OUT / "pilot_e2" / "selection.json").read_text()).items()}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--views", default=",".join(bpp.VIEW_ORDER))
    p.add_argument("--summary-only", action="store_true", help="solo results.md dai csv gia' scritti")
    a = p.parse_args()
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(bpp.NEUTRAL_EMB, fp.EVAL)
    fp.BASELINES.update(LABEL)
    if not a.summary_only:
        views = a.views.split(",")
        sens = [cc for cc in SENS if any((bp.out_dir(v, cc.split("_vbs_")[0]) / "fit_e2_sens.npz").exists() for v in views)]
        bpp.COLS[:] = bpp.NEW + bpp.NEW_E1 + COMP + sens
        bpp.EXTRA[:] = [extra_matrices]
        recs, info = [], {"primary": {}, "all_cross": {}}
        for view in views:
            r, info["primary"][view] = bpp.run_view(view, a.n_boot, a.workers)
            recs += r
        for view in [v for v in views if v in bp.CROP_VIEWS]:
            r, info["all_cross"][view] = run_all_cross(view, a.n_boot, a.workers)
            recs += r
        P = pd.DataFrame(recs)
        P.to_csv(OUT / "paired_e2.csv", index=False)
        P[P["baseline"] == "-"].to_csv(OUT / "spearman_e2.csv", index=False)
        ctrl = {"frozen": frozen(), "paired_e1_rows": controls_e1_rows(P[~P["group"].str.startswith("all_cross")]),
                "gt": gt_controls(), "info": info, "sens_columns": sens}
        (OUT / "controls_e2.json").write_text(json.dumps(ctrl, indent=1, default=str) + "\n")
        print(json.dumps({k: ctrl[k] for k in ("paired_e1_rows", "gt")}, indent=1), flush=True)
    bpp.write_results(pd.read_csv(OUT / "paired.csv"), *json.loads((OUT / "controls.json").read_text()).values())


if __name__ == "__main__":
    main()
