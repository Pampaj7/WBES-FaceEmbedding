#!/usr/bin/env python3
"""E2: e108 con e senza canonicalizzazione rigida al test, sugli stessi protocolli e repliche dei riferimenti.

    aau/run.sh aau/evidence/e2_canon/summarize_e2.py      (summarize_e2.sbatch, CPU)

Niente misure nuove: Spearman, bootstrap, riconoscimento e concordanza NoW sono le funzioni dei summarizer
esistenti, importate (``zs_summarize``, ``zs_expr_summarize``, ``now_summarize``). Righe e repliche:
  - HIFI3D, Spearman con la GT maxabs sulle righe delle pair_metrics di ``scale_e108_topology`` (a): ogni
    gruppo (``all_cross``, ``nocrop_cross``, ``subject_pair_mean``) col seme della riga e108 gia' pubblicata
    (``stable_seed(1234, "scale_e108", "maxabs", gruppo, "latent")``): la riga senza canonicalizzazione deve
    tornare IDENTICA (controllo), e tutte le altre righe e differenze del gruppo stanno sulle stesse repliche;
  - riconoscimento HIFI3D e FaceVerse con espressioni: ``bootstrap_counts`` col seme ``expr_recognition``, le
    righe senza canonicalizzazione devono tornare identiche a ``arcface_vs_scale_hifi3d/recognition.csv`` e
    ``fvexpr_partial/recognition.csv``;
  - NoW: tau per immagine sui 3 metodi pre-registrati, ``bootstrap_counts(20, 1000, 1234)`` come
    now_summarize; righe senza canonicalizzazione contro ``now_eval_scale_e108/concordance.csv``.
Baseline a parita' di allineamento: Chamfer faceBench (HIFI3D, FaceVerse) e Chamfer grezza (NoW) sulle STESSE
mesh canonicalizzate.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "aau" / "zs3dmm"))
sys.path.insert(0, str(REPO_ROOT / "aau" / "recon"))

import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zsum.base
RUNS = REPO_ROOT / "aau" / "runs"
E2 = RUNS / "evidence" / "e2"
HIFI_RUNS = RUNS / "ws_hifi3d" / "data_328f2bfc1a"
FV_RUNS = RUNS / "ws_faceverse_expr" / "data_736f96956a"
LABEL = {"e108": "e108, senza canonicalizzazione", "e108_canon": "e108, canonicalizzato",
         "chamfer_fb": "Chamfer faceBench, senza canonicalizzazione",
         "chamfer_fb_canon": "Chamfer faceBench, canonicalizzato",
         "chamfer_eval": "Chamfer eval (riferimento del summary)", "rigid_icp_chamfer": "Rigid ICP + Chamfer (riferimento)",
         "e108_ictconv": "e108, convenzione ICT fissa (Rx180, job di un altro agente)",
         "latent": "e108, senza canonicalizzazione", "latent_canon": "e108, canonicalizzato",
         "chamfer_raw": "Chamfer grezza, senza canonicalizzazione", "chamfer_raw_canon": "Chamfer grezza, canonicalizzata",
         "icp": "ICP + Chamfer (mm)", "icp_canon": "ICP + Chamfer (mm), mesh canonicalizzate"}
PAIRS_A = (("e108_canon", "e108"), ("chamfer_fb_canon", "chamfer_fb"), ("e108_canon", "chamfer_fb_canon"),
           ("e108", "chamfer_fb"))
TOK = {"all_cross": "all_cross", "nocrop_cross": "nocrop_cross", "subject_pair_mean": "spm_clean"}
PROTO = {"all_cross": "mesh_pair_all_cross", "nocrop_cross": "mesh_pair_nocrop_cross",
         "subject_pair_mean": "subject_pair_mean"}


def fmt(p, lo, hi, signed=False):
    return zes.fmt(p, lo, hi, signed)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


# ------------------------------------------------------------------- canonicalizzazione

def rots(df: pd.DataFrame) -> np.ndarray:
    return df[[f"R{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(-1, 3, 3)


def ang(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", A, B) - 1) / 2, -1, 1)))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z, ph = 1.96, k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def canon_section(name: str, df: pd.DataFrame, T: float, by: str) -> tuple[list[str], dict]:
    fail = df["residual"] > T
    lo, hi = wilson(int(fail.sum()), len(df))
    out = {"dominio": name, "n": len(df), "fallimenti": int(fail.sum()), "tasso": float(fail.mean()), "ci_low": lo,
           "ci_high": hi, "s_mediana": float(df.seconds.median()), "s_p95": float(np.percentile(df.seconds, 95)),
           "angolo_conv_mediana": float(df.angle_from_convention.median()),
           "angolo_conv_p95": float(np.percentile(df.angle_from_convention, 95)),
           "angolo_conv_gt30": int((df.angle_from_convention > 30).sum()),
           "residuo_mediana": float(df.residual.median())}
    rows = []
    for g, sub in df.groupby(by):
        f = sub["residual"] > T
        rows.append([g, len(sub), f"{int(f.sum())} ({100 * f.mean():.1f}%)", f"{sub.residual.median():.4f}",
                     f"{np.percentile(sub.residual, 95):.4f}", f"{sub.angle_from_convention.median():.2f}",
                     f"{np.percentile(sub.angle_from_convention, 95):.2f}", int((sub.angle_from_convention > 30).sum()),
                     f"{sub.seconds.median():.2f}", ", ".join(f"{k} {v}" for k, v in sub.start.value_counts().items())])
    lines = table([by, "n", "fallimenti (residuo > T)", "residuo mediano", "p95", "angolo dalla convenzione, mediana",
                   "p95", "> 30 gradi", "s/mesh mediani", "start scelti"], rows)
    return lines, out


def consistency(df: pd.DataFrame) -> list[str]:
    """Angolo fra la rotazione di ogni topologia e quella della ``original`` dello stesso soggetto."""
    o = df[df.topology == "original"].set_index("subject")
    rows = []
    for t, sub in df[df.topology != "original"].groupby("topology"):
        a = ang(rots(sub), rots(o.loc[sub.subject]))
        rows.append([t, f"{np.median(a):.2f}", f"{np.percentile(a, 95):.2f}", f"{a.max():.2f}"])
    return table(["topologia", "angolo da R(original), mediana", "p95", "max"], rows)


# ------------------------------------------------------------------------ HIFI3D (a)

def hifi_spearman(idx: zes.Index, D: dict, args) -> tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    pm = base.read_pair_metrics(HIFI_RUNS / "scale_e108_topology" / zsum.STAGE)
    keys = zsum.PAIR_KEYS
    df = pm[keys + ["gt_distance", "latent_distance", "raw_chamfer"]].rename(
        columns={"latent_distance": "e108", "raw_chamfer": "chamfer_eval"})
    df["subject_a"], df["subject_b"] = df["subject_a"].astype(str), df["subject_b"].astype(str)
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"], df["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"], df["topology_b"])])
    checks = {"e108: distanze dagli embedding contro latent_distance delle pair_metrics, max |diff|":
              float(np.abs(D["e108"][ia, ib] - df["e108"]).max())}
    for m in ("e108_canon", "chamfer_fb", "chamfer_fb_canon"):
        df[m] = D[m][ia, ib]
    cols = ["e108", "e108_canon", "chamfer_fb", "chamfer_fb_canon", "chamfer_eval"]
    frames = {"all_cross": df, "nocrop_cross": df[df.topology_a.ne("crop") & df.topology_b.ne("crop")],
              "subject_pair_mean": df.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance"] + cols].mean()}
    tasks = []
    for g, fr in frames.items():
        seed = base.stable_seed(args.seed, "scale_e108", "maxabs", TOK[g], "latent")
        for c in cols:
            tasks.append(((g, c, "ref"), fr, c, args.n_bootstrap, seed))
        # Chamfer eval anche col SUO seme pubblicato (controllo della riga 0.372 / 0.336 / 0.743)
        tasks.append(((g, "chamfer_eval", "pub"), fr, "chamfer_eval", args.n_bootstrap,
                      base.stable_seed(args.seed, "joint", "maxabs", TOK[g], "chamfer")))
        for a, b in PAIRS_A:
            tasks.append(((g, a, b), fr, (a, b), args.n_bootstrap, seed))
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = dict(pool.imap_unordered(zsum._boot_task, tasks))
    rows, pairs = [], []
    for (g, a, b), r in res.items():
        if b in ("ref", "pub"):
            rows.append({"group": g, "method": a, "seed": b, "point": r["spearman"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"], "n_pairs": r["n_pairs"]})
        else:
            pairs.append({"group": g, "a": a, "b": b, "a_point": r["a"], "b_point": r["b"], "diff": r["diff"],
                          "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p_le0": r["p_boot_le0"]})
    rows, pairs = pd.DataFrame(rows), pd.DataFrame(pairs)
    cells = pd.read_csv(RUNS / "data_scale_ood" / "hifi" / "table_cells.csv")
    ctrl = []
    for g in frames:
        for model, metric, method, seed in (("scale_e108", "latent", "e108", "ref"), ("joint", "chamfer", "chamfer_eval", "pub")):
            ref = cells[(cells["model"] == model) & (cells["gt"] == "maxabs") & (cells["protocol"] == PROTO[g])
                        & (cells.scenario == "clean")].iloc[0]
            new = rows[(rows.group == g) & (rows.method == method) & (rows.seed == seed)].iloc[0]
            pref = ref[f"{metric}_point_check"] if np.isfinite(ref[f"{metric}_point_check"]) else ref[f"{metric}_spearman"]
            ctrl.append({"riga": f"{LABEL[method]}, {g}", "pubblicato": fmt(ref[f"{metric}_spearman"], ref[f"{metric}_ci_low"],
                                                                            ref[f"{metric}_ci_high"]),
                         "ricalcolato": fmt(new.point, new.ci_low, new.ci_high),
                         "max_abs_diff": max(abs(new.point - pref), abs(new.ci_low - ref[f"{metric}_ci_low"]),
                                             abs(new.ci_high - ref[f"{metric}_ci_high"]))})
    return rows, pairs, ctrl, checks


# ------------------------------------------------------------------- riconoscimento

def recognition(D: dict, idx: zes.Index, counts: np.ndarray, blocks: dict, workers: int) -> dict:
    tasks = [((blk, m), D[m], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for m in D]
    out = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            out[key] = (vals, n_nan)
    return out


def rec_tables(rec: dict, comparisons) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, deltas = [], []
    for (blk, m), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": m, "n_nan_distances": n_nan}
        for x in ("rank1", "map", "auc"):
            r[x], (r[f"{x}_ci_low"], r[f"{x}_ci_high"]) = float(vals[x][0]), zes.ci(vals[x])
        rows.append(r)
    for blk in sorted({b for b, _ in rec}):
        for a, b in comparisons:
            if (blk, a) not in rec or (blk, b) not in rec:
                continue
            r = {"block": blk, "a": a, "b": b}
            for x in ("rank1", "map", "auc"):
                d = rec[(blk, a)][0][x] - rec[(blk, b)][0][x]
                r[x], (r[f"{x}_ci_low"], r[f"{x}_ci_high"]) = float(d[0]), zes.ci(d)
                r[f"{x}_p_le0"] = float((d[1:] <= 0).mean())
            deltas.append(r)
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def rec_control(rows: pd.DataFrame, ref_csv: Path, mapping: dict) -> list[dict]:
    ref = pd.read_csv(ref_csv).set_index(["block", "method"])
    cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
    out = []
    for mine, theirs in mapping.items():
        r = rows[(rows.block == "nocrop") & (rows.method == mine)].iloc[0]
        x = ref.loc[("nocrop", theirs)]
        out.append({"riga": LABEL[mine], "pubblicato rank-1": fmt(x["rank1"], x["rank1_ci_low"], x["rank1_ci_high"]),
                    "ricalcolato rank-1": fmt(r.rank1, r.rank1_ci_low, r.rank1_ci_high),
                    "max_abs_diff": max(abs(r[c] - x[c]) for c in cols)})
    return out


def blocks_rec() -> dict:
    nocrop = zes.NOCROP
    return {"nocrop": ([(a, b) for a in nocrop for b in nocrop if a != b],
                       [(a, b) for i, a in enumerate(nocrop) for b in nocrop[i + 1:]]),
            "crop": ([(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b and "crop" in (a, b)],
                     [("crop", b) for b in nocrop])}


# ------------------------------------------------------------------------------- NoW

def now_tau(args) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    import now_common as nc
    import now_summarize as ns
    items = nc.load_items()
    subjects = nc.subjects_of(items)
    s2i = {s: k for k, s in enumerate(subjects)}
    by_name = {it.name: it for it in items}
    counts = zes.bootstrap_counts(len(subjects), args.n_bootstrap, args.seed)
    methods = list(nc.METHODS)
    frames = []
    for m in methods:
        df = pd.read_csv(RUNS / "now_eval" / f"now_official_{m}.csv")[["name", "now_median"]]
        for path, col, new in ((RUNS / "now_eval_scale_e108" / f"gt_latent_joint_{m}.csv", "latent_joint", "latent"),
                               (E2 / "now" / f"gt_latent_joint_{m}.csv", "latent_joint", "latent_canon"),
                               (RUNS / "now_eval" / f"gt_geometric_{m}.csv", "chamfer_raw", "chamfer_raw"),
                               (E2 / "now" / f"gt_geometric_{m}.csv", "chamfer_raw", "chamfer_raw_canon"),
                               (RUNS / "now_eval" / f"gt_geometric_{m}.csv", "icp_chamfer_mm", "icp"),
                               (E2 / "now" / f"gt_geometric_{m}.csv", "icp_chamfer_mm", "icp_canon")):
            df = df.merge(pd.read_csv(path)[["name", col]].rename(columns={col: new}), on="name", how="inner")
        df["method"] = m
        frames.append(df)
    dc = pd.concat(frames, ignore_index=True)
    dc["subject"] = dc["name"].map(lambda n: by_name[n].subject)
    dc["s"] = dc["subject"].map(s2i)
    metrics = ["latent", "latent_canon", "chamfer_raw", "chamfer_raw_canon", "icp", "icp_canon"]
    ok = dc.dropna(subset=["now_median"] + metrics).groupby("name")["method"].nunique()
    names = sorted(ok[ok == len(methods)].index)
    dc = dc[dc["name"].isin(names)].copy()
    prereg = [m for m in ns.PREREGISTERED if m in methods]
    conc, reps, _ = ns.concordance(dc, prereg, metrics, names, by_name, s2i, counts)
    conc = pd.DataFrame(conc)
    deltas = []
    for a, b in (("latent_canon", "latent"), ("chamfer_raw_canon", "chamfer_raw"), ("latent_canon", "chamfer_raw_canon"),
                 ("latent", "chamfer_raw")):
        for col, j in (("tau_image", 0), ("spearman", 1)):
            d = reps[a][j] - reps[b][j]
            point = float(conc.set_index("metric").loc[a, col] - conc.set_index("metric").loc[b, col])
            lo, hi = zes.ci(np.concatenate([[point], d]))
            deltas.append({"a": a, "b": b, "concordance": col, "delta": point, "ci_low": lo, "ci_high": hi,
                           "p_le0": float((d <= 0).mean())})
    ref = pd.read_csv(RUNS / "now_eval_scale_e108" / "concordance.csv")
    ref = ref[ref.methods == "+".join(prereg)].set_index("metric")
    ctrl = []
    for mine, theirs in (("latent", "latent_joint"), ("chamfer_raw", "chamfer_raw"), ("icp", "icp_chamfer_mm")):
        r, x = conc.set_index("metric").loc[mine], ref.loc[theirs]
        ctrl.append({"riga": LABEL[mine], "pubblicato": fmt(x.tau_image, x.tau_image_ci_low, x.tau_image_ci_high),
                     "ricalcolato": fmt(r.tau_image, r.tau_image_ci_low, r.tau_image_ci_high),
                     "max_abs_diff": max(abs(r[c] - x[c]) for c in ("tau_image", "tau_image_ci_low", "tau_image_ci_high",
                                                                    "spearman", "spearman_ci_low", "spearman_ci_high"))})
    return conc, pd.DataFrame(deltas), ctrl


# ------------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--out-md", type=Path, default=E2 / "summary.md")
    args = p.parse_args()
    T = json.loads((E2 / "calibration" / "threshold.json").read_text())["T"]
    out = E2 / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# E2: canonicalizzazione rigida al test, e108 con e senza\n",
          f"Protocollo dichiarato prima: `aau/runs/evidence/e2/protocol.md`; calibrazione e soglia: "
          f"`calibration/calibration.md` (T = {T:.4f}). Script: `aau/evidence/e2_canon/`. CI 95% bootstrap per soggetto, "
          f"{args.n_bootstrap} repliche; P(<=0) = frazione di repliche con differenza <= 0.\n"]

    # canonicalizzazione: fallimenti, tempi, angoli
    canon_rows = []
    md.append("## Canonicalizzazione: fallimenti, tempo, angoli\n")
    for name, path, by in (("HIFI3D", E2 / "hifi" / "canon_records.csv", "topology"),
                           ("FaceVerse con espressioni", E2 / "fv" / "canon_records.csv", "topology"),
                           ("NoW", E2 / "now" / "canon_records.csv", "group")):
        if not path.exists():
            md += [f"### {name}\n", "(non ancora calcolato)", ""]
            continue
        df = pd.read_csv(path)
        if by == "group":
            df["group"] = df["file"].map(lambda f: f.split("/")[0] if f.startswith("scan_face") else f.split("/")[1])
        else:
            df["subject"], df["topology"] = zip(*df["file"].map(lambda f: f.split("_GTready_")))
        lines, summ = canon_section(name, df, T, by)
        canon_rows.append(summ)
        md += [f"### {name}\n", *lines, ""]
        if by == "topology":
            md += ["Consistenza fra topologie dello stesso soggetto:\n", *consistency(df), ""]
    canon_tab = pd.DataFrame(canon_rows)
    canon_tab.to_csv(out / "canon_summary.csv", index=False)
    md += ["Riepilogo (fallimento = residuo > T, IC di Wilson; tempo = secondi per mesh su 1 thread, 8 start):\n",
           *table(["dominio", "n", "fallimenti", "tasso [IC 95%]", "s/mesh mediana (p95)", "angolo dalla convenzione mediana (p95)",
                   "> 30 gradi", "residuo mediano"],
                  [[r.dominio, r.n, r.fallimenti, fmt(r.tasso, r.ci_low, r.ci_high), f"{r.s_mediana:.2f} ({r.s_p95:.2f})",
                    f"{r.angolo_conv_mediana:.2f} ({r.angolo_conv_p95:.2f})", r.angolo_conv_gt30, f"{r.residuo_mediana:.4f}"]
                   for r in canon_tab.itertuples()]), ""]

    # HIFI3D
    subj_h = select_subjects(REPO_ROOT / "datasets/HIFI3D/eval_view/npz", 1234)
    idx_h = zes.Index(subj_h)
    Dh = {"e108": zes.model_distances(HIFI_RUNS / "scale_e108_embed" / zsum.STAGE, idx_h),
          "e108_canon": zes.model_distances(E2 / "hifi" / "canon_e108", idx_h),
          "chamfer_fb": zes.facebench_distances(HIFI_RUNS / "baselines", "chamfer", idx_h),
          "chamfer_fb_canon": zes.facebench_distances(E2 / "hifi" / "canon_chamfer", "chamfer", idx_h),
          "rigid_icp_chamfer": zes.facebench_distances(HIFI_RUNS / "baselines", "rigid_icp_chamfer", idx_h)}
    sp_rows, sp_pairs, sp_ctrl, sp_checks = hifi_spearman(idx_h, Dh, args)
    sp_rows.to_csv(out / "hifi_spearman.csv", index=False)
    sp_pairs.to_csv(out / "hifi_spearman_paired.csv", index=False)
    counts_h = zes.bootstrap_counts(len(subj_h), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    comps = (("e108_canon", "e108"), ("chamfer_fb_canon", "chamfer_fb"), ("e108_canon", "chamfer_fb_canon"),
             ("e108", "chamfer_fb"), ("e108_ictconv", "e108"), ("e108_canon", "e108_ictconv"))
    rec_h = recognition(Dh, idx_h, counts_h, blocks_rec(), args.workers)
    rh_rows, rh_d = rec_tables(rec_h, comps)
    rh_rows.to_csv(out / "hifi_recognition.csv", index=False)
    rh_d.to_csv(out / "hifi_recognition_paired.csv", index=False)
    rh_ctrl = rec_control(rh_rows, RUNS / "data_scale_ood/arcface_vs_scale_hifi3d/recognition.csv",
                          {"e108": "scale_e108", "chamfer_fb": "chamfer", "rigid_icp_chamfer": "rigid_icp_chamfer"})

    # FaceVerse con espressioni
    subj_f = select_subjects(REPO_ROOT / "datasets/FACEVERSE_ZS/expr_view/npz", 1234)
    idx_f = zes.Index(subj_f)
    Df = {"e108": zes.model_distances(FV_RUNS / "scale_e108_flip_topology" / zsum.STAGE, idx_f),
          "e108_canon": zes.model_distances(E2 / "fv" / "canon_e108", idx_f),
          "chamfer_fb": zes.facebench_distances(FV_RUNS / "baselines", "chamfer", idx_f),
          "chamfer_fb_canon": zes.facebench_distances(E2 / "fv" / "canon_chamfer", "chamfer", idx_f),
          "rigid_icp_chamfer": zes.facebench_distances(FV_RUNS / "baselines", "rigid_icp_chamfer", idx_f)}
    ict_stage = FV_RUNS / "scale_e108_frame-xmymz_topology" / zsum.STAGE
    if (ict_stage / "embeddings.npz").exists():
        Df["e108_ictconv"] = zes.model_distances(ict_stage, idx_f)
    counts_f = zes.bootstrap_counts(len(subj_f), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    rec_f = recognition(Df, idx_f, counts_f, blocks_rec(), args.workers)
    rf_rows, rf_d = rec_tables(rec_f, comps)
    rf_rows.to_csv(out / "fv_recognition.csv", index=False)
    rf_d.to_csv(out / "fv_recognition_paired.csv", index=False)
    rf_ctrl = rec_control(rf_rows, RUNS / "data_scale_ood/fvexpr_partial/recognition.csv",
                          {"e108": "scale_e108@bfm", "chamfer_fb": "chamfer", "rigid_icp_chamfer": "rigid_icp_chamfer"})

    # NoW (se il job NoW ha finito)
    have_now = all((E2 / "now" / f"gt_{k}_{m}.csv").exists() for k in ("latent_joint", "geometric")
                   for m in ("3ddfa_v2", "synergynet", "prnet", "mica"))
    if have_now:
        conc, now_d, now_ctrl = now_tau(args)
        conc.to_csv(out / "now_concordance.csv", index=False)
        now_d.to_csv(out / "now_concordance_paired.csv", index=False)
    else:
        now_ctrl = []

    # ------------------------------------------------------------------ markdown
    def sp_cell(g, m):
        r = sp_rows[(sp_rows.group == g) & (sp_rows.method == m) & (sp_rows.seed == "ref")].iloc[0]
        return fmt(r.point, r.ci_low, r.ci_high)

    def sp_delta(g, a, b):
        r = sp_pairs[(sp_pairs.group == g) & (sp_pairs.a == a) & (sp_pairs.b == b)].iloc[0]
        return f"{fmt(r['diff'], r.ci_low, r.ci_high, True)} ({r.p_le0:.3f})"

    def rec_cell(rows, blk, m):
        r = rows[(rows.block == blk) & (rows.method == m)]
        if r.empty:
            return ["-"] * 3
        r = r.iloc[0]
        return [fmt(r[x], r[f"{x}_ci_low"], r[f"{x}_ci_high"]) for x in ("rank1", "map", "auc")]

    def rec_delta(d, blk, a, b):
        r = d[(d.block == blk) & (d.a == a) & (d.b == b)]
        if r.empty:
            return ["-"] * 3
        r = r.iloc[0]
        return [f"{fmt(r[x], r[f'{x}_ci_low'], r[f'{x}_ci_high'], True)} ({r[f'{x}_p_le0']:.3f})" for x in ("rank1", "map", "auc")]

    groups = ("nocrop_cross", "all_cross", "subject_pair_mean")
    md += ["## HIFI3D: Spearman con la GT maxabs\n",
           "Righe delle pair_metrics di e108 (coppie di soggetti diversi, coppie ordinate di topologie); ogni gruppo col seme "
           "della riga e108 pubblicata, per tutte le righe e le differenze del gruppo.\n",
           *table(["metodo"] + list(groups), [[LABEL[m]] + [sp_cell(g, m) for g in groups]
                                               for m in ("e108", "e108_canon", "chamfer_fb", "chamfer_fb_canon", "chamfer_eval")]),
           "\nDifferenze appaiate:\n",
           *table(["A - B"] + [f"{g} [IC] (P<=0)" for g in groups],
                  [[f"{LABEL[a]} - {LABEL[b]}"] + [sp_delta(g, a, b) for g in groups] for a, b in PAIRS_A])]
    for title, rows, d, has_ict in (("HIFI3D", rh_rows, rh_d, False), ("FaceVerse con espressioni", rf_rows, rf_d,
                                                                       "e108_ictconv" in Df)):
        ms = ["e108", "e108_canon"] + (["e108_ictconv"] if has_ict else []) + ["chamfer_fb", "chamfer_fb_canon", "rigid_icp_chamfer"]
        for blk, sub in (("nocrop", "5 topologie senza crop (primario)"), ("crop", "coppie con crop (a parte)")):
            md += [f"\n## {title}: riconoscimento, {sub}\n",
                   *table(["metodo", "rank-1", "mAP", "AUC"], [[LABEL[m]] + rec_cell(rows, blk, m) for m in ms]),
                   "\nDifferenze appaiate (stesse repliche):\n",
                   *table(["A - B", "rank-1 [IC] (P<=0)", "mAP", "AUC"],
                          [[f"{LABEL[a]} - {LABEL[b]}"] + rec_delta(d, blk, a, b) for a, b in comps
                           if (blk, a) in (rec_h if title == "HIFI3D" else rec_f) and (blk, b) in (rec_h if title == "HIFI3D" else rec_f)])]
    if not have_now:
        md += ["\n## NoW\n", "(non ancora calcolato)"]
    else:
        cidx = conc.set_index("metric")
        md += ["\n## NoW: concordanza con l'errore NoW (3 metodi pre-registrati)\n",
               *table(["metrica", "tau per immagine [IC]", "Spearman per ricostruzione [IC]"],
                      [[LABEL[m], fmt(cidx.loc[m, "tau_image"], cidx.loc[m, "tau_image_ci_low"], cidx.loc[m, "tau_image_ci_high"]),
                        fmt(cidx.loc[m, "spearman"], cidx.loc[m, "spearman_ci_low"], cidx.loc[m, "spearman_ci_high"])]
                       for m in ("latent", "latent_canon", "chamfer_raw", "chamfer_raw_canon", "icp", "icp_canon")]),
               "\nDifferenze appaiate (stesse repliche dei 20 soggetti):\n",
               *table(["A - B", "tau per immagine [IC] (P<=0)", "Spearman [IC] (P<=0)"],
                      [[f"{LABEL[a]} - {LABEL[b]}"] + [
                          f"{fmt(r.delta, r.ci_low, r.ci_high, True)} ({r.p_le0:.3f})"
                          for r in now_d[(now_d.a == a) & (now_d.b == b)].sort_values("concordance", ascending=False).itertuples()]
                       for a, b in now_d[["a", "b"]].drop_duplicates().itertuples(index=False)])]
    md += ["\n## Controlli: le righe senza canonicalizzazione riprodotte\n",
           *table(["riga", "pubblicato", "ricalcolato", "max |diff| (punto, IC)"],
                  [[c["riga"], c["pubblicato"], c["ricalcolato"], f"{c['max_abs_diff']:.2e}"] for c in sp_ctrl]),
           "",
           *table(["riga (riconoscimento HIFI3D)", "pubblicato rank-1", "ricalcolato rank-1", "max |diff| (rank-1, mAP, AUC, IC)"],
                  [[c["riga"], c["pubblicato rank-1"], c["ricalcolato rank-1"], f"{c['max_abs_diff']:.2e}"] for c in rh_ctrl]),
           "",
           *table(["riga (riconoscimento FaceVerse)", "pubblicato rank-1", "ricalcolato rank-1", "max |diff|"],
                  [[c["riga"], c["pubblicato rank-1"], c["ricalcolato rank-1"], f"{c['max_abs_diff']:.2e}"] for c in rf_ctrl]),
           "",
           *table(["riga (NoW, tau per immagine)", "pubblicato", "ricalcolato", "max |diff| (tau, Spearman, IC)"],
                  [[c["riga"], c["pubblicato"], c["ricalcolato"], f"{c['max_abs_diff']:.2e}"] for c in now_ctrl]),
           "", *[f"- {k}: {v:.2e}" for k, v in sp_checks.items()]]
    for dom in ("hifi", "fv"):
        f = E2 / dom / "canon_chamfer" / "control_chamfer.json"
        if f.exists():
            md.append(f"- Chamfer faceBench, {dom}: la funzione usata sulle mesh canonicalizzate, rifatta sulla vista "
                      f"originale per due coppie di topologie, contro le matrici esistenti: "
                      + "; ".join(f"{r['pair']} ({r['kind']}) max |diff| {r['max_abs_diff']:.2e}" for r in json.loads(f.read_text())))
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[e2-sum] scritto {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
