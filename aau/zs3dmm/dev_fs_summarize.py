#!/usr/bin/env python3
"""Numeri del set di sviluppo FaceScape: distanza graduata, riconoscimento con espressioni, punteggio dev.

    aau/run.sh aau/zs3dmm/dev_fs_summarize.py --runs-neutral aau/runs/ws_dev_facescape/data_<fp> \\
        --runs-expr aau/runs/ws_dev_facescape_expr/data_<fp> --root datasets/DEV_FACESCAPE \\
        --out-dir aau/runs/evidence/dev_facescape
    (dev_fs_summarize.sbatch)

Il set (PLAN_MASSIVE sez. 9): FaceScape bilineare, 100 soggetti estratti col seed 1234 dal pool di
500 (``zs_stage.select_subjects``, gli stessi nelle due viste), 6 topologie; vista neutra e vista
con un'espressione casuale per mesh, GT d'identita' neutra.

Bracci: le stage di zs_zeroshot.sbatch con ``.done`` e ``embeddings.npz``, cioe'
``<braccio>_topology`` (WBES_ZS_PART=topology e WBES_ZS_EMBED=1: c'e' anche il breakdown, quindi la
Chamfer eval) oppure ``<braccio>_embed`` (WBES_ZS_PART=embed, anche su CPU: il device ricade su CPU
senza CUDA); se ci sono entrambe vince ``_topology``. Nessuna trasformazione: FaceScape e' gia' nel
frame dei dati ICT (+y alto, +z naso, normali uscenti). Le coppie della graduata si costruiscono qui
come nel breakdown (soggetti a < b, 30 coppie ordinate di topologie, GT dalla matrice della vista) e
la distanza del braccio e' ||z_a - z_b|| dagli embedding, cioe' la ``latent_distance`` del breakdown
(confrontata nei controlli quando c'e'); la Chamfer eval entra solo col breakdown.

Misure, con le funzioni dei riepiloghi esistenti (importate, non riscritte):
  - GRADUATA (vista neutra): Spearman fra distanza e GT, ``nocrop_cross`` (le 20 coppie ordinate
    di topologie senza crop: il primario), ``all_cross``, ``subject_pair_mean``; GT ``maxabs``
    (primaria, sez. 14.4) e ``coef``. Metodi: i bracci, Chamfer eval se c'e', Chamfer intera e su
    regione stabile (``zs_region_chamfer.py``, 4096 punti, dev_fs_chamfer.sbatch). CI 95% bootstrap
    per soggetto e delta appaiati (``zs_summarize.paired_bootstrap``).
  - RICONOSCIMENTO (vista con espressioni): rank-1, mAP, AUC di verifica sulle 5 topologie senza
    crop (``zs_expr_summarize``: query in t1, galleria di 100 mesh in t2 != t1), crop a parte;
    delta appaiati sulle stesse repliche. La Chamfer eval non c'e': lo script non calcola le coppie
    stesso soggetto. La stessa tabella sulla vista neutra, come accessorio.
  - PUNTEGGIO DEV (sez. 9): media fra Spearman graduato senza crop (GT maxabs) e rank-1 con
    espressioni senza crop, CI sulle STESSE repliche di soggetti per le due meta' (i soggetti sono
    gli stessi nelle due viste). Per Chamfer: Chamfer intera in entrambe le meta' (stessa
    implementazione) e, se c'e', Chamfer eval nella graduata.

Scrive in ``--out-dir``: ``graded.csv``, ``recognition.csv``, ``recognition_paired.csv``,
``devscore.csv``, ``results.json`` e ``results.md`` (le tabelle, riprese da summary.md).
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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import zs_expr_summarize as zex  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import select_subjects  # noqa: E402

base = zsum.base
STAGE = "zs_zeroshot"
CHAMFERS = ("raw_chamfer", "chamfer_full", "chamfer_stable")
LABEL = {"raw_chamfer": "Chamfer eval", "chamfer_full": "Chamfer intera (4096 pt)",
         "chamfer_stable": "Chamfer regione stabile", "scale_e108": "e108 (BFM+ICT+GNM)",
         "chamfer_eval+full": "Chamfer eval (graduata) + intera (rank-1)"}
SCENARIOS = ("nocrop_cross", "all_cross", "subject_pair_mean")
KEYS = ["subject_a", "subject_b", "topology_a", "topology_b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-neutral", type=Path, required=True)
    p.add_argument("--runs-expr", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="datasets/DEV_FACESCAPE")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234, help="seme del ricampionamento")
    p.add_argument("--eval-seed", type=int, default=1234, help="WBES_EVAL_SEED: scelta dei soggetti")
    return p.parse_args()


def label(name: str) -> str:
    return LABEL.get(name, name)


def arms_of(runs: Path) -> dict[str, dict]:
    """Bracci con embedding completi; ``_topology`` (che ha anche il breakdown) vince su ``_embed``."""
    out = {}
    for tail in ("_embed", "_topology"):
        for d in sorted(runs.glob(f"*{tail}")):
            st = d / STAGE
            if (st / ".done").exists() and (st / "embeddings.npz").exists():
                out[d.name[: -len(tail)]] = {"stage": st, "breakdown": st if tail == "_topology" else None}
    return out


def pair_frame(idx: zex.Index, gt: tuple, latents: dict, region: dict, breakdown: Path | None) -> pd.DataFrame:
    """Coppie del breakdown (soggetti a < b, 30 coppie ordinate di topologie): GT, ``lat:<braccio>``
    dagli embedding, Chamfer su regione e, se c'e' il breakdown, ``raw_chamfer`` (Chamfer eval)."""
    subj = np.asarray(idx.subjects)
    iu, ju = np.triu_indices(len(subj), 1)
    frames = []
    for ta in zex.TOPOLOGIES:
        for tb in zex.TOPOLOGIES:
            if ta != tb:
                frames.append(pd.DataFrame({"subject_a": subj[iu], "subject_b": subj[ju], "topology_a": ta,
                                            "topology_b": tb, "_ia": idx.rows(ta)[iu], "_ib": idx.rows(tb)[ju]}))
    df = pd.concat(frames, ignore_index=True)
    D_gt, name_to_idx = gt
    df["gt_distance"] = D_gt[df["subject_a"].map(name_to_idx).to_numpy(int), df["subject_b"].map(name_to_idx).to_numpy(int)]
    for arm, M in latents.items():
        df[f"lat:{arm}"] = M[df["_ia"].to_numpy(), df["_ib"].to_numpy()]
    for m in ("chamfer_full", "chamfer_stable"):
        df[m] = region[m][df["_ia"].to_numpy(), df["_ib"].to_numpy()]
    if breakdown is not None:
        pm = base.read_pair_metrics(breakdown)[KEYS + ["gt_distance", "latent_distance", "raw_chamfer"]]
        pm["subject_a"], pm["subject_b"] = pm["subject_a"].astype(str), pm["subject_b"].astype(str)
        df = df.merge(pm.rename(columns={"gt_distance": "_gt_bd", "latent_distance": "_lat_bd"}), on=KEYS,
                      how="left", validate="one_to_one")
        if df["raw_chamfer"].isna().any():
            raise SystemExit(f"{breakdown}: coppie del breakdown diverse da quelle attese")
    return df


# ------------------------------------------------------------------------------ bootstrap

def _task(task):
    key, df, col_a, col_b, n, seed = task
    bm = base.load_bootstrap_module()
    rng = np.random.default_rng(seed)
    if col_b is None:
        return key, base.bootstrap_row(df, col_a, n, rng, bm)
    return key, zsum.paired_bootstrap(df, col_a, col_b, n, rng, bm)


def spearman_replicates(df: pd.DataFrame, col: str, subjects: list[str], counts: np.ndarray, bm) -> np.ndarray:
    """[punto, repliche]: righe pesate per il prodotto dei conteggi dei due soggetti (come
    ``weighted_bootstrap_spearman``), con i conteggi DATI, cosi' le repliche si appaiano al riconoscimento."""
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = df["subject_a"].astype(str).map(s2i).to_numpy()
    sb = df["subject_b"].astype(str).map(s2i).to_numpy()
    gt, v = df["gt_distance"].to_numpy(np.float64), df[col].to_numpy(np.float64)
    out = [bm.finite_spearman(gt, v)]
    for c in counts:
        w = c[sa].astype(np.int64) * c[sb].astype(np.int64)
        k = w > 0
        out.append(bm.finite_spearman(np.repeat(gt[k], w[k]), np.repeat(v[k], w[k])))
    return np.asarray(out)


def fmt(point: float, lo: float, hi: float, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


def main() -> None:
    args = parse_args()
    view_n, view_e = args.root / "eval_view", args.root / "expr_view"
    subjects = select_subjects(view_n / "npz", args.eval_seed)
    if select_subjects(view_e / "npz", args.eval_seed) != subjects:
        raise SystemExit("soggetti diversi fra vista neutra e vista con espressioni")
    idx = zex.Index(subjects)
    arms_n, arms_e = arms_of(args.runs_neutral), arms_of(args.runs_expr)
    arms = sorted(set(arms_n) & set(arms_e))
    if not arms:
        raise SystemExit(f"nessun braccio completo in entrambe le viste ({sorted(arms_n)} / {sorted(arms_e)})")
    for runs, stages in ((args.runs_neutral, arms_n), (args.runs_expr, arms_e)):
        for a in arms:
            staged = json.loads((stages[a]["stage"].parent / "subjects.json").read_text())
            if sorted(staged["subjects"]) != subjects:
                raise SystemExit(f"{runs}/{a}: zs_stage ha valutato soggetti diversi")
    reg_n = zex.region_distances(args.runs_neutral / "baselines" / "region_chamfer.npz", idx)
    reg_e = zex.region_distances(args.runs_expr / "baselines" / "region_chamfer.npz", idx)
    print(f"[devfs-sum] {len(subjects)} soggetti (primi {subjects[:3]}), bracci: "
          + ", ".join(f"{a} ({'breakdown' if arms_n[a]['breakdown'] else 'solo embedding'})" for a in arms), flush=True)

    gts = {"maxabs": zsum.load_gt(view_n / "gt_matrix.npz"), "coef": zsum.load_gt(view_n / "gt_coef_matrix.npz")}
    lat_n = {a: zex.model_distances(arms_n[a]["stage"], idx) for a in arms}
    lat_e = {a: zex.model_distances(arms_e[a]["stage"], idx) for a in arms}
    bd = {"neutral": next((arms_n[a]["breakdown"] for a in arms if arms_n[a]["breakdown"]), None),
          "expr": next((arms_e[a]["breakdown"] for a in arms if arms_e[a]["breakdown"]), None)}
    pf = {"neutral": pair_frame(idx, gts["maxabs"], lat_n, reg_n, bd["neutral"]),
          "expr": pair_frame(idx, gts["maxabs"], lat_e, reg_e, bd["expr"])}
    chamfers = [c for c in CHAMFERS if c in pf["neutral"] and c in pf["expr"]]
    checks = {"region_kept_fraction_neutral_min_median": [float(reg_n["_kept"].min()), float(np.median(reg_n["_kept"]))],
              "region_kept_fraction_expr_min_median": [float(reg_e["_kept"].min()), float(np.median(reg_e["_kept"]))],
              "chamfer_eval_available": "raw_chamfer" in chamfers,
              "arm_sources": {f"{v}/{a}": str(s[a]["stage"]) for v, s in (("neutral", arms_n), ("expr", arms_e))
                              for a in arms}}
    for view, stages in (("neutral", arms_n), ("expr", arms_e)):
        if bd[view] is None:
            continue
        df = pf[view]
        arm_bd = next(a for a in arms if stages[a]["breakdown"] == bd[view])
        checks[f"{view}: gt del breakdown contro matrice, max |diff|"] = float(np.abs(df["_gt_bd"] - df["gt_distance"]).max())
        checks[f"{view}: latent_distance del breakdown contro embedding di {arm_bd}, max |diff|"] = float(
            np.abs(df["_lat_bd"] - df[f"lat:{arm_bd}"]).max())

    # ------------------------------------------------ graduata (neutra) e secondaria (espressioni)
    tasks = []
    for view in ("neutral", "expr"):
        for gt_tag in (("maxabs", "coef") if view == "neutral" else ("maxabs",)):
            df = pf[view] if gt_tag == "maxabs" else zsum.with_gt(pf[view], gts["coef"])
            cols = [f"lat:{a}" for a in arms] + chamfers
            for scen in (SCENARIOS if view == "neutral" else ("nocrop_cross",)):
                fr = zsum.scenario_frame(df[KEYS + ["gt_distance"] + cols], scen, cols)
                tok = (view, gt_tag, scen)
                for c in chamfers:
                    tasks.append(((*tok, c, None), fr, c, None, args.n_bootstrap,
                                  base.stable_seed(args.seed, "devfs", *tok, c)))
                for a in arms:
                    tasks.append(((*tok, a, None), fr, f"lat:{a}", None, args.n_bootstrap,
                                  base.stable_seed(args.seed, "devfs", *tok, a)))
                    for c in chamfers:
                        tasks.append(((*tok, a, c), fr, f"lat:{a}", c, args.n_bootstrap,
                                      base.stable_seed(args.seed, "devfs_paired", *tok, a, c)))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    res = {}
    with mp.get_context("fork").Pool(min(workers, len(tasks))) as pool:
        for key, r in pool.imap_unordered(_task, tasks):
            res[key] = r
    graded = pd.DataFrame([{"view": k[0], "gt": k[1], "scenario": k[2], "method": k[3], "vs": k[4], **v}
                           for k, v in res.items()]).sort_values(["view", "gt", "scenario", "method", "vs"],
                                                                  na_position="first")

    # ------------------------------------------------ riconoscimento
    counts = zex.bootstrap_counts(len(subjects), args.n_bootstrap, base.stable_seed(args.seed, "devfs_recognition"))
    blocks = {
        "nocrop": ([(a, b) for a in zex.NOCROP for b in zex.NOCROP if a != b],
                   [(a, b) for i, a in enumerate(zex.NOCROP) for b in zex.NOCROP[i + 1:]]),
        "crop": ([(a, b) for a in zex.TOPOLOGIES for b in zex.TOPOLOGIES if a != b and "crop" in (a, b)],
                 [("crop", b) for b in zex.NOCROP]),
    }
    D = {}
    for view, lat, reg in (("expr", lat_e, reg_e), ("neutral", lat_n, reg_n)):
        for a in arms:
            D[(view, a)] = lat[a]
        for c in ("chamfer_full", "chamfer_stable"):
            D[(view, c)] = reg[c]
    rtasks = [((view, blk, name), M, idx, pr, pv, counts) for (view, name), M in D.items()
              for blk, (pr, pv) in blocks.items()]
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        for key, vals, n_nan in pool.imap_unordered(zex._recog_task, rtasks):
            rec[key] = (vals, n_nan)
    rec_rows, delta_rows = [], []
    for (view, blk, name), (vals, n_nan) in rec.items():
        r = {"view": view, "block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zex.ci(vals[m])
        rec_rows.append(r)
        if name in arms:
            delta_rows.append((view, blk, name))
    deltas = []
    for view, blk, name in delta_rows:
        vals = rec[(view, blk, name)][0]
        for c in ("chamfer_full", "chamfer_stable"):
            other = rec[(view, blk, c)][0]
            r = {"view": view, "block": blk, "model": name, "baseline": c}
            for m in ("rank1", "map", "auc"):
                d = vals[m] - other[m]
                r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), zex.ci(d)
                r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
            deltas.append(r)
    recog = pd.DataFrame(rec_rows).sort_values(["view", "block", "method"])
    recog_paired = pd.DataFrame(deltas).sort_values(["view", "block", "model", "baseline"])

    # ------------------------------------------------ punteggio dev (stesse repliche nelle due meta')
    bm = base.load_bootstrap_module()
    dev_rows, score = [], {}
    nocrop_n = zsum.scenario_frame(pf["neutral"], "nocrop_cross", [])
    for a in arms:
        sp = spearman_replicates(nocrop_n, f"lat:{a}", subjects, counts, bm)
        score[a] = (sp + rec[("expr", "nocrop", a)][0]["rank1"]) / 2
        dev_rows.append({"method": a, "graded": sp, "rank1": rec[("expr", "nocrop", a)][0]["rank1"]})
    r1 = rec[("expr", "nocrop", "chamfer_full")][0]["rank1"]
    sf = spearman_replicates(nocrop_n, "chamfer_full", subjects, counts, bm)
    score["chamfer_full"] = (sf + r1) / 2
    dev_rows.append({"method": "chamfer_full", "graded": sf, "rank1": r1})
    if "raw_chamfer" in chamfers:
        se = spearman_replicates(nocrop_n, "raw_chamfer", subjects, counts, bm)
        score["chamfer_eval+full"] = (se + r1) / 2
        dev_rows.append({"method": "chamfer_eval+full", "graded": se, "rank1": r1})
    dev = []
    for r in dev_rows:
        sc = score[r["method"]]
        dev.append({"method": r["method"], "graded_nocrop": float(r["graded"][0]), "rank1_expr_nocrop": float(r["rank1"][0]),
                    "dev_score": float(sc[0]), "dev_ci_low": zex.ci(sc)[0], "dev_ci_high": zex.ci(sc)[1]})
    for a in arms:
        for c in [x for x in ("chamfer_full", "chamfer_eval+full") if x in score]:
            d = score[a] - score[c]
            dev.append({"method": f"{a} - {c}", "dev_score": float(d[0]), "dev_ci_low": zex.ci(d)[0],
                        "dev_ci_high": zex.ci(d)[1], "p_le0": float((d[1:] <= 0).mean())})
    devscore = pd.DataFrame(dev)

    # ------------------------------------------------ uscite
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    graded.to_csv(out / "graded.csv", index=False)
    recog.to_csv(out / "recognition.csv", index=False)
    recog_paired.to_csv(out / "recognition_paired.csv", index=False)
    devscore.to_csv(out / "devscore.csv", index=False)
    (out / "results.json").write_text(json.dumps({"subjects": subjects, "arms": arms, "checks": checks,
                                                  "runs_neutral": str(args.runs_neutral),
                                                  "runs_expr": str(args.runs_expr),
                                                  "n_bootstrap": args.n_bootstrap}, indent=1) + "\n")

    def g(view, gt, scen, method, vs=None):
        x = graded[(graded["view"] == view) & (graded["gt"] == gt) & (graded["scenario"] == scen)
                   & (graded["method"] == method) & (graded["vs"].isna() if vs is None else graded["vs"] == vs)]
        return x.iloc[0]

    md = ["## Distanza graduata (vista neutra), Spearman con la GT [CI 95%]", "",
          "| metodo | senza crop, GT maxabs | all_cross, GT maxabs | subject-pair-mean, GT maxabs | senza crop, GT coef |",
          "| --- | --- | --- | --- | --- |"]
    for m in [*arms, *chamfers]:
        cells = [fmt(r["spearman"], r["ci_low"], r["ci_high"]) for r in
                 (g("neutral", "maxabs", sc, m) for sc in SCENARIOS)]
        r = g("neutral", "coef", "nocrop_cross", m)
        md.append(f"| {label(m)} | " + " | ".join(cells) + f" | {fmt(r['spearman'], r['ci_low'], r['ci_high'])} |")
    md += ["", "Delta appaiati braccio - Chamfer (stesse repliche bootstrap per soggetto):", "",
           "| confronto | senza crop, maxabs | all_cross, maxabs | subject-pair-mean, maxabs | senza crop, coef |",
           "| --- | --- | --- | --- | --- |"]
    for a in arms:
        for c in chamfers:
            cells = []
            for gt, sc in (("maxabs", "nocrop_cross"), ("maxabs", "all_cross"), ("maxabs", "subject_pair_mean"),
                           ("coef", "nocrop_cross")):
                r = g("neutral", gt, sc, a, c)
                cells.append(f"{fmt(r['diff'], r['ci_low'], r['ci_high'], True)} (P<=0 {r['p_boot_le0']:.3f})")
            md.append(f"| {label(a)} - {label(c)} | " + " | ".join(cells) + " |")
    for view, title in (("expr", "Riconoscimento con espressioni (vista con espressioni)"),
                        ("neutral", "Riconoscimento senza espressioni (vista neutra, accessorio)")):
        md += ["", f"## {title}", "",
               "| metodo | blocco | rank-1 | mAP | AUC verifica | NaN |", "| --- | --- | --- | --- | --- | --- |"]
        for r in recog[recog["view"] == view].itertuples():
            md.append(f"| {label(r.method)} | {r.block} | {fmt(r.rank1, r.rank1_ci_low, r.rank1_ci_high)} | "
                      f"{fmt(r.map, r.map_ci_low, r.map_ci_high)} | {fmt(r.auc, r.auc_ci_low, r.auc_ci_high)} | "
                      f"{r.n_nan_distances} |")
        md += ["", "| delta | blocco | rank-1 | mAP | AUC |", "| --- | --- | --- | --- | --- |"]
        for r in recog_paired[recog_paired["view"] == view].itertuples():
            md.append(f"| {label(r.model)} - {label(r.baseline)} | {r.block} | "
                      + " | ".join(f"{fmt(getattr(r, m), getattr(r, m + '_ci_low'), getattr(r, m + '_ci_high'), True)} "
                                   f"(P<=0 {getattr(r, m + '_p_le0'):.3f})" for m in ("rank1", "map", "auc")) + " |")
    md += ["", "## Secondario: vista con espressioni, Spearman con la GT d'identita' neutra, senza crop", "",
           "| metodo | Spearman [CI] | delta braccio - metodo [CI] |", "| --- | --- | --- |"]
    for m in [*arms, *chamfers]:
        r = g("expr", "maxabs", "nocrop_cross", m)
        d = "" if m in arms else "; ".join(
            f"{label(a)}: {fmt(x['diff'], x['ci_low'], x['ci_high'], True)}" for a in arms
            for x in [g("expr", "maxabs", "nocrop_cross", a, m)])
        md.append(f"| {label(m)} | {fmt(r['spearman'], r['ci_low'], r['ci_high'])} | {d or '-'} |")
    md += ["", "## Punteggio dev (sez. 9): media fra graduata senza crop (neutra, GT maxabs) e rank-1 con "
           "espressioni senza crop", "",
           "| metodo | graduata | rank-1 | punteggio [CI 95%] |", "| --- | --- | --- | --- |"]
    for r in devscore.itertuples():
        if " - " in r.method:
            a, c = r.method.split(" - ")
            md.append(f"| {label(a)} - {label(c)} | | | {fmt(r.dev_score, r.dev_ci_low, r.dev_ci_high, True)} "
                      f"(P<=0 {r.p_le0:.3f}) |")
        else:
            md.append(f"| {label(r.method)} | {r.graded_nocrop:.3f} | {r.rank1_expr_nocrop:.3f} | "
                      f"{fmt(r.dev_score, r.dev_ci_low, r.dev_ci_high)} |")
    md += ["", "## Controlli", "", "```", json.dumps(checks, indent=1), "```"]
    (out / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
