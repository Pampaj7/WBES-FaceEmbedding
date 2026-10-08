#!/usr/bin/env python3
"""E1: matrice celle x domini di test ai due numeri di passi e differenze appaiate, con la regola del PI.

    aau/run.sh aau/evidence/e1_factorial/e1_summarize.py        (e1_summarize.sbatch, nessuna GPU)

Non ricalcola distanze ne' embedding: legge le uscite delle pipeline lanciate da e1_eval_body.sh e, per
C3M, quelle gia' pubblicate in aau/runs/data_scale_ood/curve.md (stessi bracci, stessi file). Misure e
bootstrap sono le funzioni dei summarizer esistenti, importate e non riscritte:
  - Spearman con la GT (HIFI3D, FLAME): ``base.bootstrap_row`` per cella, col seme di zs_summarize.boot
    (``stable_seed(1234, <braccio>, "maxabs", <gruppo>, "latent")``: per C3M = i bracci scale_e036/e072,
    quindi le righe di curve.md tornano identiche, controllo in fondo); differenze appaiate con
    ``zs_summarize.paired_bootstrap`` sulle righe allineate (``merge_arms``), UN seme per (dominio, scenario):
    tutte le differenze di uno scenario stanno sulle stesse 1000 repliche per soggetto;
  - riconoscimento (HIFI3D, FaceVerse con espressioni): ``zs_expr_summarize`` (Index, model_distances,
    retrieval_queries, verification_pairs, recognition_values) con ``bootstrap_counts`` seme
    ``stable_seed(1234, "expr_recognition")``, le stesse repliche per tutte le celle (come in curve.md);
    blocco senza crop (5 topologie);
  - NoW: tau di Kendall per immagine sui 3 metodi pre-registrati, ``now_summarize.load_table`` e
    ``kendall_rows``, repliche ``bootstrap_counts(20, 1000, 1234)`` come now_summarize.
Celle mancanti (run o eval non ancora finiti): righe "-", e la regola dice "non valutabile".

Scrive in aau/runs/evidence/e1/: summary.md (= protocol.md invariato + risultati), matrix.csv, paired.csv,
controls.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "aau/runs"
OUT = RUNS / "evidence/e1"
sys.path.insert(0, str(REPO / "aau/zs3dmm"))
sys.path.insert(0, str(REPO / "aau/recon"))

import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import select_subjects  # noqa: E402

base = zsum.base
SEED = 1234
NB = 1000
CELLS = ("c3m", "c2m", "c2f", "c3f", "g1")
LABEL = {"c3m": "C3M", "c2m": "C2M", "c2f": "C2F", "c3f": "C3F", "g1": "G1"}
STEPS = {"036": 10548, "072": 21096}
CONTRASTS = (("c3m", "c2m", "varieta', molte identita'"), ("c3f", "c2f", "varieta', poche identita'"),
             ("c2m", "c2f", "quantita', 2 domini"), ("c3m", "c3f", "quantita', 3 domini"),
             ("g1", "c3m", "solo GNM contro C3M"), ("g1", "c3f", "solo GNM contro C3F"))
SCEN = ("nocrop_cross", "all_cross", "subject_pair_mean")
TOK = {"nocrop_cross": "nocrop_cross", "all_cross": "all_cross", "subject_pair_mean": "spm_clean"}
SCALE_RUN = RUNS / "data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
HIFI_C3M = RUNS / "ws_hifi3d/data_328f2bfc1a"
REF = RUNS / "data_scale_ood"


def tag(cell: str, e: str) -> str:
    return f"e{e}" if cell == "c3m" else f"e1{cell}e{e}"


def one_dir(pattern_root: Path, glob: str) -> Path | None:
    hits = sorted(pattern_root.glob(glob)) if pattern_root.is_dir() else []
    if len(hits) > 1:
        raise SystemExit(f"{pattern_root}/{glob}: piu' di una cartella {hits}")
    return hits[0] if hits else None


def eval_ckpt(stage: Path) -> str:
    key = dict(l.split("=", 1) for l in (stage.parent / "eval_key.txt").read_text().splitlines() if "=" in l)
    return os.path.realpath(key["ckpt"])


def expected_ckpt(cell: str, e: str) -> str | None:
    rr = SCALE_RUN if cell == "c3m" else None
    if rr is None:
        ptr = OUT / f"train_{cell}_s{SEED}.runs_root"
        if not ptr.exists():
            return None
        rr = Path(ptr.read_text().strip())
    hits = sorted(rr.glob(f"mixed_*/checkpoints/epoch{e}.pth"))
    return os.path.realpath(hits[0]) if hits else None


# ------------------------------------------------------------------------------- sorgenti

def hifi_stage(cell: str, e: str, part: str) -> Path | None:
    """part = topology (pair_metrics) | embed (embeddings.npz). C3M: i bracci di curve.md in ws_hifi3d
    (scale_e0NN_topology e scale_e0NN_embed); le altre celle: hifi_runs, tutto in scale_<tag>_topology
    (WBES_ZS_PART=topology con WBES_ZS_EMBED=1)."""
    if cell == "c3m":
        st = HIFI_C3M / f"scale_{tag(cell, e)}_{part}" / zsum.STAGE
    else:
        d = one_dir(OUT / "hifi_runs", "data_*")
        st = None if d is None else d / f"scale_{tag(cell, e)}_topology" / zsum.STAGE
    if st is None or not (st / ".done").exists():
        return None
    if part == "embed" and not (st / "embeddings.npz").exists():
        return None
    return st


def fv_stage(cell: str, e: str) -> Path | None:
    d = one_dir(OUT / "fv_expr", "data_*")
    st = None if d is None else d / f"scale_{tag(cell, e)}_flip_embed" / zsum.STAGE
    return st if st is not None and (st / ".done").exists() and (st / "embeddings.npz").exists() else None


def flame_stage(cell: str, e: str) -> Path | None:
    d = one_dir(OUT / "flame" / tag(cell, e), "*/joint")
    st = None if d is None else d / "flame_zeroshot"
    return st if st is not None and (st / ".done").exists() else None


def now_dir(cell: str, e: str) -> Path | None:
    d = RUNS / f"now_eval_scale_e{e}" if cell == "c3m" else OUT / "now" / tag(cell, e)
    return d if (d / "concordance.csv").exists() else None


# --------------------------------------------------------------------------- Spearman con la GT

def spm_frame(pm: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pm.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance"] + cols].mean()


def scen_frame(pm: pd.DataFrame, scen: str, cols: list[str]) -> pd.DataFrame:
    if scen == "all_cross":
        return pm
    if scen == "nocrop_cross":
        return pm[pm["topology_a"].ne("crop") & pm["topology_b"].ne("crop")]
    return spm_frame(pm, cols)


def gt_tasks(domain: str, pms: dict, arm_name, checks: list) -> list:
    """Compiti di bootstrap (zsum._boot_task) per le celle e per le differenze appaiate."""
    tasks = []
    for (cell, e), pm in pms.items():
        for scen in SCEN:
            df = scen_frame(pm, scen, ["latent_distance"])
            tasks.append(((domain, "cell", cell, e, scen), df, "latent_distance", NB,
                          base.stable_seed(SEED, arm_name(cell, e), "maxabs", TOK[scen], "latent")))
    for a, b, _ in CONTRASTS:
        for e in STEPS:
            if (a, e) not in pms or (b, e) not in pms:
                continue
            m = zsum.merge_arms(pms[(a, e)], pms[(b, e)], "latent_distance", "a", "b")
            for scen in SCEN:
                df = scen_frame(m, scen, ["latent_distance_a", "latent_distance_b"])
                tasks.append(((domain, "pair", a, b, e, scen), df, ("latent_distance_a", "latent_distance_b"), NB,
                              base.stable_seed(SEED, "e1", domain, "maxabs", scen)))
    subj = {k: sorted(set(v["subject_a"].astype(str)) | set(v["subject_b"].astype(str))) for k, v in pms.items()}
    if subj:
        first = next(iter(subj.values()))
        checks.append({"controllo": f"{domain}: stessi soggetti valutati in tutte le celle presenti",
                       "valore": str(all(s == first for s in subj.values())), "atteso": "True"})
        if not all(s == first for s in subj.values()):
            raise SystemExit(f"{domain}: soggetti diversi fra le celle")
    return tasks


# ------------------------------------------------------------------------------ riconoscimento

def recog_blocks():
    nocrop = zes.NOCROP
    return ([(a, b) for a in nocrop for b in nocrop if a != b],
            [(a, b) for i, a in enumerate(nocrop) for b in nocrop[i + 1:]])


def load_recog(domain: str, view_dir: Path, stages: dict, pms: dict | None, checks: list) -> tuple[dict, zes.Index]:
    subjects = select_subjects(view_dir, SEED)
    idx = zes.Index(subjects)
    D = {}
    for k, st in stages.items():
        staged = json.loads((st.parent / "subjects.json").read_text())["subjects"]
        if sorted(staged) != subjects:
            raise SystemExit(f"{st}: soggetti diversi da select_subjects")
        D[k] = zes.model_distances(st, idx)
        if pms is not None and k in pms:
            pm = pms[k]
            ia = np.asarray([idx.pos[x] for x in zip(pm["subject_a"].astype(str), pm["topology_a"])])
            ib = np.asarray([idx.pos[x] for x in zip(pm["subject_b"].astype(str), pm["topology_b"])])
            checks.append({"controllo": f"{domain} {LABEL[k[0]]} {STEPS[k[1]]}: distanze dagli embedding contro "
                                        f"latent_distance delle pair_metrics, max |diff|",
                           "valore": f"{float(np.abs(D[k][ia, ib] - pm['latent_distance']).max()):.2e}",
                           "atteso": "< 1e-4"})
    return D, idx


# ------------------------------------------------------------------------------------- NoW

def now_tau(o: Path, checks: list, cell: str, e: str):
    """tau per immagine del latente sui 3 metodi pre-registrati, come now_summarize.concordance."""
    import now_summarize as ns
    ns.common.OUT_ROOT = o                     # i percorsi dei csv sono letti a ogni chiamata
    items = ns.common.load_items()
    subjects = ns.common.subjects_of(items)
    s2i = {s: k for k, s in enumerate(subjects)}
    methods = list(ns.common.METHODS)
    df, metrics = ns.load_table(methods, items)
    ok = df.dropna(subset=["now_median"] + metrics).groupby("name")["method"].nunique()
    names = sorted(ok[ok == len(methods)].index)
    by_name = {it.name: it for it in items}
    dc = df[df["name"].isin(names)]
    prereg = [m for m in ns.PREREGISTERED if m in methods]
    w = {c: dc.pivot(index="name", columns="method", values=c)[prereg].loc[names] for c in ("latent_joint", "now_median")}
    tau_i = ns.kendall_rows(w["latent_joint"].to_numpy(), w["now_median"].to_numpy())
    img_s = np.asarray([s2i[by_name[n].subject] for n in names])
    pub = pd.read_csv(o / "concordance.csv")
    pub = pub[(pub["metric"] == "latent_joint") & (pub["methods"] == "+".join(prereg))].iloc[0]
    checks.append({"controllo": f"NoW {LABEL[cell]} {STEPS[e]}: tau ricalcolato contro concordance.csv della cella",
                   "valore": f"{tau_i.mean():.6f} / {pub['tau_image']:.6f}", "atteso": "uguali"})
    return names, tau_i, img_s, len(subjects)


# ------------------------------------------------------------------------------------ main

def fmt(p, lo, hi, signed=False):
    if p is None or not np.isfinite(p):
        return "-"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()
    checks: list[dict] = []
    bm = base.load_bootstrap_module()

    # checkpoint valutato = checkpoint della cella (dagli eval_key.txt)
    def ck_ok(st: Path, cell: str, e: str, what: str) -> Path | None:
        want = expected_ckpt(cell, e)
        got = eval_ckpt(st)
        if want is None or got != want:
            raise SystemExit(f"{what} {cell} e{e}: checkpoint valutato {got} != atteso {want}")
        return st

    pms_h, pms_f, emb_h, emb_f, now = {}, {}, {}, {}, {}
    for cell in CELLS:
        for e in STEPS:
            st = hifi_stage(cell, e, "topology")
            if st is not None:
                pms_h[(cell, e)] = base.read_pair_metrics(ck_ok(st, cell, e, "HIFI3D"))
            st = hifi_stage(cell, e, "embed")
            if st is not None:
                emb_h[(cell, e)] = ck_ok(st, cell, e, "HIFI3D embedding")
            st = fv_stage(cell, e)
            if st is not None:
                emb_f[(cell, e)] = ck_ok(st, cell, e, "FaceVerse")
            st = flame_stage(cell, e)
            if st is not None:
                pms_f[(cell, e)] = base.read_pair_metrics(ck_ok(st, cell, e, "FLAME"))
            o = now_dir(cell, e)
            if o is not None:
                ck = (o / "checkpoint.txt").read_text().strip().split("=", 1)[1]
                if os.path.realpath(ck) != expected_ckpt(cell, e):
                    raise SystemExit(f"NoW {cell} e{e}: checkpoint {ck} != {expected_ckpt(cell, e)}")
                now[(cell, e)] = o

    # bootstrap degli Spearman (celle e differenze), in parallelo
    tasks = gt_tasks("hifi", pms_h, lambda c, e: f"scale_{tag(c, e)}", checks)
    tasks += gt_tasks("flame", pms_f, lambda c, e: f"flame_{tag(c, e)}", checks)
    res = {}
    print(f"[e1-sum] {len(tasks)} bootstrap di Spearman su {args.workers} processi", flush=True)
    with mp.get_context("fork").Pool(args.workers) as pool:
        for key, r in pool.imap_unordered(zsum._boot_task, tasks):
            res[key] = r

    # riconoscimento
    pairs_r, pairs_v = recog_blocks()
    rec, rec_pair = {}, {}
    for domain, view, stages, pms in (("hifi", REPO / "datasets/HIFI3D/eval_view/npz", emb_h, pms_h),
                                      ("fv", REPO / "datasets/FACEVERSE_ZS/expr_view/npz", emb_f, None)):
        if not stages:
            continue
        D, idx = load_recog(domain, view, stages, pms, checks)
        counts = zes.bootstrap_counts(len(idx.subjects), NB, base.stable_seed(SEED, "expr_recognition"))
        with mp.get_context("fork").Pool(min(args.workers, 16)) as pool:
            for k, vals, n_nan in pool.imap_unordered(
                    zes._recog_task, [(k, D[k], idx, pairs_r, pairs_v, counts) for k in D]):
                rec[(domain,) + k] = vals
        for a, b, _ in CONTRASTS:
            for e in STEPS:
                if (domain, a, e) in rec and (domain, b, e) in rec:
                    rec_pair[(domain, a, b, e)] = {m: rec[(domain, a, e)][m] - rec[(domain, b, e)][m]
                                                   for m in ("rank1", "map", "auc")}

    # NoW
    tau = {}
    for k, o in now.items():
        tau[k] = now_tau(o, checks, *k)
    now_counts = zes.bootstrap_counts(20, NB, SEED)

    def now_reps(names, t, s):
        return np.concatenate([[t.mean()], [(c[s] * t).sum() / c[s].sum() for c in now_counts]])

    # ------------------------------------------------------------------- tabelle
    rows = []
    for cell in CELLS:
        for e in STEPS:
            r = {"cell": LABEL[cell], "steps": STEPS[e]}
            for dom, pms in (("hifi", pms_h), ("flame", pms_f)):
                for scen in SCEN:
                    x = res.get((dom, "cell", cell, e, scen))
                    r[f"{dom}_{scen}"], r[f"{dom}_{scen}_lo"], r[f"{dom}_{scen}_hi"] = (
                        (x["spearman"], x["ci_low"], x["ci_high"]) if x else (np.nan,) * 3)
            for dom in ("hifi", "fv"):
                v = rec.get((dom, cell, e))
                for m in ("rank1", "auc"):
                    r[f"{dom}_rec_{m}"], (r[f"{dom}_rec_{m}_lo"], r[f"{dom}_rec_{m}_hi"]) = (
                        (float(v[m][0]), zes.ci(v[m])) if v else (np.nan, (np.nan, np.nan)))
            if (cell, e) in tau:
                names, t, s, _ = tau[(cell, e)]
                reps = now_reps(names, t, s)
                r["now_tau"], (r["now_tau_lo"], r["now_tau_hi"]) = float(reps[0]), zes.ci(reps)
            else:
                r["now_tau"], r["now_tau_lo"], r["now_tau_hi"] = (np.nan,) * 3
            rows.append(r)
    matrix = pd.DataFrame(rows)

    prow = []
    for a, b, what in CONTRASTS:
        for e in STEPS:
            base_r = {"contrast": f"{LABEL[a]} - {LABEL[b]}", "reading": what, "steps": STEPS[e]}
            for dom in ("hifi", "flame"):
                for scen in SCEN:
                    x = res.get((dom, "pair", a, b, e, scen))
                    if x:
                        prow.append({**base_r, "metric": f"{dom}_{scen}", "a": x["a"], "b": x["b"], "diff": x["diff"],
                                     "ci_low": x["ci_low"], "ci_high": x["ci_high"], "p_le0": x["p_boot_le0"],
                                     "n_bootstrap": x["n_bootstrap"]})
            for dom in ("hifi", "fv"):
                d = rec_pair.get((dom, a, b, e))
                if d:
                    for m in ("rank1", "auc"):
                        lo, hi = zes.ci(d[m])
                        prow.append({**base_r, "metric": f"{dom}_rec_{m}", "a": float(rec[(dom, a, e)][m][0]),
                                     "b": float(rec[(dom, b, e)][m][0]), "diff": float(d[m][0]), "ci_low": lo,
                                     "ci_high": hi, "p_le0": float((d[m][1:] <= 0).mean()), "n_bootstrap": NB})
            if (a, e) in tau and (b, e) in tau:
                na, ta, sa, _ = tau[(a, e)]
                nb_, tb, _, _ = tau[(b, e)]
                if na != nb_:
                    raise SystemExit(f"NoW: immagini comuni diverse fra {a} e {b}")
                d = now_reps(na, ta - tb, sa)
                lo, hi = zes.ci(d)
                prow.append({**base_r, "metric": "now_tau", "a": float(ta.mean()), "b": float(tb.mean()),
                             "diff": float(d[0]), "ci_low": lo, "ci_high": hi,
                             "p_le0": float((d[1:] <= 0).mean()), "n_bootstrap": NB})
    paired = pd.DataFrame(prow)

    # controlli: le righe di C3M gia' pubblicate (curve.md) devono tornare identiche
    tc = pd.read_csv(REF / "hifi/table_cells.csv")
    for e in STEPS:
        for scen, proto in (("nocrop_cross", "mesh_pair_nocrop_cross"), ("all_cross", "mesh_pair_all_cross"),
                            ("subject_pair_mean", "subject_pair_mean")):
            x = res.get(("hifi", "cell", "c3m", e, scen))
            ref = tc[(tc["model"] == f"scale_e{e}") & (tc["gt"] == "maxabs") & (tc["protocol"] == proto)
                     & (tc["scenario"] == "clean")]
            if x and not ref.empty:
                ref = ref.iloc[0]
                d = max(abs(x["spearman"] - ref["latent_spearman"]), abs(x["ci_low"] - ref["latent_ci_low"]),
                        abs(x["ci_high"] - ref["latent_ci_high"]))
                checks.append({"controllo": f"HIFI3D C3M {STEPS[e]} {scen} contro data_scale_ood/hifi/table_cells.csv, "
                                            "max |diff| su punto e IC", "valore": f"{d:.2e}", "atteso": "0"})
    for dom, path, method in (("hifi", REF / "arcface_vs_scale_hifi3d/recognition.csv", "scale_e{e}"),
                              ("fv", REF / "fvexpr_partial/recognition.csv", "scale_e{e}@bfm")):
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        for e in STEPS:
            v = rec.get((dom, "c3m", e))
            rr = ref[(ref["block"] == "nocrop") & (ref["method"] == method.format(e=e))]
            if v is not None and not rr.empty:
                rr = rr.iloc[0]
                d = max(abs(float(v[m][0]) - rr[m]) for m in ("rank1", "map", "auc"))
                checks.append({"controllo": f"{dom} riconoscimento C3M {STEPS[e]} contro {path.relative_to(RUNS)}, "
                                            "max |diff| rank-1/mAP/AUC", "valore": f"{d:.2e}", "atteso": "0"})

    OUT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT / "matrix.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "controls.csv", index=False)
    write_summary(matrix, paired, checks)
    print(f"[e1-sum] scritto {OUT / 'summary.md'}", flush=True)


# --------------------------------------------------------------------------- markdown

def runs_info() -> list[str]:
    """Per ogni run nuovo: job, passi, blocchi, picco di memoria MISURATO dal cgroup (mem_job.log)."""
    L = ["| cella | run dir (job) | passi eseguiti | blocchi (train.log) | picco rss+shmem (GiB) | "
         "picco cgroup con page cache (GiB) | --mem |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell in CELLS[1:]:
        ptr = OUT / f"train_{cell}_s{SEED}.runs_root"
        if not ptr.exists():
            L.append(f"| {LABEL[cell]} | - | - | - | - | - | - |")
            continue
        rr = Path(ptr.read_text().strip())
        log = (rr / "train.log").read_text() if (rr / "train.log").exists() else ""
        done = re.findall(r"\[steps\] passi eseguiti: (\d+)", log)
        stop = (rr / "stop.txt").read_text() if (rr / "stop.txt").exists() else ""
        steps = (re.search(r"dopo (\d+) passi", stop).group(1) + " (arresto voluto dopo l'epoca 72)" if stop
                 else (done[0] if done else "-"))
        blk = re.search(r"\[steps\] (\d+) blocchi da \[([0-9]+)", log)
        mem = (rr / "mem_job.log").read_text() if (rr / "mem_job.log").exists() else ""
        nr = [float(a) + float(b) for a, b in re.findall(r"rss=([0-9.]+) shmem=([0-9.]+)", mem)]
        pk = [float(x) for x in re.findall(r"peak=([0-9.]+)", mem)]
        cj = json.loads((rr / "e1_cell.json").read_text()) if (rr / "e1_cell.json").exists() else {}
        L.append(f"| {LABEL[cell]} | `{rr.relative_to(REPO)}` ({cj.get('job', '-')}) | {steps} | "
                 f"{blk.group(1) + ' da ' + blk.group(2) + ' soggetti' if blk else '-'} | "
                 f"{max(nr) if nr else float('nan'):.1f} | {max(pk) if pk else float('nan'):.1f} | "
                 f"{cj.get('mem_request', '-')} |")
    return L


def write_summary(matrix: pd.DataFrame, paired: pd.DataFrame, checks: list) -> None:
    proto = (OUT / "protocol.md").read_text()
    sha = hashlib.sha256(proto.encode()).hexdigest()
    rec_sha = (OUT / "protocol.sha256").read_text().split()[0] if (OUT / "protocol.sha256").exists() else ""

    def pr(c, e, metric):
        if paired.empty:
            return None
        r = paired[(paired["contrast"] == c) & (paired["steps"] == e) & (paired["metric"] == metric)]
        return None if r.empty else r.iloc[0]

    L = [proto.rstrip(), "\n---\n", "# Risultati\n",
         f"Generato da `aau/evidence/e1_factorial/e1_summarize.py`. Protocollo qui sopra invariato: sha256 "
         f"{sha[:16]}.. {'= quello registrato prima dei numeri (protocol.sha256)' if sha == rec_sha else 'DIVERSO da protocol.sha256: CONTROLLARE'}.\n",
         "## Regola primaria: varieta' (HIFI3D, `nocrop_cross`)\n",
         "| passi | C3F - C2F [IC 95%] (P<=0) | C3M - C2M [IC 95%] (P<=0) | varieta' sostenuta? |",
         "| --- | --- | --- | --- |"]
    for e in (21096, 10548):
        cells, verdict = [], []
        for c in ("C3F - C2F", "C3M - C2M"):
            r = pr(c, e, "hifi_nocrop_cross")
            if r is None:
                cells.append("-")
                verdict.append(None)
            else:
                cells.append(f"{fmt(r['diff'], r['ci_low'], r['ci_high'], True)} ({r['p_le0']:.3f})")
                verdict.append(bool(r["diff"] > 0 and r["ci_low"] > 0))
        if None in verdict:
            v = "non valutabile (manca una cella)"
        elif all(verdict):
            v = "SI: entrambe > 0 con IC che esclude lo 0"
        else:
            v = "NO: " + ", ".join(f"{c} {'passa' if ok else 'non passa'}" for c, ok in zip(("C3F - C2F", "C3M - C2M"), verdict))
        L.append(f"| {e} | {cells[0]} | {cells[1]} | {v} |")

    cols = [("hifi_nocrop_cross", "HIFI3D nocrop"), ("hifi_all_cross", "HIFI3D all_cross"),
            ("hifi_subject_pair_mean", "HIFI3D subj-pair-mean"), ("hifi_rec_rank1", "HIFI3D rank-1"),
            ("hifi_rec_auc", "HIFI3D AUC"), ("fv_rec_rank1", "FaceVerse espr. rank-1"), ("fv_rec_auc", "FaceVerse espr. AUC"),
            ("now_tau", "NoW tau"), ("flame_nocrop_cross", "FLAME nocrop"), ("flame_all_cross", "FLAME all_cross"),
            ("flame_subject_pair_mean", "FLAME subj-pair-mean")]
    for e in (10548, 21096):
        L += [f"\n## Matrice celle x domini di test, {e} passi\n",
              "Punto [IC 95% bootstrap per soggetto, 1000 repliche]. Spearman con la GT `maxabs` (HIFI3D, FLAME), "
              "riconoscimento sulle 5 topologie senza crop (HIFI3D, FaceVerse con espressioni in convenzione BFM), "
              "tau di Kendall per immagine sui 3 metodi pre-registrati (NoW).\n",
              "| cella | " + " | ".join(n for _, n in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
        for r in matrix[matrix["steps"] == e].itertuples():
            d = r._asdict()
            L.append(f"| {r.cell} | " + " | ".join(fmt(d[k], d[k + '_lo'], d[k + '_hi']) for k, _ in cols) + " |")
    L += ["\n## Differenze appaiate\n",
          "a - b sulle stesse righe e sulle stesse repliche (un seme per dominio e scenario; riconoscimento e NoW: le "
          "repliche dei loro summarizer). Cella: differenza [IC 95%] (P(boot <= 0)).\n"]
    for e in (21096, 10548):
        L += [f"\n### {e} passi\n", "| contrasto | lettura | " + " | ".join(n for _, n in cols) + " |",
              "|" + " --- |" * (len(cols) + 2)]
        for a, b, what in CONTRASTS:
            c = f"{LABEL[a]} - {LABEL[b]}"
            cells = []
            for k, _ in cols:
                r = pr(c, e, k)
                cells.append("-" if r is None else f"{fmt(r['diff'], r['ci_low'], r['ci_high'], True)} ({r['p_le0']:.2f})")
            L.append(f"| {c} | {what} | " + " | ".join(cells) + " |")
    design = json.loads((OUT / "design.json").read_text()) if (OUT / "design.json").exists() else None
    if design:
        L += ["\n## Identita' viste per cella (design.json, calcolato prima dei numeri)\n",
              "| cella | passi | viste per dominio | viste totali | esposizioni mediana |", "| --- | --- | --- | --- | --- |"]
        for cell in CELLS:
            for ep, a in design["cells"][cell]["at"].items():
                L.append(f"| {LABEL[cell]} | {a['steps']} | {a['seen']} | {a['seen_total']} | {a['exposures_median']:.0f} |")
    L += ["\n## Run\n", *runs_info(),
          "\n## Controlli\n", "| controllo | valore | atteso |", "| --- | --- | --- |",
          *[f"| {c['controllo']} | {c['valore']} | {c['atteso']} |" for c in checks]]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
