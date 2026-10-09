#!/usr/bin/env python3
"""E1: matrice celle x domini di test, differenze appaiate e regole di protocol.md + protocol_amendment.md.

    aau/run.sh aau/evidence/e1_factorial/e1_summarize.py        (e1_summarize.sbatch, nessuna GPU)

Non ricalcola distanze ne' embedding: legge le uscite di e1_eval_body.sh (tutte le celle, C3M su L40S compresa
come c3ml, valutate sulle A100). Misure e funzioni dei summarizer esistenti, importate e non riscritte:
  - Spearman con la GT (HIFI3D, dev FaceScape, FLAME): ``finite_spearman`` e il ricampionamento per soggetto
    di ``weighted_bootstrap_spearman`` / ``zs_summarize.paired_bootstrap`` (soggetti con reinserimento, peso di
    una coppia = prodotto dei conteggi, righe ripetute). Qui UNA sola passata per (dominio, GT, passi, scenario)
    calcola lo Spearman di TUTTE le celle sulle stesse repliche: ogni differenza, e il massimo per replica della
    regola sulla varieta', sta sulle stesse 1000 repliche. GT: ``maxabs`` (gt_distance delle pair_metrics) e
    unificata (``datasets/UNIFIED_GT/gt/<set>_unified.npz``, per nome di soggetto, ``zs_summarize.with_gt``);
  - riconoscimento (HIFI3D, FaceVerse con espressioni): ``zs_expr_summarize`` con ``bootstrap_counts`` seme
    ``stable_seed(1234, "expr_recognition")``, blocco senza crop;
  - NoW: tau di Kendall per immagine sui 3 metodi pre-registrati (``now_summarize``), repliche
    ``bootstrap_counts(20, 1000, 1234)``;
  - baseline HIFI3D (ICP+Chamfer, NICP P2Tri) per la domanda su C3F-UGT: ``zs_expr_summarize.facebench_distances``
    sulle matrici di ``aau/runs/ws_hifi3d`` (sola lettura), lette sulle stesse righe del modello.
Celle mancanti: "-" e regole "non valutabile". Scrive in aau/runs/evidence/e1/: summary.md (= protocol.md +
protocol_amendment.md invariati + risultati), matrix.csv, paired.csv, controls.csv.
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
MIN_EFFECT = 0.05
CELLS = ("c3mv", "c2m", "c2f", "c3f", "c2fgnm", "c3fugt", "c3fugtraw", "c2fs2", "c3fs2", "c2f40", "c3f40", "g1", "c3ml")
LABEL = {"c3mv": "C3M", "c2m": "C2M", "c2f": "C2F", "c3f": "C3F", "c2fgnm": "C2F-GNM", "c3fugt": "C3F-UGT", "c3fugtraw": "C3F-UGT non tarata",
         "c2fs2": "C2F s2", "c3fs2": "C3F s2", "c2f40": "C2F40", "c3f40": "C3F40", "g1": "G1",
         "c3ml": "C3M L40S (rif.)"}
STEPS = {"036": 10548, "072": 21096}
# contrasti descrittivi (a, b, lettura); la regola sulla varieta' e' a parte (variety_rule)
CONTRASTS = (("c3f", "c2f", "varieta' (protocollo originale)"), ("c3f", "c2fgnm", "varieta' contro solo GNM"),
             ("c3mv", "c2m", "varieta', molte identita'"), ("c3fs2", "c2fs2", "varieta', secondo seme"),
             ("c2m", "c2f", "quantita' ~2.5x, 2 domini"), ("c3mv", "c3f", "quantita' ~2.5x, 3 domini"),
             ("c2m", "c2f40", "quantita' ~10x, 2 domini"), ("c3mv", "c3f40", "quantita' ~10x, 3 domini"),
             ("g1", "c3mv", "solo GNM contro C3M"), ("g1", "c3f", "solo GNM contro C3F"),
             ("c3fugt", "c3f", "GT unificata (tarata) contro maxabs in training"),
             ("c3fugt", "c3fugtraw", "scala della GT unificata: tarata contro non tarata"),
             ("c3mv", "c3ml", "rumore: stesso seme, V100 contro L40S"),
             ("c2f", "c2fs2", "rumore: seme 1234 contro 2345"), ("c3f", "c3fs2", "rumore: seme 1234 contro 2345"))
SCEN = ("nocrop_cross", "all_cross", "subject_pair_mean")
SCALE_RUN = RUNS / "data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
REF = RUNS / "data_scale_ood"
# GT unificate dei set di valutazione (v3_work/unified_gt/make_eval_gt.py); per HIFI3D coincide con quella di E8
# (gt/hifi3d_unified.npz) a 3e-8 dopo la normalizzazione. FLAME non ce l'ha: colonna "-".
UGT = REPO / "datasets/UNIFIED_GT/eval"
UGT_FILE = {"hifi": UGT / "hifi3d_gt_matrix.npz", "devfs": UGT / "facescape_gt_matrix.npz", "flame": UGT / "flame_gt_matrix.npz"}
HIFI_BL = RUNS / "ws_hifi3d/data_328f2bfc1a/baselines"
BASELINES = ("rigid_icp_chamfer", "nicp_p2tri")
VIEW = {"hifi": REPO / "datasets/HIFI3D/eval_view/npz", "fv": REPO / "datasets/FACEVERSE_ZS/expr_view/npz",
        "devfs": REPO / "datasets/DEV_FACESCAPE/eval_view/npz"}


def tag(cell: str, e: str) -> str:
    return f"e1{cell}e{e}"


def one_dir(root: Path, glob: str) -> Path | None:
    hits = sorted(root.glob(glob)) if root.is_dir() else []
    if len(hits) > 1:
        raise SystemExit(f"{root}/{glob}: piu' di una cartella {hits}")
    return hits[0] if hits else None


def expected_ckpt(cell: str, e: str) -> str | None:
    if cell == "c3ml":
        rr = SCALE_RUN
    else:
        ptr = OUT / f"train_{cell}.runs_root"
        if not ptr.exists():
            return None
        rr = Path(ptr.read_text().strip())
    hits = sorted(rr.glob(f"mixed_*/checkpoints/epoch{e}.pth"))
    return os.path.realpath(hits[0]) if hits else None


def eval_ckpt(stage: Path) -> str:
    key = dict(l.split("=", 1) for l in (stage.parent / "eval_key.txt").read_text().splitlines() if "=" in l)
    return os.path.realpath(key["ckpt"])


def ck_ok(st: Path, cell: str, e: str, what: str) -> Path:
    want, got = expected_ckpt(cell, e), eval_ckpt(st)
    if want is None or got != want:
        raise SystemExit(f"{what} {cell} e{e}: checkpoint valutato {got} != atteso {want}")
    return st


# ------------------------------------------------------------------------------- sorgenti

def zs_stage(domain_dir: str, cell: str, e: str, suffix: str, need_embed: bool) -> Path | None:
    d = one_dir(OUT / domain_dir, "data_*")
    st = None if d is None else d / f"scale_{tag(cell, e)}{suffix}" / zsum.STAGE
    if st is None or not (st / ".done").exists():
        return None
    if need_embed and not (st / "embeddings.npz").exists():
        return None
    return st


def flame_stage(cell: str, e: str) -> Path | None:
    d = one_dir(OUT / "flame" / tag(cell, e), "*/joint")
    st = None if d is None else d / "flame_zeroshot"
    return st if st is not None and (st / ".done").exists() else None


def now_dir(cell: str, e: str) -> Path | None:
    d = OUT / "now" / tag(cell, e)
    return d if (d / "concordance.csv").exists() else None


# ---------------------------------------------------------------------- Spearman, tutte le celle insieme

def joint_boot(task):
    """Spearman di ogni colonna e sue repliche, con UNO stesso ricampionamento per soggetto (quello di
    weighted_bootstrap_spearman). Colonne con valori non finiti: righe mascherate per quella colonna sola."""
    key, df, cols, n, seed = task
    bm = base.load_bootstrap_module()
    df = df[df["subject_a"].astype(str) != df["subject_b"].astype(str)]
    df = df[np.isfinite(df["gt_distance"].to_numpy(np.float64))]
    subjects = np.array(sorted(set(df["subject_a"].astype(str)) | set(df["subject_b"].astype(str))))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = df["subject_a"].astype(str).map(s2i).to_numpy(np.int32)
    sb = df["subject_b"].astype(str).map(s2i).to_numpy(np.int32)
    gt = df["gt_distance"].to_numpy(np.float64)
    vals = {c: df[c].to_numpy(np.float64) for c in cols}
    point = {c: bm.finite_spearman(gt, v) for c, v in vals.items()}
    rng = np.random.default_rng(seed)
    reps = {c: [] for c in cols}
    for _ in range(n):
        counts = np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects))
        w = counts[sa].astype(np.int64) * counts[sb].astype(np.int64)
        keep = w > 0
        x = np.repeat(gt[keep], w[keep])
        for c, v in vals.items():
            reps[c].append(bm.finite_spearman(x, np.repeat(v[keep], w[keep])))
    return key, point, {c: np.asarray(r) for c, r in reps.items()}, len(subjects), len(df)


def scen_frame(m: pd.DataFrame, scen: str, cols: list[str]) -> pd.DataFrame:
    if scen == "all_cross":
        return m
    if scen == "nocrop_cross":
        return m[m["topology_a"].ne("crop") & m["topology_b"].ne("crop")]
    return m.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance"] + cols].mean()


def merged_frame(pms: dict, e: str, extra: dict | None = None) -> tuple[pd.DataFrame | None, list[str]]:
    """Le pair_metrics delle celle presenti al passo e, allineate per riga (stessi soggetti e topologie)."""
    keys = zsum.PAIR_KEYS
    cells = [c for c in CELLS if (c, e) in pms]
    if not cells:
        return None, []
    m = pms[(cells[0], e)][keys + ["gt_distance"]].copy()
    for c in cells:
        pm = pms[(c, e)]
        j = m[keys + ["gt_distance"]].merge(pm[keys + ["gt_distance", "latent_distance"]], on=keys, how="left",
                                            validate="one_to_one", suffixes=("", "_c"))
        if len(pm) != len(m) or j["latent_distance"].isna().any():
            raise SystemExit(f"{c} e{e}: righe delle pair_metrics diverse da quelle di {cells[0]}")
        if not np.allclose(j["gt_distance"], j["gt_distance_c"]):
            raise SystemExit(f"{c} e{e}: GT diversa sulle stesse righe")
        m[f"lat_{c}"] = j["latent_distance"].to_numpy()
    cols = [f"lat_{c}" for c in cells]
    for name, v in (extra or {}).items():
        m[name] = v(m)
        cols.append(name)
    return m, cols


def gt_tasks(domain: str, pms: dict, checks: list, baselines=None) -> list:
    """Compiti di joint_boot per (dominio, GT, passi, scenario)."""
    tasks = []
    gts = {"maxabs": None}
    if UGT_FILE[domain].exists():
        gts["unified"] = zsum.load_gt(UGT_FILE[domain])
    for e in STEPS:
        m, cols = merged_frame(pms, e, baselines)
        if m is None:
            continue
        subj = sorted(set(m["subject_a"].astype(str)) | set(m["subject_b"].astype(str)))
        checks.append({"controllo": f"{domain} {STEPS[e]}: celle presenti (stesse righe, stessi soggetti)",
                       "valore": f"{[c[4:] for c in cols if c.startswith('lat_')]}; {len(subj)} soggetti, {len(m)} righe",
                       "atteso": "-"})
        if baselines and "unified" in gts:
            for b in BASELINES:
                # modelli sulle sole righe dove la baseline e' finita (contrasto della domanda su C3F-UGT)
                for c in [c for c in ("lat_c3fugt", "lat_c3f") if c in cols]:
                    m[f"{c}@{b}"] = np.where(np.isfinite(m[b]), m[c], np.nan)
            cols = cols + [f"{c}@{b}" for b in BASELINES for c in ("lat_c3fugt", "lat_c3f") if c in cols]
        for gname, gt in gts.items():
            mg = m if gt is None else zsum.with_gt(m, gt)
            for scen in SCEN:
                df = scen_frame(mg, scen, cols)
                tasks.append(((domain, gname, e, scen), df, cols, NB,
                              base.stable_seed(SEED, "e1v2", domain, gname, scen)))
    return tasks


# ------------------------------------------------------------------------------------- NoW

def now_tau(o: Path, checks: list, cell: str, e: str):
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
    return names, tau_i, img_s


# ------------------------------------------------------------------------------------ main

def fmt(p, lo, hi, signed=False):
    if p is None or not np.isfinite(p):
        return "-"
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5])) if len(x) else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = ap.parse_args()
    checks: list[dict] = []

    pms = {"hifi": {}, "devfs": {}, "flame": {}}
    emb = {"hifi": {}, "fv": {}}
    now = {}
    for cell in CELLS:
        for e in STEPS:
            st = zs_stage("hifi_runs", cell, e, "_topology", False)
            if st is not None:
                pms["hifi"][(cell, e)] = base.read_pair_metrics(ck_ok(st, cell, e, "HIFI3D"))
                if (st / "embeddings.npz").exists():
                    emb["hifi"][(cell, e)] = st
            st = zs_stage("devfs", cell, e, "_topology", False)
            if st is not None:
                pms["devfs"][(cell, e)] = base.read_pair_metrics(ck_ok(st, cell, e, "dev FaceScape"))
            st = zs_stage("fv_expr", cell, e, "_flip_embed", True)
            if st is not None:
                emb["fv"][(cell, e)] = ck_ok(st, cell, e, "FaceVerse")
            st = flame_stage(cell, e)
            if st is not None:
                pms["flame"][(cell, e)] = base.read_pair_metrics(ck_ok(st, cell, e, "FLAME"))
            o = now_dir(cell, e)
            if o is not None:
                ck = (o / "checkpoint.txt").read_text().strip().split("=", 1)[1]
                if os.path.realpath(ck) != expected_ckpt(cell, e):
                    raise SystemExit(f"NoW {cell} e{e}: checkpoint {ck} != {expected_ckpt(cell, e)}")
                now[(cell, e)] = o

    # baseline HIFI3D sulle righe del modello (per la domanda su C3F-UGT e come riferimento)
    hifi_idx = zes.Index(select_subjects(VIEW["hifi"], SEED))
    bl_D = {b: zes.facebench_distances(HIFI_BL, b, hifi_idx) for b in BASELINES} if pms["hifi"] else {}

    def bl_col(b):
        def f(m):
            ia = np.asarray([hifi_idx.pos[k] for k in zip(m["subject_a"].astype(str), m["topology_a"])])
            ib = np.asarray([hifi_idx.pos[k] for k in zip(m["subject_b"].astype(str), m["topology_b"])])
            return bl_D[b][ia, ib]
        return f

    tasks = gt_tasks("hifi", pms["hifi"], checks, {b: bl_col(b) for b in BASELINES} if bl_D else None)
    tasks += gt_tasks("devfs", pms["devfs"], checks)
    tasks += gt_tasks("flame", pms["flame"], checks)
    res = {}
    print(f"[e1-sum] {len(tasks)} passate di bootstrap (tutte le celle per passata) su {args.workers} processi", flush=True)
    with mp.get_context("fork").Pool(max(1, min(args.workers, len(tasks)))) as pool:
        for key, point, reps, n_s, n_r in pool.imap_unordered(joint_boot, tasks):
            res[key] = (point, reps, n_s, n_r)

    def sp(domain, gname, e, scen, col):
        r = res.get((domain, gname, e, scen))
        if r is None or col not in r[0]:
            return None
        return r[0][col], r[1][col]

    # riconoscimento
    pairs_r = [(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b]
    pairs_v = [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]
    rec = {}
    for domain, stages in emb.items():
        if not stages:
            continue
        subjects = select_subjects(VIEW[domain], SEED)
        idx = zes.Index(subjects)
        D = {}
        for k, st in stages.items():
            staged = json.loads((st.parent / "subjects.json").read_text())["subjects"]
            if sorted(staged) != subjects:
                raise SystemExit(f"{st}: soggetti diversi da select_subjects")
            D[k] = zes.model_distances(st, idx)
            if domain == "hifi" and k in pms["hifi"]:
                pm = pms["hifi"][k]
                ia = np.asarray([idx.pos[x] for x in zip(pm["subject_a"].astype(str), pm["topology_a"])])
                ib = np.asarray([idx.pos[x] for x in zip(pm["subject_b"].astype(str), pm["topology_b"])])
                checks.append({"controllo": f"hifi {LABEL[k[0]]} {STEPS[k[1]]}: embedding contro latent_distance, max |diff|",
                               "valore": f"{float(np.abs(D[k][ia, ib] - pm['latent_distance']).max()):.2e}",
                               "atteso": "< 1e-4"})
        counts = zes.bootstrap_counts(len(subjects), NB, base.stable_seed(SEED, "expr_recognition"))
        with mp.get_context("fork").Pool(max(1, min(args.workers, 16, len(D)))) as pool:
            for k, vals, _ in pool.imap_unordered(zes._recog_task, [(k, D[k], idx, pairs_r, pairs_v, counts) for k in D]):
                rec[(domain,) + k] = vals

    # NoW
    tau = {k: now_tau(o, checks, *k) for k, o in now.items()}
    now_counts = zes.bootstrap_counts(20, NB, SEED)

    def now_reps(t, s):
        return np.concatenate([[t.mean()], [(c[s] * t).sum() / c[s].sum() for c in now_counts]])

    def metric(cell, e, key):
        """(punto, repliche) di una misura: 'sp|dominio|gt|scenario', 'rec|dominio|rank1', 'now'."""
        p = key.split("|")
        if p[0] == "sp":
            return sp(p[1], p[2], e, p[3], f"lat_{cell}")
        if p[0] == "rec":
            v = rec.get((p[1], cell, e))
            return None if v is None else (float(v[p[2]][0]), v[p[2]][1:])
        if p[0] == "now" and (cell, e) in tau:
            _, t, s = tau[(cell, e)]
            r = now_reps(t, s)
            return float(r[0]), r[1:]
        return None

    COLS = [("sp|hifi|maxabs|nocrop_cross", "HIFI3D nocrop"), ("sp|hifi|unified|nocrop_cross", "HIFI3D nocrop, GT unif."),
            ("sp|hifi|maxabs|all_cross", "HIFI3D all_cross"), ("sp|hifi|maxabs|subject_pair_mean", "HIFI3D subj-pair-mean"),
            ("rec|hifi|rank1", "HIFI3D rank-1"), ("rec|hifi|auc", "HIFI3D AUC"),
            ("sp|devfs|maxabs|nocrop_cross", "dev FaceScape nocrop"), ("sp|devfs|unified|nocrop_cross", "dev FaceScape, GT unif."),
            ("rec|fv|rank1", "FaceVerse espr. rank-1"), ("rec|fv|auc", "FaceVerse espr. AUC"), ("now", "NoW tau"),
            ("sp|flame|maxabs|nocrop_cross", "FLAME nocrop"), ("sp|flame|unified|nocrop_cross", "FLAME nocrop, GT unif.")]

    rows = []
    for cell in CELLS:
        for e in STEPS:
            r = {"cell": LABEL[cell], "steps": STEPS[e]}
            for k, _ in COLS:
                x = metric(cell, e, k)
                r[k], (r[k + "_lo"], r[k + "_hi"]) = (x[0], ci(x[1])) if x else (np.nan, (np.nan, np.nan))
            rows.append(r)
    matrix = pd.DataFrame(rows)

    def diff(a, b, e, k):
        xa, xb = metric(a, e, k), metric(b, e, k)
        if xa is None or xb is None:
            return None
        d = xa[1] - xb[1]
        lo, hi = ci(d)
        return {"a": xa[0], "b": xb[0], "diff": xa[0] - xb[0], "ci_low": lo, "ci_high": hi,
                "p_le0": float(np.mean(d[np.isfinite(d)] <= 0))}

    prow = []
    for a, b, what in CONTRASTS:
        for e in STEPS:
            for k, _ in COLS:
                x = diff(a, b, e, k)
                if x:
                    prow.append({"contrast": f"{LABEL[a]} - {LABEL[b]}", "reading": what, "steps": STEPS[e], "metric": k, **x})
    paired = pd.DataFrame(prow)

    # ------------------------------------------------- regola sulla varieta' (emendamento, sezione 2)
    def delta_max(e, k):
        """C3F - max(C2F, C2F-GNM), il massimo per replica."""
        x3, x2, xg = (metric(c, e, k) for c in ("c3f", "c2f", "c2fgnm"))
        if None in (x3, x2, xg):
            return None
        d = x3[1] - np.maximum(x2[1], xg[1])
        lo, hi = ci(d)
        return {"diff": x3[0] - max(x2[0], xg[0]), "ci_low": lo, "ci_high": hi, "p_le0": float(np.mean(d <= 0)),
                "c3f": x3[0], "c2f": x2[0], "c2fgnm": xg[0]}

    rule = []
    for e in ("072", "036"):
        hm = delta_max(e, "sp|hifi|maxabs|nocrop_cross")
        fs = delta_max(e, "sp|devfs|maxabs|nocrop_cross")
        fv = delta_max(e, "rec|fv|rank1")
        s2 = diff("c3fs2", "c2fs2", e, "sp|hifi|maxabs|nocrop_cross")
        noise_parts = {n: diff(a, b, e, "sp|hifi|maxabs|nocrop_cross") for n, (a, b) in
                       {"C3M V100 - L40S": ("c3mv", "c3ml"), "C2F s1 - s2": ("c2f", "c2fs2"),
                        "C3F s1 - s2": ("c3f", "c3fs2")}.items()}
        noise_ok = all(v is not None for v in noise_parts.values())
        noise = max(abs(v["diff"]) for v in noise_parts.values()) if noise_ok else np.nan
        cond = {}
        if hm is not None:
            cond["(a) Delta >= +0.05 e IC > 0"] = hm["diff"] >= MIN_EFFECT and hm["ci_low"] > 0
        cond["(b) dev FaceScape stessa direzione"] = None if fs is None else fs["diff"] > 0
        cond["(c) FaceVerse rank-1 non inferiore (IC > -0.05)"] = None if fv is None else fv["ci_low"] > -MIN_EFFECT
        cond["(d) Delta oltre il pavimento del rumore"] = None if (hm is None or not noise_ok) else hm["diff"] > noise
        cond["(e) secondo seme C3F s2 - C2F s2 > 0"] = None if s2 is None else s2["diff"] > 0
        if hm is None:
            verdict = "NON VALUTABILE (mancano C3F, C2F o C2F-GNM su HIFI3D)"
        elif all(v is True for v in cond.values()):
            verdict = "SOSTENUTA"
        elif hm["diff"] <= 0 and hm["ci_high"] < MIN_EFFECT and fs is not None and fs["diff"] <= 0:
            verdict = "SMENTITA"
        else:
            miss = [k for k, v in cond.items() if v is not True]
            verdict = "NON CONCLUDENTE: " + "; ".join(f"{k} {'manca' if cond[k] is None else 'non vale'}" for k in miss)
        rule.append({"steps": STEPS[e], "hifi": hm, "devfs": fs, "fv": fv, "s2": s2, "noise": noise,
                     "noise_parts": noise_parts, "cond": cond, "verdict": verdict})

    # ------------------------------------------------- domanda su C3F-UGT (emendamento, sezione 4)
    ugt = []
    for e in ("072", "036"):
        for b in BASELINES:
            r = res.get(("hifi", "unified", e, "nocrop_cross"))
            col = f"lat_c3fugt@{b}"
            if r is None or col not in r[0]:
                ugt.append({"steps": STEPS[e], "baseline": b, "x": None})
                continue
            d = r[1][col] - r[1][b]
            lo, hi = ci(d)
            ugt.append({"steps": STEPS[e], "baseline": b, "x": {
                "a": r[0][col], "b": r[0][b], "diff": r[0][col] - r[0][b], "ci_low": lo, "ci_high": hi,
                "p_le0": float(np.mean(d[np.isfinite(d)] <= 0))}})

    # ------------------------------------------------- controlli contro i numeri gia' pubblicati
    tc = pd.read_csv(REF / "hifi/table_cells.csv")
    for e in STEPS:
        for scen, proto in (("nocrop_cross", "mesh_pair_nocrop_cross"), ("all_cross", "mesh_pair_all_cross"),
                            ("subject_pair_mean", "subject_pair_mean")):
            x = sp("hifi", "maxabs", e, scen, "lat_c3ml")
            ref = tc[(tc["model"] == f"scale_e{e}") & (tc["gt"] == "maxabs") & (tc["protocol"] == proto)
                     & (tc["scenario"] == "clean")]
            if x and not ref.empty:
                pt = ref.iloc[0]["latent_point_check"] if np.isfinite(ref.iloc[0]["latent_point_check"]) else ref.iloc[0]["latent_spearman"]
                checks.append({"controllo": f"HIFI3D C3M L40S {STEPS[e]} {scen}: rivalutato su A100 contro curve.md (L40S), |diff| del punto",
                               "valore": f"{abs(x[0] - pt):.2e}", "atteso": "~0 (hardware di valutazione)"})
    ms = REF.parent / "evidence/e8/methods_spearman.csv"
    if ms.exists():
        m8 = pd.read_csv(ms)
        for e in STEPS:
            for col, meth in [(b, f"fb_{b}") for b in BASELINES] + [("lat_c3ml", f"scale_e{e}")]:
                x = sp("hifi", "unified", e, "nocrop_cross", col)
                rr = m8[(m8["domain"] == "hifi3d") & (m8["group"] == "nocrop_cross") & (m8["gt"] == "unified")
                        & (m8["method"] == meth)]
                if x and not rr.empty:
                    checks.append({"controllo": f"HIFI3D GT unificata {meth} ({STEPS[e]} passi) contro e8/methods_spearman.csv, |diff| del punto",
                                   "valore": f"{abs(x[0] - rr.iloc[0]['point']):.2e}", "atteso": "~0"})
    for dom, path, meth in (("hifi", REF / "arcface_vs_scale_hifi3d/recognition.csv", "scale_e{e}"),
                            ("fv", REF / "fvexpr_partial/recognition.csv", "scale_e{e}@bfm")):
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        for e in STEPS:
            v = rec.get((dom, "c3ml", e))
            rr = ref[(ref["block"] == "nocrop") & (ref["method"] == meth.format(e=e))]
            if v is not None and not rr.empty:
                d = max(abs(float(v[m][0]) - rr.iloc[0][m]) for m in ("rank1", "map", "auc"))
                checks.append({"controllo": f"{dom} riconoscimento C3M L40S {STEPS[e]}: A100 contro {path.relative_to(RUNS)}, max |diff|",
                               "valore": f"{d:.2e}", "atteso": "~0"})
    for e in STEPS:
        o = RUNS / f"now_eval_scale_e{e}"
        if ("c3ml", e) in tau and (o / "concordance.csv").exists():
            pub = pd.read_csv(o / "concordance.csv")
            pub = pub[(pub["metric"] == "latent_joint") & (pub["methods"] == "3ddfa_v2+synergynet+prnet")].iloc[0]
            checks.append({"controllo": f"NoW C3M L40S {STEPS[e]}: A100 contro now_eval_scale_e{e}",
                           "valore": f"{abs(tau[('c3ml', e)][1].mean() - pub['tau_image']):.2e}", "atteso": "~0"})

    OUT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT / "matrix.csv", index=False)
    paired.to_csv(OUT / "paired.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "controls.csv", index=False)
    (OUT / "rule.json").write_text(json.dumps({"variety": rule, "ugt": ugt}, indent=1, default=str) + "\n")
    write_summary(matrix, paired, checks, rule, ugt, COLS)
    print(f"[e1-sum] scritto {OUT / 'summary.md'}", flush=True)


# --------------------------------------------------------------------------- markdown

def runs_info() -> list[str]:
    """Per ogni run: job, GPU, passi, blocchi, picco di memoria MISURATO dal cgroup (mem_job.log)."""
    L = ["| cella | run dir (job, GPU) | passi eseguiti | blocchi (train.log) | picco rss+shmem (GiB) | "
         "picco cgroup con page cache (GiB) | --mem |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell in CELLS[:-1]:
        ptr = OUT / f"train_{cell}.runs_root"
        if not ptr.exists():
            L.append(f"| {LABEL[cell]} | - | - | - | - | - | - |")
            continue
        rr = Path(ptr.read_text().strip())
        log = (rr / "train.log").read_text() if (rr / "train.log").exists() else ""
        done = re.findall(r"\[steps\] passi eseguiti: (\d+)", log)
        stop = (rr / "stop.txt").read_text() if (rr / "stop.txt").exists() else ""
        steps = (re.search(r"dopo (\d+) passi", stop).group(1) + " (arresto voluto)" if stop
                 else (done[0] if done else "-"))
        blk = re.search(r"\[steps\] (\d+) blocchi da \[([0-9]+)", log)
        mem = (rr / "mem_job.log").read_text() if (rr / "mem_job.log").exists() else ""
        nr = [float(a) + float(b) for a, b in re.findall(r"rss=([0-9.]+) shmem=([0-9.]+)", mem)]
        pk = [float(x) for x in re.findall(r"peak=([0-9.]+)", mem)]
        cj = json.loads((rr / "e1_cell.json").read_text()) if (rr / "e1_cell.json").exists() else {}
        L.append(f"| {LABEL[cell]} | `{rr.relative_to(REPO)}` ({cj.get('job', '-')}, {cj.get('gpu', '-').split(',')[0]}) | "
                 f"{steps} | {blk.group(1) + ' da ' + blk.group(2) + ' soggetti' if blk else '-'} | "
                 f"{max(nr) if nr else float('nan'):.1f} | {max(pk) if pk else float('nan'):.1f} | {cj.get('mem_request', '-')} |")
    return L


def write_summary(matrix, paired, checks, rule, ugt, COLS) -> None:
    parts = []
    for name, shaf in (("protocol.md", "protocol.sha256"), ("protocol_amendment.md", "protocol_amendment.sha256"),
                       ("nota_tecnica_gradvec.md", "nota_tecnica_gradvec.sha256"),
                       ("protocol_amendment_2.md", "protocol_amendment_2.sha256")):
        if not (OUT / name).exists():
            continue
        txt = (OUT / name).read_text()
        sha = hashlib.sha256(txt.encode()).hexdigest()
        rec = (OUT / shaf).read_text().split()[0] if (OUT / shaf).exists() else ""
        parts.append((name, txt, sha == rec))
    L = []
    for name, txt, ok in parts:
        L += [txt.rstrip(), "\n---\n"]
    L += ["# Risultati\n",
          "Generato da `aau/evidence/e1_factorial/e1_summarize.py`. Testi qui sopra invariati rispetto agli sha256 registrati "
          "prima dei numeri: " + ", ".join(f"{n} {'SI' if ok else 'NO, CONTROLLARE'}" for n, _, ok in parts) + ".\n",
          "## Regola sulla varieta' (emendamento, sezione 2): C3F - max(C2F, C2F-GNM), HIFI3D `nocrop_cross`, GT maxabs\n",
          "| passi | C3F / C2F / C2F-GNM | Delta [IC 95%] (P<=0) | dev FaceScape Delta | FaceVerse rank-1 Delta | "
          "secondo seme C3F s2 - C2F s2 | pavimento del rumore | esito |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rule:
        h = r["hifi"]
        f = lambda x: "-" if x is None else f"{fmt(x['diff'], x['ci_low'], x['ci_high'], True)} ({x['p_le0']:.3f})"  # noqa: E731
        L.append(f"| {r['steps']} | " + ("-" if h is None else f"{h['c3f']:.3f} / {h['c2f']:.3f} / {h['c2fgnm']:.3f}")
                 + f" | {f(h)} | {f(r['devfs'])} | {f(r['fv'])} | {f(r['s2'])} | "
                 + ("-" if not np.isfinite(r["noise"]) else f"{r['noise']:.3f} (" + ", ".join(
                     f"{k} {v['diff']:+.3f}" for k, v in r["noise_parts"].items()) + ")")
                 + f" | **{r['verdict']}** |")
    L += ["\n## Domanda su C3F-UGT (emendamento, sezione 4): GT unificata di HIFI3D, `nocrop_cross`, righe con baseline finita\n",
          "| passi | baseline | C3F-UGT / baseline | differenza [IC 95%] (P<=0) | esito |", "| --- | --- | --- | --- | --- |"]
    for u in ugt:
        x = u["x"]
        if x is None:
            L.append(f"| {u['steps']} | {u['baseline']} | - | - | non valutabile |")
            continue
        verdict = ("RAGGIUNGE" if x["ci_high"] >= 0 else "NO") if u["baseline"] == "rigid_icp_chamfer" else "(descrittivo)"
        L.append(f"| {u['steps']} | {u['baseline']} | {x['a']:.3f} / {x['b']:.3f} | "
                 f"{fmt(x['diff'], x['ci_low'], x['ci_high'], True)} ({x['p_le0']:.3f}) | {verdict} |")
    for e in (10548, 21096):
        L += [f"\n## Matrice celle x domini di test, {e} passi\n",
              "Punto [IC 95%, 1000 repliche per soggetto, le stesse per tutte le celle]. GT maxabs dove non indicato.\n",
              "| cella | " + " | ".join(n for _, n in COLS) + " |", "|" + " --- |" * (len(COLS) + 1)]
        for _, r in matrix[matrix["steps"] == e].iterrows():
            L.append(f"| {r['cell']} | " + " | ".join(fmt(r[k], r[k + '_lo'], r[k + '_hi']) for k, _ in COLS) + " |")
    L += ["\n## Differenze appaiate (descrittive)\n", "Cella: differenza [IC 95%] (P(boot <= 0)).\n"]
    for e in (21096, 10548):
        L += [f"\n### {e} passi\n", "| contrasto | lettura | " + " | ".join(n for _, n in COLS) + " |",
              "|" + " --- |" * (len(COLS) + 2)]
        for a, b, what in CONTRASTS:
            c = f"{LABEL[a]} - {LABEL[b]}"
            cells = []
            for k, _ in COLS:
                r = paired[(paired["contrast"] == c) & (paired["steps"] == e) & (paired["metric"] == k)] if not paired.empty else paired
                cells.append("-" if r.empty else f"{fmt(r.iloc[0]['diff'], r.iloc[0]['ci_low'], r.iloc[0]['ci_high'], True)} "
                                                 f"({r.iloc[0]['p_le0']:.2f})")
            L.append(f"| {c} | {what} | " + " | ".join(cells) + " |")
    design = json.loads((OUT / "design.json").read_text()) if (OUT / "design.json").exists() else None
    if design:
        L += ["\n## Identita' viste per cella (design.json, calcolato prima dei numeri)\n",
              "| cella | passi | viste per dominio | viste totali | esposizioni mediana |", "| --- | --- | --- | --- | --- |"]
        for cell in CELLS:
            dc = design["cells"].get("c3m" if cell == "c3ml" else cell)
            if dc:
                for _, a in dc["at"].items():
                    L.append(f"| {LABEL[cell]} | {a['steps']} | {a['seen']} | {a['seen_total']} | {a['exposures_median']:.0f} |")
    L += ["\n## Run\n", *runs_info(),
          "\n## Controlli\n", "| controllo | valore | atteso |", "| --- | --- | --- |",
          *[f"| {c['controllo']} | {c['valore']} | {c['atteso']} |" for c in checks]]
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
