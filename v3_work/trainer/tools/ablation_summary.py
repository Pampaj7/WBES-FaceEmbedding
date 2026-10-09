#!/usr/bin/env python3
"""Ablazioni v3 sulla cella C3F: tabella finale e regola di adozione (ablation_protocol.md + emendamento del 9 ottobre).

    aau/run.sh v3_work/trainer/tools/ablation_summary.py      (ablations/c3f/summary.sbatch, nessuna GPU)

Legge le uscite di ablations/c3f/eval_body.sh (aau/runs/evidence/trainer_v3/ablations/c3f_eval/), controlla che ogni
stage abbia valutato il checkpoint EMA atteso del braccio, e calcola con le funzioni dei riepiloghi esistenti
(importate, non riscritte):
  - PUNTEGGIO DEV (criterio primario): come aau/zs3dmm/dev_fs_summarize.py, media fra Spearman graduato senza crop
    (vista neutra, GT maxabs, ``spearman_replicates``) e rank-1 con espressioni senza crop (``zs_expr_summarize``),
    sulle STESSE repliche di soggetti (``bootstrap_counts`` seme ``stable_seed(1234, "devfs_recognition")``), quindi
    la differenza braccio - ctrl e' appaiata replica per replica;
  - HIFI3D graduata senza crop con GT maxabs e unificata (``datasets/UNIFIED_GT/eval``): ``e1_summarize.gt_tasks`` /
    ``joint_boot``, tutti i bracci sulle stesse repliche;
  - riconoscimento rank-1 senza crop, FaceVerse con espressioni e HIFI3D: ``zs_expr_summarize._recog_task``, seme
    ``stable_seed(1234, "expr_recognition")`` come E1;
  - NoW: tau per immagine sui metodi pre-registrati (``e1_summarize.now_tau``), riportato, fuori dalla regola.
Regola (ultimo checkpoint EMA, 21.096 passi): si adotta se punteggio dev - ctrl >= +0.03 con IC 95% > 0 e nessuna
fra HIFI3D GT maxabs, HIFI3D GT unificata, FaceVerse rank-1 scende di piu' di 0.05 (punto). area e arearobust: se
passano entrambe, quella col dev piu' alto. ugtmix: un'adozione resta SOSPESA (emendamento, sezione 2). loginv si
giudica da solo contro ctrl. Scrive in aau/runs/evidence/trainer_v3/ablations/: results.md, results_matrix.csv,
results_paired.csv, results_controls.csv.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "aau/evidence/e1_factorial"))
sys.path.insert(0, str(REPO / "aau/zs3dmm"))
sys.path.insert(0, str(REPO / "aau/recon"))

import dev_fs_summarize as dfs  # noqa: E402
import e1_summarize as E  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import select_subjects  # noqa: E402

base = zsum.base
EV = REPO / "aau/runs/evidence/trainer_v3"
OUT = EV / "ablations"
EVAL = OUT / "c3f_eval"
RUNS = OUT / "c3f_runs"
ARMS = ("ctrl", "area", "arearobust", "bal", "ugtmix", "loginv")
LABEL = {"ctrl": "ctrl (v2)", "area": "area", "arearobust": "arearobust", "bal": "bal", "ugtmix": "ugtmix",
         "loginv": "loginv"}
STEPS = {"036": 10548, "072": 21096}
SEED, NB = 1234, 1000
MIN_DEV, MAX_DROP = 0.03, 0.05
DEVFS = REPO / "datasets/DEV_FACESCAPE"

# le funzioni di E1 leggono questi nomi del modulo: bracci al posto delle celle, stessi passi
E.CELLS, E.STEPS = ARMS, STEPS
E.LABEL.update(LABEL)


def tag(arm: str, e: str) -> str:
    return f"v3{arm}e{e}"


def expected_ckpt(arm: str, e: str) -> str | None:
    hits = sorted(RUNS.glob(f"{arm}/v3_*/checkpoints/epoch{e}_ema.pth"))
    if len(hits) > 1:
        raise SystemExit(f"{arm}: piu' di un run dir {hits}")
    return os.path.realpath(hits[0]) if hits else None


def stage(domain_dir: str, arm: str, e: str, suffix: str, need_embed: bool) -> Path | None:
    d = E.one_dir(EVAL / domain_dir, "data_*")
    st = None if d is None else d / f"scale_{tag(arm, e)}{suffix}" / zsum.STAGE
    if st is None or not (st / ".done").exists() or (need_embed and not (st / "embeddings.npz").exists()):
        return None
    want, got = expected_ckpt(arm, e), E.eval_ckpt(st)
    if want is None or got != want:
        raise SystemExit(f"{st}: checkpoint valutato {got} != atteso {want}")
    return st


def main() -> None:
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    checks: list[dict] = []
    pms_hifi, emb = {}, {"hifi": {}, "fv": {}}
    dev_n, dev_e, now = {}, {}, {}
    for arm in ARMS:
        for e in STEPS:
            st = stage("hifi_runs", arm, e, "_topology", False)
            if st is not None:
                pms_hifi[(arm, e)] = base.read_pair_metrics(st)
                if (st / "embeddings.npz").exists():
                    emb["hifi"][(arm, e)] = st
            st = stage("fv_expr", arm, e, "_flip_embed", True)
            if st is not None:
                emb["fv"][(arm, e)] = st
            st = stage("devfs", arm, e, "_topology", True)
            if st is not None:
                dev_n[(arm, e)] = st
            st = stage("devfs_expr", arm, e, "_embed", True)
            if st is not None:
                dev_e[(arm, e)] = st
            o = EVAL / "now" / tag(arm, e)
            if (o / "concordance.csv").exists():
                ck = (o / "checkpoint.txt").read_text().strip().split("=", 1)[1]
                if os.path.realpath(ck) != expected_ckpt(arm, e):
                    raise SystemExit(f"NoW {arm} e{e}: checkpoint {ck} != {expected_ckpt(arm, e)}")
                now[(arm, e)] = o

    # ------------------------------------------------ HIFI3D graduata, GT maxabs e unificata (E1)
    tasks = E.gt_tasks("hifi", pms_hifi, checks)
    res = {}
    with mp.get_context("fork").Pool(max(1, min(workers, len(tasks) or 1))) as pool:
        for key, point, reps, _, _ in pool.imap_unordered(E.joint_boot, tasks):
            res[key] = (point, reps)

    # ------------------------------------------------ riconoscimento (FaceVerse con espressioni, HIFI3D)
    pairs_r = [(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b]
    pairs_v = [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]
    rec = {}
    for domain, stages in emb.items():
        if not stages:
            continue
        subjects = select_subjects(E.VIEW[domain], SEED)
        idx = zes.Index(subjects)
        D = {}
        for k, st in stages.items():
            if sorted(json.loads((st.parent / "subjects.json").read_text())["subjects"]) != subjects:
                raise SystemExit(f"{st}: soggetti diversi da select_subjects")
            D[k] = zes.model_distances(st, idx)
        counts = zes.bootstrap_counts(len(subjects), NB, base.stable_seed(SEED, "expr_recognition"))
        with mp.get_context("fork").Pool(max(1, min(workers, 16, len(D)))) as pool:
            for k, vals, _ in pool.imap_unordered(zes._recog_task, [(k, D[k], idx, pairs_r, pairs_v, counts) for k in D]):
                rec[(domain,) + k] = vals

    # ------------------------------------------------ punteggio dev (dev_fs_summarize, stesse repliche)
    subj = select_subjects(DEVFS / "eval_view/npz", SEED)
    if select_subjects(DEVFS / "expr_view/npz", SEED) != subj:
        raise SystemExit("dev FaceScape: soggetti diversi fra vista neutra e con espressioni")
    didx = zes.Index(subj)
    dcounts = zes.bootstrap_counts(len(subj), NB, base.stable_seed(SEED, "devfs_recognition"))
    gt_dev = zsum.load_gt(DEVFS / "eval_view/gt_matrix.npz")
    n = len(didx.subjects) * len(zes.TOPOLOGIES)
    dummy = {"chamfer_full": np.zeros((n, n)), "chamfer_stable": np.zeros((n, n))}   # pair_frame le vuole, qui non entrano
    bm = base.load_bootstrap_module()
    dev = {}
    dkeys = [k for k in dev_n if k in dev_e]
    rtasks = []
    for k in dkeys:
        for st in (dev_n[k], dev_e[k]):
            if sorted(json.loads((st.parent / "subjects.json").read_text())["subjects"]) != subj:
                raise SystemExit(f"{st}: soggetti diversi da select_subjects")
        rtasks.append((k, zes.model_distances(dev_e[k], didx), didx, pairs_r, pairs_v, dcounts))
    r1 = {}
    with mp.get_context("fork").Pool(max(1, min(workers, 16, len(rtasks) or 1))) as pool:
        for k, vals, _ in pool.imap_unordered(zes._recog_task, rtasks):
            r1[k] = vals["rank1"]
    for k in dkeys:
        col = f"lat:{k[0]}"
        pf = dfs.pair_frame(didx, gt_dev, {k[0]: zes.model_distances(dev_n[k], didx)}, dummy, dev_n[k])
        checks.append({"controllo": f"dev FaceScape {k[0]} {STEPS[k[1]]}: embedding contro latent_distance del breakdown, max |diff|",
                       "valore": f"{float(np.abs(pf['_lat_bd'] - pf[col]).max()):.2e}", "atteso": "< 1e-4"})
        sp = dfs.spearman_replicates(zsum.scenario_frame(pf, "nocrop_cross", []), col, subj, dcounts, bm)
        dev[k] = {"graded": sp, "rank1": r1[k], "score": (sp + r1[k]) / 2}

    # ------------------------------------------------ NoW
    tau = {k: E.now_tau(o, checks, *k) for k, o in now.items()}
    now_counts = zes.bootstrap_counts(20, NB, SEED)

    def metric(arm, e, key):
        """(punto, repliche): 'dev', 'devg', 'devr', 'sp|hifi|gt|scenario', 'rec|dominio|rank1', 'now'."""
        if key in ("dev", "devg", "devr"):
            x = dev.get((arm, e))
            if x is None:
                return None
            v = x[{"dev": "score", "devg": "graded", "devr": "rank1"}[key]]
            return float(v[0]), np.asarray(v[1:])
        p = key.split("|")
        if p[0] == "sp":
            r = res.get((p[1], p[2], e, p[3]))
            c = f"lat_{arm}"
            return None if r is None or c not in r[0] else (float(r[0][c]), r[1][c])
        if p[0] == "rec":
            v = rec.get((p[1], arm, e))
            return None if v is None else (float(v[p[2]][0]), np.asarray(v[p[2]][1:]))
        if key == "now" and (arm, e) in tau:
            _, t, s = tau[(arm, e)]
            r = np.concatenate([[t.mean()], [(c[s] * t).sum() / c[s].sum() for c in now_counts]])
            return float(r[0]), r[1:]
        return None

    COLS = [("dev", "punteggio dev"), ("devg", "dev graduata"), ("devr", "dev rank-1 espr."),
            ("sp|hifi|maxabs|nocrop_cross", "HIFI3D graduata, GT maxabs"),
            ("sp|hifi|unified|nocrop_cross", "HIFI3D graduata, GT unif."), ("rec|fv|rank1", "FaceVerse espr. rank-1"),
            ("rec|hifi|rank1", "HIFI3D rank-1"), ("now", "NoW tau")]
    RULE_OTHER = ("sp|hifi|maxabs|nocrop_cross", "sp|hifi|unified|nocrop_cross", "rec|fv|rank1")

    rows = []
    for arm in ARMS:
        for e in STEPS:
            r = {"arm": arm, "steps": STEPS[e]}
            for k, _ in COLS:
                x = metric(arm, e, k)
                r[k], (r[k + "_lo"], r[k + "_hi"]) = (x[0], E.ci(x[1])) if x else (np.nan, (np.nan, np.nan))
            rows.append(r)
    matrix = pd.DataFrame(rows)

    def diff(a, e, k, b="ctrl"):
        xa, xb = metric(a, e, k), metric(b, e, k)
        if xa is None or xb is None:
            return None
        d = xa[1] - xb[1]
        lo, hi = E.ci(d)
        return {"a": xa[0], "b": xb[0], "diff": xa[0] - xb[0], "ci_low": lo, "ci_high": hi,
                "p_le0": float(np.mean(d[np.isfinite(d)] <= 0))}

    paired = pd.DataFrame([{"arm": a, "steps": STEPS[e], "metric": k, **x} for a in ARMS[1:] for e in STEPS
                           for k, _ in COLS for x in [diff(a, e, k)] if x])

    # ------------------------------------------------ regola di adozione (ultimo checkpoint)
    verdict = {}
    for a in ARMS[1:]:
        dv = diff(a, "072", "dev")
        oth = {k: diff(a, "072", k) for k in RULE_OTHER}
        if dv is None or any(v is None for v in oth.values()):
            verdict[a] = ("NON VALUTABILE", "mancano misure")
            continue
        up = dv["diff"] >= MIN_DEV and dv["ci_low"] > 0
        drops = [k for k, v in oth.items() if v["diff"] < -MAX_DROP]
        why = (f"dev {dv['diff']:+.3f} [{dv['ci_low']:+.3f}, {dv['ci_high']:+.3f}]"
               + ("" if up else f" (serve >= +{MIN_DEV} con IC > 0)")
               + ("; cali oltre 0.05: " + ", ".join(drops) if drops else "; nessun calo oltre 0.05"))
        verdict[a] = ("PASSA" if up and not drops else "NO: resta il default v2", why)
    if verdict.get("ugtmix", ("",))[0] == "PASSA":
        verdict["ugtmix"] = ("SOSPESA: passa, ma va confermata con la GT tarata (emendamento, sez. 2)", verdict["ugtmix"][1])
    both = [a for a in ("area", "arearobust") if verdict.get(a, ("",))[0] == "PASSA"]
    if len(both) == 2:
        lose = min(both, key=lambda a: diff(a, "072", "dev")["diff"])
        verdict[lose] = ("PASSA, ma non si adotta: l'altro braccio d'area ha dev piu' alto", verdict[lose][1])

    OUT.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT / "results_matrix.csv", index=False)
    paired.to_csv(OUT / "results_paired.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "results_controls.csv", index=False)
    write_md(matrix, paired, verdict, checks, COLS)
    print((OUT / "results.md").read_text(), flush=True)


def write_md(matrix, paired, verdict, checks, COLS) -> None:
    sha = []
    for name in ("ablation_protocol", "ablation_protocol_emendamento_2026-10-09"):
        txt = (EV / f"{name}.md").read_bytes()
        rec = (EV / f"{name}.sha256").read_text().split()[0]
        sha.append(f"`{name}.md` {'invariato' if hashlib.sha256(txt).hexdigest() == rec else 'MODIFICATO, CONTROLLARE'}")
    L = ["# Ablazioni v3 sulla cella C3F: risultati e regola di adozione", "",
         "Generato da `v3_work/trainer/tools/ablation_summary.py`. Protocolli (sha256 registrati prima dei numeri): "
         + "; ".join(sha) + ". Un seme per braccio; IC 95% bootstrap per soggetto (1.000 repliche), differenze "
         "appaiate sulle stesse repliche. Pesi EMA.", "",
         "## Verdetti (21.096 passi)", "", "| braccio | esito | motivo |", "| --- | --- | --- |"]
    L += [f"| {LABEL[a]} | **{v[0]}** | {v[1]} |" for a, v in verdict.items()]
    for e in (21096, 10548):
        L += ["", f"## Valori, {e} passi{'' if e == 21096 else ' (descrittivo)'}", "",
              "| braccio | " + " | ".join(n for _, n in COLS) + " |", "|" + " --- |" * (len(COLS) + 1)]
        for _, r in matrix[matrix["steps"] == e].iterrows():
            L.append(f"| {LABEL[r['arm']]} | " + " | ".join(E.fmt(r[k], r[k + '_lo'], r[k + '_hi']) for k, _ in COLS) + " |")
        L += ["", f"Differenze braccio - ctrl, {e} passi: delta [IC 95%] (P(boot <= 0))", "",
              "| braccio | " + " | ".join(n for _, n in COLS) + " |", "|" + " --- |" * (len(COLS) + 1)]
        for a in ARMS[1:]:
            cells = []
            for k, _ in COLS:
                r = paired[(paired["arm"] == a) & (paired["steps"] == e) & (paired["metric"] == k)] if not paired.empty else paired
                cells.append("-" if r.empty else f"{E.fmt(r.iloc[0]['diff'], r.iloc[0]['ci_low'], r.iloc[0]['ci_high'], True)} "
                                                 f"({r.iloc[0]['p_le0']:.2f})")
            L.append(f"| {LABEL[a]} | " + " | ".join(cells) + " |")
    L += ["", "## Controlli", "", "| controllo | valore | atteso |", "| --- | --- | --- |",
          *[f"| {c['controllo']} | {c['valore']} | {c['atteso']} |" for c in checks]]
    (OUT / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
