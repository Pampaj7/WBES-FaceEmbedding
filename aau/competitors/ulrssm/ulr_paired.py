#!/usr/bin/env python3
"""Spearman FR / SR delle distanze di ULRSSM (varianti per coppia e via template) con IC bootstrap per soggetto, sulle
righe delle baseline parametriche, e i delta appaiati contro i bracci di riferimento: primo confronto.

    v3_work/unified_gt/run.sh aau/competitors/ulrssm/ulr_paired.py --tag faces_xyz --workers 32
    (ulr.sbatch, passo paired; dopo ulr_eval.py merge su tutte le viste; LEGGE LE GT di test)

Righe, GT, semi e repliche importati in sola lettura (nessuna riga riscritta qui):
  - ``nocrop`` (gruppo primario): ``fact_paired.rows_for`` (HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse con
    espressioni ``mesh_pair_nocrop``; FaceVerse neutra con le righe e la GT di FaceVerse e i bracci dalla vista
    neutra, come ``bp_paired.main``), FaMoS TEST ``fact_paired.famos_frame`` (blocco scan gallery -> scan);
  - ``all_cross`` e ``crop`` (righe col crop su almeno un lato, sottoinsieme di all_cross): ``bp_paired_e2.all_cross_rows``
    (le righe dell'emendamento 2 delle baseline parametriche);
  - repliche ``fact_paired._rep`` (per soggetto, ``default_rng(seme del gruppo)``, 1000; FaMoS
    ``famos_eval.bootstrap_counts``), riepilogo ``fact_paired.summarize`` (rho, parziale dato l'oracolo, quintile basso;
    IC 95% percentile; P(delta <= 0)).
Colonne: ``ulrssm_pair_{fr,sr}`` (variante a), ``ulrssm_tpl_{fr,sr}`` (variante b) dalle matrici di ``ulr_eval.py merge``
(``<tag>/<vista>.npz``), i bracci di ``REF_ARMS`` (``fact_paired.model_columns``), la taglia oracolo (serve alla
parziale). Maschera comune per gruppo: righe con tutte queste colonne finite. Delta = braccio - colonna di ULRSSM.

Uscite in ``aau/runs/evidence/competitors_new/ulrssm/<tag>/``: ``paired.csv`` (tutte le righe di summarize, colonna
``group``), ``spearman.csv`` (solo i valori), ``results.md`` (tabelle e costi), ``paired_info.json``.
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

import ulr_common as U

sys.path.insert(0, str(U.REPO / "v3_work" / "trainer" / "tools"))
import bp_paired as bpp  # noqa: E402  (sola lettura: importa fact_paired)
import bp_paired_e2 as bpe2  # noqa: E402  (sola lettura: all_cross_rows)

fp, be, blmm = bpp.fp, bpp.be, bpp.blmm
COLS = ["ulrssm_pair_fr", "ulrssm_pair_sr", "ulrssm_tpl_fr", "ulrssm_tpl_sr"]
LABEL = {"ulrssm_pair_fr": "ULRSSM per coppia, FR (mm)", "ulrssm_pair_sr": "ULRSSM per coppia, SR (CS_ref)",
         "ulrssm_tpl_fr": "ULRSSM via template, FR (GT)", "ulrssm_tpl_sr": "ULRSSM via template, SR (GT)"}
REF_ARMS = [c for c, _ in bpp.REF_ARMS]
VIEW_ORDER = bpp.VIEW_ORDER


def matrices(tag: str, view: str) -> tuple[dict, list]:
    """({colonna: D}, chiavi (soggetto, topologia)) di ``<tag>/<vista>.npz``."""
    with np.load(U.OUT / tag / f"{view}.npz") as z:
        keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
        return {c: np.asarray(z["D_" + c.removeprefix("ulrssm_")], np.float64) for c in COLS}, keys


def bootstrap(view: str, group: str, cols: dict, sa, sb, mask, counts: list, seed: int, workers: int) -> list:
    o = cols["oracle_size"]
    M = [m for m in REF_ARMS if m in cols]
    fp._J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
             "methods": M + COLS, "low": o[mask] <= np.quantile(o[mask], 0.2)}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(fp._rep, range(len(counts)), chunksize=4)
    V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
    recs = fp.summarize(V, view, M, COLS, int(mask.sum()), seed)
    for r in recs:
        r["group"] = group
    return recs


def subject_counts(df: pd.DataFrame, seed: int, n_boot: int):
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    return sa, sb, counts


def run_view(tag: str, view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    D, keys = matrices(tag, view)
    recs, info = [], {}
    if view == "famos":
        cols, G, sa, sb, counts, _ = fp.famos_frame(n_boot)
        with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
            gt_names = [str(x) for x in z["names"]]
        order = np.asarray([[s for s, _ in keys].index(n) for n in gt_names])
        cols = {k: v for k, v in cols.items() if k in REF_ARMS or k == "oracle_size"}
        cols.update({c: M[np.ix_(order, order)][sa, sb] for c, M in D.items()})
        cols.update({f"gt_{g}": G[g] for g in fp.GTS})
        mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
        recs += bootstrap(view, "nocrop", cols, sa, sb, mask, counts, 1234, workers)
        info["nocrop"] = {"rows": int(len(sa)), "rows_mask": int(mask.sum()), "seed": 1234}
        return recs, info
    src = "faceverse" if view == "faceverse_neutral" else view
    df, idx, seed = fp.rows_for(src)
    groups = [("nocrop", df, seed)]
    dfa, idx_a, seed_a = bpe2.all_cross_rows(view)
    if idx_a.keys != idx.keys:
        raise SystemExit(f"{view}: chiavi di all_cross diverse da quelle di rows_for")
    crop = (dfa["topology_a"].eq("crop") | dfa["topology_b"].eq("crop")).to_numpy()
    groups += [("all_cross", dfa, seed_a), ("crop", dfa[crop].reset_index(drop=True), seed_a)]
    if sorted(keys) != sorted(idx.keys):
        raise SystemExit(f"{view}: mesh di ULRSSM diverse dalle chiavi delle righe")
    on = bpp.on_index(D, keys, idx)
    Mc = {k: v for k, v in fp.model_columns(view, idx).items() if k in REF_ARMS}
    for group, d, sd in groups:
        d = be.add_columns(d, {**Mc, **on}, idx)
        if "oracle_size" not in d:
            orc = fp.oracle_S(blmm.VIEWS[src]["gt"])
            d = be.add_columns(d, {"oracle_size": be.log_abs([orc[s] for s, _ in idx.keys])}, idx)
        cols = {m: d[m].to_numpy(np.float64) for m in list(Mc) + COLS + ["oracle_size"]}
        cols.update({f"gt_{g}": d[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
        sa, sb, counts = subject_counts(d, sd, n_boot)
        mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
        recs += bootstrap(view, group, cols, sa, sb, mask, counts, sd, workers)
        info[group] = {"rows": int(len(d)), "rows_mask": int(mask.sum()), "seed": int(sd),
                       "nan_rows": {c: int((~np.isfinite(cols[c])).sum()) for c in COLS}}
        print(f"[ulr-paired] {view} {group}: {int(mask.sum())}/{len(d)} righe, seme {sd}", flush=True)
    return recs, info


# ------------------------------------------------------------------------------------------ markdown

def fmt(p, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def write_results(tag: str, P: pd.DataFrame, info: dict) -> None:
    root = U.OUT / tag
    md = [f"# ULRSSM ({tag}): Spearman con FR e SR, primo confronto", "",
          "Codice `aau/competitors/ulrssm/` (formato e definizioni nel README di questa cartella). Righe, GT e repliche "
          "importate (`fact_paired.rows_for` / `famos_frame`, `bp_paired_e2.all_cross_rows`), IC 95% percentile, 1000 "
          "repliche per soggetto. Gruppi: `nocrop` = primario, `crop` = righe di all_cross col crop su almeno un lato. "
          "Il modello NON gestisce il parziale (ULRSSM per forme complete): le righe col crop sono riportate cosi'.", ""]
    sel = U.OUT / "selection.json"
    if sel.exists():
        md += [f"Selezione (regola fissata prima della valutazione): `{json.loads(sel.read_text())}`", ""]
    R = P[(P["kind"] == "rho") & (P["baseline"] == "-")]
    for v in VIEW_ORDER:
        x = R[R["domain"] == v]
        if x.empty:
            continue
        groups = [g for g in ("nocrop", "crop", "all_cross") if g in set(x["group"])]
        md += [f"## {v}", "", "| metodo | " + " | ".join(f"{g} FR | {g} SR" for g in groups) + " |",
               "| --- |" + " --- | --- |" * len(groups)]
        rows = [(c, "baseline", LABEL[c]) for c in COLS] + [(c, c.split("|")[0], c.split("|")[1]) for c in REF_ARMS]
        for c, arm, dist in rows:
            cells = []
            for g in groups:
                for gt in fp.GTS:
                    r = x[(x["group"] == g) & (x["arm"] == arm) & (x["distance"] == dist) & (x["gt"] == gt)]
                    cells.append(fmt(r.iloc[0].arm_point, r.iloc[0].arm_ci_low, r.iloc[0].arm_ci_high) if len(r) else "-")
            md.append(f"| {LABEL.get(c, c)} | " + " | ".join(cells) + " |")
        md += ["", "righe: " + ", ".join(f"{g} {info[v][g]['rows_mask']}" for g in groups), ""]
        Dd = P[(P["domain"] == v) & (P["kind"] == "rho") & (P["group"] == "nocrop") & P["baseline"].isin(LABEL.values())]
        if len(Dd):
            md += ["Delta appaiati nocrop, braccio - ULRSSM (IC 95%, P(delta <= 0)):", "",
                   "| braccio (GT) | " + " | ".join(LABEL[c] for c in COLS) + " |", "| --- |" + " --- |" * len(COLS)]
            for a, g in (("factorized_s1234|form_cal", "fr"), ("factorized_s2345|form_cal", "fr"),
                         ("factorized_s1234|shape", "sr"), ("factorized_s2345|shape", "sr")):
                arm, dist = a.split("|")
                cells = []
                for c in COLS:
                    r = Dd[(Dd["arm"] == arm) & (Dd["distance"] == dist) & (Dd["gt"] == g) & (Dd["baseline"] == LABEL[c])]
                    cells.append(fmt(r.iloc[0].delta, r.iloc[0].ci_low, r.iloc[0].ci_high, True) +
                                 f", P {r.iloc[0].p_le0:.3f}" if len(r) else "-")
                md.append(f"| {a} ({g.upper()}) | " + " | ".join(cells) + " |")
            md.append("")
    md += ["## Costi (mediane per mesh e per coppia, secondi)", "",
           "| vista | mesh | vertici mediana / max | preparazione (CPU, 1 thread) | feature (GPU) | template -> mesh (GPU) | "
           "coppia, 2 mappe + 4 Procrustes (GPU) | ore GPU delle coppie |", "| --- |" + " --- |" * 7]
    for v in VIEW_ORDER:
        p = root / f"{v}.json"
        if p.exists():
            j = json.loads(p.read_text())
            s = j["seconds"]
            md.append(f"| {v} | {j['n_meshes']} | {j['vertices']['median']:.0f} / {j['vertices']['max']} | "
                      f"{s['prep_per_mesh_median']:.1f} | {s['feat_per_mesh_median']:.3f} | "
                      f"{s['template_map_per_mesh_median']:.3f} | {s['pair_median']:.3f} | {s['pairs_total_h']:.2f} |")
    (root / "results.md").write_text("\n".join(md) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True)
    p.add_argument("--views", default=",".join(VIEW_ORDER))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    root = U.OUT / a.tag
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")          # come bp_paired.main (in memoria)
    fp.FORM_DIR["fvn"] = os.path.relpath(bpp.NEUTRAL_EMB, fp.EVAL)
    fp.BASELINES.update(LABEL)
    if not a.summary_only:
        recs, info = [], {}
        for v in a.views.split(","):
            if not (root / f"{v}.npz").exists():
                print(f"[ulr-paired] {v}: {root / f'{v}.npz'} assente, salto", flush=True)
                continue
            r, info[v] = run_view(a.tag, v, a.n_boot, a.workers)
            recs += r
        P = pd.DataFrame(recs)
        P.to_csv(root / "paired.csv", index=False)
        P[P["baseline"] == "-"].to_csv(root / "spearman.csv", index=False)
        U.atomic_json(root / "paired_info.json", {"info": info, **U.stamp()})
    write_results(a.tag, pd.read_csv(root / "paired.csv"), json.loads((root / "paired_info.json").read_text())["info"])
    print(f"[ulr-paired] -> {root / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
