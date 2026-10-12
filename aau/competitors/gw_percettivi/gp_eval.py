#!/usr/bin/env python3
"""Gromov-Wasserstein e feature percettive: scelta del pilota (GT dei NON valutati) e Spearman FR / SR con IC.

    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py pilot-gw                (GT dei NON valutati)
    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py spearman --workers 32   (LEGGE LE GT di test)
    (gp.sbatch, passi pilot_eval e eval)

``pilot-gw``: per ogni vista del pilota (``gp_common.PILOT_VIEWS``) le distanze di ``gw/pilot/<vista>.npz`` sulle coppie
(``gp_gw.pilot_pairs``) dei 12 soggetti NON valutati (``gp_common.pilot_view``, controllato: nessuno fra i valutati) e la loro GT FR / SR
(``datasets/CANONICAL_GT/eval/<set>_{fr,sr}.npz``: copre i 500 soggetti del pool, se ne leggono solo le righe del
pilota). Punteggio di una configurazione = media sulle 4 viste di (rho(GW in mm, GT FR) + rho(GW a CS_ref, GT SR)) / 2,
Spearman su tutte le coppie del pilota (crop compreso). Scelta, fra le configurazioni entro ``gp_gw.BUDGET_S`` secondi
per soluzione (mediana): il punteggio massimo; entro ``gp_gw.TIE`` dal migliore vince la configurazione piu' economica. Uscite ``gw/pilot_choice.json`` e
``gw/pilot_scores.csv``.

``spearman``: righe, seme e GT di all_cross (``bp_paired_e2.all_cross_rows``: HIFI3D, FaceScape dev, FaceVerse con
espressioni, FaceVerse neutra con le righe di FaceVerse) e di FaMoS TEST (``fact_paired.famos_frame``, blocco scan
gallery -> scan), 1000 repliche per soggetto coi semi di quei moduli, ``fact_paired._rep`` attraverso
``bp_paired_e3.replicas`` (rho con GT FR e SR, parziale dato l'oracolo, quintile basso). Colonne ``<metodo>_{fr,sr}`` di
``<metodo>/<vista>.npz``; maschera comune fra le colonne di questo modulo (righe con tutte le distanze finite). Gruppi:
senza crop, all_cross, righe col crop (FaMoS: solo senza crop). Uscite ``spearman.csv``, ``controls.json``,
``results.md`` (tabelle e costi).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gp_common as gc
import gp_gw

GROUPS = ("senza crop", "all_cross", "righe col crop")
LABEL = {"gw": "Gromov-Wasserstein (POT)", "dinov2": "DINOv2 ViT-S/14 su render", "clip": "CLIP ViT-B/32 su render",
         "lpips": "LPIPS alex su render", "dreamsim": "DreamSim ensemble su render"}
MODE_LABEL = {"fr": "mm (FR)", "sr": "a CS_ref (SR)"}


# ------------------------------------------------------------------------------------------ pilota

def gt_of(gt_set: str) -> dict:
    import blmm_eval as be
    return {g: be.zsum.load_gt(be.EVAL_GT / f"{gt_set}_{g}.npz") for g in ("fr", "sr")}


def pilot_gw() -> None:
    from scipy.stats import spearmanr
    rows, cost = [], {}
    configs = None
    for view in gc.PILOT_VIEWS:
        with np.load(gp_gw.GW / "pilot" / f"{view}.npz") as z:
            i, j, D, S = (np.asarray(z[k]) for k in ("pair_i", "pair_j", "D", "seconds"))
            subj, topo = np.asarray(z["subjects"]).astype(str), np.asarray(z["topologies"]).astype(str)
            cf = [str(c) for c in z["configs"]]
        configs = configs or cf
        if cf != configs:
            raise SystemExit(f"{view}: configurazioni diverse dalle altre viste")
        if set(subj) & set(gc.blmm.subjects(view)):
            raise SystemExit(f"{view}: soggetti del pilota fra i valutati")
        G = gt_of(gc.blmm.VIEWS[view]["gt"])
        gt = {g: G[g][0][[G[g][1][s] for s in subj[i]], [G[g][1][s] for s in subj[j]]] for g in ("fr", "sr")}
        crop = (topo[i] == "crop") | (topo[j] == "crop")
        for c, cfg in enumerate(configs):
            cost.setdefault(cfg, []).append(S[:, c].ravel())
            for m, mode in enumerate(gc.MODES):
                d = D[:, c, m]
                for grp, sel in (("tutte", np.ones_like(crop)), ("senza crop", ~crop), ("righe col crop", crop)):
                    ok = sel & np.isfinite(d)
                    for g in ("fr", "sr"):
                        rows.append({"view": view, "config": cfg, "mode": mode, "gt": g, "group": grp,
                                     "rho": float(spearmanr(d[ok], gt[g][ok]).correlation), "n_pairs": int(ok.sum()),
                                     "n_nan": int((sel & ~np.isfinite(d)).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(gp_gw.GW / "pilot_scores.csv", index=False)
    A = R[(R["group"] == "tutte") & (R["mode"] == R["gt"])]
    score = A.groupby("config")["rho"].mean()
    sec = {c: float(np.nanmedian(np.concatenate(v))) for c, v in cost.items()}
    ok = [c for c in configs if sec[c] <= gp_gw.BUDGET_S]
    best = float(max(score[c] for c in ok))
    tied = sorted([c for c in ok if score[c] >= best - gp_gw.TIE], key=lambda c: sec[c])
    ch = tied[0]
    out = {"config": ch, "score": float(score[ch]), "best": best, "best_config": max(ok, key=lambda c: score[c]),
           "tied": tied, "in_budget": ok, "budget_s": gp_gw.BUDGET_S, "best_overall": str(score.idxmax()),
           "best_overall_score": float(score.max()), "scores": {c: float(score[c]) for c in configs},
           "s_per_solve_median": sec,
           "nan_pairs": {c: int(R[(R["config"] == c) & (R["group"] == "tutte")]["n_nan"].sum() // 2) for c in configs},
           "by_view": {f"{r.view}|{r.config}|{r.mode}": r.rho for r in A.itertuples()},
           "rule": f"fra le configurazioni con mediana dei secondi per soluzione <= {gp_gw.BUDGET_S}: massimo della media "
                   "sulle 4 viste di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 sulle coppie del pilota; entro "
                   f"{gp_gw.TIE} dal migliore la mediana dei secondi per soluzione piu' bassa",
           "pilot_subjects": gc.PILOT_SUBJECTS, "views": list(gc.PILOT_VIEWS), **gc.stamp()}
    gc.atomic_json(gp_gw.GW / "pilot_choice.json", out)
    for c in configs:
        print(f"[gp-eval] pilota {c}: punteggio {score[c]:.4f}, {sec[c]:.3f} s per soluzione", flush=True)
    print(f"[gp-eval] scelta {ch} (punteggio {score[ch]:.4f}; migliore {out['best_config']} {best:.4f}; pari {tied})",
          flush=True)


# ----------------------------------------------------------------------------------------- spearman

def methods_of(view: str) -> list[str]:
    return [m for m in gc.METHODS if gc.out_npz(m, view).exists()]


def columns(view: str, keys: list) -> tuple[dict, dict]:
    """{<metodo>_<modo>: matrice sulle chiavi ``keys``} e i controlli (mesh, NaN)."""
    out, ctrl = {}, {}
    for m in methods_of(view):
        z = gc.read_view(m, view)
        mine = list(zip(z["subjects"].astype(str), z["topologies"].astype(str)))
        pos = {k: r for r, k in enumerate(mine)}
        if sorted(mine) != sorted(keys):
            raise SystemExit(f"{m} {view}: mesh diverse dalle chiavi delle righe")
        o = np.asarray([pos[k] for k in keys])
        for mode in gc.MODES:
            out[f"{m}_{mode}"] = z[f"D_{mode}"][np.ix_(o, o)]
        ctrl[m] = {"meshes": len(mine)}
    return out, ctrl


def records(V: dict, view: str, group: str, n_rows: int, seed: int) -> list:
    recs = []
    for (gt, kind, col), a in V.items():
        b = a[1:][np.isfinite(a[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
        m, _, mode = col.rpartition("_")
        recs.append({"domain": view, "group": group, "gt": gt, "kind": kind, "column": col, "method": m, "mode": mode,
                     "label": f"{LABEL.get(m, m)}, {MODE_LABEL.get(mode, mode)}", "point": a[0], "ci_low": lo,
                     "ci_high": hi, "n_boot": int(len(b)), "n_rows": n_rows, "seed": seed})
    return recs


def run_view(view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    import bp_paired_e2 as bpe2
    import bp_paired_e3 as bpe3
    df, idx, seed = bpe2.all_cross_rows(view)
    D, ctrl = columns(view, list(idx.keys))
    df = bpe3.be.add_columns(df, D, idx)
    methods = list(D)
    cols = {m: df[m].to_numpy(np.float64) for m in methods + ["oracle_size"]}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in ("fr", "sr")})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    fin = np.all([np.isfinite(v) for v in cols.values()], axis=0)
    mask = (sa != sb) & fin
    crop = (df["topology_a"].eq("crop") | df["topology_b"].eq("crop")).to_numpy()
    recs = []
    for group, gm in (("senza crop", mask & ~crop), ("all_cross", mask), ("righe col crop", mask & crop)):
        V = bpe3.replicas(cols, sa, sb, gm, methods, counts, workers)
        recs += records(V, view, group, int(gm.sum()), seed)
    info = {"seed": seed, "rows": int(len(df)), "rows_mask": int(mask.sum()), "rows_mask_crop": int((mask & crop).sum()),
            "nan_rows_by_column": {m: int((~np.isfinite(cols[m]) & (sa != sb)).sum()) for m in methods},
            "methods": ctrl}
    print(f"[gp-eval] {view}: {int(mask.sum())}/{len(df)} righe ({int((mask & crop).sum())} col crop), {len(methods)} "
          f"colonne, seme {seed}", flush=True)
    return recs, info


def run_famos(n_boot: int, workers: int) -> tuple[list, dict]:
    import bp_paired_e3 as bpe3
    fp = bpe3.fp
    import famos_common as fc
    cols, G, sa, sb, counts, _ = fp.famos_frame(n_boot)
    with np.load(fc.VIEW_DIR / "gt_matrix.npz") as z:
        gt_names = [str(x) for x in z["names"]]
    keys = [(s, "scan_gallery") for s in gt_names]
    D, ctrl = columns("famos", keys)
    mine = {m: M[sa, sb] for m, M in D.items()}
    cols = {**{m: cols[m] for m in ("oracle_size",)}, **mine, **{f"gt_{g}": G[g] for g in ("fr", "sr")}}
    fin = np.all([np.isfinite(v) for v in cols.values()], axis=0)
    mask = (sa != sb) & fin
    V = bpe3.replicas(cols, sa, sb, mask, list(mine), counts, workers)
    recs = records(V, "famos", "senza crop", int(mask.sum()), 1234)
    info = {"seed": 1234, "rows": int(len(sa)), "rows_mask": int(mask.sum()), "methods": ctrl,
            "nan_rows_by_column": {m: int((~np.isfinite(v)).sum()) for m, v in mine.items()}}
    print(f"[gp-eval] famos: {int(mask.sum())}/{len(sa)} righe, {len(mine)} colonne", flush=True)
    return recs, info


def spearman(n_boot: int, workers: int) -> None:
    sys.path.insert(0, str(gc.AAU / "famos"))
    recs, info = [], {}
    t0 = time.time()
    for view in gc.VIEWS:
        if not methods_of(view):
            info[view] = "nessun metodo calcolato"
            continue
        r, inf = run_famos(n_boot, workers) if view == "famos" else run_view(view, n_boot, workers)
        recs += r
        info[view] = inf
    P = pd.DataFrame(recs)
    P.to_csv(gc.OUT / "spearman.csv", index=False)
    info["wall_s"] = time.time() - t0
    gc.atomic_json(gc.OUT / "controls.json", {**info, **gc.stamp()})
    write_results(P)


# ------------------------------------------------------------------------------------------ risultati

def fmt(r) -> str:
    return f"{r.point:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def cost_rows() -> list[str]:
    out = ["| metodo | vista | costo |", "|---|---|---|"]
    for m in gc.METHODS:
        for v in gc.VIEWS:
            p = gc.OUT / m / f"{v}.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            if m == "gw":
                s = j["s_per_solve"]
                c = (f"{s['fr']['median']:.3f} s per soluzione FR, {s['sr']['median']:.3f} SR (mediana, un thread); "
                     f"{s['fr']['total_h'] + s['sr']['total_h']:.1f} h-core per {j['n_pairs']} coppie; "
                     f"preparazione {sum(j['prep_s_per_mesh_median'].values()):.2f} s per mesh")
            elif m == "lpips":
                c = (f"{j['s_per_pair'] * 1e3:.2f} ms per coppia (Gram, {j['threads']} thread); render "
                     f"{j['render_s_per_mesh_median']:.2f} s per mesh (6 render)")
            else:
                c = (f"{j['s_per_image'] * 1e3:.0f} ms per immagine ({j['threads']} thread, {j['device']}); "
                     f"{j['s_per_mesh_total']:.2f} s per mesh con i 6 render")
            out.append(f"| {m} | {v} | {c} |")
    return out


def write_results(P: pd.DataFrame) -> None:
    L = ["# Gromov-Wasserstein e feature percettive su render: Spearman con GT FR / SR", "",
         "Generato da `aau/competitors/gw_percettivi/gp_eval.py spearman` (non modificare a mano). Righe, semi e repliche "
         "di all_cross (`bp_paired_e2.all_cross_rows`) e di FaMoS TEST (`fact_paired.famos_frame`), 1000 repliche per "
         "soggetto, IC al 95% percentile. Colonna mm (FR) contro GT FR, colonna a CS_ref (SR) contro GT SR; le altre "
         "combinazioni sono in `spearman.csv`.", ""]
    if (gc.OUT / "gw" / "pilot_choice.json").exists():
        ch = json.loads((gc.OUT / "gw" / "pilot_choice.json").read_text())
        L += [f"GW: configurazione `{ch['config']}` scelta sul pilota (punteggio {ch['score']:.4f}; migliore "
              f"`{ch['best_config']}` {ch['best']:.4f}; regola: {ch['rule']}).", ""]
    for group in GROUPS:
        X = P[(P["kind"] == "rho") & (P["group"] == group) & (P["gt"] == P["mode"])]
        if X.empty:
            continue
        doms = [d for d in gc.VIEWS if d in set(X["domain"])]
        L += [f"## {group}", "", "| metodo | GT | " + " | ".join(doms) + " |", "|---|---|" + "---|" * len(doms)]
        for m in gc.METHODS:
            for mode in gc.MODES:
                cells = []
                for d in doms:
                    r = X[(X["method"] == m) & (X["mode"] == mode) & (X["domain"] == d)]
                    cells.append(fmt(r.iloc[0]) if len(r) else "-")
                if any(c != "-" for c in cells):
                    L.append(f"| {LABEL[m]} | {mode.upper()} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Costi", ""] + cost_rows() + [""]
    (gc.OUT / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("pilot-gw", "spearman"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.step == "pilot-gw":
        pilot_gw()
    else:
        spearman(a.n_boot, a.workers)


if __name__ == "__main__":
    main()
