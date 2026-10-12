#!/usr/bin/env python3
"""Gromov-Wasserstein e feature percettive: scelte dei piloti (GT dei NON valutati) e Spearman FR / SR con IC.

    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py pilot-gw                (GT dei NON valutati)
    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py pilot-perc              (GT dei NON valutati)
    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py spearman --workers 32   (LEGGE LE GT di test)
    v3_work/unified_gt/run.sh aau/competitors/gw_percettivi/gp_eval.py results                 (solo results.md)
    (gp.sbatch, passi pilot_eval, perc_pilot_eval e eval)

Regole di scelta (scritte PRIMA di guardare i valutati; nessuna GT dei valutati entra in una scelta):

``pilot-gw``: le distanze di ``gp_gw.WORK/pilot/<vista>*.npz`` (piu' file per vista: configurazioni aggiunte, stesse
coppie ``gp_gw.pilot_pairs`` dei 12 soggetti NON valutati, controllato) e la GT FR / SR di quei soggetti
(``datasets/CANONICAL_GT/eval/<set>_{fr,sr}.npz``: copre i 500 soggetti del pool, se ne leggono solo le righe del
pilota). I solutori composti (``gp_gw.COMPOSITE``) si ricavano dai componenti: minimo per coppia e modo, secondi
sommati. Punteggio di una configurazione = media sulle 4 viste di (rho(GW in mm, GT FR) + rho(GW a CS_ref, GT SR)) / 2,
Spearman su tutte le coppie del pilota (crop compreso). Scelta, fra le configurazioni entro ``gp_gw.BUDGET_S`` secondi
per soluzione (mediana): il punteggio massimo; entro ``gp_gw.TIE`` dal migliore vince la configurazione piu' economica.
Uscite ``gw/pilot_choice.json`` e ``gw/pilot_scores.csv``.

``pilot-perc``: per ogni metodo percettivo e variante di render (``gp_percettivi.VARIANTS``) le distanze sulle viste
del pilota (``WORK/pilota/<metodo>/<variante>/pilot_<vista>.npz``: gli stessi 12 soggetti NON valutati x 6 topologie),
tutte le coppie di ``gp_common.pairs`` (soggetti e topologie diversi, crop compreso), stesso punteggio. Scelta per
metodo: la variante col punteggio massimo; entro ``TIE`` dalla migliore vince la prima nell'ordine ``flat``, ``smooth``,
``depth`` (dal render del repo verso quello piu' lontano). Uscite ``percettivi/pilot_choice.json`` e
``percettivi/pilot_scores.csv``. Le due scelte decidono ``declared`` in ``fm_plugin.json`` (``gp_common.declared``).

``spearman``: righe, seme e GT di all_cross (``bp_paired_e2.all_cross_rows``: HIFI3D, FaceScape dev, FaceVerse con
espressioni, FaceVerse neutra con le righe di FaceVerse) e di FaMoS TEST (``fact_paired.famos_frame``, blocco scan
gallery -> scan), 1000 repliche per soggetto coi semi di quei moduli, ``fact_paired._rep`` attraverso
``bp_paired_e3.replicas`` (rho con GT FR e SR, parziale dato l'oracolo, quintile basso). Colonne
``<metodo>/<variante>_{fr,sr}`` di ``<metodo>/<variante>/<vista>.npz``; maschera comune fra le colonne di questo modulo.
Gruppi (``GROUPS``): senza crop, all_cross, righe col crop, senza crop e senza noisy, e la scomposizione di "senza crop"
per topologia: righe con ``t`` su un lato (``con <t>``; ``con noisy`` = il complemento di "senza noisy"). FaMoS: solo
senza crop. Uscite ``spearman.csv``, ``controls.json``, ``results.md`` (tabelle e costi).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

import gp_common as gc
import gp_gw

NOCROP = ("original", "remesh", "down8k", "up60k", "noisy")
GROUPS = ("senza crop", "all_cross", "righe col crop", "senza crop, senza noisy") + tuple(f"con {t}" for t in NOCROP)
PERC_METHODS = ("lpips", "dinov2", "clip", "dreamsim")
TIE = 0.01
LABEL, MODE_LABEL = gc.LABEL, gc.MODE_LABEL


# ------------------------------------------------------------------------------------------ piloti

def gt_of(gt_set: str) -> dict:
    import blmm_eval as be
    return {g: be.zsum.load_gt(be.EVAL_GT / f"{gt_set}_{g}.npz") for g in ("fr", "sr")}


def pilot_gt(view: str, subj: np.ndarray, i: np.ndarray, j: np.ndarray) -> dict:
    """GT FR / SR delle coppie (i, j) del pilota; si ferma se un soggetto e' fra i valutati."""
    if set(subj) & set(gc.blmm.subjects(view)):
        raise SystemExit(f"{view}: soggetti del pilota fra i valutati")
    G = gt_of(gc.blmm.VIEWS[view]["gt"])
    return {g: G[g][0][[G[g][1][s] for s in subj[i]], [G[g][1][s] for s in subj[j]]] for g in ("fr", "sr")}


def rho_rows(view: str, key: str, D: dict, gt: dict, topo_i: np.ndarray, topo_j: np.ndarray) -> list:
    from scipy.stats import spearmanr
    crop = (topo_i == "crop") | (topo_j == "crop")
    noisy = (topo_i == "noisy") | (topo_j == "noisy")
    rows = []
    for mode, d in D.items():
        for grp, sel in (("tutte", np.ones_like(crop)), ("senza crop", ~crop), ("righe col crop", crop),
                         ("senza crop, senza noisy", ~crop & ~noisy), ("con noisy", ~crop & noisy)):
            ok = sel & np.isfinite(d)
            for g in ("fr", "sr"):
                rows.append({"view": view, "config": key, "mode": mode, "gt": g, "group": grp,
                             "rho": float(spearmanr(d[ok], gt[g][ok]).correlation), "n_pairs": int(ok.sum()),
                             "n_nan": int((sel & ~np.isfinite(d)).sum())})
    return rows


def pilot_gw() -> None:
    rows, cost = [], {}
    configs = None
    for view in gc.PILOT_VIEWS:
        D, S, cf, ref = {}, {}, [], None
        for p in sorted((gp_gw.WORK / "pilot").glob(f"{view}*.npz")):
            if p.stem != view and not p.stem.startswith(f"{view}__"):
                continue                                   # es. faceverse_neutral.npz quando si cerca faceverse
            with np.load(p) as z:
                i, j = np.asarray(z["pair_i"]), np.asarray(z["pair_j"])
                subj, topo = np.asarray(z["subjects"]).astype(str), np.asarray(z["topologies"]).astype(str)
                if ref is not None and not (np.array_equal(ref[0], i) and np.array_equal(ref[1], j)):
                    raise SystemExit(f"{p}: coppie diverse dagli altri file della vista")
                ref = (i, j)
                for c, cfg in enumerate(str(x) for x in z["configs"]):
                    D[cfg], S[cfg] = np.asarray(z["D"])[:, c], np.asarray(z["seconds"])[:, c]
                    cf.append(cfg)
        for cfg in [c for c in gp_gw.CONFIGS if c not in D and all(x in D for x in gp_gw.components(c))]:
            comp = gp_gw.components(cfg)
            D[cfg] = np.fmin.reduce([D[x] for x in comp])
            S[cfg] = np.sum([S[x] for x in comp], axis=0)
            cf.append(cfg)
        configs = configs or cf
        if sorted(cf) != sorted(configs):
            raise SystemExit(f"{view}: configurazioni diverse dalle altre viste")
        i, j = ref
        gt = pilot_gt(view, subj, i, j)
        for cfg in configs:
            cost.setdefault(cfg, []).append(S[cfg].ravel())
            rows += rho_rows(view, cfg, {m: D[cfg][:, k] for k, m in enumerate(gc.MODES)}, gt, topo[i], topo[j])
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
    refresh_plugins()
    for c in sorted(configs):
        print(f"[gp-eval] pilota GW {c}: punteggio {score[c]:.4f}, {sec[c]:.3f} s per soluzione", flush=True)
    print(f"[gp-eval] scelta {ch} (punteggio {score[ch]:.4f}; migliore {out['best_config']} {best:.4f}; pari {tied})",
          flush=True)


def pilot_perc() -> None:
    import gp_percettivi as gpp
    rows = []
    for view in gc.PILOT_VIEWS:
        for m in PERC_METHODS:
            for var in gpp.VARIANTS:
                z = gc.read_view(m, var, f"pilot_{view}")
                subj, topo = z["subjects"].astype(str), z["topologies"].astype(str)
                i, j = gc.pairs([(s, t, None) for s, t in zip(subj, topo)])
                gt = pilot_gt(view, subj, i, j)
                rows += [{**r, "method": m} for r in rho_rows(view, var, {md: z[f"D_{md}"][i, j] for md in gc.MODES},
                                                               gt, topo[i], topo[j])]
    R = pd.DataFrame(rows).rename(columns={"config": "variant"})
    (gc.OUT / "percettivi").mkdir(parents=True, exist_ok=True)
    R.to_csv(gc.OUT / "percettivi" / "pilot_scores.csv", index=False)
    A = R[(R["group"] == "tutte") & (R["mode"] == R["gt"])]
    score = A.groupby(["method", "variant"])["rho"].mean()
    choice, tied = {}, {}
    for m in PERC_METHODS:
        best = max(score[(m, v)] for v in gpp.VARIANTS)
        tied[m] = [v for v in gpp.VARIANTS if score[(m, v)] >= best - TIE]
        choice[m] = tied[m][0]
    out = {"choice": choice, "tied": tied, "scores": {f"{m}|{v}": float(score[(m, v)]) for m in PERC_METHODS
                                                      for v in gpp.VARIANTS},
           "by_view": {f"{r.method}|{r.variant}|{r.view}|{r.mode}": r.rho for r in A.itertuples()},
           "rule": "per metodo: massimo della media sulle 4 viste di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 su tutte le "
                   f"coppie del pilota (soggetti e topologie diversi, crop compreso); entro {TIE} dalla migliore la "
                   "prima variante nell'ordine " + ", ".join(gpp.VARIANTS),
           "pilot_subjects": gc.PILOT_SUBJECTS, "views": list(gc.PILOT_VIEWS), **gc.stamp()}
    gc.atomic_json(gc.OUT / "percettivi" / "pilot_choice.json", out)
    refresh_plugins()
    for m in PERC_METHODS:
        print(f"[gp-eval] pilota {m}: " + ", ".join(f"{v} {score[(m, v)]:.4f}" for v in gpp.VARIANTS) +
              f" -> {choice[m]} (pari {tied[m]})", flush=True)


def refresh_plugins() -> None:
    """Riscrive ``fm_plugin.json`` di ogni variante gia' calcolata (``declared`` segue le scelte dei piloti)."""
    for m in gc.METHODS:
        root = gc.OUT / m
        if root.is_dir():
            for d in sorted(x for x in root.iterdir() if x.is_dir() and any(x.glob("*.npz"))):
                gc.plugin_json(m, d.name)


# ----------------------------------------------------------------------------------------- spearman

def columns_of(view: str) -> list[tuple[str, str]]:
    return [(m, v) for m in gc.METHODS for v in gc.variants(m, view)]


def columns(view: str, keys: list) -> tuple[dict, dict]:
    """{<metodo>/<variante>_<modo>: matrice sulle chiavi ``keys``} e i controlli."""
    out, ctrl = {}, {}
    for m, var in columns_of(view):
        z = gc.read_view(m, var, view)
        mine = list(zip(z["subjects"].astype(str), z["topologies"].astype(str)))
        pos = {k: r for r, k in enumerate(mine)}
        if sorted(mine) != sorted(keys):
            raise SystemExit(f"{m}/{var} {view}: mesh diverse dalle chiavi delle righe")
        o = np.asarray([pos[k] for k in keys])
        for mode in gc.MODES:
            out[f"{m}/{var}_{mode}"] = z[f"D_{mode}"][np.ix_(o, o)]
        ctrl[f"{m}/{var}"] = {"meshes": len(mine), "declared": gc.declared(m, var)}
    return out, ctrl


def records(V: dict, view: str, group: str, n_rows: int, seed: int) -> list:
    recs = []
    for (gt, kind, col), a in V.items():
        b = a[1:][np.isfinite(a[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
        mv, _, mode = col.rpartition("_")
        m, _, var = mv.partition("/")
        recs.append({"domain": view, "group": group, "gt": gt, "kind": kind, "column": col, "method": m,
                     "variant": var, "declared": gc.declared(m, var), "mode": mode,
                     "label": f"{gc.variant_label(m, var)}, {MODE_LABEL.get(mode, mode)}", "point": a[0],
                     "ci_low": lo, "ci_high": hi, "n_boot": int(len(b)), "n_rows": n_rows, "seed": seed})
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
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    crop = (ta == "crop") | (tb == "crop")
    noisy = (ta == "noisy") | (tb == "noisy")
    sel = {"senza crop": mask & ~crop, "all_cross": mask, "righe col crop": mask & crop,
           "senza crop, senza noisy": mask & ~crop & ~noisy}
    sel.update({f"con {t}": mask & ~crop & ((ta == t) | (tb == t)) for t in NOCROP})
    recs = []
    for group in GROUPS:
        V = bpe3.replicas(cols, sa, sb, sel[group], methods, counts, workers)
        recs += records(V, view, group, int(sel[group].sum()), seed)
    info = {"seed": seed, "rows": int(len(df)), "rows_mask": int(mask.sum()),
            "rows_by_group": {g: int(sel[g].sum()) for g in GROUPS},
            "nan_rows_by_column": {m: int((~np.isfinite(cols[m]) & (sa != sb)).sum()) for m in methods},
            "columns": ctrl}
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
    info = {"seed": 1234, "rows": int(len(sa)), "rows_mask": int(mask.sum()), "columns": ctrl,
            "nan_rows_by_column": {m: int((~np.isfinite(v)).sum()) for m, v in mine.items()}}
    print(f"[gp-eval] famos: {int(mask.sum())}/{len(sa)} righe, {len(mine)} colonne", flush=True)
    return recs, info


def spearman(n_boot: int, workers: int) -> None:
    sys.path.insert(0, str(gc.AAU / "famos"))
    refresh_plugins()
    recs, info = [], {}
    t0 = time.time()
    for view in gc.VIEWS:
        if not columns_of(view):
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
    out = ["| metodo | variante | vista | costo |", "|---|---|---|---|"]
    for m in gc.METHODS:
        for var in sorted({v for view in gc.VIEWS for v in gc.variants(m, view)}):
            for view in gc.VIEWS:
                p = gc.col_dir(m, var, view) / f"{view}.json"
                if not p.exists():
                    continue
                j = json.loads(p.read_text())
                if m == "gw":
                    s = j["s_per_solve"]
                    prep = {k: x for k, x in j["prep_s_per_mesh_median"].items()
                            if j["config"].startswith("geo") or k != "geodetiche"}
                    c = (f"{s['fr']['median']:.3f} s per coppia FR, {s['sr']['median']:.3f} SR (mediana, un thread; "
                         f"composti: somma dei componenti); {s['fr']['total_h'] + s['sr']['total_h']:.1f} h-core per "
                         f"{j['n_pairs']} coppie; preparazione {sum(prep.values()):.2f} s per mesh")
                elif m == "lpips":
                    c = (f"{j['s_per_pair'] * 1e3:.2f} ms per coppia (Gram, {j['threads']} thread); render "
                         f"{j['render_s_per_mesh_median']:.2f} s per mesh (tutte le varianti)")
                else:
                    c = (f"{j['s_per_image'] * 1e3:.0f} ms per immagine ({j['threads']} thread, {j['device']}); "
                         f"{j['s_per_mesh_total']:.2f} s per mesh coi 6 render della variante")
                out.append(f"| {m} | {var} | {view} | {c} |")
    return out


def col_name(r) -> str:
    return f"{LABEL[r.method]} [{r.variant}]" + (" **(dichiarata)**" if bool(r.declared) else "")


def write_results(P: pd.DataFrame) -> None:
    P = P.copy()
    P["declared"] = P["declared"].map(lambda x: x is True or x == "True" or x == 1.0)
    L = ["# Gromov-Wasserstein e feature percettive su render: Spearman con GT FR / SR", "",
         "Generato da `aau/competitors/gw_percettivi/gp_eval.py spearman` (non modificare a mano). Righe, semi e repliche "
         "di all_cross (`bp_paired_e2.all_cross_rows`) e di FaMoS TEST (`fact_paired.famos_frame`), 1000 repliche per "
         "soggetto, IC al 95% percentile. Colonna mm (FR) contro GT FR, colonna a CS_ref (SR) contro GT SR; le altre "
         "combinazioni sono in `spearman.csv`. Dichiarata = la variante scelta dalla regola del pilota sui soggetti NON "
         "valutati (`gw/pilot_choice.json`, `percettivi/pilot_choice.json`).", ""]
    for p, what in ((gc.OUT / "gw" / "pilot_choice.json", "GW"), (gc.OUT / "percettivi" / "pilot_choice.json", "perc")):
        if p.exists():
            ch = json.loads(p.read_text())
            if what == "GW":
                L += [f"GW dichiarato: `{ch['config']}` (punteggio del pilota {ch['score']:.4f}; regola: {ch['rule']}).", ""]
            else:
                L += ["Percettivi dichiarati: " + ", ".join(f"{m} `{v}`" for m, v in ch["choice"].items()) +
                      f" (regola: {ch['rule']}).", ""]
    X0 = P[(P["kind"] == "rho") & (P["gt"] == P["mode"])]
    order = [(m, v) for m in gc.METHODS for v in sorted(set(X0[X0["method"] == m]["variant"]))]
    for group in ("senza crop", "righe col crop", "all_cross"):
        X = X0[X0["group"] == group]
        if X.empty:
            continue
        doms = [d for d in gc.VIEWS if d in set(X["domain"])]
        L += [f"## {group}", "", "| metodo [variante] | GT | " + " | ".join(doms) + " |", "|---|---|" + "---|" * len(doms)]
        for m, var in order:
            for mode in gc.MODES:
                Y = X[(X["method"] == m) & (X["variant"] == var) & (X["mode"] == mode)]
                if Y.empty:
                    continue
                cells = [fmt(Y[Y["domain"] == d].iloc[0]) if (Y["domain"] == d).any() else "-" for d in doms]
                L.append(f"| {col_name(Y.iloc[0])} | {mode.upper()} | " + " | ".join(cells) + " |")
        L.append("")
    doms = [d for d in gc.VIEWS if d != "famos" and d in set(X0["domain"])]
    sub = ("senza crop", "senza crop, senza noisy") + tuple(f"con {t}" for t in NOCROP)
    for mode in gc.MODES:
        L += [f"## Scomposizione di \"senza crop\" per topologia, GT {mode.upper()} (punto; IC in `spearman.csv`)", "",
              "Colonne: dominio | gruppo (`tutte` = senza crop; `no noisy` = senza crop e senza noisy; `con t` = una "
              "delle due mesh in topologia t, l'altra in una topologia diversa senza crop).", "",
              "| metodo [variante] | dominio | " + " | ".join(s.replace("senza crop, senza noisy", "no noisy")
                                                            .replace("senza crop", "tutte") for s in sub) + " |",
              "|---|---|" + "---|" * len(sub)]
        for m, var in order:
            for d in doms:
                Y = X0[(X0["method"] == m) & (X0["variant"] == var) & (X0["mode"] == mode) & (X0["domain"] == d)]
                if Y.empty:
                    continue
                cells = [f"{Y[Y['group'] == s].iloc[0].point:.3f}" if (Y["group"] == s).any() else "-" for s in sub]
                L.append(f"| {col_name(Y.iloc[0])} | {d} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Costi", ""] + cost_rows() + [""]
    (gc.OUT / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("pilot-gw", "pilot-perc", "spearman", "results"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.step == "pilot-gw":
        pilot_gw()
    elif a.step == "pilot-perc":
        pilot_perc()
    elif a.step == "results":
        write_results(pd.read_csv(gc.OUT / "spearman.csv"))
    else:
        spearman(a.n_boot, a.workers)


if __name__ == "__main__":
    main()
