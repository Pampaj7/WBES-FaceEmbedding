#!/usr/bin/env python3
"""Interazione tardiva: calibrazione di c' (held-out), scelta sul pilota (NON valutati), plug-in e valutazione.

    v3_work/unified_gt/run.sh aau/late_interaction/li_eval.py calib               (c' per seme e variante: held-out)
    v3_work/unified_gt/run.sh aau/late_interaction/li_eval.py pilot               (scelta: GT dei NON valutati)
    v3_work/unified_gt/run.sh aau/late_interaction/li_eval.py plugins             (fm_plugin.json + <vista>.npz)
    v3_work/unified_gt/run.sh aau/late_interaction/li_eval.py eval --workers 32   (LEGGE LE GT dei valutati)
    v3_work/unified_gt/run.sh aau/late_interaction/li_eval.py results             (solo results.md dai CSV)
    (li.sbatch, passi calib, pilot, plugins, eval)

Letture di una variante (``li_tokens.py run``: matrici d_LI per insieme e seme): ``sr`` = d_LI (con la GT SR);
``fr`` = d = sqrt((S_i - S_j)^2 + S_i S_j (c' d_LI)^2) (con la GT FR; POST HOC), S = exp(s) della testa di taglia
dello stesso seme (la forward dei token), c' = mediana(d_P GT-SR) / mediana(d_LI) sulle coppie held-out di
``li_tokens.set_pairs`` (generatori di training BFM, ICT, GNM; GT-SR di training ``c3f/gt_sr.npz`` x dP_per_unit, come
``fact_calib.calib_one``). Nessun dato dei domini di test entra in c'.

``pilot`` (regola di REGOLE.md, sez. 4, scritta prima dei numeri): punteggio di una variante = media sui due semi e sui
due piloti (HIFI3D, FaceScape: i 12 soggetti NON valutati x 5 topologie senza crop, coppie di soggetti diversi) di
(rho(fr, GT FR) + rho(sr, GT SR)) / 2; vince il massimo, entro ``TIE`` la piu' economica (mediana dei secondi per
coppia isolata di ``run.json``, media su semi e piloti). GT dei NON valutati con ``gp_eval.pilot_gt`` (si ferma se un
soggetto e' fra i valutati). Righe di riferimento (descrittive, fuori dalla regola): d_F cal. e d_P dei vettori della
stessa forward sulle stesse coppie.

``plugins``: ``OUT/factorized_<seme>/<variante>/`` con ``<vista>.npz`` (``subjects``, ``topologies``, ``D_fr``,
``D_sr`` sulle 500 mesh senza crop; NaN sulle coppie dello stesso soggetto) e ``fm_plugin.json`` (``declared`` = la
variante scelta sul pilota, entrambi i semi): plug-in di ``aau/full_matrix/fm_eval.py`` (``--plugins OUT``).

``eval``: righe, maschera comune, seme e 1000 repliche per soggetto della matrice (``fm_eval``: ``rows_view``,
``view_columns``, ``plugin_columns``, ``groups_fm``, ``replicas``; maschera ricostruita con le righe di
``fm_eval.run_view``; seme = ``seme`` di ``full_matrix/spearman.csv`` del dominio), insiemi senza crop (``nocrop``, per
perturbazione, ``stessa_topologia senza_crop``). Colonne: le letture delle varianti (due semi), i riferimenti nostri
(d_F cal., d_F non cal., d_P), NoW metrical, REALY, ICP + Chamfer in mm e il miglior concorrente della matrice per
(dominio, GT, insieme) (``migliore_vs_migliore`` di ``full_matrix/deltas.csv``). Delta appaiati = lettura - colonna,
replica per replica (positivo = interazione tardiva davanti); contro d_F / d_P solo lo stesso seme. Controllo: i rho
dei riferimenti e dei concorrenti contro ``full_matrix/spearman.csv`` (punto e IC, stesse righe: scarto atteso 0).
Uscite in ``OUT``: ``calibration.csv``, ``pilot_scores.csv``, ``pilot_choice.json``, ``spearman.csv``, ``deltas.csv``,
``controls.json``, ``results.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import li_common as lc

for _p in (lc.AAU / "full_matrix", lc.AAU / "competitors" / "gw_percettivi"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVT = lc.AAU / "runs" / "evidence" / "trainer_v3"
GT_SR = EVT / "ablations" / "c3f" / "gt_sr.npz"
CALIB = EVT / "factorized_calibration.csv"
FM = lc.AAU / "runs" / "evidence" / "full_matrix"
SETS = [("nocrop", "-")] + [("con_perturbazione", t) for t in lc.NOCROP] + [("stessa_topologia", "senza_crop")]
OTHERS = {"plugin:now_metrical|mm": "NoW metrical", "plugin:realy|mm": "REALY",
          "mm_rigid_icp_chamfer": "ICP + Chamfer in mm"}


def col(seed: str, variant: str, mode: str) -> str:
    return f"plugin:factorized_{seed}/{variant}|{mode}"


def recompose(S: np.ndarray, c: float, D: np.ndarray) -> np.ndarray:
    """d = sqrt((S_i - S_j)^2 + S_i S_j (c D)^2) per una matrice D (M, M)."""
    return np.sqrt((S[:, None] - S[None, :]) ** 2 + S[:, None] * S[None, :] * (c * D) ** 2)


def load(seed: str, name: str) -> tuple[dict, np.ndarray]:
    """({variante: D (M, M)}, S (M,)) di un insieme, con le chiavi controllate fra distanze e meta."""
    with np.load(lc.dist_path(seed, name)) as z, np.load(lc.meta_path(seed, name)) as m:
        if not (np.array_equal(z["subjects"], m["subjects"]) and np.array_equal(z["topologies"], m["topologies"])):
            raise SystemExit(f"{seed} {name}: mesh diverse fra distanze e meta")
        D = {v: np.asarray(z[f"D_{v}"], np.float64) for v in lc.VARIANTS}
        D["_keys"] = list(zip(z["subjects"].astype(str), z["topologies"].astype(str)))
        D["_pairs"] = (np.asarray(z["pair_i"]), np.asarray(z["pair_j"]))
        return D, np.exp(np.asarray(m["Z"], np.float64)[:, 0])


def calibration() -> dict:
    """{seme: {variante: c'}} da ``calibration.csv`` (``calib``)."""
    return {(r["seed"], r["variant"]): float(r["c_median"]) for r in csv.DictReader(open(lc.OUT / "calibration.csv"))}


# ------------------------------------------------------------------------------------------ calibrazione

def step_calib() -> None:
    with np.load(GT_SR, allow_pickle=True) as z:
        G = np.asarray(z["D_orig"], np.float64)
        gpos = {str(n): k for k, n in enumerate(z["names"])}
    dpu_gt = float(json.loads(GT_SR.with_suffix(".json").read_text())["dP_per_unit"])
    ho = json.loads((lc.WORK / "heldout.json").read_text())
    off = {r["key"]: r for r in csv.DictReader(open(CALIB))}
    rows = []
    for seed in lc.SEEDS:
        D, _ = load(seed, "heldout")
        i, j = D["_pairs"]
        subj = np.asarray([s for s, _ in D["_keys"]])
        gen = np.asarray([ho["generator"][s] for s in subj])
        t = G[[gpos[s] for s in subj[i]], [gpos[s] for s in subj[j]]] * dpu_gt
        with np.load(lc.meta_path(seed, "heldout")) as m:
            u = np.asarray(m["Z"], np.float64)[:, 1:]
        ref = off[f"factorized_{seed}"]
        mv = np.linalg.norm(u[i] - u[j], axis=1) * float(ref["dp_per_unit"])
        for v, m_ in [(v, D[v][i, j]) for v in lc.VARIANTS] + [("_vettore_d_P", mv)]:
            r = {"seed": seed, "variant": v, "n_pairs": int(len(i)), "n_nan": int((~np.isfinite(m_)).sum()),
                 "median_dP_gt": float(np.median(t)), "median_d": float(np.nanmedian(m_)),
                 "c_median": float(np.median(t) / np.nanmedian(m_)),
                 "c_ls": float(np.nansum(t * m_) / np.nansum(m_ * m_))}
            for g in ("bfm", "ict", "gnm"):
                k = gen[i] == g
                r[f"c_median_{g}"] = float(np.median(t[k]) / np.nanmedian(m_[k]))
            if v == "_vettore_d_P":
                r["c_ufficiale"] = float(ref["c_median"])
            rows.append(r)
            print(f"[li-eval] calib {seed} {v}: c' {r['c_median']:.5g} (LS {r['c_ls']:.5g}), {len(i)} coppie"
                  + (f"; c ufficiale {r['c_ufficiale']:.5g}" if "c_ufficiale" in r else ""), flush=True)
    pd.DataFrame(rows).to_csv(lc.OUT / "calibration.csv", index=False)


# ------------------------------------------------------------------------------------------ pilota

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def step_pilot() -> None:
    import gp_eval
    cal = calibration()
    dpu = {r["key"]: (float(r["dp_per_unit"]), float(r["c_median"])) for r in csv.DictReader(open(CALIB))}
    rows, cost = [], {}
    for seed in lc.SEEDS:
        rep = json.loads((lc.WORK / "dist" / seed / "run.json").read_text())
        for view in lc.VIEWS:
            name = f"pilot_{view}"
            D, S = load(seed, name)
            i, j = D["_pairs"]
            subj = np.asarray([s for s, _ in D["_keys"]])
            gt = gp_eval.pilot_gt(view, subj, i, j)
            for v in lc.VARIANTS:
                rd = {"sr": D[v][i, j], "fr": recompose(S, cal[(seed, v)], D[v])[i, j]}
                for mode, g in (("fr", "fr"), ("sr", "sr"), ("fr", "sr"), ("sr", "fr")):
                    rows.append({"seed": seed, "view": view, "variant": v, "reading": mode, "gt": g, "n_pairs": len(i),
                                 "rho": spearman(rd[mode], gt[g]), "rule": mode == g})
                cost.setdefault(v, []).append(rep["sets"][name]["timing_isolated"][v]["median_s"])
            with np.load(lc.meta_path(seed, name)) as m:                     # riferimento: i vettori, stesse coppie
                Z = np.asarray(m["Z"], np.float64)
            dp_u, c = dpu[f"factorized_{seed}"]
            dP = np.linalg.norm(Z[i, 1:] - Z[j, 1:], axis=1) * dp_u
            Sv = np.exp(Z[:, 0])
            ref = {"fr": np.sqrt((Sv[i] - Sv[j]) ** 2 + Sv[i] * Sv[j] * (c * dP) ** 2), "sr": dP}
            for mode in ("fr", "sr"):
                rows.append({"seed": seed, "view": view, "variant": "_vettore (d_F cal. / d_P)", "reading": mode,
                             "gt": mode, "n_pairs": len(i), "rho": spearman(ref[mode], gt[mode]), "rule": False})
    R = pd.DataFrame(rows)
    R.to_csv(lc.OUT / "pilot_scores.csv", index=False)
    A = R[R["rule"]]
    score = A.groupby("variant")["rho"].mean()
    sec = {v: float(np.mean(x)) for v, x in cost.items()}
    best = float(score.max())
    tied = sorted([v for v in lc.VARIANTS if score[v] >= best - lc.TIE], key=lambda v: sec[v])
    ch = tied[0]
    ref = R[R["variant"].str.startswith("_")].groupby("reading")["rho"].mean()
    out = {"choice": ch, "score": float(score[ch]), "best": best, "best_variant": str(score.idxmax()), "tied": tied,
           "scores": {v: float(score[v]) for v in lc.VARIANTS}, "s_per_pair_isolated": sec,
           "by_seed_view": {f"{r.seed}|{r.view}|{r.variant}|{r.reading}": r.rho for r in A.itertuples()},
           "reference_vector_mean_rho": {k: float(x) for k, x in ref.items()},
           "rule": "media su semi (s1234, s2345) e piloti (HIFI3D, FaceScape) di (rho(fr, GT FR) + rho(sr, GT SR)) / 2 "
                   f"sulle coppie del pilota; entro {lc.TIE} dalla migliore la mediana dei secondi per coppia isolata "
                   "piu' bassa",
           "pilot_subjects": lc.PILOT_SUBJECTS, "views": list(lc.VIEWS), **lc.stamp()}
    lc.atomic_json(lc.OUT / "pilot_choice.json", out)
    for v in lc.VARIANTS:
        print(f"[li-eval] pilota {v}: punteggio {score[v]:.4f}, {sec[v] * 1e3:.2f} ms per coppia isolata", flush=True)
    print(f"[li-eval] scelta {ch} (migliore {out['best_variant']} {best:.4f}; pari {tied}); vettori: "
          f"{out['reference_vector_mean_rho']}", flush=True)


def choice() -> str:
    return json.loads((lc.OUT / "pilot_choice.json").read_text())["choice"]


# ------------------------------------------------------------------------------------------ plug-in

def step_plugins() -> None:
    ch, cal = choice(), calibration()
    for seed in lc.SEEDS:
        for view in lc.VIEWS:
            D, S = load(seed, view)
            subj = np.asarray([s for s, _ in D["_keys"]])
            topo = np.asarray([t for _, t in D["_keys"]])
            for v in lc.VARIANTS:
                d = lc.plugin_dir(seed, v)
                lc.atomic_savez(d / f"{view}.npz", subjects=subj, topologies=topo,
                                D_fr=recompose(S, cal[(seed, v)], D[v]), D_sr=D[v], S=S)
                dec = v == ch
                lab = f"{lc.LABEL[v]} [factorized {seed}]" + ("" if dec else " (non dichiarata)")
                lc.atomic_json(d / "fm_plugin.json", {
                    "label": lab, "family": "interazione tardiva (feature per vertice del nostro encoder)",
                    "declared": dec, "method": "late_interaction", "variant": v, "seed": seed,
                    "c_prime": cal[(seed, v)], "post_hoc": {"fr": True, "sr": False},
                    "readings": {m: {"label": f"{lab}, {lc.MODE_LABEL[m]}", "gt": [m]} for m in lc.MODES}})
    print(f"[li-eval] plug-in scritti in {lc.OUT} (dichiarata: {ch})", flush=True)


# ------------------------------------------------------------------------------------------ valutazione

def view_frame(view: str, pdirs: list) -> tuple:
    """Colonne, GT, soggetti delle righe, maschera comune e insiemi di ``fm_eval`` per una vista (righe di
    ``fm_eval.run_view``, senza i passi che non servono qui)."""
    import fm_eval as fm
    rows, idx, _ = fm.rows_view(view)
    D, _ = fm.view_columns(view, idx)
    Pd, pmeta, _ = fm.plugin_columns(view, idx.keys, pdirs)
    D.update(Pd)
    ia = np.asarray([idx.pos[k] for k in zip(rows["subject_a"].astype(str), rows["topology_a"].astype(str))])
    ib = np.asarray([idx.pos[k] for k in zip(rows["subject_b"].astype(str), rows["topology_b"].astype(str))])
    cols = {c: Mx[ia, ib] for c, Mx in D.items()}
    gt = {g: rows[f"gt_{g}"].to_numpy(np.float64) for g in fm.GTS}
    subjects = np.array(sorted(set(rows["subject_a"]) | set(rows["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = rows["subject_a"].map(s2i).to_numpy(), rows["subject_b"].map(s2i).to_numpy()
    ta, tb = rows["topology_a"].to_numpy(str), rows["topology_b"].to_numpy(str)
    # maschera comune: le righe di fm_eval.run_view (colonne interne; i plug-in non la cambiano)
    core = [c for c in fm.METHODS if c in cols and np.isfinite(cols[c][sa != sb]).any()]
    gok = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"]) & (sa != sb)
    same = ta == tb
    fin = {c: np.isfinite(cols[c]) for c in core}
    core_same = [c for c in core if fin[c][same & gok].any()]
    mask = (gok & ~same & np.all(list(fin.values()), axis=0)) | \
        (gok & same & np.all([fin[c] for c in core_same] or [same], axis=0))
    groups = {k: g for k, g in fm.groups_fm(ta, tb, mask).items() if k in SETS}
    meta = {**{c: fm.METHODS[c] for c in fm.METHODS}, **pmeta}
    return cols, gt, sa, sb, len(subjects), mask, groups, meta


def step_eval(n_boot: int, workers: int) -> None:
    import fm_eval as fm
    fm.fp.BASELINES.update(fm.bpe2.LABEL)                          # come fm_eval.main
    ch = choice()
    S0 = pd.read_csv(FM / "spearman.csv", low_memory=False)
    S0 = S0[S0["passo"] == "comune"]
    B0 = pd.read_csv(FM / "deltas.csv", low_memory=False)
    B0 = B0[(B0["tipo"] == "migliore_vs_migliore") & (B0["passo"] == "comune")]
    pall, _ = fm.plugin_dirs([fm.PLUGIN_ROOT])
    kinds = fm.plugin_kinds(pall)
    pdirs = [(n, d) for n, d in pall if kinds[n] == "dichiarata"] + fm.plugin_dirs([lc.OUT])[0]
    srows, drows, ctrl = [], [], {"choice": ch, "plugins": [n for n, _ in pdirs]}
    t0 = time.time()
    for view in lc.VIEWS:
        cols, gt, sa, sb, n_subj, mask, groups, meta = view_frame(view, pdirs)
        seed_boot = int(S0[S0["dominio"] == view]["seme"].iloc[0])
        rng = np.random.default_rng(seed_boot)
        counts = [np.ones(n_subj, dtype=np.int64)] + \
            [np.bincount(rng.integers(0, n_subj, n_subj), minlength=n_subj) for _ in range(n_boot)]
        best = {(r.gt, r.insieme_righe, r.perturbazione): r.colonna_b for r in B0[B0["dominio"] == view].itertuples()}
        ours = [f"factorized_{s}|{d}" for s in lc.SEEDS for d in ("form_cal", "form", "shape")]
        li = [col(s, v, m) for s in lc.SEEDS for v in lc.VARIANTS for m in lc.MODES]
        others = list(OTHERS) + sorted(set(best.values()) - set(OTHERS))
        methods = ours + others + li
        miss = [c for c in methods if c not in cols]
        if miss:
            raise SystemExit(f"{view}: colonne assenti {miss}")
        for c in li:                                                   # finite su tutte le righe usate
            for k, g in groups.items():
                if not np.isfinite(cols[c][g]).all():
                    raise SystemExit(f"{view}: {c} non finita su {k}")
        V = fm.replicas({m: cols[m] for m in methods}, gt, sa, sb, counts, groups, methods, workers)
        mi = {m: k for k, m in enumerate(methods)}
        gk = list(groups)
        diffs = []
        for gi, g in enumerate(gk):
            for ti, gtn in enumerate(fm.GTS):
                for m in methods:
                    x = V[:, gi, ti, mi[m]]
                    lo, hi = fm.ci(x)
                    li_m = m.startswith("plugin:factorized_")
                    seed, var, mode = (m.split("/")[0].split("_")[-1], m.split("/")[1].split("|")[0],
                                       m.split("|")[1]) if li_m else ("", "", "")
                    srows.append({"dominio": view, "gt": gtn, "insieme_righe": g[0], "perturbazione": g[1],
                                  "n_righe": len(groups[g]), "n_soggetti": n_subj, "seme": seed_boot, "colonna": m,
                                  "metodo": meta[m][0], "interazione_tardiva": li_m, "seme_modello": seed,
                                  "variante": var, "lettura": mode, "declared": (var == ch) if li_m else True,
                                  "dichiarata": (mode == gtn) if li_m else gtn in meta[m][2],
                                  "post_hoc": (mode == "fr") if li_m else fm.post_hoc(m, view),
                                  "rho": x[0], "ic_lo": lo, "ic_hi": hi})
                    if not li_m:
                        r0 = S0[(S0["dominio"] == view) & (S0["gt"] == gtn) & (S0["insieme_righe"] == g[0]) &
                                (S0["perturbazione"] == g[1]) & (S0["colonna"] == m)]
                        if len(r0):
                            r0 = r0.iloc[0]
                            diffs.append(max(abs(x[0] - r0.rho), abs(lo - r0.ic_lo), abs(hi - r0.ic_hi),
                                             abs(len(groups[g]) - r0.n_righe)))
                for s in lc.SEEDS:
                    for v in lc.VARIANTS:
                        for mode in lc.MODES:
                            a = col(s, v, mode)
                            refs = [f"factorized_{s}|{d}" for d in (("form_cal", "form") if gtn == "fr" else ("shape",))]
                            bc = best.get((gtn, g[0], g[1]))
                            for b, tipo in [(r, "vs_nostro_vettore") for r in refs] + \
                                           [(o, "vs_concorrente") for o in OTHERS] + \
                                           ([(bc, "vs_miglior_concorrente")] if bc else []):
                                d = V[:, gi, ti, mi[a]] - V[:, gi, ti, mi[b]]
                                lo, hi = fm.ci(d)
                                bb = d[1:][np.isfinite(d[1:])]
                                drows.append({"dominio": view, "gt": gtn, "insieme_righe": g[0], "perturbazione": g[1],
                                              "n_righe": len(groups[g]), "seme": seed_boot, "tipo": tipo,
                                              "colonna_a": a, "seme_modello": s, "variante": v, "lettura": mode,
                                              "declared": v == ch, "dichiarata": mode == gtn,
                                              "colonna_b": b, "metodo_b": meta[b][0], "rho_a": V[0, gi, ti, mi[a]],
                                              "rho_b": V[0, gi, ti, mi[b]], "delta": d[0], "ic_lo": lo, "ic_hi": hi,
                                              "p_le0": float((bb <= 0).mean())})
        ctrl[view] = {"rows_mask": int(mask.sum()), "rows_by_set": {f"{a} {b}": len(x) for (a, b), x in groups.items()},
                      "seed": seed_boot, "n_boot": n_boot, "best_competitor": {"|".join(k): c for k, c in best.items()},
                      "reproduction_vs_full_matrix": {"n": len(diffs), "max_abs_diff_point_ci_rows":
                                                      float(max(diffs)) if diffs else None}}
        print(f"[li-eval] {view}: {int(mask.sum())} righe nella maschera, {len(methods)} colonne, seme {seed_boot}, "
              f"riproduzione contro la matrice {ctrl[view]['reproduction_vs_full_matrix']}, {time.time() - t0:.0f}s",
              flush=True)
    S, Dl = pd.DataFrame(srows), pd.DataFrame(drows)
    S.to_csv(lc.OUT / "spearman.csv", index=False)
    Dl.to_csv(lc.OUT / "deltas.csv", index=False)
    lc.atomic_json(lc.OUT / "controls.json", {**ctrl, **lc.stamp()})
    write_results(S, Dl)


# ------------------------------------------------------------------------------------------ risultati

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


def write_results(S: pd.DataFrame, Dl: pd.DataFrame) -> None:
    """results.md dai CSV, dal pilota, dalla calibrazione e (se c'e') dal costo."""
    ch = choice()
    pc = json.loads((lc.OUT / "pilot_choice.json").read_text())
    cal = pd.read_csv(lc.OUT / "calibration.csv")
    sname = {("nocrop", "-"): "nocrop", ("stessa_topologia", "senza_crop"): "stessa topologia senza crop",
             **{("con_perturbazione", t): f"con {t}" for t in lc.NOCROP}}
    L = ["# Interazione tardiva sulle feature per vertice: risultati", "",
         "Generato da `aau/late_interaction/li_eval.py` (non modificare a mano). Regole in `REGOLE.md` (commit "
         "separato, prima dei numeri). Righe, maschera comune, seme e 1000 repliche per soggetto della matrice "
         "(`full_matrix`), solo insiemi senza crop; IC 95% percentile; delta appaiati = interazione tardiva - colonna, "
         "replica per replica (positivo = interazione tardiva davanti). Lettura `fr` = ricomposta con S e c' (POST "
         "HOC), con la GT FR; lettura `sr` = d_LI, con la GT SR.", "",
         f"## Scelta sul pilota (soggetti NON valutati)", "",
         f"Variante dichiarata: **{ch}** ({lc.LABEL[ch]}). Regola: {pc['rule']}.", "",
         "| variante | punteggio | ms per coppia isolata (GPU) |", "|---|---|---|"]
    for v in lc.VARIANTS:
        L.append(f"| {v}{' **(dichiarata)**' if v == ch else ''} | {pc['scores'][v]:.4f} | "
                 f"{pc['s_per_pair_isolated'][v] * 1e3:.2f} |")
    L += ["", f"Riferimento sulle stesse coppie del pilota (vettori della stessa forward, fuori dalla regola): rho "
              f"medio d_F cal. con GT FR {pc['reference_vector_mean_rho'].get('fr', float('nan')):.4f}, d_P con GT SR "
              f"{pc['reference_vector_mean_rho'].get('sr', float('nan')):.4f}.", "",
          "## Calibrazione di c' (held-out dei generatori di training)", "",
          "| seme | variante | coppie | c' (mediana) | c' LS | bfm / ict / gnm |", "|---|---|---|---|---|---|"]
    for r in cal.itertuples():
        extra = f" (c ufficiale {r.c_ufficiale:.4f})" if isinstance(getattr(r, "c_ufficiale", None), float) and \
            np.isfinite(r.c_ufficiale) else ""
        L.append(f"| {r.seed} | {r.variant}{extra} | {r.n_pairs} | {r.c_median:.5g} | {r.c_ls:.5g} | "
                 f"{r.c_median_bfm:.4g} / {r.c_median_ict:.4g} / {r.c_median_gnm:.4g} |")
    L.append("")
    for view in lc.VIEWS:
        x = S[(S["dominio"] == view)]
        if x.empty:
            continue
        sets = [k for k in sname if len(x[(x["insieme_righe"] == k[0]) & (x["perturbazione"] == k[1])])]
        L += [f"## {view}: Spearman (seme delle repliche {int(x['seme'].iloc[0])})", ""]
        for g in ("fr", "sr"):
            L += [f"GT {g.upper()} (grassetto = variante dichiarata):", "",
                  "| colonna | " + " | ".join(sname[k] for k in sets) + " |", "|---|" + "---|" * len(sets)]
            y = x[x["gt"] == g]
            keep = [c for c in dict.fromkeys(y["colonna"]) if not y[y["colonna"] == c]["interazione_tardiva"].iloc[0]
                    or y[y["colonna"] == c]["lettura"].iloc[0] == g]
            for c in keep:
                z = y[y["colonna"] == c]
                b = "**" if bool(z["interazione_tardiva"].iloc[0]) and bool(z["declared"].iloc[0]) else ""
                cells = []
                for k in sets:
                    r = z[(z["insieme_righe"] == k[0]) & (z["perturbazione"] == k[1])]
                    cells.append(f3(r.iloc[0].rho, r.iloc[0].ic_lo, r.iloc[0].ic_hi) if len(r) else "-")
                L.append(f"| {b}{z['metodo'].iloc[0]}{b} | " + " | ".join(cells) + " |")
            L.append("")
        for g in ("fr", "sr"):
            y = Dl[(Dl["dominio"] == view) & (Dl["gt"] == g) & Dl["declared"] & Dl["dichiarata"]]
            if y.empty:
                continue
            L += [f"Delta appaiati della variante dichiarata ({ch}), lettura {g}, GT {g.upper()}:", "",
                  "| seme | contro | " + " | ".join(sname[k] for k in sets) + " |", "|---|---|" + "---|" * len(sets)]
            for (s, tipo, b) in dict.fromkeys(zip(y["seme_modello"], y["tipo"], y["colonna_b"])):
                if tipo == "vs_miglior_concorrente":
                    continue
                z = y[(y["seme_modello"] == s) & (y["colonna_b"] == b)]
                cells = []
                for k in sets:
                    r = z[(z["insieme_righe"] == k[0]) & (z["perturbazione"] == k[1])]
                    cells.append(f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True) if len(r) else "-")
                L.append(f"| {s} | {z['metodo_b'].iloc[0]} | " + " | ".join(cells) + " |")
            for s in lc.SEEDS:
                z = y[(y["seme_modello"] == s) & (y["tipo"] == "vs_miglior_concorrente")]
                cells = []
                for k in sets:
                    r = z[(z["insieme_righe"] == k[0]) & (z["perturbazione"] == k[1])]
                    cells.append(f"{f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True)} ({r.iloc[0].metodo_b})"
                                 if len(r) else "-")
                L.append(f"| {s} | miglior concorrente della matrice | " + " | ".join(cells) + " |")
            L.append("")
    cost = lc.OUT / "costo" / "costo.json"
    if cost.exists():
        c = json.loads(cost.read_text())
        L += ["## Costo per confronto (stesso nodo; misurato)", ""] + c.get("md", []) + [""]
    (lc.OUT / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("calib", "pilot", "plugins", "eval", "results"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args()
    if a.step == "calib":
        step_calib()
    elif a.step == "pilot":
        step_pilot()
    elif a.step == "plugins":
        step_plugins()
    elif a.step == "eval":
        step_eval(a.n_boot, a.workers)
    else:
        write_results(pd.read_csv(lc.OUT / "spearman.csv"), pd.read_csv(lc.OUT / "deltas.csv"))


if __name__ == "__main__":
    main()
