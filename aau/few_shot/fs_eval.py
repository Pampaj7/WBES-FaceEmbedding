#!/usr/bin/env python3
"""Protocollo adattabile: scelta dei concorrenti sulle k identita' di adattamento, valutazione sulle identita' restanti,
curve in k, delta appaiati e k di sorpasso; FaMoS (riconoscimento dell'identita'). Regole in
``aau/runs/evidence/few_shot/REGOLE.md`` (scritte prima dei numeri).

    v3_work/unified_gt/run.sh aau/few_shot/fs_eval.py selftest --workers 8     (nessuna GT di test)
    v3_work/unified_gt/run.sh aau/few_shot/fs_eval.py eval --workers 64        (LEGGE LE GT di test; dopo fs_head fit)
    v3_work/unified_gt/run.sh aau/few_shot/fs_eval.py famos --workers 64       (LEGGE LE GT di test)
    v3_work/unified_gt/run.sh aau/few_shot/fs_eval.py results                  (solo results.md dai CSV)
    (fs.sbatch)

Importati in sola lettura: ``fm_eval`` (righe ``rows_view``, ICP + Chamfer e NICP ``geo_columns``, fit 3DMM
``fit_columns``, colonne dei bracci ``fact_paired.model_columns``), ``fs_head`` (latenti e teste), ``diag`` (Spearman
pesato, IC), ``ei`` e ``famos_eval`` (FaMoS).

Righe (sez. 3): ``nocrop`` della matrice (soggetti diversi, topologie diverse, nessuna crop) con tutte le colonne del
confronto e le GT finite. Adattamento = righe con i due soggetti in A, test = i due nel resto. Per (dominio, k, q): ogni
famiglia di concorrenti sceglie la configurazione col rho di Spearman massimo con la GT sulle righe di adattamento (per
GT, sole letture dichiarate; pari: l'ordine di ``fs_common.GRID``); sulle righe di test, 1000 repliche per soggetto
coi conteggi della matrice (``default_rng(seme T1)`` sui 100 soggetti ordinati) RISTRETTI alle identita' di test,
Spearman pesato c_a c_b (``diag.wranks`` / ``diag.wpearson``). Per (dominio, k): media sulle 5 estrazioni replica per
replica, IC 95% percentile, delta appaiati (nostro adattato - concorrente adattato), k di sorpasso.

Uscite in ``EVID``: ``spearman.csv``, ``deltas.csv``, ``choices.csv``, ``sorpasso.csv``, ``controls.json``,
``draws.json``, ``famos_metrics.csv``, ``famos_deltas.csv``, ``famos_choices.csv``, ``results.md``; repliche in
``WORK/reps.npz`` e ``WORK/famos_reps.npz``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fs_common as fc  # noqa: E402
import fs_head as fh  # noqa: E402
import d3_stats as S  # noqa: E402
import diag  # noqa: E402

FM_DIR = fc.AAU / "runs" / "evidence" / "full_matrix"
MARK = "<!-- da qui in giu': scritto a mano, conservato da fs_eval.py results -->"
READ = {"sr": "shape", "fr": "form"}                    # lettura del nostro per GT: d_h con SR, d_F della testa con FR
BASE = {"sr": "shape", "fr": "form_cal"}                # nostro non adattato: d_P con SR, d_F cal. con FR (matrice)
_J: dict = {}


def grid_columns() -> list[str]:
    return [col for fam in fc.GRID_ALL for _, col, _ in fc.GRID_ALL[fam]]


def declared(col: str) -> tuple[str, ...]:
    for fam in fc.GRID_ALL:
        for _, c, g in fc.GRID_ALL[fam]:
            if c == col:
                return g
    raise KeyError(col)


# ------------------------------------------------------------------------------------------ righe e colonne

def frame(domain: str, info: dict) -> dict:
    """Righe ``nocrop`` della matrice, colonne del confronto, latenti dei bracci sulle mesh delle righe, GT, soggetti."""
    import fm_eval as fm
    fh.patch_fact_paired()
    fm.fp.BASELINES.update(fm.bpe2.LABEL)                                       # come fm_eval.main
    rows, idx, _ = fm.rows_view(domain)
    ta, tb = rows["topology_a"].to_numpy(str), rows["topology_b"].to_numpy(str)
    rows = rows[(ta != tb) & (ta != "crop") & (tb != "crop")].reset_index(drop=True)
    ia = np.asarray([idx.pos[k] for k in zip(rows["subject_a"].astype(str), rows["topology_a"].astype(str))])
    ib = np.asarray([idx.pos[k] for k in zip(rows["subject_b"].astype(str), rows["topology_b"].astype(str))])
    D = {}
    geo = fm.geo_columns(domain, idx)
    D.update({c: geo[c] for c in ("mm_rigid_icp_chamfer", "cs_rigid_icp_chamfer", "mm_nicp_p2tri", "cs_nicp_p2tri")})
    F, fctrl = fm.fit_columns(domain, idx)
    D.update({c: F[c] for c in grid_columns() if c in F})
    arms = fm.fp.model_columns(domain, idx)
    D.update({f"{arm}|{r}": arms[f"{arm}|{r}"] for arm in fc.ARMS for r in ("shape", "form_cal")})
    Pd, _, pmiss = fm.plugin_columns(domain, idx.keys, [(m, fm.PLUGIN_ROOT / m) for m in ("now_metrical", "now_original")])
    if pmiss:
        raise SystemExit(f"{domain}: NoW coi landmark esatti assente ({pmiss})")
    D.update(Pd)
    with np.load(fc.NOW_WORK / f"{domain}.npz") as z:
        o = fm.ordered(fm.keys_of(z), idx)
        for m in ("now_metrical", "now_original"):
            D[f"now_det:{m}"] = np.asarray(z[f"D_{m}"], np.float64)[np.ix_(o, o)]
    cols = {c: M[ia, ib] for c, M in D.items()}
    miss = [c for c in grid_columns() if c not in cols]
    if miss:
        raise SystemExit(f"{domain}: colonne assenti {miss}")
    Zs = fh.store_Z(domain, list(idx.keys))
    for arm, (_, ck) in Zs.items():
        if ck.resolve() != info["ckpt"][arm].resolve():
            raise SystemExit(f"{domain} {arm}: checkpoint {ck}, atteso {info['ckpt'][arm]}")
    gt = {g: rows[f"gt_{g}"].to_numpy(np.float64) for g in fc.GTS}
    subjects = sorted(set(rows["subject_a"].astype(str)) | set(rows["subject_b"].astype(str)))
    if subjects != fh.domain_subjects(domain):
        raise SystemExit(f"{domain}: soggetti delle righe diversi da quelli delle estrazioni")
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = rows["subject_a"].astype(str).map(s2i).to_numpy()
    sb = rows["subject_b"].astype(str).map(s2i).to_numpy()
    need = grid_columns() + [f"{arm}|{r}" for arm in fc.ARMS for r in ("shape", "form_cal")]
    fin = {c: np.isfinite(cols[c]) for c in need}
    mask = np.isfinite(gt["fr"]) & np.isfinite(gt["sr"]) & (sa != sb) & np.all(list(fin.values()), axis=0)
    k1 = mask & np.isfinite(cols["mm_nicp_p2tri"]) & np.isfinite(cols["cs_nicp_p2tri"])
    return {"cols": cols, "gt": gt, "sa": sa, "sb": sb, "ia": ia, "ib": ib, "Z": {a: z for a, (z, _) in Zs.items()},
            "subjects": subjects, "mask": mask, "k1": k1, "n_nocrop": int(len(rows)),
            "dropped": {c: int((~f).sum()) for c, f in fin.items() if (~f).any()}, "fits_ctrl": fctrl}


def head_columns(T: dict, rows: np.ndarray, arm: str, path, dpu: float) -> dict:
    """d_h e d_F della testa sulle righe date."""
    h, c, meta = fh.load_head(path)
    Z = T["Z"][arm]
    Sv, X = np.exp(Z[:, 0]), Z[:, 1:] * dpu
    a, b = T["ia"][rows], T["ib"][rows]
    d = h.dist(X[a], X[b])
    return {"shape": d, "form": np.sqrt((Sv[a] - Sv[b]) ** 2 + Sv[a] * Sv[b] * (c * d) ** 2), "meta": meta}


def counts_of(domain: str, n: int, n_boot: int) -> np.ndarray:
    """(1 + n_boot, n): i conteggi della matrice (``fm_eval.run_view``: default_rng(seme T1) sui soggetti ordinati)."""
    return S.fp_counts(n, fc.BOOT_SEED[domain], n_boot)


# ------------------------------------------------------------------------------------------ repliche

def _rep(task):
    """rho di ogni (colonna, GT) dell'insieme ``key`` per le repliche b0..b1-1 (Spearman pesato c_a c_b)."""
    key, b0, b1 = task
    J = _J[key]
    out = {cg: np.empty(b1 - b0) for cg in J["want"]}
    for n, b in enumerate(range(b0, b1)):
        c = J["counts"][b]
        w = c[J["sa"]] * c[J["sb"]]
        k = w > 0
        wk = w[k]
        rg = {g: diag.wranks(J["gt"][g][k], wk) for g in fc.GTS}
        for col, gts in J["cols"].items():
            r = diag.wranks(J["x"][col][k], wk)
            for g in gts:
                out[(col, g)][n] = diag.wpearson(r, rg[g], wk)
    return key, b0, out


def replicas(keys: list, n_boot: int, workers: int, chunk: int = 50) -> dict:
    """{key: {(colonna, GT): (1 + n_boot,)}}."""
    tasks = [(key, b0, min(b0 + chunk, n_boot + 1)) for key in keys for b0 in range(0, n_boot + 1, chunk)]
    V = {key: {cg: np.empty(n_boot + 1) for cg in _J[key]["want"]} for key in keys}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, b0, out in pool.imap_unordered(_rep, tasks, chunksize=1):
            for cg, x in out.items():
                V[key][cg][b0:b0 + len(x)] = x
    return V


def setj(key, sa, sb, gt, x: dict, cols: dict, counts) -> None:
    """``cols`` = {colonna: GT con cui si legge}."""
    _J[key] = {"sa": sa, "sb": sb, "gt": gt, "x": x, "cols": cols, "counts": counts,
               "want": [(c, g) for c, gs in cols.items() for g in gs]}


# ------------------------------------------------------------------------------------------ eval

def select(T: dict, rows: np.ndarray) -> dict:
    """{(famiglia, GT): (configurazione scelta, colonna, {configurazione: rho su A})} + ("tutte", GT): la migliore fra
    tutte le configurazioni delle tre famiglie di ``GRID`` (descrittiva; ``GRID_DESC`` fuori)."""
    out = {}
    one = np.ones(int(rows.sum()), dtype=np.int64)
    for g in fc.GTS:
        y = T["gt"][g][rows]
        allc = []
        for fam in fc.GRID_ALL:
            rho = {cfg: diag.wspearman(T["cols"][col][rows], y, one) for cfg, col in fc.configs(fam, g)}
            best = max(rho.values())
            cfg = next(c for c, _ in fc.configs(fam, g) if rho[c] == best)        # pari: l'ordine della griglia
            out[(fam, g)] = (cfg, dict(fc.configs(fam, g))[cfg], rho)
            if fam in fc.GRID:
                allc += [(rho[c], fam, c, col) for c, col in fc.configs(fam, g)]
        best = max(r for r, *_ in allc)
        r, fam, cfg, col = next(x for x in allc if x[0] == best)
        out[("tutte", g)] = (f"{fam}:{cfg}", col, {f"{f}:{c}": rr for rr, f, c, _ in allc})
    return out


def step_eval(a) -> None:
    t0 = time.time()
    info = fh.arm_info()
    S0 = pd.read_csv(FM_DIR / "spearman.csv", low_memory=False)
    S0 = S0[(S0["passo"] == "comune") & (S0["insieme_righe"] == "nocrop")]
    frames, ctrl, choices = {}, {"domains": {}}, []
    for d in fc.DOMAINS:
        frames[d] = frame(d, info)
        T = frames[d]
        seed_m = sorted(set(S0[S0["dominio"] == d]["seme"].astype(int)))
        if seed_m != [fc.BOOT_SEED[d]]:
            raise SystemExit(f"{d}: seme delle repliche {fc.BOOT_SEED[d]}, matrice {seed_m}")
        # K-emb: d_P dai latenti = colonna d_P della matrice (stesse righe, a meno della scala)
        kemb = {}
        for arm in fc.ARMS:
            Z = T["Z"][arm]
            u = Z[:, 1:]
            dP = np.linalg.norm(u[T["ia"][T["mask"]]] - u[T["ib"][T["mask"]]], axis=1) * info["dpu"][arm]
            ref = T["cols"][f"{arm}|shape"][T["mask"]]
            kemb[arm] = float(np.max(np.abs(dP - ref) / ref))
        # K-GT: la G del blocco (GT SR dei soggetti) proporzionale alla gt_sr delle righe
        G = fh.gt_sub(d, T["subjects"])
        rr = G[T["sa"][T["mask"]], T["sb"][T["mask"]]] / T["gt"]["sr"][T["mask"]]
        ctrl["domains"][d] = {"rows_nocrop": T["n_nocrop"], "rows_mask": int(T["mask"].sum()),
                              "rows_k1": int(T["k1"].sum()), "dropped_by_column": T["dropped"], "seed": fc.BOOT_SEED[d],
                              "K_emb_max_rel_diff_dP": kemb,
                              "K_gt_block_over_rows_ratio": {"min": float(rr.min()), "max": float(rr.max())},
                              "K_gt_pass": bool(np.ptp(rr) <= 1e-9 * abs(rr.mean())),
                              "fits_nocrop_vs_saved_max_abs": T["fits_ctrl"]}
        print(f"[fs-eval] {d}: {T['n_nocrop']} righe nocrop, {int(T['mask'].sum())} nella maschera, K1 "
              f"{int(T['k1'].sum())}; K-emb {kemb}; ({time.time() - t0:.0f}s)", flush=True)
    # ------------------------------------------------------------------ insiemi: K1 e (dominio, k, q)
    keys, meta = [], {}
    for d, T in frames.items():
        counts = counts_of(d, len(T["subjects"]), a.n_boot)
        base = {f"{arm}|{BASE[g]}": (g,) for arm in fc.ARMS for g in fc.GTS}
        gridc = {c: declared(c) for c in grid_columns()}
        r = T["k1"]
        setj(("K1", d), T["sa"][r], T["sb"][r], {g: v[r] for g, v in T["gt"].items()},
             {c: T["cols"][c][r] for c in list(base) + list(gridc)}, {**base, **gridc}, counts)
        keys.append(("K1", d))
        for k in fc.KS:
            for q in range(fc.R_DRAWS):
                A, test = fc.drawn(d, k, q)
                ad = T["mask"] & np.isin(T["sa"], A) & np.isin(T["sb"], A)
                te = T["mask"] & np.isin(T["sa"], test) & np.isin(T["sb"], test)
                sel = select(T, ad)
                for (fam, g), (cfg, col, rho) in sel.items():
                    choices.append({"dominio": d, "k": k, "q": q, "famiglia": fam, "gt": g, "scelta": cfg,
                                    "colonna": col, "rho_adattamento": rho[cfg], "righe_adattamento": int(ad.sum()),
                                    **{f"rho_A[{c}]": v for c, v in rho.items()}})
                x = {c: T["cols"][c][te] for c in list(base) + list(gridc)}
                cols = {**base, **gridc}
                hm = {}
                for arm in fc.ARMS:
                    for kk, tag in ((k, "head"), (0, "k0")):
                        hc = head_columns(T, te, arm, fc.head_path(arm, d, kk, q), info["dpu"][arm])
                        if kk and not np.array_equal(hc["meta"]["subjects"], A):
                            raise SystemExit(f"{d} k={k} q={q} {arm}: testa con soggetti diversi dall'estrazione")
                        hm[(arm, tag)] = {"success": hc["meta"]["success"], "nit": hc["meta"]["nit"],
                                          "host": hc["meta"]["host"]}
                        for g in fc.GTS:
                            x[f"{arm}|{tag}|{READ[g]}"] = hc[READ[g]]
                            cols[f"{arm}|{tag}|{READ[g]}"] = (g,)
                bad = [c for c, v in x.items() if not np.isfinite(v).all()]
                if bad:
                    raise SystemExit(f"{d} k={k} q={q}: colonne non finite {bad[:3]}")
                setj((d, k, q), T["sa"][te], T["sb"][te], {g: v[te] for g, v in T["gt"].items()}, x, cols, counts)
                keys.append((d, k, q))
                meta[(d, k, q)] = {"sel": sel, "rows_test": int(te.sum()), "rows_ad": int(ad.sum()),
                                   "n_test": len(test), "heads": hm}
    print(f"[fs-eval] {len(keys)} insiemi, repliche... ({time.time() - t0:.0f}s)", flush=True)
    V = replicas(keys, a.n_boot, a.workers)
    print(f"[fs-eval] repliche finite ({time.time() - t0:.0f}s)", flush=True)
    # ------------------------------------------------------------------ K1: contro full_matrix/spearman.csv
    k1, worst = {}, 0.0
    for d in fc.DOMAINS:
        for (c, g), x in V[("K1", d)].items():
            r0 = S0[(S0["dominio"] == d) & (S0["gt"] == g) & (S0["colonna"] == c)]
            if not len(r0):
                continue
            r0 = r0.iloc[0]
            p, lo, hi, _ = diag.ci(x)
            dd = max(abs(p - r0.rho), abs(lo - r0.ic_lo), abs(hi - r0.ic_hi))
            k1[f"{d}|{c}|{g}"] = {"diff": dd, "rows": int(frames[d]["k1"].sum()), "rows_matrix": int(r0.n_righe)}
            worst = max(worst, dd)
    rows_ok = all(v["rows"] == v["rows_matrix"] for v in k1.values())
    ctrl["K1"] = {"cells": k1, "max_abs_diff": worst, "rows_equal": rows_ok, "pass": bool(worst <= 1e-9 and rows_ok)}
    print(f"[fs-eval] K1 contro la matrice: {worst:.1e}, righe uguali {rows_ok}", flush=True)
    # ------------------------------------------------------------------ aggregati
    srecs, drecs, sorp = [], [], []
    AG = {}

    def agg(d, k, fn) -> tuple[np.ndarray, np.ndarray]:
        """(media sulle estrazioni replica per replica, valori puntuali per estrazione)."""
        X = np.stack([fn(q) for q in range(fc.R_DRAWS)])
        return X.mean(0), X[:, 0]

    def chosen(d, k, q, fam, g):
        return V[(d, k, q)][(meta[(d, k, q)]["sel"][(fam, g)][1], g)]

    for d in fc.DOMAINS:
        for k in fc.KS:
            nt = float(np.mean([meta[(d, k, q)]["rows_test"] for q in range(fc.R_DRAWS)]))
            for g in fc.GTS:
                series = {}
                for arm in fc.ARMS:
                    series[("nostro", arm)] = agg(d, k, lambda q: V[(d, k, q)][(f"{arm}|head|{READ[g]}", g)])
                    series[("nostro_k0", arm)] = agg(d, k, lambda q: V[(d, k, q)][(f"{arm}|k0|{READ[g]}", g)])
                    series[("nostro_base", arm)] = agg(d, k, lambda q: V[(d, k, q)][(f"{arm}|{BASE[g]}", g)])
                for fam in list(fc.GRID_ALL) + ["tutte"]:
                    series[("concorrente", fam)] = agg(d, k, lambda q: chosen(d, k, q, fam, g))
                for c in grid_columns():
                    if g in declared(c):
                        series[("config_fissa", c)] = agg(d, k, lambda q: V[(d, k, q)][(c, g)])
                for (kind, who), (x, pts) in series.items():
                    p, lo, hi, _ = diag.ci(x)
                    srecs.append({"dominio": d, "k": k, "gt": g, "tipo": kind, "colonna": who, "rho": p, "ic_lo": lo,
                                  "ic_hi": hi, "sd_estrazioni": float(pts.std(ddof=1)),
                                  "estrazioni": " ".join(f"{v:.4f}" for v in pts), "righe_test_media": nt,
                                  "soggetti_test": fc.N_SUBJ[d] - k})
                    AG[(d, k, g, kind, who)] = (x, pts)
                best = max(fc.GRID, key=lambda f: series[("concorrente", f)][0][0])
                for arm in fc.ARMS:
                    xa, pa = series[("nostro", arm)]
                    for contro, (xb, pb) in [(f, series[("concorrente", f)]) for f in fc.GRID_ALL] + \
                            [("migliore", series[("concorrente", best)]), ("tutte", series[("concorrente", "tutte")]),
                             ("nostro_base", series[("nostro_base", arm)]), ("nostro_k0", series[("nostro_k0", arm)])]:
                        p, lo, hi, ple = diag.ci(xa - xb)
                        drecs.append({"dominio": d, "k": k, "gt": g, "braccio": arm, "contro": contro,
                                      "famiglia_migliore": best if contro == "migliore" else "", "delta": p, "ic_lo": lo,
                                      "ic_hi": hi, "p_le0": ple, "rho_nostro": xa[0], "rho_contro": xb[0],
                                      "sd_estrazioni": float((pa - pb).std(ddof=1)),
                                      "estrazioni": " ".join(f"{v:+.4f}" for v in pa - pb)})
    D_ = pd.DataFrame(drecs)
    for d in fc.DOMAINS:
        for g in fc.GTS:
            ks = {}
            for arm in fc.ARMS:
                ok = []
                for k in fc.KS:
                    x = D_[(D_["dominio"] == d) & (D_["gt"] == g) & (D_["braccio"] == arm) & (D_["k"] == k) &
                           D_["contro"].isin(list(fc.GRID))]
                    ok.append(bool((x["ic_lo"] > 0).all() and len(x) == len(fc.GRID)))
                kmin = next((k for k, o in zip(fc.KS, ok) if o), None)
                ks[arm] = kmin
                sorp.append({"dominio": d, "gt": g, "braccio": arm, "k_sorpasso": kmin if kmin else "nessuno",
                             "sopra_a_k": " ".join(str(k) for k, o in zip(fc.KS, ok) if o),
                             "stabile_dopo": bool(kmin is not None and all(o for k, o in zip(fc.KS, ok) if k >= kmin))})
            both = max(ks.values()) if all(v is not None for v in ks.values()) else None
            sorp.append({"dominio": d, "gt": g, "braccio": "entrambi", "k_sorpasso": both if both else "nessuno",
                         "sopra_a_k": "", "stabile_dopo": ""})
    # ------------------------------------------------------------------ uscite
    fc.EVID.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(srecs).to_csv(fc.EVID / "spearman.csv", index=False)
    D_.to_csv(fc.EVID / "deltas.csv", index=False)
    pd.DataFrame(choices).to_csv(fc.EVID / "choices.csv", index=False)
    pd.DataFrame(sorp).to_csv(fc.EVID / "sorpasso.csv", index=False)
    heads = [m["heads"] for m in meta.values()]
    ctrl.update(draws=fc.check_draws(), n_boot=a.n_boot, workers=a.workers, seconds=time.time() - t0,
                heads={"n_read": sum(len(h) for h in heads),
                       "success": sum(x["success"] for h in heads for x in h.values()),
                       "hosts": sorted({x["host"] for h in heads for x in h.values()})},
                rows_by_set={f"{d}|{k}|{q}": {"test": m["rows_test"], "adattamento": m["rows_ad"]}
                             for (d, k, q), m in meta.items()}, **fc.stamp())
    fc.atomic_json(fc.EVID / "controls.json", ctrl)
    fc.atomic_json(fc.EVID / "draws.json", fc.draws_record({d: frames[d]["subjects"] for d in fc.DOMAINS}))
    fc.atomic_savez(fc.WORK / "reps.npz", **{f"{key}|{c}|{g}": x for key, Vk in V.items() for (c, g), x in Vk.items()})
    write_results()
    print(f"[fs-eval] finito ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------ FaMoS

_F: dict = {}


def _famos_task(task):
    """(k, q): scelte sulle scansioni di adattamento e metriche delle colonne sulle scansioni di test, per replica."""
    import ei
    import famos_eval as fe
    k, q = task
    F = _F
    A, test = fc.drawn(fc.FAMOS, k, q)
    sa_, st_ = np.isin(F["subj"], A), np.isin(F["subj"], test)
    out = {"choices": {}, "metrics": {}}
    for fam, grid in fc.GRID_FAMOS.items():                                       # scelta: mAP sulle scansioni di A
        m = {}
        for cfg, col in grid:
            _, ap = ei.retrieval(F["D"][col][np.ix_(sa_, sa_)], F["subj"][sa_])
            m[cfg] = float(ap.mean())
        best = max(m.values())
        cfg = next(c for c, _ in grid if m[c] == best)
        out["choices"][fam] = (cfg, dict(grid)[cfg], m)
    cols = dict(F["base"])
    for arm in fc.ARMS:
        for kk, tag in ((k, "head"), (0, "k0")):
            h, c, meta = fh.load_head(fc.head_path(arm, fc.FAMOS, kk, q))
            if kk and not np.array_equal(meta["subjects"], A):
                raise SystemExit(f"FaMoS k={k} q={q} {arm}: testa con soggetti diversi dall'estrazione")
            Sv, X = F["S"][arm], F["X"][arm]
            dh = np.zeros((len(Sv), len(Sv)))
            iu, ju = np.triu_indices(len(Sv), 1)
            d = h.dist(X[iu], X[ju])
            dh[iu, ju] = dh[ju, iu] = d
            dF = np.sqrt((Sv[:, None] - Sv[None]) ** 2 + Sv[:, None] * Sv[None] * (c * dh) ** 2)
            cols[f"{arm}|{tag}|form"], cols[f"{arm}|{tag}|shape"] = dF, dh
    for _, grid in fc.GRID_FAMOS.items():
        for _, col in grid:
            cols[col] = F["D"][col]
    subj = F["subj"][st_]
    n = len(subj)
    iu, ju = np.triu_indices(n, 1)
    gen = subj[iu] == subj[ju]
    counts = F["counts"]
    for col, Dm in cols.items():
        Dt = Dm[np.ix_(st_, st_)]
        r1, ap = ei.retrieval(Dt, subj)
        wq = counts[:, subj].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            res = {"rank1": (wq @ r1) / wq.sum(1), "mAP": (wq @ ap) / wq.sum(1)}
        res["AUC"], res["EER"] = np.full(len(counts), np.nan), np.full(len(counts), np.nan)
        dv = Dt[iu, ju]
        for b, c in enumerate(counts):
            w = np.where(gen, c[subj[iu]], c[subj[iu]] * c[subj[ju]]).astype(np.float64)
            if (w[gen] > 0).any() and (w[~gen] > 0).any():
                res["AUC"][b] = fe.weighted_auc(-dv, gen, w)
                res["EER"][b] = ei.eer(dv, gen, w)
        out["metrics"][col] = res
    out["n_test_scans"] = n
    return task, out


def step_famos(a) -> None:
    import ei
    import famos_eval as fe
    t0 = time.time()
    info = fh.arm_info()
    cols, it, failed = ei.load_columns("eval")
    names = [p.stem for _, _, p in it]
    subjects = sorted({x for x, _, _ in it})
    if subjects != fh.domain_subjects(fc.FAMOS):
        raise SystemExit("FaMoS: soggetti diversi da quelli delle estrazioni")
    subj = np.asarray([subjects.index(x) for x, _, _ in it])
    n = len(it)
    iu, ju = ei.pairs(n)

    def full(v):
        M = np.zeros((n, n))
        M[iu, ju] = M[ju, iu] = v
        return M

    _F.update(subj=subj, D={col: full(cols[col]) for g in fc.GRID_FAMOS.values() for _, col in g},
              base={f"{arm}|base|{r}": full(cols[f"{arm}|{r}"]) for arm in fc.ARMS for r in ("form", "form_cal", "shape")},
              counts=np.concatenate([np.ones((1, len(subjects)), np.int64),
                                     fe.bootstrap_counts(len(subjects), a.n_boot, fc.FAMOS_BOOT_SEED)]), S={}, X={})
    for arm in fc.ARMS:
        with np.load(fh.FAMOS_EMB / f"{arm}.npz") as z:
            if [str(x) for x in z["names"]] != names:
                raise SystemExit(f"FaMoS {arm}: latenti con scansioni diverse")
            Z = np.asarray(z["Z"], np.float64)
        _F["S"][arm], _F["X"][arm] = np.exp(Z[:, 0]), Z[:, 1:] * info["dpu"][arm]
        # K-emb: la d_P dei latenti = la d_P di esterna_famos_id
        dP = np.linalg.norm(_F["X"][arm][iu] - _F["X"][arm][ju], axis=1)
        kemb = float(np.max(np.abs(dP - cols[f"{arm}|shape"]) / cols[f"{arm}|shape"]))
        _F.setdefault("kemb", {})[arm] = kemb
    tasks = [(k, q) for k in fc.KS_FAMOS for q in range(fc.R_DRAWS)]
    with mp.get_context("fork").Pool(min(a.workers, len(tasks))) as pool:
        res = dict(pool.map(_famos_task, tasks, chunksize=1))
    print(f"[fs-famos] {len(tasks)} estrazioni ({time.time() - t0:.0f}s)", flush=True)
    mrecs, drecs, crecs, reps = [], [], [], {}

    def ci(x):
        p, lo, hi, ple = diag.ci(x)
        return p, lo, hi, ple

    for k in fc.KS_FAMOS:
        R = [res[(k, q)] for q in range(fc.R_DRAWS)]
        for q, r in enumerate(R):
            for fam, (cfg, col, m) in r["choices"].items():
                crecs.append({"k": k, "q": q, "famiglia": fam, "scelta": cfg, "colonna": col,
                              **{f"mAP_A[{c}]": v for c, v in m.items()}})
        series = {}
        for arm in fc.ARMS:
            for tag in ("head", "k0"):
                for rd in ("form", "shape"):
                    series[(f"nostro_{tag}_{rd}", arm)] = f"{arm}|{tag}|{rd}"
            for rd in ("form", "form_cal", "shape"):
                series[(f"nostro_base_{rd}", arm)] = f"{arm}|base|{rd}"
        for fam, grid in fc.GRID_FAMOS.items():
            for cfg, col in grid:
                series[("config_fissa", col)] = col
        for met in fc.FAMOS_METRICS:
            sgn = -1.0 if met == "EER" else 1.0
            vals = {}
            for key, col in series.items():
                X = np.stack([r["metrics"][col][met] for r in R])
                vals[key] = X
            for fam in fc.GRID_FAMOS:
                vals[("concorrente", fam)] = np.stack([r["metrics"][r["choices"][fam][1]][met] for r in R])
            for (kind, who), X in vals.items():
                x = X.mean(0)
                p, lo, hi, _ = ci(x)
                mrecs.append({"k": k, "metrica": met, "tipo": kind, "colonna": who, "valore": p, "ic_lo": lo,
                              "ic_hi": hi, "sd_estrazioni": float(X[:, 0].std(ddof=1)),
                              "estrazioni": " ".join(f"{v:.4f}" for v in X[:, 0]),
                              "scansioni_test": R[0]["n_test_scans"], "soggetti_test": fc.N_SUBJ[fc.FAMOS] - k,
                              "repliche_non_finite": int((~np.isfinite(x[1:])).sum())})
                reps[f"{k}|{met}|{kind}|{who}"] = x
            best = max(fc.GRID_FAMOS, key=lambda f: sgn * vals[("concorrente", f)].mean(0)[0])
            for arm in fc.ARMS:
                xa = vals[("nostro_head_form", arm)]
                for contro, xb in [(f, vals[("concorrente", f)]) for f in fc.GRID_FAMOS] + \
                        [("migliore", vals[("concorrente", best)]), ("nostro_base_form", vals[("nostro_base_form", arm)]),
                         ("nostro_k0_form", vals[("nostro_k0_form", arm)])]:
                    dlt = sgn * (xa - xb)
                    p, lo, hi, ple = ci(dlt.mean(0))
                    drecs.append({"k": k, "metrica": met, "braccio": arm, "contro": contro,
                                  "famiglia_migliore": best if contro == "migliore" else "", "delta": p, "ic_lo": lo,
                                  "ic_hi": hi, "p_le0": ple, "sd_estrazioni": float(dlt[:, 0].std(ddof=1)),
                                  "estrazioni": " ".join(f"{v:+.4f}" for v in dlt[:, 0]),
                                  "verso": "positivo = il nostro meglio" + (" (EER: concorrente - nostro)" if sgn < 0 else "")})
    pd.DataFrame(mrecs).to_csv(fc.EVID / "famos_metrics.csv", index=False)
    pd.DataFrame(drecs).to_csv(fc.EVID / "famos_deltas.csv", index=False)
    pd.DataFrame(crecs).to_csv(fc.EVID / "famos_choices.csv", index=False)
    fc.atomic_savez(fc.WORK / "famos_reps.npz", **reps)
    fc.atomic_json(fc.EVID / "famos_controls.json", {
        "n_scans": n, "n_subjects": len(subjects), "failed_pairs": failed, "K_emb_max_rel_diff_dP": _F["kemb"],
        "boot": f"famos_eval.bootstrap_counts({len(subjects)}, {a.n_boot}, {fc.FAMOS_BOOT_SEED}) ristretti ai soggetti "
                "di test", "draws": fc.draws_record({fc.FAMOS: subjects})[fc.FAMOS], "seconds": time.time() - t0,
        **fc.stamp()})
    write_results()
    print(f"[fs-famos] finito ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------ selftest

def step_selftest(a) -> None:
    """Nessuna GT dei soggetti di test letta per un metodo: estrazioni, blocchi, checkpoint, soggetti, file."""
    t0 = time.time()
    info = fh.arm_info()
    rep = {"draws": fc.check_draws(), "blocks": {}}
    for arm in fc.ARMS:
        for d in fc.DOMAINS + (fc.FAMOS,):
            b = fh.domain_block(arm, d, info)
            A, _ = fc.drawn(d, fc.ks_of(d)[0], 0)
            bt = b.take(A)
            rep["blocks"][f"{arm}|{d}"] = {"meshes": int(len(b.X)), "subjects": int(len(b.G)), "p": int(b.X.shape[1]),
                                           "labels": int(len(np.unique(b.lab))), "k_block_meshes": int(len(bt.X)),
                                           "k_block_subjects": int(len(np.unique(bt.subj)))}
    rep["sources"] = {d: list(S.settings()[f"{fc.TARGET[d]}|all"]) for d in fc.DOMAINS + (fc.FAMOS,)}
    rep["now_det_files"] = {d: (fc.NOW_WORK / f"{d}.npz").exists() for d in fc.DOMAINS}
    rep.update(seconds=time.time() - t0, **fc.stamp())
    fc.atomic_json(fc.EVID / "selftest.json", rep)
    print(f"[fs-selftest] {json.dumps(rep['blocks'])}; NoW {rep['now_det_files']} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------ results.md

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


def write_results() -> None:
    L = ["# Protocollo adattabile (few-shot): risultati", "",
         "Generato da `aau/few_shot/fs_eval.py` (non modificare a mano sopra il segno). Regole in `REGOLE.md` (commit "
         "separato, prima dei numeri). Per (dominio, k): media sulle 5 estrazioni; IC 95% percentile con 1000 repliche "
         "per soggetto ristrette alle identita' di test (stesso ricampionamento per tutte le estrazioni); sd = "
         "deviazione standard fra le estrazioni. Nostro adattato = testa di R6 (r = 64, lambda = 1e-6, EQ) con le k "
         "identita' in coda alle 7 sorgenti senza il generatore del dominio; letture d_h con SR, d_F della testa con FR. "
         "Concorrente adattato = configurazione della famiglia scelta sulle stesse k identita'. Delta = nostro - "
         "concorrente, replica per replica (positivo = il nostro davanti).", ""]
    if (fc.EVID / "deltas.csv").exists():
        Sp = pd.read_csv(fc.EVID / "spearman.csv")
        Dl = pd.read_csv(fc.EVID / "deltas.csv")
        So = pd.read_csv(fc.EVID / "sorpasso.csv")
        Ch = pd.read_csv(fc.EVID / "choices.csv")
        ct = json.loads((fc.EVID / "controls.json").read_text())
        L += ["## k di sorpasso", "",
              "Il k minimo della griglia al quale l'IC 95% del delta contro OGNUNA delle tre famiglie adattate sta sopra "
              "lo 0 (sez. 6 delle regole); `entrambi` = il massimo dei due semi, `nessuno` se un seme non sorpassa.", "",
              "| dominio | GT | s1234 (k sopra) | s2345 (k sopra) | entrambi |", "|---|---|---|---|---|"]
        for d in fc.DOMAINS:
            for g in fc.GTS:
                x = So[(So["dominio"] == d) & (So["gt"] == g)].set_index("braccio")
                cell = lambda arm: f"{x.loc[arm, 'k_sorpasso']} ({x.loc[arm, 'sopra_a_k'] if isinstance(x.loc[arm, 'sopra_a_k'], str) else '-'})"  # noqa: E731
                L.append(f"| {fc.NAMES[d]} | {g.upper()} | {cell('factorized_s1234')} | {cell('factorized_s2345')} | "
                         f"{x.loc['entrambi', 'k_sorpasso']} |")
        L.append("")
        for d in fc.DOMAINS:
            L += [f"## {fc.NAMES[d]}", ""]
            for g in fc.GTS:
                L += [f"**GT {g.upper()}: rho sulle identita' di test**", "",
                      "| colonna | " + " | ".join(f"k = {k}" for k in fc.KS) + " |", "|---|" + "---|" * len(fc.KS)]
                y = Sp[(Sp["dominio"] == d) & (Sp["gt"] == g)]
                order = [("nostro", a) for a in fc.ARMS] + [("nostro_base", a) for a in fc.ARMS] + \
                    [("nostro_k0", a) for a in fc.ARMS] + [("concorrente", f) for f in list(fc.GRID_ALL) + ["tutte"]]
                lab = {"nostro": "nostro adattato", "nostro_base": "nostro non adattato", "nostro_k0": "testa k = 0",
                       "concorrente": "concorrente adattato"}
                for kind, who in order:
                    z = y[(y["tipo"] == kind) & (y["colonna"] == who)].set_index("k")
                    name = f"{lab[kind]} {fc.ARM_NAMES.get(who, fc.FAMILY_NAMES.get(who, 'migliore fra tutte (scelto su A)'))}"
                    L.append(f"| {name} | " + " | ".join(f3(z.loc[k, 'rho'], z.loc[k, 'ic_lo'], z.loc[k, 'ic_hi']) +
                                                        f" sd {z.loc[k, 'sd_estrazioni']:.3f}" for k in fc.KS) + " |")
                L += ["", f"**GT {g.upper()}: delta nostro adattato - contro**", "",
                      "| seme | contro | " + " | ".join(f"k = {k}" for k in fc.KS) + " |", "|---|---|" + "---|" * len(fc.KS)]
                y = Dl[(Dl["dominio"] == d) & (Dl["gt"] == g)]
                for arm in fc.ARMS:
                    for contro in list(fc.GRID_ALL) + ["migliore", "tutte", "nostro_base", "nostro_k0"]:
                        z = y[(y["braccio"] == arm) & (y["contro"] == contro)].set_index("k")
                        nm = {"migliore": "miglior famiglia (post hoc)", "tutte": "migliore fra tutte (scelto su A)",
                              "nostro_base": "nostro non adattato", "nostro_k0": "testa k = 0"}.get(
                            contro, fc.FAMILY_NAMES.get(contro, contro))
                        cells = []
                        for k in fc.KS:
                            c = f3(z.loc[k, "delta"], z.loc[k, "ic_lo"], z.loc[k, "ic_hi"], True)
                            if contro == "migliore":
                                c += f" ({fc.FAMILY_NAMES[z.loc[k, 'famiglia_migliore']]})"
                            cells.append(c)
                        L.append(f"| {fc.ARM_NAMES[arm]} | {nm} | " + " | ".join(cells) + " |")
                L += ["", f"Configurazioni scelte sulle k identita' (GT {g.upper()}, estrazioni per configurazione):", ""]
                y = Ch[(Ch["dominio"] == d) & (Ch["gt"] == g)]
                for fam in list(fc.GRID_ALL) + ["tutte"]:
                    parts = []
                    for k in fc.KS:
                        vc = y[(y["famiglia"] == fam) & (y["k"] == k)]["scelta"].value_counts()
                        parts.append(f"k = {k}: " + ", ".join(f"{c} {n}" for c, n in vc.items()))
                    L.append(f"- {fc.FAMILY_NAMES.get(fam, 'tutte le famiglie')}: " + "; ".join(parts))
                L.append("")
        k1 = ct.get("K1", {})
        L += ["## Controlli", "",
              f"- K1 (rho sulle righe della matrice, tutti i 100 soggetti, contro `full_matrix/spearman.csv`): scarto "
              f"massimo {k1.get('max_abs_diff', float('nan')):.1e}, righe uguali {k1.get('rows_equal')}, passa "
              f"{k1.get('pass')}.",
              "- Righe: " + "; ".join(f"{fc.NAMES[d]} {x['rows_mask']} di {x['rows_nocrop']} nocrop (K1 {x['rows_k1']})"
                                      for d, x in ct["domains"].items()) + ".",
              "- K-emb (d_P dai latenti contro la colonna della matrice, scarto relativo massimo): " +
              "; ".join(f"{d} {max(x['K_emb_max_rel_diff_dP'].values()):.1e}" for d, x in ct["domains"].items()) +
              "; K-GT (G del blocco proporzionale alla GT delle righe): " +
              ", ".join(f"{d} {x['K_gt_pass']}" for d, x in ct["domains"].items()) + ".",
              f"- Teste lette: {ct['heads']['success']} / {ct['heads']['n_read']} con success, nodi "
              f"{', '.join(ct['heads']['hosts'])}.", ""]
        hj = fc.EVID / "heads.json"
        if hj.exists():
            h = json.loads(hj.read_text())
            L += [f"- Fit: {h['success']} / {h['n']} con success, iterazioni {h['nit_min']}-{h['nit_max']}, nodi "
                  f"{', '.join(h['hosts'])}; K0 (testa joint di R6 contro d3_e2_heads.csv): " +
                  "; ".join(f"{fc.ARM_NAMES[a_]} c {x['rel_c']:.1e}, alpha {x['rel_alpha']:.1e}, passa {x['pass']}"
                            for a_, x in h["K0"].items()) + ".", ""]
    if (fc.EVID / "famos_deltas.csv").exists():
        M = pd.read_csv(fc.EVID / "famos_metrics.csv")
        Dd = pd.read_csv(fc.EVID / "famos_deltas.csv")
        L += ["## FaMoS TEST: riconoscimento dell'identita' (descrittivo, limiti in REGOLE.md sez. 7)", "",
              "Scansioni dei soggetti di test (galleria = le altre scansioni di test); nostro = d_F della testa adattata; "
              "concorrenti = configurazione scelta col mAP sulle scansioni di adattamento. Fit 3DMM vB assente su FaMoS.",
              "", "| colonna | metrica | " + " | ".join(f"k = {k}" for k in fc.KS_FAMOS) + " |",
              "|---|---|" + "---|" * len(fc.KS_FAMOS)]
        keep = [("nostro_head_form", a) for a in fc.ARMS] + [("nostro_base_form", a) for a in fc.ARMS] + \
            [("concorrente", f) for f in fc.GRID_FAMOS]
        for met in fc.FAMOS_METRICS:
            for kind, who in keep:
                z = M[(M["metrica"] == met) & (M["tipo"] == kind) & (M["colonna"] == who)].set_index("k")
                L.append(f"| {kind} {fc.ARM_NAMES.get(who, fc.FAMILY_NAMES.get(who, who))} | {met} | " +
                         " | ".join(f3(z.loc[k, 'valore'], z.loc[k, 'ic_lo'], z.loc[k, 'ic_hi']) for k in fc.KS_FAMOS) + " |")
        L += ["", "| seme | metrica | contro | " + " | ".join(f"k = {k}" for k in fc.KS_FAMOS) + " |",
              "|---|---|---|" + "---|" * len(fc.KS_FAMOS)]
        for met in fc.FAMOS_METRICS:
            for arm in fc.ARMS:
                for contro in list(fc.GRID_FAMOS) + ["migliore", "nostro_base_form"]:
                    z = Dd[(Dd["metrica"] == met) & (Dd["braccio"] == arm) & (Dd["contro"] == contro)].set_index("k")
                    L.append(f"| {fc.ARM_NAMES[arm]} | {met} | {fc.FAMILY_NAMES.get(contro, contro)} | " +
                             " | ".join(f3(z.loc[k, 'delta'], z.loc[k, 'ic_lo'], z.loc[k, 'ic_hi'], True) for k in fc.KS_FAMOS)
                             + " |")
        L.append("")
    out = fc.EVID / "results.md"
    tail = ""
    if out.exists() and MARK in out.read_text():
        tail = out.read_text().split(MARK, 1)[1]
    out.write_text("\n".join(L) + "\n" + MARK + tail)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("selftest", "eval", "famos", "results"))
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=fc.N_BOOT)
    a = p.parse_args()
    {"selftest": step_selftest, "eval": step_eval, "famos": step_famos,
     "results": lambda _: write_results()}[a.step](a)


if __name__ == "__main__":
    main()
