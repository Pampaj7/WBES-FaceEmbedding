#!/usr/bin/env python3
"""D3, emendamento 2 (POST HOC, ``PROTOCOL_D3_emendamento_2.md``): catena LOGO con la griglia di lambda estesa e la
perdita riscalata per sorgente, testa congiunta su tutte le sorgenti (descrittiva), curva, contributo di FaMoS, d_P con
la stessa etichetta ai due lati, controlli contro i preregistrati e contro i numeri del critic.

    v3_work/unified_gt/run.sh aau/diagnostics/d3_e2.py cv --arm <braccio> --workers 64     (d3_e2.sbatch, uno per braccio)
    v3_work/unified_gt/run.sh aau/diagnostics/d3_e2.py eval --workers 64                   (dopo i tre cv)

File NUOVO: ``d3_head.py``, ``d3_stats.py`` e ``diag.py`` sono importati in sola lettura (la catena dell'ablazione 2x2
importa ``diag`` e ``d1_stats`` dal repo vivo). Perdita riscalata: la GT di ogni blocco divisa per l'alpha0 del blocco
(``Problem([b]).alpha0`` sulle sole coppie di fit, come il critic), poi ``d3_head.fit``; c_h dai blocchi con la GT
originale. ``cv`` scrive un risultato per riga in ``d3/e2/cv_<braccio>.jsonl`` e salta quelli gia' presenti.

Uscite (``eval``): ``d3_e2_cv.csv``, ``d3_e2_heads.csv``, ``d3_e2_spearman.csv``, ``d3_e2_delta.csv``, ``d3_e2_curve.csv``,
``d3/e2/readings.json``, ``d3/e2/controls.json``, ``d3/e2/reps.npz``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import d3_head as H  # noqa: E402
import d3_stats as S  # noqa: E402
import diag  # noqa: E402

E2 = S.D3 / "e2"
LAMBDAS_E2 = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CONFIGS_E2 = [(0, 0.0)] + [(r, lam) for r in H.R_GRID for lam in LAMBDAS_E2]
N_HALF = 100                                   # testa congiunta: FLAME soggetti 0-99 in training, 100-199 in test
SEED_HALF = 20261124
CRIT = (32, 1e-5)                              # configurazione fissa del critic (exp2.py, exp4.py)
# delta SR puntuali del critic a CRIT con la GT riscalata (exp2.json, "logo|<braccio>|<T>|32|1e-05|True" e "joint|...")
CRITIC = {("factorized_s1234", "logo"): {"facescape": 0.0501, "hifi3d": 0.0217, "faceverse": 0.0268, "flame2023_s1": 0.0332},
          ("factorized_s2345", "logo"): {"facescape": 0.0312, "hifi3d": 0.0262, "faceverse": 0.0080, "flame2023_s1": 0.0175},
          ("factorized_s1234", "joint"): {"facescape": 0.1326, "hifi3d": 0.1224, "faceverse": 0.0898, "flame2023_s1": 0.0763},
          ("factorized_s2345", "joint"): {"facescape": 0.1043, "hifi3d": 0.1236, "faceverse": 0.0525, "flame2023_s1": 0.0655}}
SAME = tuple(f"same_{lab}" for lab in diag.LABELS)


def settings_e2() -> dict:
    """Le 9 impostazioni LOGO dell'emendamento 1 piu' ``joint`` (tutte le 8 sorgenti, FLAME a meta')."""
    out = dict(S.settings())
    out["joint"] = S.SOURCES
    return out


def half(b: H.Block) -> H.Block:
    """Il blocco FLAME ristretto ai soggetti 0-99 (i primi 100 per id), G ristretta."""
    keep = b.subj < N_HALF
    return H.Block(b.X[keep], b.subj[keep], b.lab[keep], b.G[:N_HALF, :N_HALF], b.name)


def blocks_for(arm: str, key: str) -> list[H.Block]:
    srcs = settings_e2()[key]
    bl = [S._B[(arm, s)] for s in srcs]
    return [half(b) if key == "joint" and b.name == "flame2023_s1" else b for b in bl]


def rescale(blocks: list[H.Block]) -> list[H.Block]:
    """GT di ogni blocco divisa per il suo alpha0 (scala di d_P sulla GT sulle coppie del blocco): perdita riscalata."""
    return [H.Block(b.X, b.subj, b.lab, b.G / H.Problem([b]).alpha0, b.name) for b in blocks]


def fit_e2(blocks: list[H.Block], r: int, lam: float, scaled: bool = True) -> H.Head:
    return H.fit(rescale(blocks) if scaled else blocks, r, lam)


# ------------------------------------------------------------------------------------------ cv (un braccio)

def _cv(task):
    arm, key, k, q, f = task
    blocks = blocks_for(arm, key)
    srcs = [b.name for b in blocks]
    fo = H.folds([len(b.G) for b in blocks], q, [S.SOURCES.index(s) for s in srcs])
    fit_b, val_b = H.split(blocks, fo, f)
    r, lam = CONFIGS_E2[k]
    t0 = time.time()
    head = fit_e2(fit_b, r, lam)
    out = {"task": [arm, key, k, q, f], "score": H.score(head, val_b), "nit": head.nit, "success": head.success,
           "seconds": time.time() - t0}
    if r == 0:
        out["k5"] = abs(out["score"] - H.score(H.Head(1.0, np.zeros((0, head.W.shape[1])), 0, 0.0, 1.0), val_b))
    return out


def main_cv(a) -> None:
    t0 = time.time()
    info = {"ckpt": {}, "dpu": {}}
    _, _, ck = S.store_by_names(diag.CALIB_EMB / a.arm / "embeddings.npz", diag.set_names("bfm"))
    info["ckpt"][a.arm], info["dpu"][a.arm] = ck, diag.dp_per_unit(ck)
    S.build_sources(a.arm, info)
    E2.mkdir(parents=True, exist_ok=True)
    cache = E2 / f"cv_{a.arm}{a.tag}.jsonl"                   # --tag: un aiutante su un altro nodo scrive a parte
    done = set()
    for c in E2.glob(f"cv_{a.arm}*.jsonl"):
        for line in c.read_text().splitlines():
            if line.strip():
                done.add(tuple(json.loads(line)["task"]))
    tasks = [(a.arm, key, k, q, f) for key in settings_e2() for k in range(len(CONFIGS_E2))
             for q in range(H.N_REPEAT) for f in range(H.N_FOLD)]
    todo = [t for t in tasks if tuple(t) not in done]
    todo.sort(key=lambda t: (-CONFIGS_E2[t[2]][0], CONFIGS_E2[t[2]][1]), reverse=a.reverse)   # i piu' lenti prima
    print(f"[d3-e2] cv {a.arm}: {len(tasks)} fit, {len(done)} gia' fatti ({time.time() - t0:.0f}s)", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool, open(cache, "a") as fh:
        for n, r in enumerate(pool.imap_unordered(_cv, todo, chunksize=1), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if n % 500 == 0:
                print(f"[d3-e2] cv {a.arm}: {n} / {len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[d3-e2] cv {a.arm} finito ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------ eval

def _refit(task):
    """(chiave della testa, Head, c): ``e2`` = catena corretta, ``pre`` = configurazione preregistrata senza
    riscalare, ``crit`` = configurazione fissa del critic con la perdita riscalata; c dai blocchi con la GT originale."""
    kind, arm, key, r, lam = task
    blocks = blocks_for(arm, key)
    head = fit_e2(blocks, r, lam, scaled=kind != "pre")
    return task, head, H.calib_c(lambda b, i, j: head.dist(b.X[i], b.X[j]), blocks)


def _curve(task):
    arm, target, test, subset, r, lam = task
    from scipy.stats import spearmanr
    T = S._TS[test]
    X = T["Z"][arm][:, 1:] * S._TS["_dpu"][arm]
    Xa, Xb = X[T["ma"]], X[T["mb"]]
    g = T["gt"]["sr"]
    ref = spearmanr(np.linalg.norm(Xa - Xb, axis=1), g).correlation
    head = fit_e2([S._B[(arm, s)] for s in subset], r, lam)
    return {"arm": arm, "target": target, "test": test, "k": len(subset), "sources": "+".join(subset),
            "r": r, "lambda": lam, "delta_i_e2": float(spearmanr(head.dist(Xa, Xb), g).correlation - ref),
            "rho_dP": float(ref)}


def rowsets_e2(T: dict) -> dict:
    """Righe incrociate; FLAME anche le righe fra i soggetti 100-199 (``cross_half``, testa congiunta); per i bersagli
    neutri le righe con la stessa etichetta ai due lati (``same_<etichetta>``) e mediate (``lavg``), una per coppia."""
    out = {"cross": (T["sa"], T["sb"], T["gt"], T["Z"], T["ma"], T["mb"], T["counts"])}
    if T["name"] == "flame2023_s1":
        k = (T["sa"] >= N_HALF) & (T["sb"] >= N_HALF)
        cnt = diag.boot_counts(len(T["subjects"]) - N_HALF, 0, seed=SEED_HALF)
        out["cross_half"] = (T["sa"][k] - N_HALF, T["sb"][k] - N_HALF, {g: v[k] for g, v in T["gt"].items()}, T["Z"],
                             T["ma"][k], T["mb"][k], cnt)
    if T["name"] == "faceverse":
        return out
    orig, grp = S.mesh_index(T)
    i, j = np.triu_indices(len(T["subjects"]), 1)
    gt = {g: T["Gsub"][g][i, j] for g in ("sr", "fr")}
    for n, lab in enumerate(diag.LABELS):
        out[f"same_{lab}"] = (i, j, gt, {a: Z[grp[:, n]] for a, Z in T["Z"].items()}, i, j, T["counts"])
    out["lavg"] = (i, j, gt, {a: Z[grp].mean(1) for a, Z in T["Z"].items()}, i, j, T["counts"])
    return out


def columns_e2(T: dict, rs: tuple, rsn: str, heads: dict, cal: dict, info: dict) -> dict:
    """d_P e d_F calibrata; sulle righe incrociate anche (i-e2), (i-e2) senza FaMoS, (i) preregistrata, la testa
    congiunta e le configurazioni del critic."""
    sa, sb, gt, Zs, ia, ib, _ = rs
    target = S.TESTS[T["name"]]
    cols = dict(T["B"]) if rsn == "cross" else {}            # B solo sulle righe incrociate di fact_paired
    for arm, Z in Zs.items():
        S_ = np.exp(Z[:, 0])
        X = Z[:, 1:] * info["dpu"][arm]
        Sa, Sb, Xa, Xb = S_[ia], S_[ib], X[ia], X[ib]

        def form(d, c):
            return np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (c * d) ** 2)

        dP = np.linalg.norm(Xa - Xb, axis=1)
        cols[f"{arm}|dP|shape"], cols[f"{arm}|dP|form"] = dP, form(dP, cal[arm])
        if not rsn.startswith("cross"):
            continue
        for m, key in (("i_e2", ("e2", f"{target}|all")), ("i_nf_e2", ("e2", f"{target}|nofamos")),
                       ("i_pre", ("pre", f"{target}|all")), ("i_nf_pre", ("pre", f"{target}|nofamos")),
                       ("joint", ("e2", "joint")), ("crit_logo", ("crit", f"{target}|all")),
                       ("crit_joint", ("crit", "joint"))):
            h = heads.get((key[0], arm, key[1]))
            if h is None:
                continue
            d = h[0].dist(Xa, Xb)
            cols[f"{arm}|{m}|shape"], cols[f"{arm}|{m}|form"] = d, form(d, h[1])
    bad = [k for k, v in cols.items() if not np.isfinite(v).all()]
    if bad:
        raise SystemExit(f"{T['name']} {rsn}: colonne non finite {bad[:4]}")
    return cols


def main_eval(a) -> None:
    t0 = time.time()
    cal = diag.calibration()
    info = {"ckpt": {}, "dpu": {}}
    for arm in diag.ARMS:
        _, _, ck = S.store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
        info["ckpt"][arm], info["dpu"][arm] = ck, diag.dp_per_unit(ck)
        S.build_sources(arm, info)
    sets = settings_e2()
    # ------------------------------------------------------------------ CV: completezza e scelta
    cv, k5 = {}, []
    for arm in diag.ARMS:
        for c in sorted(E2.glob(f"cv_{arm}*.jsonl")):              # anche gli aiutanti (--tag); doppioni identici
            for line in c.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                cv[tuple(r["task"])] = r
                if "k5" in r:
                    k5.append(r["k5"])
    want = [(arm, key, k, q, f) for arm in diag.ARMS for key in sets for k in range(len(CONFIGS_E2))
            for q in range(H.N_REPEAT) for f in range(H.N_FOLD)]
    miss = [t for t in want if t not in cv]
    if miss:
        raise SystemExit(f"CV incompleta: mancano {len(miss)} fit (es. {miss[:2]})")
    recs_cv, chosen = [], {}
    for arm in diag.ARMS:
        for key in sets:
            means = {}
            for k, cfg in enumerate(CONFIGS_E2):
                rs = [cv[(arm, key, k, q, f)] for q in range(H.N_REPEAT) for f in range(H.N_FOLD)]
                sc = np.asarray([x["score"] for x in rs])
                means[cfg] = float(sc.mean())
                recs_cv.append({"arm": arm, "setting": key, "r": cfg[0], "lambda": cfg[1], "cv_score": means[cfg],
                                "cv_sd": float(sc.std(ddof=1)), "n_fits": len(rs),
                                "nit_median": float(np.median([x["nit"] for x in rs])),
                                "success_frac": float(np.mean([x["success"] for x in rs])),
                                "seconds_median": float(np.median([x["seconds"] for x in rs]))})
            chosen[(arm, key)] = H.choose(means)
            chosen[(arm, key, "cv")] = (means[chosen[(arm, key)]], means[(0, 0.0)])
            print(f"[d3-e2] {arm} {key}: scelta {chosen[(arm, key)]}, CV {means[chosen[(arm, key)]]:.4f} contro d_P "
                  f"{means[(0, 0.0)]:.4f}", flush=True)
    pre = {(r["arm"], r["setting"]): (int(r["r"]), float(r["lambda"]))
           for r in csv.DictReader(open(diag.EV / "d3_heads.csv")) if r["method"] == "i"}
    # ------------------------------------------------------------------ teste: e2, preregistrate, critic
    jobs = [("e2", arm, key, *chosen[(arm, key)]) for arm in diag.ARMS for key in sets]
    jobs += [("pre", arm, key, *pre[(arm, key)]) for arm in diag.ARMS for key in S.settings()]
    jobs += [("crit", arm, key, *CRIT) for arm in diag.RULE_ARMS for key in [f"{t}|all" for t in S.TARGETS] + ["joint"]]
    with mp.get_context("fork").Pool(a.workers) as pool:
        fits = pool.map(_refit, jobs, chunksize=1)
    heads = {(kind, arm, key): (h, c) for (kind, arm, key, r, lam), h, c in fits}
    print(f"[d3-e2] {len(heads)} teste ({time.time() - t0:.0f}s)", flush=True)
    # ------------------------------------------------------------------ insiemi di test
    with mp.get_context("fork").Pool(len(S.FP_VIEWS)) as pool:
        tests = dict(zip(S.FP_VIEWS, pool.map(S.view_data, S.FP_VIEWS)))
    for name in ("flame2023_s1", "famos"):
        tests[name] = S.set_data(name, info)
    for t, Tn in tests.items():
        for arm, ck in Tn["ckpt"].items():
            if ck.resolve() != info["ckpt"][arm].resolve():
                raise SystemExit(f"{t} {arm}: checkpoint {ck}")
    print(f"[d3-e2] insiemi di test ({time.time() - t0:.0f}s)", flush=True)
    # ------------------------------------------------------------------ curva (i-e2)
    S._TS.update(tests)
    S._TS["_dpu"] = info["dpu"]
    ctasks = [(arm, t, test, sub, *chosen[(arm, f"{t}|all")]) for arm in diag.ARMS
              for t, test in zip(S.TARGETS, S.PRIMARY) for k in S.CURVE_K
              for sub in itertools.combinations(sets[f"{t}|all"], k)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        recs_curve = pool.map(_curve, ctasks, chunksize=1)
    # ------------------------------------------------------------------ colonne e repliche
    for t, Tn in tests.items():
        for rsn, rs in rowsets_e2(Tn).items():
            S._J[(t, rsn)] = {"counts": rs[6][: a.n_boot + 1], "sa": rs[0], "sb": rs[1], "gt": rs[2],
                              "cols": columns_e2(Tn, rs, rsn, heads, cal, info)}
    keys = [(key, b) for key in S._J for b in range(a.n_boot + 1)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        reps = pool.map(S._rep, keys, chunksize=8)
    V = {key: {} for key in S._J}
    for (key, b), r in zip(keys, reps):
        for m, x in r.items():
            V[key].setdefault(m, np.empty(a.n_boot + 1))[b] = x
    for arm in diag.ARMS:                                          # k = 7: le teste e2 di "tutte tranne T"
        for t, test in zip(S.TARGETS, S.PRIMARY):
            Vt = V[(test, "cross")]
            recs_curve.append({"arm": arm, "target": t, "test": test, "k": 7, "sources": "+".join(sets[f"{t}|all"]),
                               "r": chosen[(arm, f"{t}|all")][0], "lambda": chosen[(arm, f"{t}|all")][1],
                               "delta_i_e2": float(Vt[(f"{arm}|i_e2|shape", "sr")][0] - Vt[(f"{arm}|dP|shape", "sr")][0]),
                               "rho_dP": float(Vt[(f"{arm}|dP|shape", "sr")][0])})
    # ------------------------------------------------------------------ uscite
    recs_rho, recs_d, dd = [], [], {}
    for (t, rsn), Vt in V.items():
        for (m, g), x in Vt.items():
            p, lo, hi, _ = diag.ci(x)
            arm, _, rest = m.partition("|")
            recs_rho.append({"test": t, "rows": rsn, "column": m, "arm": arm, "method": rest.split("|")[0], "gt": g,
                             "rho": p, "ci_low": lo, "ci_high": hi, "n_rows": int(len(S._J[(t, rsn)]["sa"]))})

    def add(kind, t, rsn, arm, meth, tag, what, x, ref):
        p, lo, hi, ple = diag.ci(x - ref)
        rec = {"kind": kind, "test": t, "rows": rsn, "arm": arm, "method": meth, "gt": tag, "what": what, "delta": p,
               "ci_low": lo, "ci_high": hi, "p_le0": ple, "rho_method": float(x[0]), "rho_ref": float(ref[0])}
        recs_d.append(rec)
        return rec

    trip = (("sr", "shape", "sr", "sr"), ("fr", "form", "fr", "fr"), ("fr_nosize", "shape", "fr", "fr"))
    for (t, rsn), Vt in V.items():
        if not rsn.startswith("cross"):
            continue
        # la testa congiunta ha visto FaMoS e i soggetti 0-99 di FLAME: si legge solo fuori dal suo training
        joint_ok = (t in S.FP_VIEWS and rsn == "cross") or (t == "flame2023_s1" and rsn == "cross_half")
        for arm in tests[t]["Z"]:
            for meth, ref in (("i_e2", "dP"), ("i_nf_e2", "dP"), ("i_pre", "dP"), ("i_nf_pre", "dP"), ("joint", "dP"),
                              ("crit_logo", "dP"), ("crit_joint", "dP"), ("i_e2", "i_pre"), ("i_e2", "i_nf_e2"),
                              ("i_nf_e2", "i_nf_pre"), ("joint", "i_e2")):
                if "joint" in meth and not joint_ok:
                    continue
                for tag, dist, g, _ in trip:
                    col, rcol = f"{arm}|{meth}|{dist}", f"{arm}|{ref}|{dist}"
                    if (col, g) not in Vt or (rcol, g) not in Vt:
                        continue
                    kind = "ref" if ref == "dP" else "pair"
                    rec = add(kind, t, rsn, arm, f"{meth}-{ref}", tag, f"{col} - {rcol} ({g})", Vt[(col, g)],
                              Vt[(rcol, g)])
                    dd[(t, rsn, arm, f"{meth}-{ref}", tag)] = rec
    for (t, rsn), Vt in V.items():                                 # d_P con la stessa etichetta contro le incrociate
        if rsn in SAME + ("lavg",):
            for arm in tests[t]["Z"]:
                for tag, dist, g in (("sr", "shape", "sr"), ("fr", "form", "fr")):
                    add("same", t, rsn, arm, "dP", tag, f"{arm}|dP|{dist} {rsn} - cross ({g})",
                        Vt[(f"{arm}|dP|{dist}", g)], V[(t, "cross")][(f"{arm}|dP|{dist}", g)])
    primary_rows = {t: ("cross_half" if t == "flame2023_s1" else "cross") for t in S.PRIMARY}
    for arm in diag.ARMS:                                          # medie sui 4 bersagli, replica per replica
        for meth, rows in (("i_e2-dP", {t: "cross" for t in S.PRIMARY}), ("i_pre-dP", {t: "cross" for t in S.PRIMARY}),
                           ("joint-dP", primary_rows)):
            for tag, dist, g, _ in trip[:2]:
                m = meth.split("-")[0]
                if not all((f"{arm}|{m}|{dist}", g) in V[(t, rows[t])] for t in S.PRIMARY):
                    continue
                x = np.mean([V[(t, rows[t])][(f"{arm}|{m}|{dist}", g)] for t in S.PRIMARY], axis=0)
                ref = np.mean([V[(t, rows[t])][(f"{arm}|dP|{dist}", g)] for t in S.PRIMARY], axis=0)
                dd[("mean", "-", arm, meth, tag)] = add("mean", "media dei 4 bersagli", "-", arm, meth, tag,
                                                        f"media: {meth} ({g})", x, ref)
    for name, recs in (("d3_e2_cv.csv", recs_cv), ("d3_e2_spearman.csv", recs_rho), ("d3_e2_delta.csv", recs_d),
                       ("d3_e2_curve.csv", recs_curve)):
        with open(diag.EV / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in recs for k in r)))
            w.writeheader()
            w.writerows(recs)
    recs_h = []
    for (kind, arm, key), (h, c) in sorted(heads.items()):
        cvs = chosen.get((arm, key, "cv"), (np.nan, np.nan)) if kind == "e2" else (np.nan, np.nan)
        recs_h.append({"kind": kind, "arm": arm, "setting": key, "r": h.r, "lambda": h.lam, "cv_score": cvs[0],
                       "cv_score_dP": cvs[1], "edge": bool(kind == "e2" and (h.r == H.R_GRID[-1] or (
                           h.r and h.lam in (LAMBDAS_E2[0], LAMBDAS_E2[-1])))),
                       "alpha_over_alpha0": h.alpha / h.alpha0, "c": c, "nit": h.nit, "success": h.success,
                       "W_fro_over_alpha": float(np.linalg.norm(h.W) / h.alpha) if h.r else 0.0})
    with open(diag.EV / "d3_e2_heads.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs_h[0]))
        w.writeheader()
        w.writerows(recs_h)
    # ------------------------------------------------------------------ letture (post hoc) e controlli
    rule = {(t, arm, "i_e2", tag): dd[(t, "cross", arm, "i_e2-dP", tag)] for t in S.PRIMARY for arm in diag.ARMS
            for tag in ("sr", "fr")}
    jt = {}
    for arm in diag.ARMS:
        for t in S.PRIMARY:
            x = dd[(t, primary_rows[t], arm, "joint-dP", "sr")]
            jt.setdefault(arm, {})[t] = {"sr": x, "fr": dd[(t, primary_rows[t], arm, "joint-dP", "fr")],
                                         "pass": bool(x["delta"] >= S.SR_MIN and x["ci_low"] > 0)}
    curve = {}
    for arm in diag.ARMS:
        ok, means = 0, {}
        for t in S.TARGETS:
            m = [float(np.mean([r["delta_i_e2"] for r in recs_curve if r["arm"] == arm and r["target"] == t
                                and r["k"] == k])) for k in S.CURVE_K + (7,)]
            means[t] = m
            ok += all(b >= x for x, b in zip(m, m[1:]))
        curve[arm] = {"targets_non_decreasing": ok, "means_k1_2_4_7": means}
    grows = [curve[arm]["targets_non_decreasing"] >= 3 for arm in diag.RULE_ARMS]
    readings = {"post_hoc": True, "R_LOGO_i_e2": S.verdict(rule, "i_e2"), "joint": jt,
                "mean": {f"{arm}|{meth}|{tag}": rec for (k0, _, arm, meth, tag), rec in dd.items() if k0 == "mean"},
                "curve_i_e2": {"grows": "si" if all(grows) else ("no" if not any(grows) else "non risolto"),
                               "arms": curve},
                "chosen": {f"{arm}|{key}": list(chosen[(arm, key)]) for arm in diag.ARMS for key in sets},
                "preregistered_verdict": "R_LOGO NO (results_d3.md, f0c643b): invariato"}
    # K_pre: le teste preregistrate rifittate ridanno i delta di d3_delta.csv (punto e IC)
    pub = {(r["test"], r["arm"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_delta.csv"))
           if r["kind"] == "ref" and r["rows"] == "cross" and r["method"] == "i" and r["arm"] in diag.ARMS}
    kpre = 0.0
    for (t, arm, tag), r in pub.items():
        x = dd[(t, "cross", arm, "i_pre-dP", tag)]
        kpre = max(kpre, abs(x["delta"] - float(r["delta"])), abs(x["ci_low"] - float(r["ci_low"])),
                   abs(x["ci_high"] - float(r["ci_high"])))
    kcrit = {}
    for (arm, kind), vals in CRITIC.items():
        for t, v in vals.items():
            test = dict(zip(S.TARGETS, S.PRIMARY))[t]
            rows = "cross_half" if kind == "joint" and t == "flame2023_s1" else "cross"
            ours = dd[(test, rows, arm, f"crit_{kind}-dP", "sr")]["delta"]
            kcrit[f"{arm}|{kind}|{t}"] = {"critic": v, "ours": round(ours, 4), "abs_diff": abs(round(ours, 4) - v)}
    controls = {"K1": S.k1({k: v for k, v in V.items() if k[1] == "cross"}, tests),
                "K_pre": {"max_abs_diff": kpre, "n_cells": len(pub), "pass": bool(kpre <= 1e-9)},
                "K_critic": {"cells": kcrit, "max_abs_diff": max(x["abs_diff"] for x in kcrit.values()),
                             "pass": bool(max(x["abs_diff"] for x in kcrit.values()) <= 1e-4)},
                "K5_r0_max_abs_diff": float(max(k5)), "n_cv_fits": len(cv),
                "rows": {f"{t}|{rsn}": int(len(S._J[(t, rsn)]["sa"])) for (t, rsn) in S._J},
                "seeds": {"flame_half": f"diag.boot_counts(100, 0, seed={SEED_HALF})"},
                "lambdas": list(LAMBDAS_E2), "n_configs": len(CONFIGS_E2), "n_boot": a.n_boot,
                "seconds": time.time() - t0, "workers": a.workers}
    diag.atomic_json(E2 / "readings.json", readings)
    diag.atomic_json(E2 / "controls.json", controls)
    diag.atomic_savez(E2 / "reps.npz", **{f"{t}|{rsn}|{m}|{g}": x for (t, rsn), Vt in V.items()
                                          for (m, g), x in Vt.items()})
    print(f"[d3-e2] R_LOGO (i-e2, post hoc) {readings['R_LOGO_i_e2']['verdict']}; K1 {controls['K1']['pass']}, K_pre "
          f"{kpre:.1e}, K_critic {controls['K_critic']['max_abs_diff']:.1e}; {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("cv", "eval"))
    ap.add_argument("--arm", choices=diag.ARMS)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=diag.N_BOOT)
    ap.add_argument("--tag", default="", help="cv: suffisso della cache (un aiutante su un altro nodo)")
    ap.add_argument("--reverse", action="store_true", help="cv: dai fit piu' veloci (aiutante)")
    a = ap.parse_args()
    if a.step == "cv":
        if not a.arm:
            raise SystemExit("cv: serve --arm")
        main_cv(a)
    else:
        main_eval(a)


if __name__ == "__main__":
    main()
