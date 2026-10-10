#!/usr/bin/env python3
"""D2 (PROTOCOL_D.md sez. 4): sonda lineare dall'embedding congelato alla forma GT. DIAGNOSTICA, non un metodo.

    v3_work/unified_gt/run.sh aau/diagnostics/d2_probe.py --workers 32          (diag.sbatch, passo d2)

Per vista (FaceScape dev primaria, HIFI3D descrittiva): righe e GT di ``factorized_paired.csv`` (``fact_paired.rows_for``
+ la maschera comune di ``fact_paired.main``, ricostruita con le stesse colonne), embedding [s, u] degli store
ufficiali (``fact_paired.embeddings``) delle 500 mesh senza crop, bersagli per soggetto = vettori GT (``cgt``, la
fattorizzazione di ``train_fr_sr._factor_chunk``, copiata): SR = sqrt(w) z_i, FR = sqrt(w) a_i.

Sonda (``fit_predict``): standardizzazione pesata delle 257 feature sulle mesh di training, PCA pesata dei bersagli dei
soggetti di training (componenti con valore singolare > 1e-10 del massimo), ridge sui punteggi con lambda di
``LAMBDAS`` scelto dalla CV interna a 5 fold per soggetto (somma pesata dell'errore quadratico; standardizzazione e
base PCA del fold esterno, i bersagli di validazione stanno nella base: e' l'errore sul vettore intero), rifit; distanza
euclidea fra i bersagli previsti. CV esterna a 2 fold per soggetto, R = 50 split (stima puntuale = media), IC dal
bootstrap per soggetto dell'intera procedura (1000 repliche, uno split per replica), IC di sola valutazione
(emendamento 1, descrittivo: predizioni fisse dei 50 split, conteggi ``SEED_EVAL``), sonda coi bersagli permutati
(K1: riferimento "mappa lineare senza la supervisione giusta", NON un controllo a rho ~ 0, vedi l'emendamento 1).

Uscite: ``d2_probe.csv`` (sonda, riferimento d_P / form_cal sulle stesse righe, delta, IC, lambda, permutazione),
``d2_controls.json`` (K2, K3, conteggi), ``d2/reps.npz`` (tutte le repliche).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import diag  # noqa: E402

sys.path.insert(0, str(diag.REPO / "v3_work/trainer/tools"))
sys.path.insert(0, str(diag.REPO / "v3_work/canonical_gt"))

VIEWS = {"facescape": "facescape", "hifi3d": "hifi3d"}      # vista di fact_paired -> set della GT di eval
PRIMARY = "facescape"
LAMBDAS = np.logspace(-1, 5, 25)
N_INNER, N_REP = 5, 50
REF = {"sr": "shape", "fr": "form_cal"}                      # riferimento del braccio per GT
SEED_EVAL = 20261117                                         # IC di sola valutazione (emendamento 1, descrittivo)
_D: dict = {}           # dati per vista, ereditati dai worker (fork)


# ------------------------------------------------------------------------------------------------ dati

def official_rows(view: str) -> tuple:
    """(df delle righe di factorized_paired.csv, colonne sulle righe, soggetti, sa, sb): ``fact_paired.main`` fino alla
    maschera comune (copiato)."""
    import fact_paired as fp          # sola lettura; importato qui: le prove della sonda non ne hanno bisogno
    df, idx, _ = fp.rows_for(view)
    Mc = fp.model_columns(view, idx)
    df = fp.be.add_columns(df, Mc, idx)
    cols = {m: df[m].to_numpy(np.float64) for m in list(Mc) + [b for b in fp.BASELINES if b in df]}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
    return df[mask].reset_index(drop=True), {k: v[mask] for k, v in cols.items()}, subjects, sa[mask], sb[mask]


def gt_vectors(gset: str) -> tuple[list[str], dict, float]:
    """(id, {fr, sr: (n, 3m)}, A): ``train_fr_sr._factor_chunk`` (copiata) sui punti di ``cgt.native_points``."""
    import cgt
    cn = cgt.Canon()
    nat = cgt.native_points(gset, cn)
    rb = cn.rigid_robust(cn.to_F(nat["domain"], nat["P"]))
    a = rb["a"]
    S, c = cn.centroid_size(a), cn.centroid(a)
    z = (a - c[:, None]) / S[:, None, None]
    sq = np.sqrt(cn.sp.w)[None, :, None]
    return list(nat["ids"]), {"fr": (sq * a).reshape(len(a), -1), "sr": (sq * z).reshape(len(a), -1)}, float(cn.sp.A)


def load_view(view: str) -> dict:
    """Righe, mesh senza crop, embedding dei bracci, bersagli e controlli K2, K3 di una vista."""
    import fact_paired as fp
    from scipy.stats import spearmanr
    df, cols, subjects, sa, sb = official_rows(view)
    ids, T, A = gt_vectors(VIEWS[view])
    pos = {s: k for k, s in enumerate(ids)}
    T = {g: np.asarray(v[[pos[s] for s in subjects]], np.float64) for g, v in T.items()}
    meshes = [(s, t) for s in subjects for t in diag.LABELS]
    mpos = {m: k for k, m in enumerate(meshes)}
    ma = np.asarray([mpos[k] for k in zip(df["subject_a"], df["topology_a"])])
    mb = np.asarray([mpos[k] for k in zip(df["subject_b"], df["topology_b"])])
    ms = np.repeat(np.arange(len(subjects)), len(diag.LABELS))
    ctrl = {"n_rows": int(len(df)), "n_subjects": int(len(subjects)), "K2": {}, "K3": {}}
    for g in ("sr", "fr"):                                  # K2: distanze dei bersagli contro la GT delle righe
        d = np.linalg.norm(T[g][sa] - T[g][sb], axis=1) / np.sqrt(A)
        r = d / cols[f"gt_{g}"]
        ctrl["K2"][g] = {"ratio_median": float(np.median(r)), "ratio_rel_spread": float((r.max() - r.min()) / np.median(r))}
    ref = {}
    with open(fp.EV / "factorized_paired.csv") as fh:
        pub = {(r["gt"], r["arm"], r["distance"]): float(r["arm_point"]) for r in csv.DictReader(fh)
               if r["domain"] == view and r["kind"] == "rho" and r["baseline"] == "-"}
    X = {}
    for pre, v, e in fp.MODELS:
        if pre not in diag.ARMS:
            continue
        with np.load(fp.embeddings(fp.VIEWS[view][0], v, e), allow_pickle=True) as z:
            keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
            kpos = {k: i for i, k in enumerate(keys)}
            X[pre] = np.asarray(z["Z"], np.float64)[[kpos[m] for m in meshes]]
        for g, dname in REF.items():                       # K3: rho dei riferimenti = factorized_paired.csv
            ref[(pre, g)] = cols[f"{pre}|{dname}"]
            rho = float(spearmanr(ref[(pre, g)], cols[f"gt_{g}"]).correlation)
            ctrl["K3"][f"{pre}|{dname}|{g}"] = {"rho": rho, "published": pub.get((g, pre, dname), float("nan")),
                                                "abs_diff": abs(rho - pub.get((g, pre, dname), float("nan")))}
    return {"sa": sa, "sb": sb, "ma": ma, "mb": mb, "ms": ms, "T": T, "X": X, "ref": ref,
            "gt": {g: cols[f"gt_{g}"] for g in ("sr", "fr")}, "n_subj": len(subjects), "ctrl": ctrl}


# ------------------------------------------------------------------------------------------------ sonda

def ridge_paths(Xi: np.ndarray, Yi: np.ndarray, wi: np.ndarray):
    """Ridge pesato per tutti i lambda: (media di X, media di Y, funzione lambda -> W)."""
    wn = wi / wi.sum()
    xm, ym = wn @ Xi, wn @ Yi
    sw = np.sqrt(wi)[:, None]
    U, s, Vt = np.linalg.svd(sw * (Xi - xm), full_matrices=False)
    UtB = U.T @ (sw * (Yi - ym))
    return xm, ym, lambda lam: Vt.T @ ((s / (s * s + lam))[:, None] * UtB)


def fit_predict(X: np.ndarray, ms: np.ndarray, T: np.ndarray, tr: np.ndarray, c: np.ndarray,
                rng: np.random.Generator, perm: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Sonda addestrata sui soggetti ``tr`` (distinti, pesi ``c[tr]``): (punteggi previsti di TUTTE le mesh, lambda).
    ``perm``: bersagli dei soggetti di training permutati (controllo K1)."""
    in_tr = np.zeros(len(c), dtype=bool)
    in_tr[tr] = True
    if perm is not None and not np.array_equal(np.sort(perm), np.sort(tr)):
        raise ValueError("la permutazione deve restare fra i soggetti di training")
    mtr = np.flatnonzero(in_tr[ms])
    wm = c[ms[mtr]].astype(np.float64)
    mu = (wm @ X[mtr]) / wm.sum()
    sd = np.sqrt((wm @ (X[mtr] - mu) ** 2) / wm.sum())
    Xs = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    Ttr = T[tr if perm is None else perm]
    cs = c[tr].astype(np.float64)
    muT = (cs @ Ttr) / cs.sum()
    _, sv, Vt = np.linalg.svd(np.sqrt(cs)[:, None] * (Ttr - muT), full_matrices=False)
    Vt = Vt[sv > 1e-10 * sv.max()]
    spos = np.full(len(c), -1)
    spos[tr] = np.arange(len(tr))
    Y = ((Ttr - muT) @ Vt.T)[spos[ms[mtr]]]                # punteggi del soggetto di ogni mesh di training
    fold = rng.permutation(len(tr)) % N_INNER
    fold_m = fold[spos[ms[mtr]]]
    err = np.zeros(len(LAMBDAS))
    for f in range(N_INNER):
        it, iv = fold_m != f, fold_m == f
        xm, ym, W = ridge_paths(Xs[mtr][it], Y[it], wm[it])
        Xv = Xs[mtr][iv] - xm
        for k, lam in enumerate(LAMBDAS):
            err[k] += float(wm[iv] @ ((Xv @ W(lam) + ym - Y[iv]) ** 2).sum(1))
    lam = float(LAMBDAS[int(np.argmin(err))])
    xm, ym, W = ridge_paths(Xs[mtr], Y, wm)
    return (Xs - xm) @ W(lam) + ym, lam


def evaluate(P: np.ndarray, D: dict, ref: np.ndarray, gt: np.ndarray, te: np.ndarray, c: np.ndarray) -> tuple:
    """(rho sonda, rho riferimento) sulle righe con entrambi i soggetti in ``te``, pesi c_a c_b."""
    in_te = np.zeros(len(c), dtype=bool)
    in_te[te] = True
    k = in_te[D["sa"]] & in_te[D["sb"]]
    w = (c[D["sa"]] * c[D["sb"]])[k]
    d = np.linalg.norm(P[D["ma"][k]] - P[D["mb"][k]], axis=1)
    return diag.wspearman(d, gt[k], w), diag.wspearman(ref[k], gt[k], w)


def evaluate_boot(P: np.ndarray, D: dict, ref: np.ndarray, gt: np.ndarray, te: np.ndarray) -> np.ndarray:
    """IC di sola valutazione (emendamento 1, descrittivo): predizioni fisse, (n_eval, 2) rho di sonda e riferimento
    sulle righe di ``te`` coi conteggi ``D["eval_counts"]`` per soggetto (pesi c_a c_b)."""
    in_te = np.zeros(D["n_subj"], dtype=bool)
    in_te[te] = True
    k = in_te[D["sa"]] & in_te[D["sb"]]
    sa, sb, q, y = D["sa"][k], D["sb"][k], ref[k], gt[k]
    x = np.linalg.norm(P[D["ma"][k]] - P[D["mb"][k]], axis=1)
    out = np.full((len(D["eval_counts"]), 2), np.nan)
    for b, c in enumerate(D["eval_counts"]):
        w = c[sa] * c[sb]
        m = w > 0
        if m.sum() < 3:
            continue
        ww = w[m]
        ry = diag.wranks(y[m], ww)
        out[b] = diag.wpearson(diag.wranks(x[m], ww), ry, ww), diag.wpearson(diag.wranks(q[m], ww), ry, ww)
    return out


def _task(task):
    """Una ripetizione (``point`` / ``perm``: soggetti tutti a peso 1) o una replica bootstrap (``boot``): due fold."""
    view, arm, g, kind, r = task
    D = _D[view]
    n = D["n_subj"]
    seed = {"point": diag.SEED_SPLIT, "boot": diag.SEED_BBOOT, "perm": diag.SEED_PERM}[kind]
    rng = np.random.default_rng(np.random.SeedSequence([seed, r]))
    if kind == "boot":
        c = np.bincount(rng.integers(0, n, n), minlength=n)
        pool = rng.permutation(np.flatnonzero(c > 0))
    else:
        c = np.ones(n, dtype=np.int64)
        pool = rng.permutation(n)
    half = len(pool) // 2
    out, ev = [], []
    for tr, te in ((pool[:half], pool[half:]), (pool[half:], pool[:half])):
        if np.intersect1d(tr, te).size:                       # nessuna fuga: soggetti disgiunti per costruzione
            raise RuntimeError("soggetti comuni fra training e test")
        tr = np.sort(tr)
        perm = rng.permutation(tr) if kind == "perm" else None     # il soggetto tr[i] riceve il bersaglio di perm[i]
        P, lam = fit_predict(D["X"][arm], D["ms"], D["T"][g], tr, c, rng, perm)
        out.append((*evaluate(P, D, D["ref"][(arm, g)], D["gt"][g], te, c), lam))
        if kind == "point":
            ev.append(evaluate_boot(P, D, D["ref"][(arm, g)], D["gt"][g], te))
    a = np.asarray(out)
    return task, float(a[:, 0].mean()), float(a[:, 1].mean()), a[:, 2].tolist(), \
        (np.mean(ev, axis=0) if ev else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--n-boot", type=int, default=diag.N_BOOT)
    ap.add_argument("--n-eval", type=int, default=diag.N_BOOT, help="repliche dell'IC di sola valutazione")
    a = ap.parse_args()
    t0 = time.time()
    views = a.views.split(",")
    for v in views:
        _D[v] = load_view(v)
        n = _D[v]["n_subj"]
        _D[v]["eval_counts"] = [np.bincount(np.random.default_rng(np.random.SeedSequence([SEED_EVAL, b])).integers(
            0, n, n), minlength=n) for b in range(a.n_eval)]
        print(f"[d2] {v}: {_D[v]['ctrl']['n_rows']} righe, {_D[v]['n_subj']} soggetti, K2 {_D[v]['ctrl']['K2']}, "
              f"K3 max {max(x['abs_diff'] for x in _D[v]['ctrl']['K3'].values()):.1e} ({time.time() - t0:.0f}s)",
              flush=True)
    tasks = [(v, arm, g, kind, r) for v in views for arm in diag.ARMS for g in ("sr", "fr")
             for kind, nr in (("point", N_REP), ("perm", N_REP), ("boot", a.n_boot)) for r in range(nr)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(_task, tasks, chunksize=4)
    R, E = {}, {}
    for (v, arm, g, kind, r), p, q, lam, ev in res:
        R.setdefault((v, arm, g, kind), []).append((r, p, q, lam))
        if ev is not None:
            E.setdefault((v, arm, g), []).append(ev)
    recs, reps = [], {}
    for v in views:
        for arm in diag.ARMS:
            for g in ("sr", "fr"):
                pt = sorted(R[(v, arm, g, "point")])
                bt = sorted(R[(v, arm, g, "boot")])
                pm = sorted(R[(v, arm, g, "perm")])
                p_pt, q_pt = np.asarray([x[1] for x in pt]), np.asarray([x[2] for x in pt])
                p_bt, q_bt = np.asarray([x[1] for x in bt]), np.asarray([x[2] for x in bt])
                lam = np.asarray([x[3] for x in pt]).ravel()
                dlt = p_bt - q_bt
                ev = np.mean(E[(v, arm, g)], axis=0)           # (n_eval, 2), media sui 50 split
                dev = (ev[:, 0] - ev[:, 1])[np.isfinite(ev).all(1)]
                rec = {"domain": v, "arm": arm, "gt": g, "reference": REF[g],
                       "probe": float(p_pt.mean()), "probe_ci_low": float(np.percentile(p_bt, 2.5)),
                       "probe_ci_high": float(np.percentile(p_bt, 97.5)),
                       "ref": float(q_pt.mean()), "ref_ci_low": float(np.percentile(q_bt, 2.5)),
                       "ref_ci_high": float(np.percentile(q_bt, 97.5)),
                       "delta": float((p_pt - q_pt).mean()), "delta_ci_low": float(np.percentile(dlt, 2.5)),
                       "delta_ci_high": float(np.percentile(dlt, 97.5)), "p_le0": float((dlt <= 0).mean()),
                       "delta_split_sd": float((p_pt - q_pt).std(ddof=1)),
                       "delta_evalci_low": float(np.percentile(dev, 2.5)),
                       "delta_evalci_high": float(np.percentile(dev, 97.5)),
                       "perm_mean": float(np.mean([x[1] for x in pm])), "perm_min": float(np.min([x[1] for x in pm])),
                       "perm_max": float(np.max([x[1] for x in pm])),
                       "lambda_median": float(np.median(lam)),
                       "lambda_edge_frac": float(np.isin(lam, LAMBDAS[[0, -1]]).mean()),
                       "n_rows": _D[v]["ctrl"]["n_rows"], "n_subjects": _D[v]["n_subj"], "n_rep": len(pt),
                       "n_boot": len(bt)}
                rec["r3_pass"] = bool(rec["delta"] >= 0.05 and rec["delta_ci_low"] > 0) if g == "sr" else None
                recs.append(rec)
                reps.update({f"{v}|{arm}|{g}|{k}": np.asarray(x) for k, x in
                             (("point_probe", p_pt), ("point_ref", q_pt), ("boot_probe", p_bt), ("boot_ref", q_bt),
                              ("perm_probe", [x[1] for x in pm]), ("lambda", lam), ("eval_boot", ev))})
                print(f"[d2] {v} {arm} {g}: sonda {rec['probe']:.3f} rif {rec['ref']:.3f} delta {rec['delta']:+.3f} "
                      f"[{rec['delta_ci_low']:+.3f}, {rec['delta_ci_high']:+.3f}] perm {rec['perm_mean']:+.3f}",
                      flush=True)
    with open(diag.EV / "d2_probe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    r3 = [r["r3_pass"] for r in recs if r["domain"] == PRIMARY and r["gt"] == "sr" and r["arm"] in diag.RULE_ARMS]
    verdict = "si" if all(r3) else ("no" if not any(r3) else "non risolto (semi discordi)")
    diag.atomic_json(diag.EV / "d2_controls.json", {
        "views": {v: _D[v]["ctrl"] for v in views}, "R3_information_in_embedding": verdict,
        "lambdas": LAMBDAS.tolist(), "n_inner": N_INNER, "n_rep": N_REP, "n_boot": a.n_boot,
        "seeds": {"split": diag.SEED_SPLIT, "boot": diag.SEED_BBOOT, "perm": diag.SEED_PERM, "eval": SEED_EVAL},
        "n_eval": a.n_eval,
        "seconds": time.time() - t0})
    diag.atomic_savez(diag.EV / "d2" / "reps.npz", **reps)
    print(f"[d2] R3 (FaceScape SR, entrambi i semi): {verdict}; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
