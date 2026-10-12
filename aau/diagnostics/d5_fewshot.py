#!/usr/bin/env python3
"""D5 (``aau/runs/evidence/diagnostics/PROTOCOL_D5.md``): adattamento few-shot della testa (i-e2) di D3 a un generatore
nuovo. Per bersaglio T le 7 sorgenti del LOGO di D3 e2 piu' k soggetti del pool NON valutato di T, configurazione
(r, lambda) fissa (scelta di D3 e2 per ``T|all``), perdita riscalata, R = 10 estrazioni annidate; valutazione sui soggetti
di test di T col bootstrap per soggetto di D3. DIAGNOSTICA, non un metodo.

    v3_work/unified_gt/run.sh aau/diagnostics/d5_fewshot.py fit --workers 64     (d5.sbatch, ripartibile)
    v3_work/unified_gt/run.sh aau/diagnostics/d5_fewshot.py eval --workers 64    (dopo fit)

File NUOVO: ``diag.py``, ``d3_head.py``, ``d3_stats.py`` e ``d3_e2.py`` sono importati in sola lettura. Pesi: EQ =
``d3_head.Problem`` (ogni blocco pesa lo stesso, ``d3_e2.fit_e2``); W = blocco del bersaglio 1/2, le 7 sorgenti 1/14
(``WProblem``, le formule di Problem con un peso per blocco). ``fit`` scrive una testa per file in ``d5/heads/`` e salta
quelle gia' presenti.

Uscite (``eval``): ``d5_draws.csv``, ``d5_delta.csv``, ``d5/readings.json``, ``d5/controls.json``, ``d5/draws.json``,
``d5/reps.npz``, ``results_d5.md``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import d3_e2 as E2  # noqa: E402
import d3_head as H  # noqa: E402
import d3_pools  # noqa: E402
import d3_stats as S  # noqa: E402
import diag  # noqa: E402

D5 = diag.EV / "d5"
HEADS = D5 / "heads"
ARMS = diag.RULE_ARMS                                   # factorized s1234 e s2345; C3M fuori da D5
KS = (0, 5, 10, 20, 50, 100, 200)
R_DRAWS = 10
SEED_D5 = 20261125
W_TARGET = 0.5                                          # variante W: il blocco del bersaglio pesa 1/2
VARIANTS = ("eq", "w")
K_RULE, SR_MIN, FR_MIN = 20, 0.05, -0.03                # lettura (sez. 4)
NODE_E2 = "i256-a10-10"                                 # nodo della valutazione di D3 e2 (job 1068260), per K0
NAMES = {"facescape": "FaceScape", "hifi3d": "HIFI3D", "faceverse": "FaceVerse neutra", "flame2023_s1": "FLAME 2023"}
ARM_NAMES = {"factorized_s1234": "fact. s1234", "factorized_s2345": "fact. s2345"}
TEST_OF = dict(zip(S.TARGETS, S.PRIMARY))               # bersaglio -> insieme di test
ROWS_OF = {t: ("cross_half" if t == "flame2023_s1" else "cross") for t in S.TARGETS}   # righe di test in D3 e2


# ------------------------------------------------------------------------------------------ pool ed estrazioni

def pool_subjects(t: str) -> list[str]:
    """Soggetti del pool non valutato di T nell'ordine del blocco (id crescente, ``diag.subject_index``): le 200
    sorgenti di D3 per i pool, i soggetti 0-99 di FLAME."""
    if t == "flame2023_s1":
        return diag.subject_index(diag.set_names(t))[1][:E2.N_HALF]
    names = sorted(f"{s}_GTready_{lab}.npz" for s in S.pool_info(t)["source"] for lab in diag.LABELS)
    return diag.subject_index(names)[1]


def n_pool(t: str) -> int:
    return E2.N_HALF if t == "flame2023_s1" else len(S.pool_info(t)["source"])


def ks_of(t: str) -> list[int]:
    """I k effettivi di T: quelli della griglia, al piu' il pool intero (FLAME: 200 -> 100)."""
    return sorted({min(k, n_pool(t)) for k in KS})


def n_draws(t: str, k: int) -> int:
    """k = 0 e il pool intero: un fit solo (le estrazioni coincidono)."""
    return 1 if k in (0, n_pool(t)) else R_DRAWS


def drawn(t: str, k: int, q: int) -> np.ndarray:
    """Indici (nel blocco) dei k soggetti della ripetizione q: i primi k della permutazione di (T, q), ordinati."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED_D5, S.TARGETS.index(t), q]))
    return np.sort(rng.permutation(n_pool(t))[:k])


def tasks() -> list[tuple]:
    """(braccio, T, k, q, variante) di tutte le teste; k = 0 solo EQ (non c'e' blocco del bersaglio)."""
    out = []
    for arm in ARMS:
        for t in S.TARGETS:
            for k in ks_of(t):
                for q in range(n_draws(t, k)):
                    for var in (VARIANTS[:1] if k == 0 else VARIANTS):
                        out.append((arm, t, k, q, var))
    return out


def head_path(arm: str, t: str, k: int, q: int, var: str):
    return HEADS / f"{arm}__{t}__k{k:03d}__q{q}__{var}.npz"


# ------------------------------------------------------------------------------------------ testa pesata

class WProblem(H.Problem):
    """``d3_head.Problem`` con un peso per blocco (normalizzato a somma 1) al posto di 1 / numero di blocchi: w_p, Z e
    alpha0 ricalcolati con le formule di Problem; perdita e gradiente sono quelli di Problem."""

    def __init__(self, blocks: list[H.Block], weights) -> None:
        super().__init__(blocks)
        w = np.asarray(weights, np.float64)
        w = w / w.sum()
        self.Om = [O * len(blocks) * wb for O, wb in zip(self.Om, w)]
        self.Z = float(sum((O * G * G).sum() for O, G in zip(self.Om, self.Gm)))
        num = sum((O * G * np.sqrt(D)).sum() for O, G, D in zip(self.Om, self.Gm, self.Dx2))
        den = sum((O * D).sum() for O, D in zip(self.Om, self.Dx2))
        self.alpha0 = float(num / den)


def fit_w(blocks: list[H.Block], weights, r: int, lam: float, maxiter: int = H.MAXITER) -> H.Head:
    """``d3_head.fit`` con ``WProblem`` (copiata: d3_head non si tocca); r > 0."""
    from scipy.optimize import minimize
    P = WProblem(blocks, weights)
    W0 = H.INIT_SCALE * P.alpha0 * P.basis(r)
    th0 = np.concatenate([[np.log(P.alpha0)], W0.ravel()])
    res = minimize(P.loss, th0, args=(r, lam), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return H.Head(float(np.exp(res.x[0])), res.x[1:].reshape(r, P.p), r, lam, P.alpha0, int(res.nit),
                  bool(res.success), float(res.fun))


# ------------------------------------------------------------------------------------------ fit

def config() -> dict:
    """{(braccio, T): (r, lambda)} scelti dalla CV di D3 e2 per ``T|all`` (d3_e2_heads.csv, kind e2)."""
    out = {}
    for r in csv.DictReader(open(diag.EV / "d3_e2_heads.csv")):
        if r["kind"] == "e2" and r["arm"] in ARMS and r["setting"].endswith("|all"):
            out[(r["arm"], r["setting"].split("|")[0])] = (int(r["r"]), float(r["lambda"]))
    return out


def train_blocks(arm: str, t: str, k: int, q: int) -> tuple[list[H.Block], int]:
    """(blocchi nell'ordine di ``d3_stats.SOURCES``, indice del blocco del bersaglio o -1): le 7 sorgenti di
    ``T|all`` e, se k > 0, il blocco di T ristretto ai k soggetti estratti."""
    bl = [S._B[(arm, s)] for s in S.settings()[f"{t}|all"]]
    if k == 0:
        return bl, -1
    tb = S._B[(arm, t)].take(drawn(t, k, q))
    pos = S.SOURCES.index(t)
    return bl[:pos] + [tb] + bl[pos:], pos


_CFG: dict = {}


def _fit(task):
    arm, t, k, q, var = task
    r, lam = _CFG[(arm, t)]
    blocks, pos = train_blocks(arm, t, k, q)
    t0 = time.time()
    if var == "eq":
        head = E2.fit_e2(blocks, r, lam)
    else:
        w = np.full(len(blocks), (1.0 - W_TARGET) / (len(blocks) - 1))
        w[pos] = W_TARGET
        head = fit_w(E2.rescale(blocks), w, r, lam)
    c = H.calib_c(lambda b, i, j: head.dist(b.X[i], b.X[j]), blocks)      # GT originale, pesi uguali (D3 e2)
    sec = time.time() - t0
    nt = int(len(np.unique(blocks[pos].subj))) if pos >= 0 else 0
    diag.atomic_savez(head_path(*task), alpha=head.alpha, W=head.W, r=head.r, lam=head.lam, alpha0=head.alpha0,
                      nit=head.nit, success=head.success, loss=head.loss, c=c, seconds=sec, n_target=nt,
                      subjects=drawn(t, k, q), host=socket.gethostname())
    return task, sec


def main_fit(a) -> None:
    t0 = time.time()
    info = {"ckpt": {}, "dpu": {}}
    for arm in ARMS:
        _, _, ck = S.store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
        info["ckpt"][arm], info["dpu"][arm] = ck, diag.dp_per_unit(ck)
        S.build_sources(arm, info)
    _CFG.update(config())
    k6 = check_draws()
    HEADS.mkdir(parents=True, exist_ok=True)
    todo = [x for x in tasks() if not head_path(*x).exists()]
    todo.sort(key=lambda x: -x[2])                                          # i blocchi piu' grandi prima
    print(f"[d5] fit: {len(tasks())} teste, {len(todo)} da fare, K6 {k6['pass']}, host {socket.gethostname()} "
          f"({time.time() - t0:.0f}s)", flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for n, (task, sec) in enumerate(pool.imap_unordered(_fit, todo, chunksize=1), 1):
            if n % 25 == 0 or n == len(todo):
                print(f"[d5] fit {n} / {len(todo)} (ultimo {task}, {sec:.0f}s; {time.time() - t0:.0f}s)", flush=True)
    print(f"[d5] fit finiti ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------ controlli sulle estrazioni

def check_draws(tests: dict | None = None) -> dict:
    """K6: estrazioni dentro il pool, annidate in k, disgiunte dai soggetti di test; k soggetti per blocco."""
    out = {}
    for t in S.TARGETS:
        subj = pool_subjects(t)
        if len(subj) != n_pool(t):
            raise SystemExit(f"K6 {t}: pool di {len(subj)} soggetti")
        if t == "flame2023_s1":
            test = set(diag.subject_index(diag.set_names(t))[1][E2.N_HALF:])
        else:
            test = set(d3_pools.subjects(t)["evaluated"])     # i 100 valutati (store ufficiale, d3_pools)
        if tests is not None:                                   # i soggetti delle righe di test davvero usate
            test |= set(tests[TEST_OF[t]]["subjects"])
        if test & set(subj):
            raise SystemExit(f"K6 {t}: soggetti di test nel pool")
        for q in range(R_DRAWS):
            prev = np.zeros(0, dtype=int)
            for k in ks_of(t)[1:]:
                d = drawn(t, k, q)
                if len(d) != k or len(np.unique(d)) != k or d.max() >= n_pool(t) or not np.isin(prev, d).all():
                    raise SystemExit(f"K6 {t} k={k} q={q}: estrazione non valida o non annidata")
                prev = d
        out[t] = {"n_pool": n_pool(t), "n_test_or_evaluated": len(test), "ks": ks_of(t)}
    return {"targets": out, "pass": True}


def draws_record() -> dict:
    """{T: {q: [soggetti nell'ordine della permutazione, il pool intero]}}: i primi k sono l'estrazione (T, k, q)."""
    out = {}
    for t in S.TARGETS:
        subj = pool_subjects(t)
        out[t] = {"seed": f"SeedSequence([{SEED_D5}, {S.TARGETS.index(t)}, q]).permutation({n_pool(t)})",
                  "permutations": {str(q): [subj[i] for i in np.random.default_rng(np.random.SeedSequence(
                      [SEED_D5, S.TARGETS.index(t), q])).permutation(n_pool(t))] for q in range(R_DRAWS)}}
    return out


# ------------------------------------------------------------------------------------------ eval

_J: dict = {}


def _rep(task):
    """Una replica di un insieme di test: rho(shape, SR) e rho(form, FR) di ogni colonna (``diag.wspearman``)."""
    test, b = task
    J = _J[test]
    c = J["counts"][b]
    w = c[J["sa"]] * c[J["sb"]]
    k = w > 0
    w = w[k]
    rg = {g: diag.wranks(x[k], w) for g, x in J["gt"].items()}
    return {m: diag.wpearson(diag.wranks(x[k], w), rg["sr" if m.endswith("shape") else "fr"], w)
            for m, x in J["cols"].items()}


def load_head(arm, t, k, q, var) -> tuple[H.Head, float, dict]:
    with np.load(head_path(arm, t, k, q, var)) as z:
        h = H.Head(float(z["alpha"]), np.asarray(z["W"]), int(z["r"]), float(z["lam"]), float(z["alpha0"]),
                   int(z["nit"]), bool(z["success"]), float(z["loss"]))
        meta = {"c": float(z["c"]), "seconds": float(z["seconds"]), "n_target": int(z["n_target"]),
                "host": str(z["host"]), "subjects": np.asarray(z["subjects"])}
    return h, meta["c"], meta


def test_sets(info: dict) -> dict:
    """{T: righe di test}: viste di fact_paired (righe incrociate), FLAME righe 100-199 (``cross_half`` di D3 e2)."""
    views = [TEST_OF[t] for t in S.TARGETS if t != "flame2023_s1"]
    with mp.get_context("fork").Pool(len(views)) as pool:
        vd = dict(zip(views, pool.map(S.view_data, views)))
    out = {}
    for t in S.TARGETS:
        if t == "flame2023_s1":
            T = S.set_data(t, info)
            rs = E2.rowsets_e2(T)["cross_half"]
            sa, sb, gt, Z, ma, mb, cnt = rs
            subjects = T["subjects"][E2.N_HALF:]
            T["seed"] = f"diag.boot_counts({E2.N_HALF}, 0, seed={E2.SEED_HALF}) (cross_half di D3 e2)"
        else:
            T = vd[TEST_OF[t]]
            sa, sb, gt, Z, ma, mb, cnt = T["sa"], T["sb"], T["gt"], T["Z"], T["ma"], T["mb"], T["counts"]
            subjects = T["subjects"]
        for arm in ARMS:
            if T["ckpt"][arm].resolve() != info["ckpt"][arm].resolve():
                raise SystemExit(f"{t} {arm}: checkpoint {T['ckpt'][arm]}")
        out[TEST_OF[t]] = {"sa": sa, "sb": sb, "gt": {g: gt[g] for g in ("sr", "fr")}, "Z": Z, "ma": ma, "mb": mb,
                           "counts": cnt, "subjects": list(subjects), "seed": T["seed"]}
    return out


def columns(T: dict, heads: dict, cal: dict, info: dict, t: str) -> dict:
    """d_P e d_F cal. dei bracci, d_h e d_F della testa per ogni (k, q, variante) di T."""
    cols = {}
    for arm in ARMS:
        Z = T["Z"][arm]
        S_ = np.exp(Z[:, 0])
        X = Z[:, 1:] * info["dpu"][arm]
        Sa, Sb, Xa, Xb = S_[T["ma"]], S_[T["mb"]], X[T["ma"]], X[T["mb"]]

        def form(d, c):
            return np.sqrt((Sa - Sb) ** 2 + Sa * Sb * (c * d) ** 2)

        dP = np.linalg.norm(Xa - Xb, axis=1)
        cols[f"{arm}|dP|shape"], cols[f"{arm}|dP|form"] = dP, form(dP, cal[arm])
        for (a_, t_, k, q, var), (h, c, _) in heads.items():
            if a_ == arm and t_ == t:
                d = h.dist(Xa, Xb)
                cols[f"{arm}|k{k}q{q}{var}|shape"], cols[f"{arm}|k{k}q{q}{var}|form"] = d, form(d, c)
    bad = [k for k, v in cols.items() if not np.isfinite(v).all()]
    if bad:
        raise SystemExit(f"{t}: colonne non finite {bad[:4]}")
    return cols


def main_eval(a) -> None:
    t0 = time.time()
    cal = diag.calibration()
    info = {"ckpt": {}, "dpu": {}}
    for arm in ARMS:
        _, _, ck = S.store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
        info["ckpt"][arm], info["dpu"][arm] = ck, diag.dp_per_unit(ck)
    miss = [x for x in tasks() if not head_path(*x).exists()]
    if miss:
        raise SystemExit(f"teste mancanti: {len(miss)} (es. {miss[:2]})")
    heads = {x: load_head(*x) for x in tasks()}
    tests = test_sets(info)
    k6 = check_draws(tests)
    print(f"[d5] {len(heads)} teste, insiemi di test {[(n, len(T['sa'])) for n, T in tests.items()]} "
          f"({time.time() - t0:.0f}s)", flush=True)
    for t in S.TARGETS:
        T = tests[TEST_OF[t]]
        _J[TEST_OF[t]] = {"counts": T["counts"][: a.n_boot + 1], "sa": T["sa"], "sb": T["sb"], "gt": T["gt"],
                          "cols": columns(T, heads, cal, info, t)}
    keys = [(test, b) for test in _J for b in range(a.n_boot + 1)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        reps = pool.map(_rep, keys, chunksize=4)
    V = {test: {} for test in _J}
    for (test, b), r in zip(keys, reps):
        for m, x in r.items():
            V[test].setdefault(m, np.empty(a.n_boot + 1))[b] = x
    print(f"[d5] repliche ({time.time() - t0:.0f}s)", flush=True)
    # ------------------------------------------------------------------ delta per estrazione
    DIST = {"sr": "shape", "fr": "form"}

    def delta(arm, t, k, q, var, g):
        """Delta testa - d_P di (braccio, T, k, q, variante) per replica; k = 0 e' lo stesso fit per le due varianti."""
        Vt = V[TEST_OF[t]]
        v = "eq" if k == 0 else var
        return Vt[f"{arm}|k{k}q{q}{v}|{DIST[g]}"] - Vt[f"{arm}|dP|{DIST[g]}"]

    def draws_of(arm, t, k, var, g) -> np.ndarray:
        """(R', n_boot + 1): i delta delle estrazioni di (braccio, T, k)."""
        return np.stack([delta(arm, t, k, q, var, g) for q in range(n_draws(t, k))])

    recs_draw = []
    for (arm, t, k, q, var), (h, c, meta) in sorted(heads.items()):
        recs_draw.append({"arm": arm, "target": t, "test": TEST_OF[t], "k": k, "q": q, "variant": var,
                          "n_target_subjects": meta["n_target"], "r": h.r, "lambda": h.lam,
                          "alpha_over_alpha0": h.alpha / h.alpha0, "c": c, "nit": h.nit, "success": h.success,
                          "seconds": meta["seconds"], "host": meta["host"],
                          "delta_sr": float(delta(arm, t, k, q, var, "sr")[0]),
                          "delta_fr": float(delta(arm, t, k, q, var, "fr")[0]),
                          "rho_sr": float(V[TEST_OF[t]][f"{arm}|k{k}q{q}{var}|shape"][0]),
                          "rho_fr": float(V[TEST_OF[t]][f"{arm}|k{k}q{q}{var}|form"][0])})
    # ------------------------------------------------------------------ aggregati
    recs, AG = [], {}

    def add(arm, target, kcol, keff, var, what, g, x, pts):
        """x = replica per replica del delta medio sulle estrazioni; pts = i delta puntuali delle estrazioni."""
        p, lo, hi, ple = diag.ci(x)
        pts = np.asarray(pts, np.float64)
        rec = {"arm": arm, "target": target, "k": kcol, "k_eff": keff, "variant": var, "what": what, "gt": g,
               "delta": p, "ci_low": lo, "ci_high": hi, "p_le0": ple, "n_draws": int(len(pts)),
               "draws_sd": float(pts.std(ddof=1)) if len(pts) > 1 else 0.0, "draws_min": float(pts.min()),
               "draws_max": float(pts.max()), "draws_frac_ge_sr_min": float((pts >= SR_MIN).mean())}
        recs.append(rec)
        AG[(arm, target, kcol, var, what, g)] = (rec, x, pts)
        return rec

    for arm in ARMS:
        for var in VARIANTS:
            for g in ("sr", "fr"):
                for kcol in KS:
                    xs, ps = [], []
                    for t in S.TARGETS:
                        keff = min(kcol, n_pool(t))
                        D = draws_of(arm, t, keff, var, g)
                        x = D.mean(0)
                        add(arm, t, kcol, keff, var, "vs_dP", g, x, D[:, 0])
                        x0 = delta(arm, t, 0, 0, "eq", g)
                        add(arm, t, kcol, keff, var, "vs_k0", g, x - x0, D[:, 0] - x0[0])
                        xs.append(x)
                        ps.append(np.resize(D[:, 0], R_DRAWS))      # pool intero / k = 0: lo stesso valore per ogni q
                    add(arm, "mean4", kcol, "-", var, "vs_dP", g, np.mean(xs, 0), np.mean(ps, 0))
                    x0 = np.mean([delta(arm, t, 0, 0, "eq", g) for t in S.TARGETS], 0)
                    add(arm, "mean4", kcol, "-", var, "vs_k0", g, np.mean(xs, 0) - x0, np.mean(ps, 0) - x0[0])
            if var == "w":                                              # W - EQ, replica per replica
                for g in ("sr", "fr"):
                    for kcol in KS[1:]:
                        xs, ps = [], []
                        for t in S.TARGETS:
                            keff = min(kcol, n_pool(t))
                            D = draws_of(arm, t, keff, "w", g) - draws_of(arm, t, keff, "eq", g)
                            add(arm, t, kcol, keff, "w", "w_minus_eq", g, D.mean(0), D[:, 0])
                            xs.append(D.mean(0))
                            ps.append(np.resize(D[:, 0], R_DRAWS))
                        add(arm, "mean4", kcol, "-", "w", "w_minus_eq", g, np.mean(xs, 0), np.mean(ps, 0))
    # ------------------------------------------------------------------ letture
    readings = {"protocol": "PROTOCOL_D5.md", "rule": {"k": K_RULE, "sr_min": SR_MIN, "fr_min": FR_MIN},
                "F": {}, "verdict": {}, "k_min_F": {}, "k_half": {}, "k_half_increment": {}, "k_half_target": {},
                "frac_draws_mean4_ge_sr_min": {}, "n_targets_sr_at_k20": {}}

    def F(arm, kcol, var):
        m = AG[(arm, "mean4", kcol, var, "vs_dP", "sr")][0]
        fr = {t: AG[(arm, t, kcol, var, "vs_dP", "fr")][0]["delta"] for t in S.TARGETS}
        ok = bool(m["delta"] >= SR_MIN and m["ci_low"] > 0 and all(v >= FR_MIN for v in fr.values()))
        return {"pass": ok, "mean4_sr": m["delta"], "ci": [m["ci_low"], m["ci_high"]], "fr": fr}

    for var in VARIANTS:
        per = {arm: F(arm, K_RULE, var) for arm in ARMS}
        n = sum(per[arm]["pass"] for arm in ARMS)
        readings["F"][var] = per
        readings["verdict"][var] = "FUNZIONA" if n == len(ARMS) else ("NON CONFERMATO" if n else "NO")
        for arm in ARMS:
            ok = [k for k in KS[1:] if F(arm, k, var)["pass"]]
            readings["k_min_F"][f"{arm}|{var}"] = ok[0] if ok else None
            G = {k: AG[(arm, "mean4", k, var, "vs_dP", "sr")][0]["delta"] for k in KS}
            half = [k for k in KS if G[200] > 0 and G[k] >= G[200] / 2]
            readings["k_half"][f"{arm}|{var}"] = {"k": half[0] if half else None, "G": G}
            inc = [k for k in KS if G[200] > G[0] and G[k] - G[0] >= (G[200] - G[0]) / 2]
            readings["k_half_increment"][f"{arm}|{var}"] = inc[0] if inc else None
            for t in S.TARGETS:
                Gt = {k: AG[(arm, t, k, var, "vs_dP", "sr")][0]["delta"] for k in KS}
                ht = [k for k in KS if Gt[200] > 0 and Gt[k] >= Gt[200] / 2]
                readings["k_half_target"][f"{arm}|{var}|{t}"] = ht[0] if ht else None
            readings["frac_draws_mean4_ge_sr_min"][f"{arm}|{var}"] = {
                k: AG[(arm, "mean4", k, var, "vs_dP", "sr")][0]["draws_frac_ge_sr_min"] for k in (5, 10, 20)}
            readings["n_targets_sr_at_k20"][f"{arm}|{var}"] = int(sum(
                AG[(arm, t, K_RULE, var, "vs_dP", "sr")][0]["delta"] >= SR_MIN
                and AG[(arm, t, K_RULE, var, "vs_dP", "sr")][0]["ci_low"] > 0 for t in S.TARGETS))
    # ------------------------------------------------------------------ controlli
    pub = {(r["test"], r["rows"], r["arm"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_e2_delta.csv"))
           if r["kind"] == "ref" and r["method"] == "i_e2-dP"}
    k0, worst0 = {}, 0.0
    for arm in ARMS:
        for t in S.TARGETS:
            for g in ("sr", "fr"):
                r = pub[(TEST_OF[t], ROWS_OF[t], arm, g)]
                p, lo, hi, _ = diag.ci(delta(arm, t, 0, 0, "eq", g))
                d = max(abs(p - float(r["delta"])), abs(lo - float(r["ci_low"])), abs(hi - float(r["ci_high"])))
                worst0 = max(worst0, d)
                k0[f"{arm}|{t}|{g}"] = {"d5": p, "d3_e2": float(r["delta"]), "max_abs_diff": d}
    pubr = {(r["test"], r["rows"], r["column"], r["gt"]): r for r in csv.DictReader(open(diag.EV / "d3_e2_spearman.csv"))}
    k1, worst1 = {}, 0.0
    for arm in ARMS:
        for t in S.TARGETS:
            for g in ("sr", "fr"):
                col = f"{arm}|dP|{DIST[g]}"
                r = pubr[(TEST_OF[t], ROWS_OF[t], col, g)]
                p, lo, hi, _ = diag.ci(V[TEST_OF[t]][col])
                d = max(abs(p - float(r["rho"])), abs(lo - float(r["ci_low"])), abs(hi - float(r["ci_high"])))
                rows_ok = int(r["n_rows"]) == len(tests[TEST_OF[t]]["sa"])
                worst1 = max(worst1, d)
                k1[f"{arm}|{t}|{g}"] = {"max_abs_diff": d, "rows": int(len(tests[TEST_OF[t]]["sa"])), "rows_equal": rows_ok}
    hosts = sorted({m["host"] for _, _, m in heads.values()})
    controls = {"K0": {"cells": k0, "max_abs_diff": worst0, "pass": bool(worst0 <= 1e-9)},
                "K1": {"cells": k1, "max_abs_diff": worst1,
                       "pass": bool(worst1 <= 1e-9 and all(x["rows_equal"] for x in k1.values()))},
                "K6": k6,
                "fits": {"n": len(heads), "success": int(sum(h.success for h, _, _ in heads.values())),
                         "nit_min": int(min(h.nit for h, _, _ in heads.values())),
                         "nit_max": int(max(h.nit for h, _, _ in heads.values())),
                         "n_target_ok": bool(all(m["n_target"] == k for (_, _, k, _, _), (_, _, m) in heads.items())),
                         "seconds_median": float(np.median([m["seconds"] for _, _, m in heads.values()])),
                         "hosts": hosts, "host_e2": NODE_E2},
                "config": {f"{arm}|{t}": list(v) for (arm, t), v in config().items()},
                "tests": {n: {"rows": int(len(T["sa"])), "subjects": len(T["subjects"]), "seed": str(T["seed"])}
                          for n, T in tests.items()},
                "n_boot": a.n_boot, "seconds": time.time() - t0, "workers": a.workers, "host_eval": socket.gethostname()}
    readings["controls_pass"] = {"K0": controls["K0"]["pass"], "K1": controls["K1"]["pass"]}
    # ------------------------------------------------------------------ uscite
    for name, rr in (("d5_draws.csv", recs_draw), ("d5_delta.csv", recs)):
        with open(diag.EV / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rr for k in r)))
            w.writeheader()
            w.writerows(rr)
    diag.atomic_json(D5 / "readings.json", readings)
    diag.atomic_json(D5 / "controls.json", controls)
    diag.atomic_json(D5 / "draws.json", draws_record())
    diag.atomic_savez(D5 / "reps.npz", **{f"{test}|{m}": x for test, Vt in V.items() for m, x in Vt.items()})
    report(AG, readings, controls)
    print(f"[d5] verdetto EQ {readings['verdict']['eq']}, W {readings['verdict']['w']}; K0 {worst0:.1e}, K1 "
          f"{worst1:.1e}; {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------------------------------ results_d5.md

def fmt(x: float) -> str:
    return f"{x:+.3f}"


def cell(rec: dict, sd: bool = True) -> str:
    s = f"{fmt(rec['delta'])} [{fmt(rec['ci_low'])}, {fmt(rec['ci_high'])}]"
    return s + (f" sd {rec['draws_sd']:.3f}" if sd and rec["draws_sd"] > 1e-12 else "")


def joint_d3() -> dict:
    """Testa congiunta di D3 e2 (descrittiva): {(braccio, T, gt): rec} sulle righe di test di D5."""
    out = {}
    for r in csv.DictReader(open(diag.EV / "d3_e2_delta.csv")):
        if r["kind"] == "ref" and r["method"] == "joint-dP" and r["arm"] in ARMS:
            for t in S.TARGETS:
                if r["test"] == TEST_OF[t] and r["rows"] == ROWS_OF[t]:
                    out[(r["arm"], t, r["gt"])] = {k: float(r[k]) for k in ("delta", "ci_low", "ci_high")}
    return out


def report(AG: dict, rd: dict, ct: dict) -> None:
    sha = (diag.EV / "PROTOCOL_D5.sha256").read_text().split()[0]
    L = ["# Diagnostica D5: adattamento few-shot della testa (i-e2) a un generatore nuovo, risultati", "",
         f"Protocollo `PROTOCOL_D5.md` (sha256 {sha[:8]}..., commit 484d638), scritto prima dei numeri. FaceScape, "
         "HIFI3D e FaceVerse sono campioni di 3DMM non visti dai bracci, non dati reali. Generato da "
         "`aau/diagnostics/d5_fewshot.py`; numeri in `d5_delta.csv` (aggregati) e `d5_draws.csv` (per estrazione). "
         "Delta = rho(testa) - rho(d_P) dello stesso braccio sulle righe di test di T (viste: righe incrociate dei 100 "
         "soggetti valutati; FLAME: righe fra i soggetti 100-199). Punto e IC 95% = delta medio sulle estrazioni, "
         "bootstrap per soggetto di test (1000 repliche, condizionato alle estrazioni e ai fit); sd = deviazione standard "
         "fra le 10 estrazioni (k = 0 e pool intero: un fit). FLAME ha un pool di 100: nella colonna k = 200 entra con "
         "k = 100.", "", "## Lettura preregistrata", ""]
    for var, lab in (("eq", "EQ (primaria, peso uguale per sorgente)"), ("w", "variante W (bersaglio 1/2, secondaria)")):
        L.append(f"- **{lab}: {rd['verdict'][var]}**")
        for arm in ARMS:
            f = rd["F"][var][arm]
            fr = ", ".join(f"{NAMES[t]} {fmt(v)}" for t, v in f["fr"].items())
            kh = rd["k_half"][f"{arm}|{var}"]
            L.append(f"  - {ARM_NAMES[arm]}: F(k = 20) {'SI' if f['pass'] else 'NO'}; media sui 4 SR {fmt(f['mean4_sr'])} "
                     f"[{fmt(f['ci'][0])}, {fmt(f['ci'][1])}]; FR per bersaglio {fr}; k minimo con F: "
                     f"{rd['k_min_F'][f'{arm}|{var}']}; k a meta' guadagno {kh['k']} (G(200) = {fmt(kh['G'][200])}), "
                     f"sull'incremento {rd['k_half_increment'][f'{arm}|{var}']}; bersagli con SR >= +0.05 e IC > 0 a "
                     f"k = 20: {rd['n_targets_sr_at_k20'][f'{arm}|{var}']} su 4")
    L += ["", f"Controlli: K0 (k = 0 = LOGO di D3 e2) scarto massimo {ct['K0']['max_abs_diff']:.1e}, passa "
          f"{ct['K0']['pass']}; K1 (d_P) {ct['K1']['max_abs_diff']:.1e}, passa {ct['K1']['pass']}; K6 {ct['K6']['pass']}; "
          f"fit {ct['fits']['success']} / {ct['fits']['n']} con success, iterazioni {ct['fits']['nit_min']}-"
          f"{ct['fits']['nit_max']}, nodi {', '.join(ct['fits']['hosts'])}.", ""]
    for var, lab in (("eq", "EQ"), ("w", "variante W")):
        for g, gl in (("sr", "d_P con SR (primaria)"), ("fr", "d_F cal. con FR (vincolo)")):
            L += [f"## Guadagno contro d_P per bersaglio e k, {lab}, {gl}", ""]
            for arm in ARMS:
                L += [f"**{ARM_NAMES[arm]}**", "", "| bersaglio | " + " | ".join(f"k = {k}" for k in KS) + " |",
                      "|---|" + "---|" * len(KS)]
                for t in S.TARGETS + ("mean4",):
                    name = "media dei 4" if t == "mean4" else NAMES[t]
                    L.append(f"| {name} | " + " | ".join(cell(AG[(arm, t, k, var, 'vs_dP', g)][0]) for k in KS) + " |")
                L.append("")
    L += ["## Incremento su k = 0 (media dei 4, SR), EQ e W", "", "| braccio | variante | " +
          " | ".join(f"k = {k}" for k in KS[1:]) + " |", "|---|---|" + "---|" * (len(KS) - 1)]
    for arm in ARMS:
        for var in VARIANTS:
            L.append(f"| {ARM_NAMES[arm]} | {var.upper()} | " +
                     " | ".join(cell(AG[(arm, 'mean4', k, var, 'vs_k0', 'sr')][0]) for k in KS[1:]) + " |")
    L += ["", "## W - EQ (SR), replica per replica", "", "| braccio | bersaglio | " +
          " | ".join(f"k = {k}" for k in KS[1:]) + " |", "|---|---|" + "---|" * (len(KS) - 1)]
    for arm in ARMS:
        for t in S.TARGETS + ("mean4",):
            L.append(f"| {ARM_NAMES[arm]} | {'media dei 4' if t == 'mean4' else NAMES[t]} | " +
                     " | ".join(cell(AG[(arm, t, k, 'w', 'w_minus_eq', 'sr')][0], sd=False) for k in KS[1:]) + " |")
    L += ["", "## Dispersione fra estrazioni (media dei 4, SR, EQ)", "",
          "| braccio | k | sd | min | max | estrazioni >= +0.05 |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        for k in KS[1:-1]:
            r = AG[(arm, "mean4", k, "eq", "vs_dP", "sr")][0]
            L.append(f"| {ARM_NAMES[arm]} | {k} | {r['draws_sd']:.3f} | {fmt(r['draws_min'])} | {fmt(r['draws_max'])} | "
                     f"{round(r['draws_frac_ge_sr_min'] * R_DRAWS)} / {R_DRAWS} |")
    J = joint_d3()
    L += ["", "## k = 200 accanto alla testa congiunta di D3 e2 (descrittivo, SR)", "",
          "La congiunta ha in piu' i pool degli altri bersagli e FLAME 0-99: non e' lo stesso disegno.", "",
          "| bersaglio | braccio | D5 k = 0 | D5 k = 200 EQ | D5 k = 200 W | congiunta D3 e2 |", "|---|---|---|---|---|---|"]
    for t in S.TARGETS:
        for arm in ARMS:
            j = J.get((arm, t, "sr"))
            L.append(f"| {NAMES[t]} | {ARM_NAMES[arm]} | {cell(AG[(arm, t, 0, 'eq', 'vs_dP', 'sr')][0])} | "
                     f"{cell(AG[(arm, t, 200, 'eq', 'vs_dP', 'sr')][0])} | {cell(AG[(arm, t, 200, 'w', 'vs_dP', 'sr')][0])} | "
                     + (f"{fmt(j['delta'])} [{fmt(j['ci_low'])}, {fmt(j['ci_high'])}]" if j else "-") + " |")
    L += ["", "## k a meta' guadagno per bersaglio (SR, G(k) >= G(200) / 2)", "", "| braccio | variante | " +
          " | ".join(NAMES[t] for t in S.TARGETS) + " | media dei 4 |", "|---|---|" + "---|" * (len(S.TARGETS) + 1)]
    for arm in ARMS:
        for var in VARIANTS:
            L.append(f"| {ARM_NAMES[arm]} | {var.upper()} | " +
                     " | ".join(str(rd['k_half_target'][f'{arm}|{var}|{t}']) for t in S.TARGETS) +
                     f" | {rd['k_half'][f'{arm}|{var}']['k']} |")
    L.append("")
    (diag.EV / "results_d5.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("fit", "eval"))
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=diag.N_BOOT)
    a = ap.parse_args()
    if a.step == "fit":
        main_fit(a)
    else:
        main_eval(a)


if __name__ == "__main__":
    main()
