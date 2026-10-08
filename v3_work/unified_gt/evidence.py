#!/usr/bin/env python3
"""Passo 5: evidenze E8 sulla GT unificata (a, b, c) e la variante con Procrustes per coppia.

    v3_work/unified_gt/run.sh v3_work/unified_gt/evidence.py      (dopo shapes.py)

(a) Spearman fra GT unificata, GT maxabs e GT dei coefficienti (e variante per coppia, e le GT
    intermedie sulla patch nativa di native_gt.py) sugli stessi
    100 soggetti valutati di HIFI3D e FaceVerse (``subjects.json`` delle run esistenti), 4.950
    coppie; IC 95% bootstrap per soggetto (1000 repliche, seme 1234), ricampionamento di
    ``zs_summarize.paired_bootstrap`` (coppia pesata per il prodotto dei conteggi). Anche sul pool
    di 500 (solo punto).
(b) quasi-duplicati FRA domini: per ogni identita' di test (100 HIFI3D, 100 FaceVerse) la distanza
    unificata dal vicino piu' prossimo in ogni pool di training del run su scala (split esplicito
    ``aau/data_scale/split_scale_all.json``: BFM 392, ICT 54.008, GNM 10.000), contro il vicino piu'
    prossimo DENTRO il test (fra i 100 e fra i 500 del pool). Il vicino piu' prossimo dipende dalla
    taglia del pool, quindi il confronto fra domini e' anche a taglia uguale: 392 (tutti e tre) e
    10.000 (ICT contro GNM), 50 sottocampioni a seme fisso; per identita' di test la mediana sui
    sottocampioni, poi differenze appaiate con IC bootstrap sui soggetti di test.
(c) distanze fra le medie dei domini nello spazio unificato (le medie dei template, allineate a mu),
    accanto alla dispersione interna di ogni dominio.

Scrive i csv in ``aau/runs/evidence/e8/`` e ``evidence.json`` (letto da report.py).
"""

from __future__ import annotations

import json
import sys

import numpy as np

import ugt as C
import domains
from shapes import SHAPES_DIR, GT_DIR, Space, unified_distances, pairwise_distances

sys.path.insert(0, str(C.REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic"))
from intrinsic_utils import SUBJECT_RE_ANY, load_gt_distance_matrix  # noqa: E402

RUNS = C.REPO_ROOT / "aau" / "runs"
EVAL_SUBJECTS = {
    "hifi3d": RUNS / "ws_hifi3d" / "data_328f2bfc1a" / "bfm_only" / "subjects.json",
    "faceverse": RUNS / "ws_faceverse" / "data_184ec4171e" / "joint" / "subjects.json",
}
OLD_GT = {
    "hifi3d": {"maxabs": C.REPO_ROOT / "datasets/HIFI3D/eval_view/gt_matrix.npz",
               "coef": C.REPO_ROOT / "datasets/HIFI3D/eval_view/gt_coef_matrix.npz"},
    "faceverse": {"maxabs": C.REPO_ROOT / "datasets/FACEVERSE_ZS/eval_view/gt_matrix.npz",
                  "coef": C.REPO_ROOT / "datasets/FACEVERSE_ZS/eval_view/gt_coef_matrix.npz"},
}
SPLIT = C.REPO_ROOT / "aau" / "data_scale" / "split_scale_all.json"
N_BOOT = 1000
SEED = 1234
NATIVE = ("native_maxabs_rms", "native_sim", "native_sim_area", "native_sim_area_region")  # native_gt.py
N_SUB = 50


def load_set(name: str) -> dict:
    with np.load(SHAPES_DIR / f"{name}.npz") as z:
        return {"s": z["s"], "ids": [str(x) for x in z["ids"]], "scale": z["scale"], "rms": z["rms_to_mu"]}


def gt_on(path, subjects: list[str]) -> np.ndarray:
    D, idx = load_gt_distance_matrix(str(path), subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    ii = np.array([idx[s] for s in subjects])
    return D[np.ix_(ii, ii)]


def rankdata(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata as rd
    return rd(x)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def boot_spearman(mats: dict, n_boot: int, seed: int) -> dict:
    """Spearman fra tutte le coppie di matrici (n, n) sulle coppie i<j, punto e IC bootstrap per
    soggetto (stesse repliche per tutte le coppie di GT)."""
    names = list(mats)
    n = len(next(iter(mats.values())))
    iu, ju = np.triu_indices(n, 1)
    vals = {k: m[iu, ju] for k, m in mats.items()}
    rng = np.random.default_rng(seed)
    reps = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1:]}
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        wt = counts[iu].astype(np.int64) * counts[ju]
        keep = wt > 0
        rep = {k: np.repeat(v[keep], wt[keep]) for k, v in vals.items()}
        for a, b in reps:
            reps[(a, b)].append(spearman(rep[a], rep[b]))
    out = {}
    for (a, b), r in reps.items():
        lo, hi = np.percentile(r, [2.5, 97.5])
        out[f"{a}|{b}"] = {"point": spearman(vals[a], vals[b]), "ci_low": float(lo), "ci_high": float(hi)}
    return out


def part_a(sp: Space, sets: dict) -> dict:
    res = {}
    for dom in ("hifi3d", "faceverse"):
        subj = json.loads(EVAL_SUBJECTS[dom].read_text())["subjects"]
        ids = sets[dom]["ids"]
        pos = {s: i for i, s in enumerate(ids)}
        mats = {"unified": gt_on(GT_DIR / f"{dom}_unified.npz", subj),
                "unified_pairwise": gt_on(GT_DIR / f"{dom}_unified_pairwise.npz", subj),
                "maxabs": gt_on(OLD_GT[dom]["maxabs"], subj),
                "coef": gt_on(OLD_GT[dom]["coef"], subj),
                **{k: gt_on(GT_DIR / f"{dom}_{k}.npz", subj) for k in NATIVE}}
        # controllo: la GT unificata riletta coincide col ricalcolo da s
        S = sets[dom]["s"][[pos[s] for s in subj]]
        D = unified_distances(S, S, sp.A)
        np.fill_diagonal(D, 0)
        Du = mats["unified"]
        iu0 = np.triu_indices(len(D), 1)
        check = float(np.abs(Du[iu0] / Du[iu0].max() - D[iu0] / D[iu0].max()).max())
        r = boot_spearman(mats, N_BOOT, SEED)
        pool = {"unified": gt_on(GT_DIR / f"{dom}_unified.npz", ids),
                "unified_pairwise": gt_on(GT_DIR / f"{dom}_unified_pairwise.npz", ids),
                "maxabs": gt_on(OLD_GT[dom]["maxabs"], ids), "coef": gt_on(OLD_GT[dom]["coef"], ids),
                **{k: gt_on(GT_DIR / f"{dom}_{k}.npz", ids) for k in NATIVE}}
        iu = np.triu_indices(len(ids), 1)
        pool_r = {}
        names = list(pool)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pool_r[f"{a}|{b}"] = spearman(pool[a][iu], pool[b][iu])
        iu1 = np.triu_indices(len(subj), 1)
        g = D[iu1]
        gp_raw = pairwise_distances(S, sp)[iu1]
        res[dom] = {"n_subjects": len(subj), "n_pairs": int(len(g)), "subjects_first": subj[:3],
                    "spearman_eval": r, "spearman_pool500": pool_r,
                    "pearson_unified_vs_pairwise": float(np.corrcoef(g, gp_raw)[0, 1]),
                    "pairwise_over_global_median": float(np.median(gp_raw / g)),
                    "unified_mm": {"median": float(np.median(g)), "min": float(g.min()), "max": float(g.max()),
                                   "p5": float(np.percentile(g, 5))},
                    "reload_check_max_abs_rel": check}
    return res


def nn_dist(S_test: np.ndarray, S_pool: np.ndarray, A: float, chunk: int = 20000) -> np.ndarray:
    best = np.full(len(S_test), np.inf)
    for k in range(0, len(S_pool), chunk):
        best = np.minimum(best, unified_distances(S_test, S_pool[k:k + chunk], A).min(1))
    return best


def part_b(sp: Space, sets: dict) -> dict:
    split = json.loads(SPLIT.read_text())
    train = set(split["train"])
    pools = {}
    for dom in ("bfm", "ict", "gnm"):
        ids = sets[dom]["ids"]
        sel = np.array([s in train for s in ids])
        pools[dom] = sets[dom]["s"][sel]
    sizes = {d: int(len(v)) for d, v in pools.items()}
    res = {"pool_sizes": sizes}
    rng_master = np.random.default_rng(SEED)
    sub_idx = {n: {d: [rng_master.choice(len(pools[d]), n, replace=False) for _ in range(N_SUB)]
                   for d in pools if len(pools[d]) >= n} for n in (392, 10000)}
    for dom in ("hifi3d", "faceverse"):
        subj = json.loads(EVAL_SUBJECTS[dom].read_text())["subjects"]
        pos = {s: i for i, s in enumerate(sets[dom]["ids"])}
        St = sets[dom]["s"][[pos[s] for s in subj]]
        S500 = sets[dom]["s"]
        Dtt = unified_distances(St, St, sp.A)
        np.fill_diagonal(Dtt, np.inf)
        D500 = unified_distances(St, S500, sp.A)
        D500[np.arange(len(subj)), [pos[s] for s in subj]] = np.inf
        within100, within500 = Dtt.min(1), D500.min(1)
        full = {d: nn_dist(St, pools[d], sp.A) for d in pools}
        eq = {}
        for n, per in sub_idx.items():
            eq[n] = {d: np.median(np.stack([nn_dist(St, pools[d][ix], sp.A) for ix in per[d]]), axis=0)
                     for d in per}
        # differenze appaiate (per soggetto di test) a taglia uguale, IC bootstrap sui soggetti
        rng = np.random.default_rng(SEED)
        boots = [rng.integers(0, len(subj), len(subj)) for _ in range(N_BOOT)]

        def paired(x, y):
            d = np.log(x) - np.log(y)
            b = [np.mean(d[ix]) for ix in boots]
            lo, hi = np.percentile(b, [2.5, 97.5])
            return {"mean_log_ratio": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                    "ratio_geomean": float(np.exp(d.mean())), "frac_first_closer": float((d < 0).mean())}

        cmp = {"n392_gnm_vs_ict": paired(eq[392]["gnm"], eq[392]["ict"]),
               "n392_gnm_vs_bfm": paired(eq[392]["gnm"], eq[392]["bfm"]),
               "n392_ict_vs_bfm": paired(eq[392]["ict"], eq[392]["bfm"]),
               "n10000_gnm_vs_ict": paired(eq[10000]["gnm"], eq[10000]["ict"])}
        best_train = np.min(np.stack(list(full.values())), axis=0)
        q = lambda x: {"median": float(np.median(x)), "p5": float(np.percentile(x, 5)),  # noqa: E731
                       "min": float(np.min(x)), "p95": float(np.percentile(x, 95))}
        res[dom] = {
            "n_test": len(subj),
            "nn_within_test100": q(within100), "nn_within_pool500": q(within500),
            "nn_train_full": {d: q(v) for d, v in full.items()},
            "nn_train_eqsize": {str(n): {d: q(v) for d, v in per.items()} for n, per in eq.items()},
            "paired_eqsize": cmp,
            "n_train_closer_than_within500": int((best_train < within500).sum()),
            "n_train_closer_than_within100": int((best_train < within100).sum()),
            "n_train_below_min_within500": int((best_train < within500.min()).sum()),
            "argmin_domain_full": {d: int((np.argmin(np.stack([full[k] for k in pools]), 0) == i).sum())
                                   for i, d in enumerate(pools)},
        }
        np.savez(C.EVID_DIR / f"nn_{dom}.npz", subjects=np.array(subj), within100=within100, within500=within500,
                 **{f"full_{d}": v for d, v in full.items()},
                 **{f"eq{n}_{d}": v for n, per in eq.items() for d, v in per.items()})
    return res


def part_c(sp: Space, sets: dict) -> dict:
    z = sp.z
    doms = list(domains.DOMAINS)
    M = np.stack([sp.svec(z[f"mean_{d}"][None])[0] for d in doms])
    Dm = unified_distances(M, M, sp.A)
    np.fill_diagonal(Dm, 0)
    spread = {}
    rng = np.random.default_rng(SEED)
    for name, st in sets.items():
        S = st["s"]
        if len(S) > 2000:
            S = S[rng.choice(len(S), 2000, replace=False)]
        D = unified_distances(S, S, sp.A)
        iu = np.triu_indices(len(S), 1)
        np.fill_diagonal(D, np.inf)
        dom = name
        dm = unified_distances(S, M[[doms.index(dom)]], sp.A)[:, 0]
        emp = S.astype(np.float64).mean(0)
        spread[name] = {"n": int(len(st["s"])), "pairwise_median": float(np.median(D[iu])),
                        "nn_median": float(np.median(D.min(1))),
                        "to_template_mean_median": float(np.median(dm)),
                        "empirical_mean_vs_template_mean": float(
                            unified_distances(emp[None].astype(np.float32), M[[doms.index(dom)]], sp.A)[0, 0])}
    # distanza di ogni identita' di test dalle medie dei domini di training
    to_means = {}
    subj_sets = {d: json.loads(EVAL_SUBJECTS[d].read_text())["subjects"] for d in EVAL_SUBJECTS}
    for dom, subj in subj_sets.items():
        pos = {s: i for i, s in enumerate(sets[dom]["ids"])}
        St = sets[dom]["s"][[pos[s] for s in subj]]
        Dm_t = unified_distances(St, M, sp.A)
        to_means[dom] = {d: float(np.median(Dm_t[:, i])) for i, d in enumerate(doms)}
    np.savetxt(C.EVID_DIR / "domain_means_mm.csv", Dm, delimiter=",", header=",".join(doms), fmt="%.3f")
    return {"domains": doms, "means_mm": Dm.tolist(), "spread": spread, "test_to_means_median": to_means}


def cross_domain_variant(sp: Space, sets: dict) -> dict:
    """Variante per coppia contro globale su un campione FRA domini: 100 identita' per dominio."""
    rng = np.random.default_rng(SEED)
    S, lab = [], []
    for name, st in sets.items():
        k = min(100, len(st["s"]))
        S.append(st["s"][rng.choice(len(st["s"]), k, replace=False)])
        lab += [name] * k
    S = np.concatenate(S)
    lab = np.array(lab)
    g = unified_distances(S, S, sp.A)
    gp = pairwise_distances(S, sp)
    iu = np.triu_indices(len(S), 1)
    cross = lab[iu[0]] != lab[iu[1]]
    return {"n_shapes": int(len(S)), "spearman_all": spearman(g[iu], gp[iu]),
            "spearman_cross_domain_pairs": spearman(g[iu][cross], gp[iu][cross]),
            "pearson_all": float(np.corrcoef(g[iu], gp[iu])[0, 1]),
            "pairwise_over_global_median": float(np.median(gp[iu] / g[iu]))}


def main() -> None:
    sp = Space()
    names = ["hifi3d", "faceverse", "bfm", "ict", "gnm", "flame", "multiface"]
    sets = {n: load_set(n) for n in names}
    out = {"a": part_a(sp, sets)}
    print("[evid] (a) fatto", flush=True)
    out["b"] = part_b(sp, sets)
    print("[evid] (b) fatto", flush=True)
    out["c"] = part_c(sp, sets)
    out["variant_cross_domain"] = cross_domain_variant(sp, sets)
    out["space"] = {"n_vertices": int(len(sp.mu)), "area_mm2": sp.A}
    C.save_json(C.EVID_DIR / "evidence.json", out)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
