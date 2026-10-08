#!/usr/bin/env python3
"""E3, E3b, E3c: da dove vengono gli errori di riconoscimento di e108, e se un rimesh al test li toglie.

    aau/run.sh aau/evidence/e3_breakdown/breakdown.py        (breakdown.sbatch, CPU)

Distanze (matrici 600 x 600 indicizzate da (soggetto, topologia), ``zs_expr_summarize.Index``):
  - e108: ||z_i - z_j|| dagli embedding gia' salvati (HIFI3D ``scale_e108_embed``, FaceVerse con espressioni
    ``scale_e108_flip_topology``, convenzione BFM come la riga di riferimento);
  - Chamfer faceBench e Rigid ICP + Chamfer: le matrici esistenti (``zs_expr_summarize.facebench_distances``);
  - E3b: e108 sulle mesh rimeshate (``e3b_eval.sbatch``); diagnosi: e108 con centro per area / pooling per area;
  - FaceVerse NEUTRO, e108 convenzione BFM (``e3b_eval.sbatch``): separa espressione e topologia.
Repliche: ``bootstrap_counts`` col seme ``expr_recognition`` (quelle del riconoscimento pubblicato) per
(1)-(3) ed E3b; per E3c il seme della riga e108 ``nocrop_cross`` pubblicata, cosi' il globale torna 0.630.

(1) rank-1 e AUC per coppia ordinata di topologie (query t1, galleria t2): 100 query, verifica sulla coppia.
(2) per ogni mesh (i, t): dispersione intra = media delle distanze dalle mesh della stessa identita' nelle
    altre topologie senza crop; distanza dal vicino = min sulle altre identita' j della media delle distanze
    dalle mesh di j nelle altre topologie; rapporto = intra / vicino (> 1: la mesh e' piu' vicina, in media,
    a un'altra identita' che alla propria). Per coppia (t1, t2): genuina / impostore piu' vicino nella
    galleria t2 (> 1 <=> errore di rank-1).
(3) rango del match vero contro la distanza GT dall'impostore piu' vicino; per le query sbagliate, il
    percentile GT (fra i 99 impostori del soggetto) dell'impostore restituito al primo posto (caso: uniforme).
E3c: HIFI3D ``nocrop_cross``, Spearman con la GT ristretto alle coppie per decile di GT (decili sulle 4950
    coppie di soggetti, fissati sul campione) e residuo relativo |d/media(d) - g/media(g)| / (g/media(g)).
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "aau" / "zs3dmm"))

import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

base = zsum.base
RUNS = REPO_ROOT / "aau" / "runs"
E3 = RUNS / "evidence" / "e3"
HIFI_RUNS = RUNS / "ws_hifi3d" / "data_328f2bfc1a"
FV_RUNS = RUNS / "ws_faceverse_expr" / "data_736f96956a"
NOCROP = zes.NOCROP
LABEL = {"e108": "e108", "chamfer": "Chamfer faceBench", "rigid_icp_chamfer": "Rigid ICP + Chamfer",
         "chamfer_eval": "Chamfer eval", "e108_remesh": "e108, rimesh uniforme al test (E3b)",
         "e108_base": "e108, ricalcolato (controllo)", "e108_areacenter": "e108, centro per area",
         "e108_areapool": "e108, pooling medio pesato per area", "e108_areaboth": "e108, centro e pooling per area",
         "e108_neutral": "e108, FaceVerse NEUTRO"}
_BM = None


def fmt(p, lo, hi, signed=False):
    return zes.fmt(p, lo, hi, signed)


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def gt_sub(path: Path, subjects: list[str]) -> np.ndarray:
    D, n2i = zsum.load_gt(path)
    ix = np.asarray([n2i[s] for s in subjects])
    return D[np.ix_(ix, ix)]


# ------------------------------------------------------------- (1) per coppia di topologie

def _pair_task(task):
    key, D, idx, t1, t2, counts = task
    v = zes.recognition_values(zes.retrieval_queries(D, idx, [(t1, t2)]), zes.verification_pairs(D, idx, [(t1, t2)]),
                               idx, counts)
    return key, {m: (float(v[m][0]), *zes.ci(v[m])) for m in ("rank1", "auc")}, v["rank1"]


def per_pair(Ds: dict, idx, counts, workers) -> tuple[pd.DataFrame, dict]:
    tasks = [((m, t1, t2), D, idx, t1, t2, counts) for m, D in Ds.items() for t1 in TOPOLOGIES for t2 in TOPOLOGIES if t1 != t2]
    rows, reps = [], {}
    with mp.get_context("fork").Pool(workers) as pool:
        for (m, t1, t2), r, rep in pool.imap_unordered(_pair_task, tasks):
            rows.append({"method": m, "query": t1, "gallery": t2, **{f"{k}{s}": v for k, vals in r.items()
                                                                      for s, v in zip(("", "_ci_low", "_ci_high"), vals)}})
            reps[(m, t1, t2)] = rep
    return pd.DataFrame(rows), reps


def matrix_md(df: pd.DataFrame, m: str, col: str) -> list[str]:
    sub = df[df.method == m].set_index(["query", "gallery"])
    return table(["query \\ galleria"] + list(TOPOLOGIES),
                 [[a] + ["-" if a == b else f"{sub.loc[(a, b), col]:.2f}" for b in TOPOLOGIES] for a in TOPOLOGIES])


# --------------------------------------------------------------------- (2) dispersione

def dispersion(D: np.ndarray, idx, topos=NOCROP) -> pd.DataFrame:
    n = len(idx.subjects)
    R = {t: idx.rows(t) for t in TOPOLOGIES}
    rows = []
    for t in topos:
        others = [u for u in topos if u != t]
        # M[i, j] = media su u delle distanze fra (i, t) e (j, u)
        M = np.mean([D[np.ix_(R[t], R[u])] for u in others], axis=0)
        intra = np.diag(M).copy()
        off = M + np.diag(np.full(n, np.inf))
        nn = off.min(1)
        for i, s in enumerate(idx.subjects):
            rows.append({"subject": s, "topology": t, "intra": intra[i], "nearest_other": nn[i], "ratio": intra[i] / nn[i],
                         "nearest_id": idx.subjects[int(off[i].argmin())]})
    return pd.DataFrame(rows)


def pair_ratio(D: np.ndarray, idx) -> pd.DataFrame:
    rows = []
    for t1 in TOPOLOGIES:
        for t2 in TOPOLOGIES:
            if t1 == t2:
                continue
            M = D[np.ix_(idx.rows(t1), idx.rows(t2))]
            g = np.diag(M).copy()
            nn = (M + np.diag(np.full(len(g), np.inf))).min(1)
            rows.append({"query": t1, "gallery": t2, "ratio_median": float(np.median(g / nn)),
                         "ratio_p90": float(np.percentile(g / nn, 90)), "frac_gt1": float((g / nn > 1).mean())})
    return pd.DataFrame(rows)


def boot_median_by_subject(df: pd.DataFrame, col: str, subjects: list[str], counts: np.ndarray) -> tuple[float, float, float]:
    """Mediana pesata (repliche: conteggi dei soggetti)."""
    s2i = {s: k for k, s in enumerate(subjects)}
    x = df[col].to_numpy()
    si = df["subject"].map(s2i).to_numpy()
    order = np.argsort(x)
    x, si = x[order], si[order]
    reps = []
    for c in counts:
        w = c[si].astype(float)
        cw = np.cumsum(w)
        reps.append(x[np.searchsorted(cw, 0.5 * cw[-1])])
    return float(np.median(x)), *np.percentile(reps, [2.5, 97.5])


def boot_mean_by_subject(x: np.ndarray, si: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    """Media con IC; le repliche senza osservazioni (p.es. nessun errore) si scartano."""
    reps = [(c[si] * x).sum() / c[si].sum() for c in counts if c[si].sum() > 0]
    return float(x.mean()), *np.percentile(reps, [2.5, 97.5])


# ---------------------------------------------------------------- (3) vicini in GT

def gt_neighbors(D: np.ndarray, G: np.ndarray, idx, counts) -> tuple[dict, pd.DataFrame]:
    n = len(idx.subjects)
    Gi = G + np.diag(np.full(n, np.inf))
    gnn = Gi.min(1)
    # percentile GT di ogni impostore j per il soggetto i (0 = il piu' vicino in GT)
    gt_rank = np.argsort(np.argsort(Gi, axis=1), axis=1) / (n - 2)
    q = []
    for t1 in NOCROP:
        for t2 in NOCROP:
            if t1 == t2:
                continue
            M = D[np.ix_(idx.rows(t1), idx.rows(t2))]
            M = np.where(np.isfinite(M), M, np.inf)
            g = np.diag(M)[:, None]
            rank = 1 + (M < g).sum(1) + 0.5 * ((M == g).sum(1) - 1)
            top = (M + np.diag(np.full(n, np.inf))).argmin(1)
            for i in range(n):
                q.append({"si": i, "t1": t1, "t2": t2, "rank": rank[i], "err": rank[i] > 1, "gnn": gnn[i],
                          "top_imp_gt_pct": gt_rank[i, top[i]], "top_imp_is_gt_nn": int(top[i] == Gi[i].argmin())})
    q = pd.DataFrame(q)
    si = q["si"].to_numpy()
    rho = base.load_bootstrap_module().finite_spearman
    point = rho(q["rank"].to_numpy(float), q["gnn"].to_numpy())
    reps = []
    for c in counts:
        w = c[si]
        keep = w > 0
        reps.append(rho(np.repeat(q["rank"].to_numpy(float)[keep], w[keep]), np.repeat(q["gnn"].to_numpy()[keep], w[keep])))
    quart = np.digitize(gnn, np.percentile(gnn, [25, 50, 75]))
    q["gnn_quartile"] = quart[si]
    out = {"spearman_rank_gnn": (point, *np.nanpercentile(reps, [2.5, 97.5]))}
    for k in range(4):
        sub = q[q.gnn_quartile == k]
        out[f"err_q{k + 1}"] = boot_mean_by_subject(sub["err"].to_numpy(float), sub["si"].to_numpy(), counts)
    e = q[q.err]
    out["n_err"] = int(len(e))
    if len(e):
        out["err_top_pct_median"] = float(e.top_imp_gt_pct.median())
        out["err_top_in_gt5pct"] = boot_mean_by_subject((e.top_imp_gt_pct <= 0.05).to_numpy(float), e["si"].to_numpy(), counts)
        out["err_top_is_gt_nn"] = boot_mean_by_subject(e.top_imp_is_gt_nn.to_numpy(float), e["si"].to_numpy(), counts)
    return out, q


# ---------------------------------------------------------------------------- E3c

def _e3c_task(task):
    global _BM
    if _BM is None:
        _BM = base.load_bootstrap_module()
    key, gt, x, sa, sb, counts = task
    rho = _BM.finite_spearman
    point = rho(gt, x)
    reps = np.empty(len(counts))
    for k, c in enumerate(counts):
        w = (c[sa] * c[sb]).astype(np.int64)
        keep = w > 0
        reps[k] = rho(np.repeat(gt[keep], w[keep]), np.repeat(x[keep], w[keep])) if keep.sum() >= 3 else np.nan
    return key, point, reps


def e3c(df: pd.DataFrame, methods: list[str], subjects: list[str], seed: int, n_boot: int, workers: int):
    s2i = {s: k for k, s in enumerate(subjects)}
    sa = df["subject_a"].map(s2i).to_numpy()
    sb = df["subject_b"].map(s2i).to_numpy()
    counts = zes.bootstrap_counts(len(subjects), n_boot, seed)
    sp = df.groupby(["subject_a", "subject_b"])["gt_distance"].first()
    edges = np.percentile(sp.to_numpy(), np.arange(10, 100, 10))
    dec = np.digitize(df["gt_distance"].to_numpy(), edges)          # 0 = decile piu' vicino
    groups = {"globale": np.ones(len(df), bool), "quintile 1": dec <= 1}
    groups.update({f"decile {k + 1}": dec == k for k in range(10)})
    gt = df["gt_distance"].to_numpy()
    tasks = [((g, m), gt[mask], df[m].to_numpy()[mask], sa[mask], sb[mask], counts)
             for g, mask in groups.items() for m in methods]
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, point, reps in pool.imap_unordered(_e3c_task, tasks):
            res[key] = (point, reps)
    rows = []
    for (g, m), (point, reps) in res.items():
        rows.append({"group": g, "method": m, "spearman": point, "ci_low": np.nanpercentile(reps, 2.5),
                     "ci_high": np.nanpercentile(reps, 97.5), "n_rows": int(groups[g].sum())})
    deltas = []
    for g in groups:
        for m in methods[1:]:
            d = res[(g, methods[0])][1] - res[(g, m)][1]
            deltas.append({"group": g, "a": methods[0], "b": m, "diff": res[(g, methods[0])][0] - res[(g, m)][0],
                           "ci_low": np.nanpercentile(d, 2.5), "ci_high": np.nanpercentile(d, 97.5), "p_le0": float((d <= 0).mean())})
    # caduta locale (globale - locale) di e108 contro quella di ogni baseline, stesse repliche
    drops = []
    for g in ("decile 1", "quintile 1"):
        for m in methods:
            dm = res[("globale", m)][1] - res[(g, m)][1]
            drops.append({"group": g, "method": m, "drop": res[("globale", m)][0] - res[(g, m)][0],
                          "ci_low": np.nanpercentile(dm, 2.5), "ci_high": np.nanpercentile(dm, 97.5)})
        d0 = res[("globale", methods[0])][1] - res[(g, methods[0])][1]
        for m in methods[1:]:
            dd = d0 - (res[("globale", m)][1] - res[(g, m)][1])
            point = (res[("globale", methods[0])][0] - res[(g, methods[0])][0]) - (res[("globale", m)][0] - res[(g, m)][0])
            drops.append({"group": g, "method": f"caduta {methods[0]} - caduta {m}", "drop": point,
                          "ci_low": np.nanpercentile(dd, 2.5), "ci_high": np.nanpercentile(dd, 97.5),
                          "p_le0": float((dd <= 0).mean())})
    # residuo relativo per decile (normalizzazione con le medie globali), media con IC e mediana
    rr_rows = []
    gn = gt / gt.mean()
    for m in methods:
        x = df[m].to_numpy()
        rr = np.abs(x / x.mean() - gn) / gn
        for g, mask in groups.items():
            w_reps = [((c[sa] * c[sb])[mask] * rr[mask]).sum() / (c[sa] * c[sb])[mask].sum() for c in counts]
            rr_rows.append({"group": g, "method": m, "rel_resid_mean": float(rr[mask].mean()),
                            "ci_low": np.percentile(w_reps, 2.5), "ci_high": np.percentile(w_reps, 97.5),
                            "rel_resid_median": float(np.median(rr[mask])), "gt_mean_in_group": float(gt[mask].mean())})
    return pd.DataFrame(rows), pd.DataFrame(deltas), pd.DataFrame(drops), pd.DataFrame(rr_rows), edges


# ------------------------------------------------------------------- geometria di down8k

def geometry_stats(view: Path, idx) -> pd.DataFrame:
    """Per mesh: quanto il centro e la scala del loader (media dei vertici, max|V|) dipendono dalla discretizzazione."""
    rows = []
    for s in idx.subjects:
        ref = None
        for t in ("original",) + tuple(x for x in TOPOLOGIES if x != "original"):   # la original per prima: e' il riferimento
            with np.load(view / f"{s}_GTready_{t}.npz") as z:
                V = np.asarray(z["V"] if "V" in z else z["verts"], np.float64)
                F = np.asarray(z["F"] if "F" in z else z["faces"], np.int64)
            tri = V[F]
            a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            ca = (tri.mean(1) * a[:, None]).sum(0) / a.sum()
            cv = V.mean(0)
            mx = np.abs(V - cv).max()
            va = np.bincount(F.ravel(), weights=np.repeat(a / 3, 3), minlength=len(V))
            E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
            _, cnt = np.unique(E[:, 0] * len(V) + E[:, 1], return_counts=True)
            if t == "original":
                ref = (cv, mx, ca)
            rows.append({"subject": s, "topology": t, "n_verts": len(V), "vertex_area_cv": float(va.std() / va.mean()),
                         "boundary_edge_frac": float((cnt == 1).mean()),
                         "center_shift_vs_area": float(np.linalg.norm(cv - ca) / mx),
                         "center_shift_vs_original": float(np.linalg.norm(cv - ref[0]) / ref[1]) if ref is not None else np.nan,
                         "maxabs_ratio_vs_original": float(mx / ref[1]) if ref is not None else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--out-md", type=Path, default=E3 / "summary.md")
    args = p.parse_args()
    out = E3 / "tables"
    out.mkdir(parents=True, exist_ok=True)
    seed_rec = base.stable_seed(args.seed, "expr_recognition")
    md = ["# E3 / E3b / E3c: da dove vengono gli errori di riconoscimento di e108\n",
          "Script: `aau/evidence/e3_breakdown/breakdown.py`. IC 95% bootstrap per soggetto, "
          f"{args.n_bootstrap} repliche (seme `expr_recognition`, le repliche del riconoscimento pubblicato; E3c: seme della "
          "riga e108 `nocrop_cross` pubblicata). Distanze e embedding gia' salvati, salvo dove indicato.\n"]
    checks = []

    doms = {}
    for dom, view, emb, bl in (("hifi", REPO_ROOT / "datasets/HIFI3D/eval_view", HIFI_RUNS / "scale_e108_embed" / zsum.STAGE,
                                HIFI_RUNS / "baselines"),
                               ("fv", REPO_ROOT / "datasets/FACEVERSE_ZS/expr_view", FV_RUNS / "scale_e108_flip_topology" / zsum.STAGE,
                                FV_RUNS / "baselines")):
        subjects = select_subjects(view / "npz", 1234)
        idx = zes.Index(subjects)
        Ds = {"e108": zes.model_distances(emb, idx),
              "chamfer": zes.facebench_distances(bl, "chamfer", idx),
              "rigid_icp_chamfer": zes.facebench_distances(bl, "rigid_icp_chamfer", idx)}
        extra = {}
        rm = E3 / dom / "remesh_e108"
        if (rm / "embeddings.npz").exists():
            extra["e108_remesh"] = zes.model_distances(rm, idx)
        if dom == "hifi":
            for v in ("base", "areacenter", "areapool", "areaboth"):
                f = E3 / "hifi" / "variants_e108" / f"embeddings_{v}.npz"
                if f.exists():
                    tmp = E3 / "hifi" / "variants_e108" / f"_{v}"
                    tmp.mkdir(exist_ok=True)
                    if not (tmp / "embeddings.npz").exists():
                        (tmp / "embeddings.npz").symlink_to(f)
                    extra[f"e108_{v}"] = zes.model_distances(tmp, idx)
            if "e108_base" in extra:
                checks.append(f"HIFI3D, e108 ricalcolato nella catena di E3 contro gli embedding esistenti, max |diff| delle "
                              f"distanze: {np.abs(extra['e108_base'] - Ds['e108']).max():.2e}")
        neu = E3 / "fv" / "neutral_e108_flip"
        if dom == "fv" and (neu / "embeddings.npz").exists():
            extra["e108_neutral"] = zes.model_distances(neu, idx)
        G = gt_sub(view / "gt_matrix.npz", subjects)
        counts = zes.bootstrap_counts(len(subjects), args.n_bootstrap, seed_rec)
        doms[dom] = (subjects, idx, Ds, extra, G, counts, view)

    # (1) per coppia di topologie ------------------------------------------------------
    for dom, title in (("hifi", "HIFI3D"), ("fv", "FaceVerse con espressioni (e108 in convenzione BFM)")):
        subjects, idx, Ds, extra, G, counts, view = doms[dom]
        allD = {**Ds, **extra}
        pp, reps = per_pair(allD, idx, counts, args.workers)
        pp.to_csv(out / f"{dom}_per_pair.csv", index=False)
        pr = pd.concat([pair_ratio(D, idx).assign(method=m) for m, D in allD.items()])
        pr.to_csv(out / f"{dom}_pair_ratio.csv", index=False)
        md += [f"## (1) {title}: rank-1 per coppia ordinata di topologie\n"]
        for m in allD:
            md += [f"### {LABEL[m]}\n", *matrix_md(pp, m, "rank1"), ""]
        md += [f"AUC di verifica per coppia (e108):\n", *matrix_md(pp, "e108", "auc"), ""]
        # IC sulle coppie con down8k e noisy per e108 e le diagnosi
        key_pairs = [("original", "down8k"), ("down8k", "original"), ("remesh", "down8k"), ("up60k", "down8k"),
                     ("original", "noisy"), ("remesh", "up60k"), ("crop", "original")]
        md += ["Coppie chiave, rank-1 [IC] (e AUC):\n",
               *table(["metodo"] + [f"{a} -> {b}" for a, b in key_pairs],
                      [[LABEL[m]] + [f"{fmt(*pp[(pp.method == m) & (pp['query'] == a) & (pp.gallery == b)][['rank1', 'rank1_ci_low', 'rank1_ci_high']].iloc[0])}"
                                     f" ({pp[(pp.method == m) & (pp['query'] == a) & (pp.gallery == b)].auc.iloc[0]:.3f})"
                                     for a, b in key_pairs] for m in allD]), ""]
        # delta appaiati per le varianti di e108 sulle coppie con down8k
        var = [m for m in allD if m.startswith("e108_") and m != "e108_neutral"]
        if var:
            rows = []
            for m in var:
                cells = [LABEL[m]]
                for a, b in key_pairs:
                    d = reps[(m, a, b)] - reps[("e108", a, b)]
                    cells.append(f"{fmt(d[0], *zes.ci(d), signed=True)}")
                rows.append(cells)
            md += ["Varianti di e108 meno e108, rank-1 (stesse repliche):\n",
                   *table(["variante - e108"] + [f"{a} -> {b}" for a, b in key_pairs], rows), ""]
        md += ["Rapporto genuina / impostore piu' vicino della galleria, mediana per coppia (e108; > 1 = errore):\n",
               *table(["query \\ galleria"] + list(TOPOLOGIES),
                      [[a] + ["-" if a == b else f"{pr[(pr.method == 'e108') & (pr['query'] == a) & (pr.gallery == b)].ratio_median.iloc[0]:.2f}"
                              for b in TOPOLOGIES] for a in TOPOLOGIES]), ""]

        # (2) dispersione ------------------------------------------------------------
        disp = []
        for m, D in allD.items():
            d = dispersion(D, idx).assign(method=m)
            disp.append(d)
        disp = pd.concat(disp)
        disp.to_csv(out / f"{dom}_dispersion.csv", index=False)
        rows = []
        for m in allD:
            for t in NOCROP:
                sub = disp[(disp.method == m) & (disp.topology == t)]
                med = boot_median_by_subject(sub, "ratio", subjects, counts)
                gt1 = boot_mean_by_subject((sub.ratio > 1).to_numpy(float), np.arange(len(sub)), counts)
                rows.append([LABEL[m], t, fmt(*med), f"{np.percentile(sub.ratio, 25):.2f}-{np.percentile(sub.ratio, 75):.2f}",
                             fmt(*gt1)])
        md += [f"## (2) {title}: dispersione intra-identita' / distanza dall'identita' piu' vicina, per topologia\n",
               "Rapporto per mesh (i, t) sulle 5 topologie senza crop: media delle distanze dalle altre topologie della stessa "
               "identita' / min sulle altre identita' della stessa media.\n",
               *table(["metodo", "topologia", "mediana [IC]", "IQR", "frazione > 1 [IC]"], rows), ""]

        # (3) vicini in GT -----------------------------------------------------------
        rows = []
        for m in allD:
            if m == "e108_neutral":
                continue
            o, q = gt_neighbors(allD[m], G, idx, counts)
            q.to_csv(out / f"{dom}_queries_{m}.csv", index=False)
            rows.append([LABEL[m], fmt(*o["spearman_rank_gnn"]), *[fmt(*o[f"err_q{k}"]) for k in range(1, 5)], o["n_err"],
                         f"{o.get('err_top_pct_median', np.nan):.3f}",
                         fmt(*o["err_top_in_gt5pct"]) if "err_top_in_gt5pct" in o else "-",
                         fmt(*o["err_top_is_gt_nn"]) if "err_top_is_gt_nn" in o else "-"])
        md += [f"## (3) {title}: gli errori cadono sulle identita' vicine in GT?\n",
               "Query senza crop (2000). Spearman fra rango del match vero e distanza GT dell'identita' query dal suo impostore "
               "piu' vicino (negativo = piu' errori quando c'e' un vicino stretto); tasso d'errore di rank-1 per quartile di quella "
               "distanza (Q1 = vicino piu' stretto); per le query sbagliate, percentile GT dell'impostore messo al primo posto "
               "(0 = il vicino GT; caso: mediana 0.5, entro il 5% con probabilita' 0.05).\n",
               *table(["metodo", "Spearman(rango, d_GT vicino) [IC]", "errore Q1", "Q2", "Q3", "Q4", "query sbagliate",
                       "percentile GT del primo impostore, mediana", "primo impostore nel 5% GT piu' vicino", "= vicino GT"], rows), ""]

        if dom == "fv" and "e108_neutral" in extra:
            neu = extra["e108_neutral"]
            # spostamento d'espressione a topologia fissa: z_expr(i,t) - z_neu(i,t)
            ze = np.load(FV_RUNS / "scale_e108_flip_topology" / zsum.STAGE / "embeddings.npz")
            zn = np.load(E3 / "fv" / "neutral_e108_flip" / "embeddings.npz")

            def Zof(z):
                k = {(str(s), str(t)): i for i, (s, t) in enumerate(zip(z["subjects"], z["topologies"]))}
                return np.stack([z["Z"][k[key]] for key in idx.keys]).astype(np.float64)
            Ze, Zn = Zof(ze), Zof(zn)
            dexp = np.linalg.norm(Ze - Zn, axis=1)
            no = idx.rows("original")
            Mn = neu[np.ix_(no, no)] + np.diag(np.full(len(no), np.inf))
            nn_neu = Mn.min(1)
            rows = []
            for t in TOPOLOGIES:
                r = idx.rows(t)
                dtopo = np.linalg.norm(Zn[r] - Zn[no], axis=1) if t != "original" else np.full(len(r), np.nan)
                rows.append([t, f"{np.median(dexp[r] / nn_neu):.2f}", f"{np.nanmedian(dtopo / nn_neu):.2f}" if t != "original" else "-"])
            md += ["### FaceVerse: espressione contro topologia nell'embedding di e108\n",
                   "Per ogni identita' i: d_vicino = distanza fra la sua `original` NEUTRA e quella neutra dell'identita' piu' vicina. "
                   "Colonne: mediana di ||z_espr(i,t) - z_neutro(i,t)|| / d_vicino (solo espressione, stessa topologia; per t diversa "
                   "da original anche la triangolazione e' rigenerata) e di ||z_neutro(i,t) - z_neutro(i,original)|| / d_vicino "
                   "(solo topologia).\n",
                   *table(["topologia", "espressione / d_vicino", "topologia / d_vicino"], rows), ""]

    # E3c ------------------------------------------------------------------------------
    subjects, idx, Ds, extra, G, counts, view = doms["hifi"]
    pm = base.read_pair_metrics(HIFI_RUNS / "scale_e108_topology" / zsum.STAGE)
    pm = pm[pm.topology_a.ne("crop") & pm.topology_b.ne("crop")].copy()
    pm["subject_a"], pm["subject_b"] = pm["subject_a"].astype(str), pm["subject_b"].astype(str)
    ia = np.asarray([idx.pos[k] for k in zip(pm["subject_a"], pm["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(pm["subject_b"], pm["topology_b"])])
    df = pd.DataFrame({"subject_a": pm.subject_a, "subject_b": pm.subject_b, "gt_distance": pm.gt_distance,
                       "e108": pm.latent_distance, "chamfer_eval": pm.raw_chamfer,
                       "chamfer": Ds["chamfer"][ia, ib], "rigid_icp_chamfer": Ds["rigid_icp_chamfer"][ia, ib]})
    meths = ["e108", "chamfer_eval", "chamfer", "rigid_icp_chamfer"]
    if "e108_remesh" in extra:
        df["e108_remesh"] = extra["e108_remesh"][ia, ib]
        meths.append("e108_remesh")
    sp, dl, drops, rr, edges = e3c(df, meths, subjects, base.stable_seed(args.seed, "scale_e108", "maxabs", "nocrop_cross", "latent"),
                                   args.n_bootstrap, args.workers)
    for name, t in (("e3c_spearman", sp), ("e3c_paired", dl), ("e3c_drops", drops), ("e3c_rel_resid", rr)):
        t.to_csv(out / f"{name}.csv", index=False)
    glob = sp[(sp.group == "globale") & (sp.method == "e108")].iloc[0]
    checks.append(f"E3c, e108 globale nocrop_cross: {fmt(glob.spearman, glob.ci_low, glob.ci_high)} (pubblicato 0.630 [0.569, 0.689])")
    gorder = ["globale", "quintile 1"] + [f"decile {k}" for k in range(1, 11)]
    md += ["## E3c: struttura locale (HIFI3D, nocrop_cross, GT maxabs)\n",
           f"Decili delle 4950 coppie di soggetti per distanza GT (decile 1 = coppie piu' vicine; limiti {', '.join(f'{e:.3f}' for e in edges)}); "
           "tutte le 20 coppie ordinate di topologie di una coppia di soggetti stanno nello stesso decile. Ristringere l'intervallo "
           "della GT abbassa lo Spearman di QUALUNQUE metrica (attenuazione): il confronto giusto e' fra metodi nello stesso decile, "
           "e la caduta globale - locale di e108 contro quella delle baseline.\n",
           *table(["gruppo"] + [LABEL[m] for m in meths],
                  [[g] + [fmt(*sp[(sp.group == g) & (sp.method == m)][["spearman", "ci_low", "ci_high"]].iloc[0]) for m in meths]
                   for g in gorder]),
           "\nDifferenze appaiate e108 - baseline per gruppo:\n",
           *table(["gruppo"] + [f"e108 - {LABEL[m]} (P<=0)" for m in meths[1:]],
                  [[g] + [f"{fmt(r['diff'], r.ci_low, r.ci_high, True)} ({r.p_le0:.3f})"
                          for r in [dl[(dl.group == g) & (dl.b == m)].iloc[0] for m in meths[1:]]] for g in gorder]),
           "\nCaduta globale - locale (stesse repliche) e differenza delle cadute e108 - baseline (> 0: e108 perde di piu' nel locale):\n",
           *table(["gruppo", "metodo / confronto", "caduta [IC]", "P(<=0)"],
                  [[r.group, LABEL.get(r.method, r.method), fmt(r.drop, r.ci_low, r.ci_high, True),
                    f"{r.p_le0:.3f}" if "p_le0" in drops.columns and np.isfinite(r.p_le0) else "-"] for r in drops.itertuples()]),
           "\nResiduo relativo |d/media(d) - g/media(g)| / (g/media(g)) per gruppo: media [IC] (mediana):\n",
           *table(["gruppo"] + [LABEL[m] for m in meths],
                  [[g] + [f"{fmt(*rr[(rr.group == g) & (rr.method == m)][['rel_resid_mean', 'ci_low', 'ci_high']].iloc[0])} "
                          f"({rr[(rr.group == g) & (rr.method == m)].rel_resid_median.iloc[0]:.2f})" for m in meths] for g in gorder]), ""]

    # geometria: perche' down8k --------------------------------------------------------
    geo = geometry_stats(view / "npz", idx)
    geo.to_csv(out / "hifi_geometry.csv", index=False)
    no, d8 = idx.rows("original"), idx.rows("down8k")
    zdist = np.diag(Ds["e108"][np.ix_(no, d8)])
    g8 = geo[geo.topology == "down8k"].set_index("subject").loc[subjects]
    rho = base.load_bootstrap_module().finite_spearman
    corr = {c: rho(zdist, g8[c].to_numpy()) for c in ("center_shift_vs_original", "maxabs_ratio_vs_original", "vertex_area_cv")}
    md += ["## Geometria delle topologie HIFI3D (perche' down8k)\n",
           "Il loader centra sulla MEDIA DEI VERTICI e divide per max|V|; il pooling medio del modello e' una media sui vertici. "
           "Colonne: mediane sui 100 soggetti.\n",
           *table(["topologia", "vertici", "CV area per vertice", "frazione di spigoli di bordo",
                   "|media vertici - baricentro area| / max|V|", "|media vertici - quella della original| / max|V|",
                   "max|V| / quello della original"],
                  [[t, int(g.n_verts.median()), f"{g.vertex_area_cv.median():.2f}", f"{g.boundary_edge_frac.median():.3f}",
                    f"{g.center_shift_vs_area.median():.4f}", f"{g.center_shift_vs_original.median():.4f}",
                    f"{g.maxabs_ratio_vs_original.median():.3f}"] for t, g in geo.groupby("topology")]),
           "\nSpearman, sui 100 soggetti, fra ||z(original) - z(down8k)|| di e108 e: "
           + ", ".join(f"{k} {v:+.2f}" for k, v in corr.items()) + "\n",
           "## Controlli\n", *[f"- {c}" for c in checks]]
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[e3] scritto {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
