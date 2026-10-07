#!/usr/bin/env python3
"""Tabelle della valutazione su NoW validation: (a) NoW ufficiale, (b) distanze dalla
scansione, (c) classifiche e concordanza con NoW, (d) identita' fra ricostruzioni.

    aau/run.sh aau/recon/now_summarize.py

Protocollo: ``aau/runs/now_eval/protocol.md``, scritto prima dei numeri e copiato in testa a
``summary.md``.  Il bootstrap ricampiona i 20 soggetti (1000 repliche, seme 1234) e le
stesse repliche servono tutte le righe, cosi' i delta sono appaiati.  Le funzioni del
protocollo d'identita' (``weighted_auc``, ``bootstrap_counts``, ``ci``) sono importate da
``aau/zs3dmm/zs_expr_summarize.py``, non riscritte.

Ingressi: ``now_official_<metodo>.csv`` e le distanze per vertice in
``<WORK_ROOT>/official`` (now_official.py), ``gt_<metrica>_<metodo>.csv`` (now_latent.py,
now_geometric.py, now_arcface.py), le matrici fra ricostruzioni in ``<WORK_ROOT>/pairs``.
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
from scipy.stats import rankdata

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))

import now_common as common  # noqa: E402
from zs_expr_summarize import bootstrap_counts, ci, weighted_auc  # noqa: E402

# metrica -> (prefisso del csv gt, colonna, file della matrice, chiave nella matrice)
METRICS = {
    "latent_joint": ("latent_joint", "latent_joint", "latent_joint", "D"),
    "chamfer_raw": ("geometric", "chamfer_raw", "geometric", "chamfer_raw"),
    "icp_chamfer_mm": ("geometric", "icp_chamfer_mm", "geometric", "icp_chamfer_mm"),
    "arcface_normals": ("arcface_normals", "arcface_normals", "arcface_normals", "D"),
    "arcface_shaded": ("arcface_shaded", "arcface_shaded", "arcface_shaded", "D"),
}
LABEL = {
    "now": "NoW ufficiale (mm)",
    "latent_joint": "latente congiunto BFM+ICT",
    "chamfer_raw": "Chamfer grezza (maxabs)",
    "icp_chamfer_mm": "ICP + Chamfer (mm)",
    "arcface_normals": "ArcFace su normal map",
    "arcface_shaded": "ArcFace su render ombreggiato (secondaria)",
}
PRIMARY_DELTA = ("latent_joint", "chamfer_raw")
BIN_MM = 0.0005  # risoluzione della mediana bootstrap delle distanze NoW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default=",".join(common.METHODS))
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def fmt(point: float, lo: float, hi: float, digits: int = 3, signed: bool = False) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


# ----------------------------------------------------------------------- caricamento

def load_table(methods, items) -> tuple[pd.DataFrame, list[str]]:
    """Una riga per (ricostruzione, metodo): NoW per immagine e tutte le distanze dalla scansione."""
    frames = []
    for m in methods:
        df = pd.DataFrame(common.read_rows(common.official_csv_path(m)))[["name", "now_median", "now_mean"]]
        for metric, (prefix, col, _, _) in METRICS.items():
            path = common.gt_csv_path(prefix, m)
            if not path.exists():
                print(f"[now-sum] ATTENZIONE: manca {path.name}, {metric} saltata per {m}", flush=True)
                continue
            df = df.merge(pd.DataFrame(common.read_rows(path))[["name", col]], on="name", how="outer")
        df["method"] = m
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame([{"name": it.name, "subject": it.subject, "challenge": it.challenge} for it in items])
    df = df.merge(meta, on="name", how="left")
    cols = ["now_median", "now_mean"] + [c for c in METRICS if c in df]
    df[cols] = df[cols].astype(float)
    metrics = [c for c in METRICS if c in df and df[c].notna().all()]
    return df, metrics


def official_distances(method: str) -> dict[str, np.ndarray]:
    res = np.load(common.WORK_ROOT / "official" / f"{method}_computed_distances.npy", allow_pickle=True).item()
    return {str(f): np.asarray(d, dtype=np.float64) for f, d in zip(res["input_files"], res["computed_distances"])}


# ----------------------------------------------------------------------- (a) e score NoW

def now_stats(dists: dict, images: list[str], subj_of: dict, subjects: list[str], counts: np.ndarray) -> dict:
    """Mediana/media/std di tutte le distanze concatenate, piu' repliche bootstrap (soggetti)."""
    cat = np.concatenate([dists[i] for i in images])
    out = {"median": float(np.median(cat)), "mean": float(cat.mean()), "std": float(cat.std()),
           "n_images": len(images), "n_distances": int(cat.size)}
    edges = np.arange(0.0, cat.max() + 2 * BIN_MM, BIN_MM)
    H = np.zeros((len(subjects), len(edges) - 1))
    sums, ns = np.zeros(len(subjects)), np.zeros(len(subjects))
    s2i = {s: k for k, s in enumerate(subjects)}
    for i in images:
        k = s2i[subj_of[i]]
        H[k] += np.histogram(dists[i], bins=edges)[0]
        sums[k] += dists[i].sum()
        ns[k] += dists[i].size
    med, mean = [], []
    for c in counts:
        cum = np.cumsum(c @ H)
        j = int(np.searchsorted(cum, 0.5 * cum[-1]))
        med.append(0.5 * (edges[j] + edges[j + 1]))
        mean.append(float(c @ sums / (c @ ns)))
    out["median_reps"], out["mean_reps"] = np.asarray(med), np.asarray(mean)
    return out


# ----------------------------------------------------------------------- (c) concordanza

def kendall_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kendall tau per riga fra due ordinamenti degli stessi k metodi (righe = immagini)."""
    k = a.shape[1]
    num, n = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            num = num + np.sign(a[:, i] - a[:, j]) * np.sign(b[:, i] - b[:, j])
            n += 1
    return num / n


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


# ----------------------------------------------------------------------- (d) identita'

def retrieval(D: np.ndarray, subj: np.ndarray, n_subj: int, queries: np.ndarray, gallery: np.ndarray):
    """rank del soggetto giusto per ogni query: distanza query-soggetto = minimo sulle mesh
    di quel soggetto in galleria, query esclusa."""
    Dq = D[np.ix_(queries, gallery)].astype(np.float64)
    Dq[queries[:, None] == gallery[None, :]] = np.inf
    dS = np.full((len(queries), n_subj), np.inf)
    for s in range(n_subj):
        cols = subj[gallery] == s
        if cols.any():
            dS[:, s] = Dq[:, cols].min(axis=1)
    true = dS[np.arange(len(queries)), subj[queries]]
    keep = np.isfinite(true)
    rank = 1 + (dS < true[:, None]).sum(1) + 0.5 * ((dS == true[:, None]).sum(1) - 1)
    return queries[keep], rank[keep]


def recog_values(D, subj, n_subj, block, chal, counts) -> dict:
    n = len(subj)
    if block == "all":
        queries, gallery = np.arange(n), np.arange(n)
        iu, ju = np.triu_indices(n, 1)
    else:  # query non neutre, galleria neutra
        neutral = chal == "multiview_neutral"
        queries, gallery = np.flatnonzero(~neutral), np.flatnonzero(neutral)
        iu, ju = (x.ravel() for x in np.meshgrid(queries, gallery, indexing="ij"))
    q, rank = retrieval(D, subj, n_subj, queries, gallery)
    hit, ap, qs = (rank == 1).astype(float), 1.0 / rank, subj[q]
    sa, sb = subj[iu], subj[ju]
    score, gen = -D[iu, ju], sa == sb
    out = {"rank1": [hit.mean()], "map": [ap.mean()], "auc": [weighted_auc(score, gen, np.ones(len(score)))],
           "n_queries": len(q), "n_pairs": len(score), "n_genuine": int(gen.sum())}
    for c in counts:
        wq = c[qs].astype(np.float64)
        out["rank1"].append((wq * hit).sum() / wq.sum())
        out["map"].append((wq * ap).sum() / wq.sum())
        out["auc"].append(weighted_auc(score, gen, np.where(gen, c[sa], c[sa] * c[sb]).astype(np.float64)))
    for k in ("rank1", "map", "auc"):
        out[k] = np.asarray(out[k])
    return out


def _recog_task(task):
    key, D, subj, n_subj, block, chal, counts = task
    return key, recog_values(D, subj, n_subj, block, chal, counts)


def load_pair_matrix(metric: str, method: str, names: list[str]) -> np.ndarray | None:
    _, _, stem, key = METRICS[metric]
    path = common.pair_matrix_path(stem, method)
    if not path.exists():
        return None
    with np.load(path) as z:
        pos = {str(n): k for k, n in enumerate(z["names"])}
        idx = np.asarray([pos[n] for n in names])
        return np.asarray(z[key], dtype=np.float64)[np.ix_(idx, idx)]


# ----------------------------------------------------------------------- main

def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    items = common.load_items()
    subjects = common.subjects_of(items)
    s2i = {s: k for k, s in enumerate(subjects)}
    counts = bootstrap_counts(len(subjects), args.n_bootstrap, args.seed)
    df, metrics = load_table(methods, items)
    print(f"[now-sum] metriche complete: {metrics}", flush=True)

    # Insieme comune: immagini con NoW e tutte le metriche per tutti i metodi.
    ok = df.dropna(subset=["now_median"] + metrics).groupby("name")["method"].nunique()
    common_names = sorted(ok[ok == len(methods)].index)
    by_name = {it.name: it for it in items}
    subj_of_image = {it.image: it.subject for it in items}
    common_images = [by_name[n].image for n in common_names]
    dc = df[df["name"].isin(common_names)].copy()
    dc["s"] = dc["subject"].map(s2i)

    # (a) NoW ufficiale -----------------------------------------------------------------
    official, a_rows = {}, []
    for m in methods:
        dists = official_distances(m)
        official[m] = now_stats(dists, common_images, subj_of_image, subjects, counts)
        n_all = len(dists)
        all_stats = json.loads(common.official_csv_path(m).with_suffix(".json").read_text())["params"]
        for block, imgs in [("comune", common_images)] + [
                (c, [i for i in common_images if f"/{c}/" in i]) for c in common.CHALLENGES]:
            cat = np.concatenate([dists[i] for i in imgs]) if imgs else np.zeros(1)
            a_rows.append({"method": m, "block": block, "n_images": len(imgs), "median": float(np.median(cat)),
                           "mean": float(cat.mean()), "std": float(cat.std())})
        a_rows.append({"method": m, "block": "tutte (json ufficiale)", "n_images": n_all,
                       **{k: all_stats["challenge_stats"]["all"][k] for k in ("median", "mean", "std")},
                       "missing": all_stats["num_missing_files"]})
    a_table = pd.DataFrame(a_rows)

    # (b) + (c) classifiche -------------------------------------------------------------
    scores = {"now": {m: (official[m]["median"], official[m]["median_reps"]) for m in methods}}
    for metric in metrics:
        scores[metric] = {}
        for m in methods:
            sub = dc[dc["method"] == m]
            v, s = sub[metric].to_numpy(), sub["s"].to_numpy()
            reps = np.asarray([(c[s] * v).sum() / c[s].sum() for c in counts])
            scores[metric][m] = (float(v.mean()), reps)
    rank_rows, tau_rows = [], []
    now_point = np.asarray([scores["now"][m][0] for m in methods])
    now_reps = np.stack([scores["now"][m][1] for m in methods], axis=1)
    for metric, per in scores.items():
        point = np.asarray([per[m][0] for m in methods])
        reps = np.stack([per[m][1] for m in methods], axis=1)
        first = np.bincount(reps.argmin(1), minlength=len(methods)) / len(reps)
        for r, k in enumerate(np.argsort(point)):
            lo, hi = ci(np.concatenate([[point[k]], reps[:, k]]))
            rank_rows.append({"metric": metric, "rank": r + 1, "method": methods[k], "score": point[k],
                              "ci_low": lo, "ci_high": hi, "p_first": first[k]})
        if metric != "now":
            t_point = float(kendall_rows(point[None], now_point[None])[0])
            t_reps = kendall_rows(reps, now_reps)
            tau_rows.append({"metric": metric, "tau": t_point, "tau_boot_mean": float(t_reps.mean()),
                             "p_tau1": float((t_reps == 1).mean())})
    rank_table, tau_table = pd.DataFrame(rank_rows), pd.DataFrame(tau_rows)

    # Concordanza per immagine e per ricostruzione
    wide = {col: dc.pivot(index="name", columns="method", values=col)[methods].loc[common_names]
            for col in ["now_median"] + metrics}
    img_s = np.asarray([s2i[by_name[n].subject] for n in common_names])
    rows_s = dc["s"].to_numpy()
    conc, conc_reps = [], {}
    for metric in metrics:
        tau_i = kendall_rows(wide[metric].to_numpy(), wide["now_median"].to_numpy())
        t_reps = np.asarray([(c[img_s] * tau_i).sum() / c[img_s].sum() for c in counts])
        x, y = dc[metric].to_numpy(), dc["now_median"].to_numpy()
        rho = spearman(x, y)
        r_reps = []
        for c in counts:
            idx = np.repeat(np.arange(len(x)), c[rows_s])
            r_reps.append(spearman(x[idx], y[idx]))
        r_reps = np.asarray(r_reps)
        conc_reps[metric] = (t_reps, r_reps)
        per_method = {f"spearman_{m}": spearman(dc.loc[dc["method"] == m, metric], dc.loc[dc["method"] == m, "now_median"])
                      for m in methods}
        conc.append({"metric": metric, "tau_image": float(tau_i.mean()),
                     **dict(zip(("tau_image_ci_low", "tau_image_ci_high"), ci(np.concatenate([[tau_i.mean()], t_reps])))),
                     "spearman": rho, **dict(zip(("spearman_ci_low", "spearman_ci_high"), ci(np.concatenate([[rho], r_reps])))),
                     **per_method})
    conc_table = pd.DataFrame(conc)
    deltas_c = []
    a_, b_ = PRIMARY_DELTA
    if a_ in conc_reps and b_ in conc_reps:
        ra, rb = conc_table.set_index("metric").loc[a_], conc_table.set_index("metric").loc[b_]
        for k, (col, j) in enumerate((("tau_image", 0), ("spearman", 1))):
            d = conc_reps[a_][j] - conc_reps[b_][j]
            point = float(ra[col] - rb[col])
            lo, hi = ci(np.concatenate([[point], d]))
            deltas_c.append({"concordance": col, "delta": point, "ci_low": lo, "ci_high": hi,
                             "p_le0": float((d <= 0).mean())})

    # (d) identita' fra ricostruzioni -----------------------------------------------------
    tasks = []
    for m in methods:
        names = [n for n in common_names]
        subj = np.asarray([s2i[by_name[n].subject] for n in names])
        chal = np.asarray([by_name[n].challenge for n in names])
        for metric in metrics:
            D = load_pair_matrix(metric, m, names)
            if D is None:
                print(f"[now-sum] ATTENZIONE: matrice {metric} {m} assente", flush=True)
                continue
            for block in ("all", "neutral_gallery"):
                tasks.append(((m, metric, block), D, subj, len(subjects), block, chal, counts))
    rec = {}
    with mp.get_context("fork").Pool(min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)) as pool:
        for key, vals in pool.imap_unordered(_recog_task, tasks):
            rec[key] = vals
    rec_rows, rec_deltas = [], []
    for (m, metric, block), v in sorted(rec.items()):
        r = {"method": m, "metric": metric, "block": block, "n_queries": v["n_queries"],
             "n_pairs": v["n_pairs"], "n_genuine": v["n_genuine"]}
        for k in ("rank1", "map", "auc"):
            r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(v[k][0]), ci(v[k])
        rec_rows.append(r)
        if metric != "latent_joint" and (m, "latent_joint", block) in rec:
            a = rec[(m, "latent_joint", block)]
            r = {"method": m, "block": block, "baseline": metric}
            for k in ("rank1", "map", "auc"):
                d = a[k] - v[k]
                r[k], (r[f"{k}_ci_low"], r[f"{k}_ci_high"]) = float(d[0]), ci(d)
                r[f"{k}_p_le0"] = float((d[1:] <= 0).mean())
            rec_deltas.append(r)
    rec_table, rec_delta_table = pd.DataFrame(rec_rows), pd.DataFrame(rec_deltas)

    # Uscite ----------------------------------------------------------------------------
    out = common.OUT_ROOT
    a_table.to_csv(out / "official.csv", index=False)
    rank_table.to_csv(out / "ranking.csv", index=False)
    tau_table.to_csv(out / "ranking_kendall.csv", index=False)
    conc_table.to_csv(out / "concordance.csv", index=False)
    pd.DataFrame(deltas_c).to_csv(out / "concordance_paired.csv", index=False)
    rec_table.to_csv(out / "recognition.csv", index=False)
    rec_delta_table.to_csv(out / "recognition_paired.csv", index=False)

    L = [(out / "protocol.md").read_text().rstrip(), "\n---\n", "# Risultati\n",
         f"Metodi: {', '.join(methods)}. Insieme comune: {len(common_names)} immagini su {len(items)} "
         f"(ricostruite da tutti i metodi e con tutte le metriche), {len(subjects)} soggetti. "
         f"CI 95% bootstrap sui soggetti, {args.n_bootstrap} repliche, seme {args.seed}.\n",
         "## (a) Errore NoW ufficiale (mm)\n",
         "| metodo | blocco | immagini | mediana | media | std |", "| --- | --- | --- | --- | --- | --- |"]
    for r in a_rows:
        L.append(f"| {r['method']} | {r['block']} | {r['n_images']} | {r['median']:.3f} | {r['mean']:.3f} | {r['std']:.3f} |")
    sc_path = out / "official_selfcheck.json"
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        L.append("\nControllo del codice ufficiale (scansione come predizione di se stessa): "
                 + "; ".join(f"{Path(r['image']).parts[0]} mediana {r['median_mm']:.4f} mm, max {r['max_mm']:.4f} mm"
                             for r in sc) + ".")

    L += ["\n## (b) Distanza ricostruzione-scansione e (c) classifica dei metodi\n",
          "Punteggio: NoW = mediana ufficiale sull'insieme comune; le altre = media delle distanze per "
          "ricostruzione. Piccolo = meglio. `p_first` = frazione di repliche in cui il metodo e' primo.\n",
          "| metrica | " + " | ".join(f"{k + 1}." for k in range(len(methods)))
          + " | Kendall tau con NoW (punto; media bootstrap; P(tau=1)) |",
          "| --- | " + " | ".join("---" for _ in methods) + " | --- |"]
    taus = tau_table.set_index("metric") if not tau_table.empty else None
    for metric in ["now"] + metrics:
        sub = rank_table[rank_table["metric"] == metric].sort_values("rank")
        d = 2 if metric in ("now", "icp_chamfer_mm") else 4
        cells = [f"{r.method} {fmt(r.score, r.ci_low, r.ci_high, d)} ({r.p_first:.2f})" for r in sub.itertuples()]
        t = "-" if metric == "now" else (f"{taus.loc[metric, 'tau']:+.2f}; {taus.loc[metric, 'tau_boot_mean']:+.2f}; "
                                         f"{taus.loc[metric, 'p_tau1']:.2f}")
        L.append(f"| {LABEL[metric]} | " + " | ".join(cells) + f" | {t} |")

    L += ["\n### Concordanza con NoW per immagine e per ricostruzione\n",
          "Per immagine: Kendall tau fra l'ordine dei metodi secondo la metrica e secondo l'errore NoW mediano "
          "dell'immagine, medio sulle immagini (PRIMARIA). Per ricostruzione: Spearman su tutte le ricostruzioni "
          "di tutti i metodi; accanto lo Spearman dentro ciascun metodo.\n",
          "| metrica | tau per immagine [CI] | Spearman [CI] | " + " | ".join(f"Spearman {m}" for m in methods) + " |",
          "| --- | --- | --- | " + " | ".join("---" for _ in methods) + " |"]
    for r in conc:
        L.append(f"| {LABEL[r['metric']]} | {fmt(r['tau_image'], r['tau_image_ci_low'], r['tau_image_ci_high'])} | "
                 f"{fmt(r['spearman'], r['spearman_ci_low'], r['spearman_ci_high'])} | "
                 + " | ".join(f"{r[f'spearman_{m}']:.3f}" for m in methods) + " |")
    if deltas_c:
        L.append(f"\nDelta appaiato pre-registrato {LABEL[a_]} - {LABEL[b_]}: "
                 + "; ".join(f"{r['concordance']} {fmt(r['delta'], r['ci_low'], r['ci_high'], signed=True)} "
                             f"(P<=0 {r['p_le0']:.3f})" for r in deltas_c) + ".")

    L += ["\n## (d) Identita' fra ricostruzioni\n",
          "Retrieval: query = ogni ricostruzione, distanza dal soggetto = minimo sulle altre ricostruzioni di quel "
          "soggetto, rank fra 20 soggetti (caso: rank-1 0.05). Verifica: AUC su tutte le coppie i < j. "
          "Blocco `neutral_gallery`: query non neutre, galleria e coppie solo verso multiview_neutral.\n"]
    for block in ("all", "neutral_gallery"):
        L += [f"\n### Blocco {block}\n", "| metodo | metrica | rank-1 | mAP | AUC verifica | query / coppie |",
              "| --- | --- | --- | --- | --- | --- |"]
        sub = rec_table[rec_table["block"] == block]
        for m in methods:
            for metric in metrics:
                rr = sub[(sub["method"] == m) & (sub["metric"] == metric)]
                if rr.empty:
                    continue
                r = rr.iloc[0]
                L.append(f"| {m} | {LABEL[metric]} | " + " | ".join(fmt(r[k], r[f"{k}_ci_low"], r[f"{k}_ci_high"])
                                                                for k in ("rank1", "map", "auc"))
                         + f" | {r['n_queries']} / {r['n_pairs']} |")
        L += ["\nDelta appaiati latente - baseline:\n",
              "| metodo | baseline | rank-1 delta (P<=0) | mAP delta (P<=0) | AUC delta (P<=0) |", "| --- | --- | --- | --- | --- |"]
        sub = rec_delta_table[rec_delta_table["block"] == block] if not rec_delta_table.empty else rec_delta_table
        for r in sub.itertuples():
            L.append(f"| {r.method} | {LABEL[r.baseline]} | " + " | ".join(
                f"{fmt(getattr(r, k), getattr(r, k + '_ci_low'), getattr(r, k + '_ci_high'), signed=True)} "
                f"({getattr(r, k + '_p_le0'):.3f})" for k in ("rank1", "map", "auc")) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[now-sum] scritto {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
