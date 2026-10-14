#!/usr/bin/env python3
"""NoW ranking (test di YJz1): classifiche dei metodi di ricostruzione con NoW e con la nostra metrica, e concordanza.

    aau/run.sh aau/recon/now_ranking.py [--methods 3ddfa_v2,synergynet,prnet,3ddfa_v3,hrn]

Regole: ``aau/runs/evidence/now_ranking/REGOLE.md`` (scritte prima dei numeri, commit f1bf27e).  Ingressi, per metodo:
``now_official_<m>.csv`` e ``gt_geometric_<m>.csv`` (nella cartella ``tables`` della corsa se ci sono, altrimenti in
``aau/runs/now_eval``, 7 ottobre), ``now_metrical_<m>.csv`` e ``gt_fact_<key>_<m>.csv`` (``tables``), le distanze per
vertice ufficiali in ``<WORK_ROOT>/official``.  Funzioni di ``now_summarize.py`` (tau per riga, Spearman, mediana
bootstrap delle distanze concatenate) e di ``zs_expr_summarize.py`` (repliche, IC) importate, non riscritte.

Uscite in ``aau/runs/evidence/now_ranking``: ``concordance.csv`` (tau per immagine e Spearman, IC), ``deltas.csv``,
``ranking.csv``, ``by_challenge.csv``, ``within_method.csv``, ``verdict.json`` e ``numbers.md`` (le tabelle in
markdown, da cui ``results.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "zs3dmm"))

import now_common as common  # noqa: E402
from now_summarize import kendall_rows, now_stats, spearman  # noqa: E402
from zs_expr_summarize import bootstrap_counts, ci  # noqa: E402

OUT = common.AAU_DIR / "runs" / "evidence" / "now_ranking"
TABLES = OUT / "tables"
OLD = common.AAU_DIR / "runs" / "now_eval"
KEYS = ("factorized_s1234", "factorized_s2345")
SEED_OF = {"factorized_s1234": "s1234", "factorized_s2345": "s2345"}
PRE_REGISTERED_7OCT = ("3ddfa_v2", "synergynet", "prnet")
COMPARATOR = "chamfer_raw"
LABEL = {
    "now": "NoW originale (mm)", "metrical": "NoW metrical (mm)", "chamfer_raw": "Chamfer grezza (maxabs)",
    "icp_chamfer_mm": "ICP + Chamfer (mm)",
    **{f"dP_{s}": f"d_P {s}" for s in SEED_OF.values()},
    **{f"dF_{s}": f"d_F {s} (taglia del template)" for s in SEED_OF.values()},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", type=str, default="3ddfa_v2,synergynet,prnet,3ddfa_v3,hrn")
    p.add_argument("--metric-methods", type=str, default="3ddfa_v3,hrn", help="metodi con unita' note (d_F metrica)")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def fmt(point: float, lo: float, hi: float, digits: int = 3, signed: bool = True) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


def first_existing(name: str) -> Path:
    for d in (TABLES, OLD):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"{name}: assente da {TABLES} e da {OLD}")


def load_long(methods: list[str]) -> pd.DataFrame:
    """Una riga per (immagine, metodo) con tutti i criteri."""
    frames = []
    for m in methods:
        df = pd.DataFrame(common.read_rows(first_existing(f"now_official_{m}.csv")))[
            ["name", "subject", "challenge", "now_median"]].rename(columns={"now_median": "now"})
        met = pd.DataFrame(common.read_rows(TABLES / f"now_metrical_{m}.csv"))[["name", "now_median"]]
        df = df.merge(met.rename(columns={"now_median": "metrical"}), on="name", how="left")
        geo = pd.DataFrame(common.read_rows(first_existing(f"gt_geometric_{m}.csv")))[
            ["name", "chamfer_raw", "icp_chamfer_mm"]]
        df = df.merge(geo, on="name", how="left")
        for key, s in SEED_OF.items():
            f = pd.DataFrame(common.read_rows(TABLES / f"gt_fact_{key}_{m}.csv"))[["name", "d_P", "d_F"]]
            df = df.merge(f.rename(columns={"d_P": f"dP_{s}", "d_F": f"dF_{s}"}), on="name", how="left")
        df["method"] = m
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if c not in ("name", "subject", "challenge", "method"):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def official_dists(tag: str) -> dict[str, np.ndarray]:
    res = np.load(common.WORK_ROOT / "official" / f"{tag}_computed_distances.npy", allow_pickle=True).item()
    return {str(f): np.asarray(d, dtype=np.float64) for f, d in zip(res["input_files"], res["computed_distances"])}


def tau_image(dc: pd.DataFrame, methods: list[str], names: list[str], crit: str, ref: str) -> np.ndarray:
    a = dc.pivot(index="name", columns="method", values=crit)[methods].loc[names].to_numpy()
    b = dc.pivot(index="name", columns="method", values=ref)[methods].loc[names].to_numpy()
    return kendall_rows(a, b)


def weighted_mean_reps(v: np.ndarray, s: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.asarray([(c[s] * v).sum() / c[s].sum() for c in counts])


def boot_spearman(x: np.ndarray, y: np.ndarray, s: np.ndarray, counts: np.ndarray) -> tuple[float, float, float]:
    rho = spearman(x, y)
    reps = np.asarray([spearman(x[idx], y[idx]) for idx in (np.repeat(np.arange(len(x)), c[s]) for c in counts)])
    lo, hi = ci(np.concatenate([[rho], reps]))
    return rho, lo, hi


def concordance_block(dc, methods, names, crits, ref, s_img, counts, tag):
    """tau per immagine (punto, repliche) di ogni criterio contro ``ref``, su ``methods`` e ``names``."""
    rows, reps = [], {}
    for crit in crits:
        t = tau_image(dc, methods, names, crit, ref)
        r = np.asarray([(c[s_img] * t).sum() / c[s_img].sum() for c in counts])
        lo, hi = ci(np.concatenate([[t.mean()], r]))
        reps[crit] = r
        rows.append({"set": tag, "methods": "+".join(methods), "n_images": len(names), "ref": ref, "criterion": crit,
                     "tau_image": float(t.mean()), "ci_low": lo, "ci_high": hi})
    return rows, reps


def verdict(tau_lo: float, tau_hi: float, d_lo: float, d_hi: float) -> str:
    """REGOLE.md sez. 4, per seme."""
    if tau_hi < 0 or d_hi < 0:
        return "NEGATIVO"
    if tau_lo > 0 and not d_hi < 0:
        return "INFORMATIVO POSITIVO"
    return "NON CONCLUSIVO"


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    metric_methods = [m.strip() for m in args.metric_methods.split(",") if m.strip()]
    items = common.load_items()
    by_name = {it.name: it for it in items}
    subjects = common.subjects_of(items)
    s2i = {s: k for k, s in enumerate(subjects)}
    counts = bootstrap_counts(len(subjects), args.n_bootstrap, args.seed)
    df = load_long(methods)
    crits = ["now", "metrical", "chamfer_raw", "icp_chamfer_mm"] + [f"{p}_{s}" for s in SEED_OF.values()
                                                                    for p in ("dP", "dF")]
    md = []

    def complete(ms: list[str]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
        sub = df[df["method"].isin(ms)].dropna(subset=crits)
        n_ok = sub.groupby("name")["method"].nunique()
        names = sorted(n_ok[n_ok == len(ms)].index)
        dc = sub[sub["name"].isin(names)].copy()
        dc["s"] = dc["subject"].map(s2i)
        return dc, names, np.asarray([s2i[by_name[n].subject] for n in names])

    # ---------------------------------------------------------------- primaria e comparatori (REGOLE sez. 3-4)
    dc, names, s_img = complete(methods)
    missing = {m: int(len(items) - df[(df["method"] == m)].dropna(subset=crits)["name"].nunique()) for m in methods}
    conc, deltas = [], []
    for ref in ("now", "metrical"):
        rows, reps = concordance_block(dc, methods, names, [c for c in crits if c != ref], ref, s_img, counts, "tutti")
        conc += rows
        if ref == "now":
            primary_reps = reps
    point = {r["criterion"]: r for r in conc if r["ref"] == "now" and r["set"] == "tutti"}
    seeds = {}
    for s in SEED_OF.values():
        crit = f"dP_{s}"
        for other in (COMPARATOR, "icp_chamfer_mm"):
            d = primary_reps[crit] - primary_reps[other]
            p_ = point[crit]["tau_image"] - point[other]["tau_image"]
            lo, hi = ci(np.concatenate([[p_], d]))
            deltas.append({"criterion": crit, "minus": other, "delta": p_, "ci_low": lo, "ci_high": hi,
                           "p_le0": float((d <= 0).mean())})
        dlt = next(x for x in deltas if x["criterion"] == crit and x["minus"] == COMPARATOR)
        seeds[s] = {"tau": point[crit], "delta": dlt,
                    "verdict": verdict(point[crit]["ci_low"], point[crit]["ci_high"], dlt["ci_low"], dlt["ci_high"])}
    vs = {v["verdict"] for v in seeds.values()}
    final = vs.pop() if len(vs) == 1 else "NON CONCLUSIVO"

    # sensibilita': i tre metodi del 7 ottobre
    dc3, names3, s3 = complete(list(PRE_REGISTERED_7OCT))
    rows3, reps3 = concordance_block(dc3, list(PRE_REGISTERED_7OCT), names3, [c for c in crits if c != "now"], "now",
                                     s3, counts, "tre_7ottobre")
    conc += rows3
    for s in SEED_OF.values():
        d = reps3[f"dP_{s}"] - reps3[COMPARATOR]
        r_ = {r["criterion"]: r["tau_image"] for r in rows3}
        p_ = r_[f"dP_{s}"] - r_[COMPARATOR]
        lo, hi = ci(np.concatenate([[p_], d]))
        deltas.append({"criterion": f"dP_{s}", "minus": COMPARATOR, "set": "tre_7ottobre", "delta": p_,
                       "ci_low": lo, "ci_high": hi, "p_le0": float((d <= 0).mean())})

    # per sfida
    chal_rows = []
    for ch in common.CHALLENGES:
        nm = [n for n in names if by_name[n].challenge == ch]
        si = np.asarray([s2i[by_name[n].subject] for n in nm])
        rows, _ = concordance_block(dc, methods, nm, [c for c in crits if c != "now"], "now", si, counts, ch)
        chal_rows += rows

    # ---------------------------------------------------------------- classifiche dei metodi (descrittive)
    imgs = [by_name[n].image for n in names]
    subj_of_image = {it.image: it.subject for it in items}
    rank_rows, scores = [], {}
    for tag, label in (("now_official", ""), ("metrical_official", "metrical_")):
        per = {}
        for m in methods:
            st = now_stats(official_dists(f"{label}{m}"), imgs, subj_of_image, subjects, counts)
            per[m] = (st["median"], st["median_reps"])
        scores[tag] = per
    for crit in crits:
        per = {}
        for m in methods:
            sub = dc[dc["method"] == m]
            v, s = sub[crit].to_numpy(), sub["s"].to_numpy()
            per[m] = (float(v.mean()), weighted_mean_reps(v, s, counts))
        scores[crit] = per
    order = {}
    for crit, per in scores.items():
        pt = np.asarray([per[m][0] for m in methods])
        reps = np.stack([per[m][1] for m in methods], axis=1)
        first = np.bincount(reps.argmin(1), minlength=len(methods)) / len(reps)
        order[crit] = pt
        for r, k in enumerate(np.argsort(pt)):
            lo, hi = ci(np.concatenate([[pt[k]], reps[:, k]]))
            rank_rows.append({"criterion": crit, "rank": r + 1, "method": methods[k], "score": float(pt[k]),
                              "ci_low": lo, "ci_high": hi, "p_first": float(first[k])})
    glob = {c: {ref: float(kendall_rows(order[c][None], order[ref][None])[0]) for ref in ("now_official", "metrical_official")}
            for c in order}

    # ---------------------------------------------------------------- dentro il metodo (descrittivo)
    within = []
    for m in methods:
        sub = dc[dc["method"] == m]
        s = sub["s"].to_numpy()
        for crit, ref in [(f"dP_{x}", "now") for x in SEED_OF.values()] + [(COMPARATOR, "now"), ("icp_chamfer_mm", "now")]:
            rho, lo, hi = boot_spearman(sub[crit].to_numpy(), sub[ref].to_numpy(), s, counts)
            within.append({"method": m, "criterion": crit, "ref": ref, "n": len(sub), "spearman": rho,
                           "ci_low": lo, "ci_high": hi})
    for m in metric_methods:
        sub = df[df["method"] == m][["name", "subject", "metrical", "now"]].copy()
        for key, sd in SEED_OF.items():
            f = pd.DataFrame(common.read_rows(TABLES / f"gt_fact_{key}_{m}_metric.csv"))[["name", "d_P", "d_F"]]
            f[["d_P", "d_F"]] = f[["d_P", "d_F"]].apply(pd.to_numeric)
            sub = sub.merge(f.rename(columns={"d_P": f"dPm_{sd}", "d_F": f"dFm_{sd}"}), on="name", how="inner")
        sub = sub.dropna()
        s = sub["subject"].map(s2i).to_numpy()
        for crit in [f"{p}_{x}" for x in SEED_OF.values() for p in ("dFm", "dPm")]:
            for ref in ("metrical", "now"):
                rho, lo, hi = boot_spearman(sub[crit].to_numpy(), sub[ref].to_numpy(), s, counts)
                within.append({"method": m, "criterion": f"{crit} (patch metrica)", "ref": ref, "n": len(sub),
                               "spearman": rho, "ci_low": lo, "ci_high": hi})

    # d_F metrica contro NoW metrical fra i metodi con unita' note: tau per immagine se sono almeno due (REGOLE sez. 5)
    if len(metric_methods) >= 2:
        fr = []
        for m in metric_methods:
            x = df[df["method"] == m][["name", "subject", "method", "metrical", "now"]].copy()
            for key, sd in SEED_OF.items():
                f = pd.DataFrame(common.read_rows(TABLES / f"gt_fact_{key}_{m}_metric.csv"))[["name", "d_P", "d_F"]]
                f[["d_P", "d_F"]] = f[["d_P", "d_F"]].apply(pd.to_numeric)
                x = x.merge(f.rename(columns={"d_P": f"dPm_{sd}", "d_F": f"dFm_{sd}"}), on="name", how="inner")
            fr.append(x.dropna())
        dm = pd.concat(fr, ignore_index=True)
        n_ok = dm.groupby("name")["method"].nunique()
        nm = sorted(n_ok[n_ok == len(metric_methods)].index)
        dm = dm[dm["name"].isin(nm)]
        sm = np.asarray([s2i[by_name[n].subject] for n in nm])
        for ref in ("metrical", "now"):
            rows, _ = concordance_block(dm, metric_methods, nm, [f"{p}_{x}" for x in SEED_OF.values()
                                                                 for p in ("dFm", "dPm")], ref, sm, counts, "metrici")
            conc += rows

    # ---------------------------------------------------------------- uscite
    pd.DataFrame(conc).to_csv(OUT / "concordance.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "deltas.csv", index=False)
    pd.DataFrame(chal_rows).to_csv(OUT / "by_challenge.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(OUT / "ranking.csv", index=False)
    pd.DataFrame(within).to_csv(OUT / "within_method.csv", index=False)
    (OUT / "verdict.json").write_text(json.dumps({
        "methods": methods, "n_images_complete": len(names), "missing_per_method": missing,
        "per_seed": {s: {"tau": v["tau"], "delta_vs_chamfer_raw": v["delta"], "verdict": v["verdict"]}
                     for s, v in seeds.items()},
        "final": final, "global_tau": glob, "n_bootstrap": args.n_bootstrap, "seed": args.seed}, indent=2),
        encoding="utf-8")

    md.append(f"Metodi: {', '.join(methods)}; immagini con tutti i metodi e tutti i criteri: {len(names)} / "
              f"{len(items)}; mancanti per metodo: {missing}. IC 95% bootstrap per soggetto ({args.n_bootstrap} "
              f"repliche, seme {args.seed}).\n")
    md.append("### Primaria: tau per immagine contro NoW originale\n")
    md.append("| criterio | tutti i metodi | tre del 7 ottobre |\n|---|---|---|")
    r3 = {r["criterion"]: r for r in rows3}
    for crit in [c for c in crits if c != "now"]:
        a = point[crit]
        md.append(f"| {LABEL.get(crit, crit)} | {fmt(a['tau_image'], a['ci_low'], a['ci_high'])} | "
                  f"{fmt(r3[crit]['tau_image'], r3[crit]['ci_low'], r3[crit]['ci_high'])} |")
    md.append("\n| delta | insieme | valore | P(<= 0) |\n|---|---|---|---|")
    for d in deltas:
        md.append(f"| {LABEL[d['criterion']]} - {LABEL[d['minus']]} | {d.get('set', 'tutti')} | "
                  f"{fmt(d['delta'], d['ci_low'], d['ci_high'])} | {d['p_le0']:.3f} |")
    md.append("\n| seme | esito |\n|---|---|")
    for s, v in seeds.items():
        md.append(f"| {s} | {v['verdict']} |")
    md.append(f"| finale | **{final}** |\n")
    md.append("### Contro NoW metrical (descrittivo)\n\n| criterio | tau per immagine |\n|---|---|")
    for r in conc:
        if r["ref"] == "metrical" and r["set"] == "tutti":
            md.append(f"| {LABEL.get(r['criterion'], r['criterion'])} | {fmt(r['tau_image'], r['ci_low'], r['ci_high'])} |")
    if any(r["set"] == "metrici" for r in conc):
        md.append("\n### Metodi con unita' note, patch metriche (descrittivo)\n\n| criterio | contro | tau per immagine |\n|---|---|---|")
        for r in conc:
            if r["set"] == "metrici":
                md.append(f"| {r['criterion']} ({r['methods']}) | {r['ref']} | {fmt(r['tau_image'], r['ci_low'], r['ci_high'])} |")
    md.append("\n### Per sfida (tau per immagine contro NoW originale)\n")
    md.append("| criterio | " + " | ".join(f"{c} (n)" for c in common.CHALLENGES) + " |\n|---|" + "---|" * 4)
    for crit in [c for c in crits if c != "now"]:
        cells = []
        for ch in common.CHALLENGES:
            r = next(x for x in chal_rows if x["set"] == ch and x["criterion"] == crit)
            cells.append(f"{fmt(r['tau_image'], r['ci_low'], r['ci_high'], 2)} ({r['n_images']})")
        md.append(f"| {LABEL.get(crit, crit)} | " + " | ".join(cells) + " |")
    md.append("\n### Classifiche dei metodi (descrittive)\n")
    md.append("| criterio | ordine (punteggio [IC], P(primo)) | tau globale con NoW orig. / metrical |\n|---|---|---|")
    rk = pd.DataFrame(rank_rows)
    for crit in scores:
        sub = rk[rk["criterion"] == crit].sort_values("rank")
        cells = "; ".join(f"{r.method} {r.score:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}] {r.p_first:.2f}"
                          for r in sub.itertuples())
        md.append(f"| {LABEL.get(crit, crit)} | {cells} | {glob[crit]['now_official']:+.2f} / "
                  f"{glob[crit]['metrical_official']:+.2f} |")
    md.append("\n### Dentro il metodo: Spearman fra le immagini (descrittivo)\n\n| metodo | criterio | contro | n | rho |\n|---|---|---|---|---|")
    for w in within:
        md.append(f"| {w['method']} | {LABEL.get(w['criterion'], w['criterion'])} | {w['ref']} | {w['n']} | "
                  f"{fmt(w['spearman'], w['ci_low'], w['ci_high'])} |")
    (OUT / "numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
