#!/usr/bin/env python3
"""FaMoS: effetto della taglia per ogni concorrente, con IC appaiato. Delta = lettura con taglia - lettura senza
(ICP mm - cs, NoW metrical - originale, ULRSSM per coppia mm - cs, GW mm - cs) in rank-1, mAP, AUC, EER (EER: senza -
con, positivo = la taglia aiuta), sulle stesse 11.175 coppie e con lo stesso bootstrap per soggetto di
``esterna_famos_id`` / ``famos_dominio`` (``ei.N_BOOT`` repliche, seme ``ei.BOOT_SEED``, repliche appaiate).

    sbatch aau/famos_dominio/size_pairs.sbatch          (solo CPU, pochi minuti)

Nessuna distanza nuova: colonne di ``ei.load_columns("eval")``. Controlli: metriche globali uguali a
``esterna_famos_id/metrics.csv``; delta ICP e NoW in rank-1 e AUC uguali al numeratore H3 di ``famos_dominio/size.csv``.
POST HOC e DESCRITTIVO (FaMoS TEST gia' visto). Uscite in ``aau/runs/evidence/famos_dominio/size_pairs.{csv,md}``.
"""

from __future__ import annotations

import csv
import time

import numpy as np

import fd  # noqa: E402  (path di esterna_famos_id e famos, ci, wmean)
import ei  # noqa: E402

PAIRS = (("ICP", "icp|mm", "icp|cs"), ("NoW", "now_metrical|mm", "now_original|mm"),
         ("ULRSSM", "ulrssm|pair_fr", "ulrssm|pair_sr"), ("GW", "gw|fr", "gw|sr"))


def main() -> None:
    import famos_eval as fe
    t0 = time.time()
    cols, it, _ = ei.load_columns("eval")
    subj_names = sorted({x for x, _, _ in it})
    subj = np.asarray([subj_names.index(x) for x, _, _ in it])
    n, ns = len(it), len(subj_names)
    iu, ju = ei.pairs(n)
    gen = subj[iu] == subj[ju]
    counts = np.concatenate([np.ones((1, ns), np.int64), fe.bootstrap_counts(ns, ei.N_BOOT, ei.BOOT_SEED)])
    W = np.where(gen[None], counts[:, subj[iu]], counts[:, subj[iu]] * counts[:, subj[ju]]).astype(np.float64)
    wq = counts[:, subj].astype(np.float64)
    R = {}
    for key in {k for _, a, b in PAIRS for k in (a, b)}:
        D = np.zeros((n, n))
        D[iu, ju] = D[ju, iu] = cols[key]
        r1, ap = ei.retrieval(D, subj)
        R[key, "rank1"] = fd.wmean(r1, wq)
        R[key, "mAP"] = fd.wmean(ap, wq)
        R[key, "AUC"] = np.asarray([fe.weighted_auc(-cols[key], gen, w) for w in W])
        R[key, "EER"] = np.asarray([ei.eer(cols[key], gen, w) for w in W])
        print(f"[size_pairs] {key}: AUC {R[key, 'AUC'][0]:.4f}", flush=True)
    with open(ei.EVID / "metrics.csv", newline="") as fh:
        ref = {(r["colonna"], r["metrica"]): float(r["valore"]) for r in csv.DictReader(fh)}
    check = max(abs(R[k][0] - ref[k]) for k in R)
    if check > 1e-12:
        raise SystemExit(f"metriche globali diverse da esterna_famos_id/metrics.csv: {check}")
    rows = []
    for fam, km, kn in PAIRS:
        for met in ei.METRICS:
            d = R[kn, met] - R[km, met] if met == "EER" else R[km, met] - R[kn, met]
            lo, hi, nx = fd.ci(d)
            rows.append({"famiglia": fam, "con_taglia": km, "senza_taglia": kn, "metrica": met,
                         "con": float(R[km, met][0]), "senza": float(R[kn, met][0]), "delta": float(d[0]),
                         "ic_lo": lo, "ic_hi": hi, "repliche_escluse": nx, "repliche": ei.N_BOOT,
                         "seme": ei.BOOT_SEED})
    # controllo contro il numeratore H3 di famos_dominio (ICP, NoW; AUC e rank-1)
    with open(fd.EVID / "size.csv", newline="") as fh:
        h3 = [r for r in csv.DictReader(fh) if r["grandezza"].startswith("numeratore")]
    dmax = 0.0
    for r in h3:
        x = next(y for y in rows if y["famiglia"] == r["famiglia"] and y["metrica"] == r["metrica"])
        dmax = max(dmax, *(abs(x[k] - float(r[k2])) for k, k2 in (("delta", "valore"), ("ic_lo", "ic_lo"),
                                                                   ("ic_hi", "ic_hi"))))
    if dmax > 1e-12:
        raise SystemExit(f"delta diversi dal numeratore H3 di famos_dominio/size.csv: {dmax}")
    out = fd.EVID / "size_pairs.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["# FaMoS: effetto della taglia per concorrente (con taglia - senza, IC appaiato)", "",
          f"Generato da `aau/famos_dominio/size_pairs.py` ({ei.stamp()['date']}, job {ei.stamp()['job']}). POST HOC e "
          f"DESCRITTIVO. 15 soggetti, 150 scansioni, 11.175 coppie; {ei.N_BOOT} repliche per soggetto, seme "
          f"{ei.BOOT_SEED}, appaiate. EER: senza - con. Controlli: metriche uguali a esterna_famos_id/metrics.csv "
          f"(max {check:.1e}); ICP e NoW uguali al numeratore H3 di size.csv (max {dmax:.1e}).", "",
          "| famiglia | metrica | con taglia | senza | delta [IC 95%] |", "|---|---|---|---|---|"]
    md += [f"| {r['famiglia']} | {r['metrica']} | {r['con']:.3f} | {r['senza']:.3f} | {r['delta']:+.3f} "
           f"[{r['ic_lo']:+.3f}, {r['ic_hi']:+.3f}] |" for r in rows]
    (fd.EVID / "size_pairs.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[6:]), f"\n[size_pairs] {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
