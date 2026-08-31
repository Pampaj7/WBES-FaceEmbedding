#!/usr/bin/env python
"""La compressione riportata nel paper e' in parte un cambio di unita'.

Il paper (`paper.tex`, tab:distance_compression) misura quanto la registrazione comprime le
distanze inter-soggetto con il rapporto degli IQR ASSOLUTI rispetto a raw Chamfer, e conclude:

    «Rigid ICP retains only 53.7% of the raw Chamfer IQR, while NICP P2P retains only 21.9%»

e poi afferma:

    «Registration does not merely rescale distances; it reduces the dynamic range needed to
     distinguish subject pairs.»

Il problema e' che l'IQR assoluto non puo' distinguere le due cose. Se moltiplichi ogni
distanza per una costante k, l'IQR si moltiplica per k e il rapporto crolla, ma NESSUN rango
cambia -- e la metrica principale del paper e' lo Spearman, che e' invariante per
trasformazioni monotone. La parte del calo di IQR dovuta a un puro cambio di scala e' quindi
inerte: non puo' causare il degrado di ranking che dovrebbe spiegare.

E un riscalamento nella pipeline c'e'. `prealign_by_bbox` (faceBench/facebench/rigid_aligners/
icp.py:92-118) riscala la sorgente al raggio del bersaglio:

    source_scaled = source_centered / source_scale * target_scale

Viene applicato a rigid_p2p e a entrambe le NICP (run_facebench_remesh.py:166-170 e :185), e
NON a raw_chamfer (:158-161). Le quattro colonne della tabella non sono nelle stesse unita'.
Coerentemente, le mediane calano al 79%/65%/58% di raw.

La statistica giusta -- IQR/mediana, oppure il coefficiente di variazione std/media -- e'
invariante per riscalamento. Il `cv` e' gia' calcolato dal codice
(`analyze_distance_compression.py:73`) e salvato nel CSV, ma non e' mai stato riportato.

Questo script rilegge l'artefatto e riscrive la tabella nella forma invariante.
"""
from __future__ import annotations

import argparse, csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = (REPO / "faceBench" / "latentVSpipeline" / "outputs"
       / "distance_compression_clean_100subj_norm" / "distance_distribution_overall.csv")
ORDER = ["raw_chamfer", "rigid_p2p", "nicp_p2p", "nicp_p2tri"]
LABEL = {"raw_chamfer": "Raw Chamfer", "rigid_p2p": "Rigid ICP P2P",
         "nicp_p2p": "NICP P2P", "nicp_p2tri": "NICP P2Tri"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "gt_matrices")
    args = ap.parse_args()

    rows = {r["metric"]: r for r in csv.DictReader(open(args.artifact))
            if r["group_type"] == "overall"}
    missing = [m for m in ORDER if m not in rows]
    if missing:
        raise SystemExit(f"metriche assenti dall'artefatto: {missing}")

    n = int(rows["raw_chamfer"]["n"])
    base_rel = float(rows["raw_chamfer"]["iqr"]) / float(rows["raw_chamfer"]["p50"])
    base_cv = float(rows["raw_chamfer"]["cv"])

    out = []
    for m in ORDER:
        r = rows[m]
        med, iqr, cv = float(r["p50"]), float(r["iqr"]), float(r["cv"])
        rel = iqr / med
        out.append({
            "metric": m,
            "n": int(r["n"]),
            "median": med,
            "iqr": iqr,
            "iqr_over_raw_PAPER": float(r["iqr_vs_raw"]),      # quello stampato
            "median_over_raw": float(r["median_vs_raw"]),      # il fattore di scala inerte
            "iqr_over_median": rel,
            "rel_spread_over_raw": rel / base_rel,             # la correzione
            "cv": cv,
            "cv_over_raw": cv / base_cv,
        })

    # il conto che conta: quanto della compressione riportata e' puro cambio di unita'
    print(f"n = {n} coppie\n")
    print(f"{'metrica':16s} {'IQR/raw':>9s} {'mediana/raw':>12s} "
          f"{'(IQR/med)/raw':>14s} {'CV/raw':>8s}")
    print(f"{'':16s} {'(paper)':>9s} {'(inerte)':>12s} {'(corretto)':>14s} {'(corretto)':>8s}")
    for r in out:
        print(f"{LABEL[r['metric']]:16s} {r['iqr_over_raw_PAPER']:9.3f} "
              f"{r['median_over_raw']:12.3f} {r['rel_spread_over_raw']:14.3f} "
              f"{r['cv_over_raw']:8.3f}")

    # identita' algebrica: (IQR/raw) = (IQR/med)/raw * (mediana/raw). Se non torna, il CSV
    # e' incoerente e nessuna delle due letture vale.
    for r in out:
        lhs = r["iqr_over_raw_PAPER"]
        rhs = r["rel_spread_over_raw"] * r["median_over_raw"]
        assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(lhs)), (
            f"{r['metric']}: la decomposizione non torna ({lhs:.6f} vs {rhs:.6f}); "
            "le colonne dell'artefatto sono incoerenti fra loro")

    print("\nla compressione sopravvive ma e' piu' piccola di quanto riportato:")
    for r in out[1:]:
        pap, cor = r["iqr_over_raw_PAPER"], r["rel_spread_over_raw"]
        print(f"  {LABEL[r['metric']]:16s} paper 'trattiene {pap*100:.1f}%'  ->  "
              f"in realta' {cor*100:.1f}%   (sovrastima {cor/pap:.2f}x)")

    args.out.mkdir(parents=True, exist_ok=True)
    p = args.out / "compression_scale_invariant.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"\nscritto in {p}")


if __name__ == "__main__":
    main()
