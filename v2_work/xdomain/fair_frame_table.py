#!/usr/bin/env python
"""Ogni metrica valutata nel frame in cui e' effettivamente calcolata.

La Tabella 1 del paper confronta otto metriche contro un D_GT costruito in `raw`. Ma solo una
delle otto vive in `raw`: le altre sono calcolate ognuna nel proprio frame, e valutarle contro
un bersaglio in un frame diverso le penalizza per una scelta nostra, non per la loro qualita'.

Mappa dei frame, accertata leggendo il codice (non dedotta):

    D_GT                        raw            compute_gt_distance_matrix_normalized.py
    latent_distance             raw (bersaglio)  ma ingresso in maxabs: dataset_gtready.py:299-306
    raw_chamfer                 maxabs         mesh_npz_utils.py:17-22
    varifold, currents          area           measure_distances.py:116-118
    arcface/clip/dinov2/lpips   bbox camera    render_mesh.py

L'ultimo caso e' un'approssimazione dichiarata: il renderer centra e riscala ogni mesh sul suo
bounding box, che e' una similarita' per-mesh della stessa famiglia di `maxabs` ma non identica.
Si riporta sotto `maxabs` come analogo piu' vicino, e lo si dice invece di fingere precisione.

`latent_distance` e' un caso a se' e non va letta come le altre: e' stata addestrata contro
D_GT in `raw`, quindi il suo valore sotto `raw` include il vantaggio di essere valutata nel
frame in cui e' stata ottimizzata. Le altre sette non hanno mai visto D_GT.
"""
from __future__ import annotations

import csv
from pathlib import Path

THIS = Path(__file__).resolve().parent
SRC = THIS / "gt_matrices" / "frame_metric_ranking.csv"

OWN = {
    "latent_distance": ("raw", "bersaglio dell'addestramento -- non confrontabile con le altre"),
    "raw_chamfer": ("maxabs", "mesh_npz_utils.py:17-22"),
    "varifold_dist": ("area", "measure_distances.py:116-118"),
    "currents_dist": ("area", "measure_distances.py:116-118"),
    "arcface_dist": ("maxabs", "bbox della camera, approssimato con maxabs"),
    "clip_dist": ("maxabs", "bbox della camera, approssimato con maxabs"),
    "dinov2_dist": ("maxabs", "bbox della camera, approssimato con maxabs"),
    "lpips_dist": ("maxabs", "bbox della camera, approssimato con maxabs; 5.700/148.500 coppie"),
}


def main() -> None:
    rows = {r["frame"]: r for r in csv.DictReader(open(SRC)) if r["pair_label"] == "OVERALL"}
    if "raw" not in rows:
        raise SystemExit(f"manca il frame di riferimento in {SRC}")

    out = []
    for m, (frame, note) in OWN.items():
        pub = float(rows["raw"][m])          # come riportato nel paper: tutto contro raw
        own = float(rows[frame][m])
        out.append({"metric": m, "own_frame": frame, "as_published_vs_raw_GT": pub,
                    "in_own_frame": own, "delta": own - pub, "source": note})

    out.sort(key=lambda r: -r["in_own_frame"])
    w = max(len(r["metric"]) for r in out)
    print(f"{'metrica':{w}s} {'frame':>7s} {'pubblicato':>11s} {'nel suo':>9s} {'delta':>8s}")
    for r in out:
        print(f"{r['metric']:{w}s} {r['own_frame']:>7s} {r['as_published_vs_raw_GT']:11.4f} "
              f"{r['in_own_frame']:9.4f} {r['delta']:+8.4f}")

    worse = [r for r in out if r["delta"] > 0 and r["metric"] != "latent_distance"]
    print(f"\n{len(worse)} baseline su {len(out)-1} sono sottostimate dalla valutazione pubblicata.")
    if worse:
        big = max(worse, key=lambda r: r["delta"])
        print(f"la piu' penalizzata e' {big['metric']}: {big['as_published_vs_raw_GT']:.4f} "
              f"-> {big['in_own_frame']:.4f} ({big['delta']:+.4f})")

    # il margine del nostro metodo, riletto onestamente
    lat = next(r for r in out if r["metric"] == "latent_distance")
    best = max((r for r in out if r["metric"] != "latent_distance"),
               key=lambda r: r["in_own_frame"])
    print(f"\nmargine sulla migliore baseline:")
    print(f"  come pubblicato   {lat['as_published_vs_raw_GT'] - best['as_published_vs_raw_GT']:+.4f}"
          f"  ({lat['metric']} {lat['as_published_vs_raw_GT']:.4f} vs "
          f"{best['metric']} {best['as_published_vs_raw_GT']:.4f})")
    print(f"  nei rispettivi    {lat['in_own_frame'] - best['in_own_frame']:+.4f}"
          f"  ({lat['in_own_frame']:.4f} vs {best['in_own_frame']:.4f})")

    p = THIS / "gt_matrices" / "fair_frame_table.csv"
    with open(p, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(out[0])); wr.writeheader(); wr.writerows(out)
    print(f"\nscritto in {p}")


if __name__ == "__main__":
    main()
