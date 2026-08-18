#!/usr/bin/env python
"""Reads the alpha-sweep parts and prints the trade-off the well is subject to.

Two quantities, deliberately reported together:

  within    mean spectral distance between two TOPOLOGIES of the SAME identity.
            The well is supposed to push this down -- that is its entire purpose.
  between   mean spectral distance between DIFFERENT identities in the SAME topology.
            A well that has eaten the face pushes this down as well.

Reporting `within` alone would make alpha -> 0 look like a triumph, because a small enough
well turns every mesh into the same tiny disc: perfectly consistent and perfectly useless.
The ratio between/within is the quantity that has to improve for the well to be worth it, and
it is scale-free, so it can be compared across alphas whose spectra differ by orders of
magnitude.
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
NAME = re.compile(r"(?P<subj>.+?)_GTready_(?P<topo>.+)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts-dir", type=Path, default=THIS / "sweep_parts")
    ap.add_argument("--out", type=Path, default=THIS / "alpha_sweep_summary.json")
    args = ap.parse_args()

    data: dict[str, dict[str, list[float]]] = {}
    for p in sorted(args.parts_dir.glob("part_*.json")):
        data.update(json.loads(p.read_text()))
    if not data:
        raise FileNotFoundError(f"no part_*.json in {args.parts_dir}")

    parsed = {}
    for stem, rec in data.items():
        m = NAME.match(stem)
        if m:
            parsed[(m["subj"], m["topo"])] = {k: np.asarray(v) for k, v in rec.items()}
    subjects = sorted({s for s, _ in parsed})
    topos = sorted({t for _, t in parsed})
    alphas = sorted({k for r in parsed.values() for k in r},
                    key=lambda k: (k != "plain", k))
    print(f"{len(subjects)} identità, {len(topos)} topologie: {', '.join(topos)}\n")

    print(f"{'alpha':>8s} {'within':>10s} {'between':>10s} {'ratio':>8s} {'crop-within':>12s}")
    summary = {}
    for a in alphas:
        within, crop_within, between = [], [], []
        for s in subjects:
            for t1, t2 in combinations(topos, 2):
                x, y = parsed.get((s, t1)), parsed.get((s, t2))
                if x is None or y is None or a not in x or a not in y:
                    continue
                dist = float(np.linalg.norm(x[a] - y[a]))
                within.append(dist)
                if "crop" in (t1, t2):
                    crop_within.append(dist)
        for t in topos:
            for s1, s2 in combinations(subjects, 2):
                x, y = parsed.get((s1, t)), parsed.get((s2, t))
                if x is None or y is None or a not in x or a not in y:
                    continue
                between.append(float(np.linalg.norm(x[a] - y[a])))
        if not within or not between:
            continue
        w, b, cw = float(np.mean(within)), float(np.mean(between)), float(np.mean(crop_within))
        summary[a] = {"within": w, "between": b, "ratio": b / max(w, 1e-12),
                      "crop_within": cw, "n_within": len(within), "n_between": len(between)}
        print(f"{a:>8s} {w:10.4f} {b:10.4f} {b/max(w,1e-12):8.3f} {cw:12.4f}")

    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nscritto in {args.out}")
    print("\nLettura: `ratio` più alto = identità più separabili rispetto al rumore di "
          "topologia. Se scende passando da plain a alpha piccoli, il pozzo sta togliendo "
          "più identità che bordo.")


if __name__ == "__main__":
    main()
