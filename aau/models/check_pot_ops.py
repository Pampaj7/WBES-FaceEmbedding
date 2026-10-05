#!/usr/bin/env python3
"""Controllo degli operatori col pozzo contro quelli standard, file per file.

Il pozzo cambia SOLO lo spettro (L' = L + U, potential_operators.py): verts, faces, mass, L e
gradienti devono restare identici agli operatori standard, e devono esserci `roi_mask` e uno
spettro che differisce davvero. Un set di operatori "col pozzo" uguale a quello standard
(p.es. un alpha che non accende il pozzo da nessuna parte) produrrebbe un braccio identico al
controllo senza nessun errore: qui fallisce.

    aau/run.sh aau/models/check_pot_ops.py --pot-dir <out> --ref-dir <withops> --n 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STD_KEYS = ("verts", "faces", "mass", "evals", "evecs",
            "L_indices", "L_values", "L_shape",
            "gradX_indices", "gradX_values", "gradX_shape",
            "gradY_indices", "gradY_values", "gradY_shape")
SAME = ("verts", "faces", "mass", "L_values", "gradX_values", "gradY_values")
MIN_SPECTRAL_REL_DIFF = 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pot-dir", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60, help="file controllati, equispaziati nell'ordine")
    args = ap.parse_args()

    files = sorted(args.pot_dir.glob("*.npz"))
    if not files:
        sys.exit(f"ERRORE: nessun npz in {args.pot_dir}")
    pick = np.unique(np.linspace(0, len(files) - 1, min(args.n, len(files))).round().astype(int))
    errors = []
    rows = []
    for i in pick:
        p = files[int(i)]
        r = args.ref_dir / p.name
        with np.load(p) as a, np.load(r) as b:
            missing = [k for k in STD_KEYS + ("roi_mask",) if k not in a.files]
            if missing:
                errors.append(f"{p.name}: chiavi mancanti {missing}")
                continue
            for k in SAME:
                if a[k].shape != b[k].shape:
                    errors.append(f"{p.name}: {k} shape {a[k].shape} vs {b[k].shape}")
                elif not np.allclose(a[k], b[k], rtol=1e-4, atol=1e-6 * float(np.abs(b[k]).max() + 1e-30)):
                    errors.append(f"{p.name}: {k} diverso dallo standard "
                                  f"(max |d| {float(np.abs(a[k] - b[k]).max()):.3g})")
            for k in ("evals", "evecs", "mass", "roi_mask"):
                if not np.isfinite(a[k]).all():
                    errors.append(f"{p.name}: {k} non finito")
            ea, eb = a["evals"].astype(np.float64), b["evals"].astype(np.float64)
            # Il loader divide gli autovalori per il proprio massimo: e' la forma normalizzata
            # quella che il modello vede, quindi e' su quella che deve vedersi la differenza.
            na, nb = ea / (ea.max() + 1e-30), eb / (eb.max() + 1e-30)
            rel = float(np.abs(na[1:] - nb[1:]).mean() / (np.abs(nb[1:]).mean() + 1e-30))
            # Overlap del primo autovettore non costante nella metrica di massa: 1 = stesso modo.
            m = b["mass"].astype(np.float64)
            va, vb = a["evecs"][:, 1].astype(np.float64), b["evecs"][:, 1].astype(np.float64)
            ov = abs(float((va * m * vb).sum())) / np.sqrt(float((va * m * va).sum()) * float((vb * m * vb).sum()))
            roi = float((a["roi_mask"] > 0.5).mean())
            rows.append((p.name, ea.max() / max(eb.max(), 1e-30), rel, ov, roi))
            if rel < MIN_SPECTRAL_REL_DIFF:
                errors.append(f"{p.name}: spettro normalizzato quasi identico allo standard (rel {rel:.2e})")

    print(f"{'file':34s} {'lmax pot/std':>12s} {'d spettro':>10s} {'ovl evec1':>10s} {'ROI':>6s}")
    for name, lr, rel, ov, roi in rows:
        print(f"{name:34s} {lr:12.3g} {rel:10.4f} {ov:10.3f} {roi:6.3f}")
    if rows:
        arr = np.array([r[1:] for r in rows])
        print(f"{'mediana':34s} {np.median(arr[:, 0]):12.3g} {np.median(arr[:, 1]):10.4f} "
              f"{np.median(arr[:, 2]):10.3f} {np.median(arr[:, 3]):6.3f}")
        roi_by_topo: dict[str, list[float]] = {}
        for name, *_rest, roi in rows:
            roi_by_topo.setdefault(name.rsplit("_", 1)[-1][:-4], []).append(roi)
        print("frazione di vertici nella ROI per topologia: "
              + ", ".join(f"{t} {np.mean(v):.3f}" for t, v in sorted(roi_by_topo.items())))
    print(f"controllati {len(pick)} file di {len(files)}")
    if errors:
        print("\nERRORI:", file=sys.stderr)
        for e in errors[:30]:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print("OK: chiavi, verts/faces/mass/L/grad identici allo standard, roi_mask presente, spettro diverso")


if __name__ == "__main__":
    main()
