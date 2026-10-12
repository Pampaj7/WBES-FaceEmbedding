#!/usr/bin/env python3
"""Ablazione 2x2, rumore del checkpoint finale (POST HOC, lettura corretta, punto 7): A e A' a epoch090_ema contro
epoch100_ema sulle stesse righe e repliche delle famiglie di test.

    v3_work/unified_gt/run.sh v3_work/stream/eval_ablation_epoch.py --workers 32

Celle: ``A100``, ``Ap100`` (``<cella>/eval``) e ``A090``, ``Ap090`` (``<cella>/eval_e090``, ablation_2x2_eval.sbatch con
ABL_EPOCH=090), ognuna con la sua c (calib.json dello stesso checkpoint). Righe, repliche e distanze di eval_ablation.py
(importato in sola lettura: ``families``). Per (famiglia, gruppo, GT): rho, delta e090 - e100 per seme e medio,
divario fra semi A - A' a e090 e a e100. Uscite in ``ablation_2x2/epoch_noise``: ``spearman.csv``, ``deltas.csv``,
``results.md``. Ipotesi in prova: con lr costante l'iterato finale oscilla di 0.03-0.07.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import eval_ablation as EA

CELLS = {"A100": ("A", "eval", 100), "A090": ("A", "eval_e090", 90), "Ap100": ("Ap", "eval", 100),
         "Ap090": ("Ap", "eval_e090", 90)}


def load(root: Path) -> dict:
    out = {}
    for name, (cell, ev, ep) in CELLS.items():
        ck = sorted((root / cell / "runs").glob(f"v3_*/checkpoints/epoch{ep:03d}_ema.pth"))
        if len(ck) != 1:
            raise SystemExit(f"{name}: {len(ck)} checkpoint")
        ck = ck[0].resolve()
        cal = json.loads((root / cell / ev / "calib/calib.json").read_text())
        if Path(cal["checkpoint"]).resolve() != ck:
            raise SystemExit(f"{name}: calib.json di {cal['checkpoint']}")
        emb = {}
        for f in EA.FAMILIES:
            hits = sorted((root / cell / ev / EA.FORM_DIR[f]).glob(
                f"data_*/scale_v3*fulle{ep:03d}*/zs_zeroshot/embeddings.npz"))
            if len(hits) != 1:
                raise SystemExit(f"{name}, {f}: {len(hits)} embedding")
            with np.load(hits[0], allow_pickle=True) as z:
                if Path(str(z["checkpoint"])).resolve() != ck:
                    raise SystemExit(f"{name}, {f}: embedding di un altro checkpoint")
            emb[f] = hits[0]
        out[name] = {"ckpt": ck, "c": float(cal["c_median"]), "c_ls": float(cal["c_ls"]), "emb": emb}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--root", type=Path, default=EA.ABL)
    a = ap.parse_args()
    out = a.root / "epoch_noise"
    cells = load(a.root)
    V, ctrl = EA.families(cells, a.n_boot, a.workers)
    sp, dl = [], []
    for (f, g), reps in V.items():
        if g == "nocrop_all":
            continue
        for gt, dist in EA.PRIMARY.items():
            R = {x: reps[(gt, f"{x}|{dist}")] for x in CELLS}
            for x, v in R.items():
                sp.append({"family": f, "group": g, "gt": gt, "distance": dist, "cell": x, **EA.ci(v)})
            con = {"A: e090 - e100": R["A090"] - R["A100"], "A': e090 - e100": R["Ap090"] - R["Ap100"],
                   "media: e090 - e100": (R["A090"] - R["A100"] + R["Ap090"] - R["Ap100"]) / 2,
                   "A - A' a e100": R["A100"] - R["Ap100"], "A - A' a e090": R["A090"] - R["Ap090"]}
            for k, v in con.items():
                dl.append({"family": f, "group": g, "gt": gt, "distance": dist, "contrast": k, **EA.ci(v)})
    out.mkdir(parents=True, exist_ok=True)
    EA.write_csv(out / "spearman.csv", sp)
    EA.write_csv(out / "deltas.csv", dl)
    abs_d = [abs(r["point"]) for r in dl if r["contrast"] in ("A: e090 - e100", "A': e090 - e100") and r["group"] == "nocrop"]
    md = ["# Ablazione 2x2: epoch090_ema contro epoch100_ema di A e A' (POST HOC)", "",
          "Generato da `v3_work/stream/eval_ablation_epoch.py`; c calibrata per checkpoint (" + ", ".join(
              f"{x} {i['c']:.4f}" for x, i in cells.items()) + f"); {a.n_boot} repliche, righe e semi di eval_ablation.",
          "", f"Righe nocrop (3 famiglie x FR, SR, 2 semi): |e090 - e100| mediana {np.median(abs_d):.3f}, "
          f"massimo {max(abs_d):.3f}, RMS {np.sqrt(np.mean(np.square(abs_d))):.3f}.", "",
          "| famiglia | gruppo | GT | A e090 - e100 | A' e090 - e100 | media | A - A' e100 | A - A' e090 |",
          "|---|---|---|---|---|---|---|---|"]
    D = {(r["family"], r["group"], r["gt"], r["contrast"]): r for r in dl}
    for f in EA.FAMILIES:
        for g in ("nocrop", "all_cross", "crop"):
            for gt in EA.PRIMARY:
                md.append(f"| {f} | {g} | {gt} | " + " | ".join(EA.fmt(D[(f, g, gt, k)]) for k in (
                    "A: e090 - e100", "A': e090 - e100", "media: e090 - e100", "A - A' a e100", "A - A' a e090")) + " |")
    md += ["", "Controlli: " + json.dumps({f: {g: i["n_rows"] for g, i in c["groups"].items()} for f, c in ctrl.items()})]
    EA.atomic_text(out / "results.md", "\n".join(md) + "\n")
    print(f"[abl-epoch] |e090 - e100| nocrop: mediana {np.median(abs_d):.3f}, massimo {max(abs_d):.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
