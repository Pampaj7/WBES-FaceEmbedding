#!/usr/bin/env python3
"""Secondo giro della prova di riproduzione: A / P / B su tre semi (trainer_repro2.sbatch).

    aau/run.sh aau/data_scale/trainer_repro_compare2.py --out-json aau/data_scale/trainer_repro2.json

Stessa valutazione fissa del primo giro (``trainer_repro_compare.heldout_eval``: ultimo checkpoint,
100 held-out BFM x 6 topologie, Spearman latente-GT su tutte le coppie di soggetti diversi e sulle
sole cross-topologia). Per ogni condizione media e deviazione standard sui semi; per P e B anche la
differenza APPAIATA con A dello stesso seme, che toglie la parte di rumore dovuta al seme.
Criterio: la media della differenza appaiata sta entro due volte la deviazione standard di A fra
semi (il rumore del seme della ricetta attuale).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from trainer_repro_compare import GT, SPLIT, VIEW, GTReadyDataset, heldout_eval, run_dir  # noqa: E402

ROOT = THIS_DIR.parents[1] / "aau/runs/data_scale_trainer_repro2"
SEEDS = (1234, 2345, 3456)
TAGS = ("A", "P", "B")
METRICS = ("spearman_all", "spearman_xtopo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, required=True)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    held = set(json.loads(SPLIT.read_text())["heldout"])
    ds = GTReadyDataset(str(VIEW))
    idx = [i for i, n in enumerate(ds.files) if n.split("_GTready_")[0] in held]
    sids = [ds.files[i].split("_GTready_")[0] for i in idx]
    with np.load(GT) as z:
        names = [str(n).split("_GTready")[0] for n in z["names"]]
        G = z["D_orig"]
    pos = {n: k for k, n in enumerate(names)}
    gi = [pos[s] for s in sids]
    D_gt = G[np.ix_(gi, gi)].astype(np.float64)

    res = {t: {m: [] for m in METRICS} for t in TAGS}
    per_seed = {}
    for s in SEEDS:
        for t in TAGS:
            r = heldout_eval(run_dir(ROOT / f"s{s}", t), ds, idx, sids, D_gt, dev)
            per_seed[f"{t}_s{s}"] = r
            for m in METRICS:
                res[t][m].append(r[m])
            print(t, s, r, flush=True)
    out: dict = {"per_seed": per_seed, "summary": {}}
    for m in METRICS:
        A = np.array(res["A"][m])
        sd = float(A.std(ddof=1))
        out["summary"][m] = {"A_mean": float(A.mean()), "A_sd_between_seeds": sd}
        for t in ("P", "B"):
            X = np.array(res[t][m])
            d = X - A
            out["summary"][m][f"{t}_mean"] = float(X.mean())
            out["summary"][m][f"{t}_minus_A_paired"] = [float(v) for v in d]
            out["summary"][m][f"{t}_minus_A_mean"] = float(d.mean())
            out["summary"][m][f"{t}_within_2sd"] = bool(abs(d.mean()) <= 2 * sd)
    text = json.dumps(out, indent=1)
    print(text)
    a.out_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
