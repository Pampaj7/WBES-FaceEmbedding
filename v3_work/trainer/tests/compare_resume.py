#!/usr/bin/env python
"""Ripresa da checkpoint: il run interrotto e ripreso (I) contro due run non interrotti (R, R2 = pavimento).

Per ciascuna coppia: loss passo per passo (steps_loss.csv), log per epoca e metriche dell'eval online
(train_log.csv, xtopo_mesh_log.csv, extra_eval.csv), pesi e pesi EMA all'ultimo checkpoint.

    compare_resume.py --i <runs_root I> --r <runs_root R> --r2 <runs_root R2> --out resume.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from compare_runs import cmp_csv, run_dir


def steps(d: Path) -> dict[int, float]:
    with open(d / "steps_loss.csv") as fh:
        return {int(r["step"]): float(r["loss"]) for r in csv.DictReader(fh)}


def last_ckpt(d: Path) -> Path:
    c = sorted(p for p in (d / "checkpoints").glob("epoch*.pth") if "_ema" not in p.name)
    return c[-1]


def pair(a: Path, b: Path) -> dict:
    da, db = run_dir(a), run_dir(b)
    sa, sb = steps(da), steps(db)
    common = sorted(set(sa) & set(sb))
    diffs = [abs(sa[k] - sb[k]) for k in common]
    out = {"a": str(da), "b": str(db), "steps_a": len(sa), "steps_b": len(sb),
           "steps_missing_or_extra": sorted(set(sa) ^ set(sb))[:20],
           "loss_max_abs": max(diffs) if diffs else None,
           "loss_first_diff_step": next((k for k in common if sa[k] != sb[k]), None),
           "final_step_loss": [sa[common[-1]], sb[common[-1]]] if common else None}
    for f in ("train_log.csv", "xtopo_mesh_log.csv", "extra_eval.csv"):
        out[f] = cmp_csv(da / f, db / f)
    ca, cb = last_ckpt(da), last_ckpt(db)
    pa = torch.load(ca, map_location="cpu", weights_only=False)
    pb = torch.load(cb, map_location="cpu", weights_only=False)
    out["checkpoint"] = [ca.name, cb.name]
    out["weights_max_abs"] = max(float((pa["state_dict"][k].float() - pb["state_dict"][k].float()).abs().max())
                                 for k in pa["state_dict"])
    if "ema_state_dict" in pa and "ema_state_dict" in pb:
        out["ema_weights_max_abs"] = max(float((pa["ema_state_dict"][k].float() - pb["ema_state_dict"][k].float())
                                               .abs().max()) for k in pa["ema_state_dict"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", type=Path, required=True)
    ap.add_argument("--r", type=Path, required=True)
    ap.add_argument("--r2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    x = ap.parse_args()
    res = {"interrotto_vs_continuo": pair(x.i, x.r), "pavimento_continuo_vs_continuo": pair(x.r2, x.r)}
    x.out.write_text(json.dumps(res, indent=1))
    for k, r in res.items():
        print(f"== {k}: passi {r['steps_a']}/{r['steps_b']}, loss max|delta| {r['loss_max_abs']}, "
              f"primo passo diverso {r['loss_first_diff_step']}, pesi {r['weights_max_abs']:.3e}, "
              f"EMA {r.get('ema_weights_max_abs', float('nan')):.3e}")
        print(f"   train_log {r['train_log.csv']['max_abs']}")
        print(f"   extra_eval {r['extra_eval.csv']['max_abs']}")


if __name__ == "__main__":
    main()
