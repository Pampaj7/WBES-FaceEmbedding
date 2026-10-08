#!/usr/bin/env python
"""Confronto di due run (livello 2 dell'equivalenza): log per epoca, eval online e pesi finali.

    compare_runs.py --a <runs_root v2> --b <runs_root v3> [--floor <runs_root v2 bis>] --out confronto.json

Ogni runs_root contiene una sola run dir (quella che il trainer crea). Si confrontano colonna per colonna
train_log.csv, xtopo_mesh_log.csv, mixed_train_log.csv, extra_eval.csv e, per ogni epochNNN.pth comune, lo
state_dict (max |delta|). Con --floor anche a contro floor: il non determinismo della GPU fra due run v2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch


def run_dir(root: Path) -> Path:
    subs = [p for p in root.iterdir() if p.is_dir() and (p / "train_log.csv").exists()]
    if len(subs) != 1:
        raise SystemExit(f"{root}: attesa una run dir, trovate {[p.name for p in subs]}")
    return subs[0]


def read_csv(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def cmp_csv(a: Path, b: Path) -> dict:
    ra, rb = read_csv(a), read_csv(b)
    out = {"rows": [len(ra), len(rb)], "max_abs": {}}
    for x, y in zip(ra, rb):
        for k in x:
            if k not in y:
                continue
            try:
                fx, fy = float(x[k]), float(y[k])
            except ValueError:
                if x[k] != y[k]:
                    out["max_abs"][k] = "diverso"
                continue
            if fx != fx and fy != fy:
                continue
            out["max_abs"][k] = max(out["max_abs"].get(k, 0.0), abs(fx - fy))
    return out


def cmp_ckpt(a: Path, b: Path) -> dict:
    out = {}
    for pa in sorted((a / "checkpoints").glob("epoch*.pth")):
        if "_ema" in pa.name:
            continue
        pb = b / "checkpoints" / pa.name
        if not pb.exists():
            continue
        sa = torch.load(pa, map_location="cpu", weights_only=False)["state_dict"]
        sb = torch.load(pb, map_location="cpu", weights_only=False)["state_dict"]
        if set(sa) != set(sb):
            out[pa.name] = "chiavi diverse"
            continue
        out[pa.name] = max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)
    return out


def compare(a: Path, b: Path) -> dict:
    da, db = run_dir(a), run_dir(b)
    res = {"a": str(da), "b": str(db)}
    for f in ("train_log.csv", "xtopo_mesh_log.csv", "mixed_train_log.csv", "extra_eval.csv"):
        res[f] = cmp_csv(da / f, db / f)
    res["checkpoints_max_abs"] = cmp_ckpt(da, db)
    for f in ("best_by_auc.txt", "best_by_clean.txt", "best_by_xtopo_mesh_clean.txt"):
        ta, tb = da / f, db / f
        res[f] = [ta.read_text().split()[0] if ta.exists() else None, tb.read_text().split()[0] if tb.exists() else None]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--floor", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    x = ap.parse_args()
    out = {"v2_vs_v3": compare(x.a, x.b)}
    if x.floor is not None:
        out["v2_vs_v2bis"] = compare(x.a, x.floor)
    x.out.write_text(json.dumps(out, indent=1))
    for k, r in out.items():
        print(f"== {k}: {Path(r['a']).name} contro {Path(r['b']).name}")
        for f in ("train_log.csv", "xtopo_mesh_log.csv", "mixed_train_log.csv", "extra_eval.csv"):
            print(f"  {f}: righe {r[f]['rows']} max|delta| {r[f]['max_abs']}")
        print(f"  pesi: {r['checkpoints_max_abs']}")
        print(f"  best: auc {r['best_by_auc.txt']} clean {r['best_by_clean.txt']} xtopo {r['best_by_xtopo_mesh_clean.txt']}")


if __name__ == "__main__":
    main()
