#!/usr/bin/env python3
"""Split ridotti per lo smoke di e1_train_body.sh (WBES_E1_SMOKE=1): primi soggetti di ogni sorgente
dello split vero della cella; held-out ed eval online invariati. Solo stdlib.

    python3 aau/evidence/e1_factorial/smoke/make_smoke_splits.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAKE = {"bfm": 10, "ict5000": 10, "ictnew": 30, "gnm": 30}


def source_of(sid: str) -> str:
    num = int(sid[2:])
    return ("bfm" if num < 1000 else "gnm" if 100000 <= num < 200000
            else "ict5000" if num < 15000 else "ictnew")


for cell in ("c2m", "c2f", "c3f", "g1"):
    s = json.loads((HERE.parent / f"split_{cell}.json").read_text())
    by = {}
    for sid in s["train"]:
        by.setdefault(source_of(sid), []).append(sid)
    train = [x for src, ids in sorted(by.items()) for x in ids[:TAKE[src]]]
    out = dict(s, train=train, note=f"smoke di {cell}: {TAKE} dai primi soggetti di split_{cell}.json",
               counts={src: min(len(ids), TAKE[src]) for src, ids in by.items()})
    (HERE / f"split_{cell}_smoke.json").write_text(json.dumps(out, indent=0) + "\n")
    print(cell, out["counts"])
