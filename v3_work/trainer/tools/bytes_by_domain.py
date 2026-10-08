#!/usr/bin/env python3
"""Byte di cache per mesh, per dominio e topologia: viste dagli header, tar dall'indice (cache_budget)."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "aau/data_scale"))
from cache_budget import load_index, predicted_bytes  # noqa: E402


def dom(n):
    num = int(n.split("_GTready_")[0][2:])
    if 100000 <= num < 200000:
        return "gnm"
    return "bfm" if num < 1000 else ("ict" if num >= 10000 else "flame")


agg = collections.defaultdict(list)
per_subj = collections.defaultdict(lambda: collections.defaultdict(int))
vb = json.loads((REPO / "aau/data_scale/view_bytes_all.json").read_text())
for k, v in vb.items():
    lab = k[:-4].split("_GTready_")[1]
    agg[(dom(k), "view", lab)].append(v)
    per_subj[(dom(k), "view")][k.split("_GTready_")[0]] += v
idx = load_index(REPO / "datasets/SCALE_ALL/shards/index.npz")
for i, n in enumerate(idx["names"]):
    n = str(n)
    b = predicted_bytes(int(idx["n"][i]), int(idx["m"][i]), int(idx["E"][i]))
    lab = n[:-4].split("_GTready_")[1]
    agg[(dom(n), "tar", lab)].append(b)
    per_subj[(dom(n), "tar")][n.split("_GTready_")[0]] += b
for k in sorted(agg):
    v = np.asarray(agg[k])
    print(k, len(v), "MiB/mesh %.1f" % (v.mean() / 2 ** 20))
for k, d in sorted(per_subj.items()):
    v = np.asarray(list(d.values()))
    print("per soggetto", k, len(v), "MiB %.1f (min %.1f max %.1f)" % (v.mean() / 2 ** 20, v.min() / 2 ** 20, v.max() / 2 ** 20))
