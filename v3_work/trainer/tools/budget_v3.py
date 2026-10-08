#!/usr/bin/env python3
"""Memoria per blocco di un run v3 a blocchi: cache nel formato v2 e compatta, /tmp del pre-pass.

    aau/run.sh v3_work/trainer/tools/budget_v3.py --spec <spec.json> --split <split.json>

Stessa partizione del trainer (data_v3.partition_blocks) e stesse formule di aau/data_scale/cache_budget.py:
mesh dei tar dall'indice (n, m, spigoli), viste dagli header (view_bytes_all.json). Compatta: facce e indici
COO in int32 e niente L, cioe' meno 12 m + 36 nnz byte; per le viste (solo il totale e' noto) il rapporto
compatta/v2 misurato sulle mesh dei tar della stessa etichetta. /tmp del pre-pass: npz compressi delle sole
mesh dei tar, stimati come 0.70 x il formato v2 (6.2 contro 8.9 MiB per una ICT original, PLAN dati).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[2]
sys.path.insert(0, str(THIS.parent))
sys.path.insert(0, str(REPO / "aau/data_scale"))
from cache_budget import load_index, predicted_bytes  # noqa: E402

from common import domain_of, split_name  # noqa: E402

TMP_RATIO = 0.70


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    import data_v3 as dv
    spec = json.loads(a.spec.read_text())
    split = json.loads(a.split.read_text())
    sources = dv.collect_sources(spec)
    by_subj = collections.defaultdict(list)
    for n in sources:
        by_subj[split_name(n)[0]].append(n)
    vb = {k: int(v) for k, v in json.loads(Path(spec["view_bytes_json"]).read_text()).items()}
    idx = load_index(Path(spec["tar_index"]))
    pos = {str(n): i for i, n in enumerate(idx["names"])}
    ratio_by_label = collections.defaultdict(list)

    def tar_bytes(n):
        i = pos[n]
        nv, m, E = int(idx["n"][i]), int(idx["m"][i]), int(idx["E"][i])
        v2 = predicted_bytes(nv, m, E)
        nnz = nv + 2 * E
        return v2, v2 - 12 * m - 36 * nnz

    per = {}
    for s in split["train"]:
        v2 = cp = tmp = 0
        for n in by_subj[s]:
            if sources[n][0] == "tar":
                b2, bc = tar_bytes(n)
                ratio_by_label[split_name(n)[1]].append(bc / b2)
                v2, cp, tmp = v2 + b2, cp + bc, tmp + TMP_RATIO * b2
            else:
                v2 += vb[n]
        per[s] = [v2, cp, tmp]
    ratio = {k: float(np.mean(v)) for k, v in ratio_by_label.items()}
    r_all = float(np.mean([x for v in ratio_by_label.values() for x in v]))
    for s in split["train"]:          # viste: rapporto compatta/v2 dell'etichetta, misurato sui tar
        for n in by_subj[s]:
            if sources[n][0] != "tar":
                per[s][1] += vb[n] * ratio.get(split_name(n)[1], r_all)
    blocks = dv.partition_blocks(split["train"], spec)
    gib = 2 ** 30
    rows = []
    for k, b in enumerate(blocks):
        t = np.sum([per[s] for s in b], axis=0) / gib
        dom = collections.Counter(domain_of(s) for s in b)
        rows.append({"block": k, "subjects": len(b), "domains": dict(dom), "cache_v2_gib": round(float(t[0]), 1),
                     "cache_compact_gib": round(float(t[1]), 1), "prepass_tmp_gib": round(float(t[2]), 1)})
        print(rows[-1])
    worst = max(r["cache_compact_gib"] + r["prepass_tmp_gib"] for r in rows)
    worst_v2 = max(r["cache_v2_gib"] + r["prepass_tmp_gib"] for r in rows)
    out = {"blocks": rows, "compact_over_v2_by_label": ratio,
           "peak_cache_plus_next_prepass_gib": {"compact": round(worst, 1), "v2": round(worst_v2, 1)}}
    print(json.dumps(out["peak_cache_plus_next_prepass_gib"]), "rapporti compatta/v2:", ratio)
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
