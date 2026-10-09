#!/usr/bin/env python3
"""Riepilogo del registro delle viste usate da un run a streaming (train_stream.py --stream-log-views).

    aau/run.sh v3_work/stream/views_log.py <run_dir> [--merge out.npz]

Legge ``<run_dir>/views_used/*.npz`` (una riga per vista alla prima estrazione, per rank ed epoca; consumer.py:
colonne categoriche come codici + ``<nome>_vocab``, numeriche compatte) e stampa/scrive
(``views_used_summary.json``): viste uniche, identita' (gruppi: il seme, o la persona FaMoS) distinte per dominio,
origine e licenza, viste per discretizzazione ed espressione, quota delle fonti ridistribuibili (rango 0), CV di
S_i per dominio. ``--merge``: un solo npz con tutte le righe, decodificate (la descrizione del dataset di training).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

CAT = ("domain", "origin", "license", "label", "expr", "person", "key")


def load(run_dir: Path) -> dict:
    files = sorted((run_dir / "views_used").glob("rank*_e*.npz"))
    if not files:
        raise SystemExit(f"{run_dir}/views_used: nessun registro")
    parts = []
    for f in files:
        with np.load(f) as z:
            p = {k: z[k] for k in z.files if not k.endswith("_vocab")}
            for k in CAT:
                if f"{k}_vocab" in z.files:
                    p[k] = z[f"{k}_vocab"][z[k]]
                elif k in z.files:            # registro della prima versione: stringhe
                    p[k] = z[k]
            parts.append(p)
    keys = sorted(set.intersection(*(set(p) for p in parts)))
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["_files"] = len(files)
    return out


def identity_ids(t: dict) -> np.ndarray:
    """Un'identita' = il seme del gruppo; FaMoS = la persona (stesso soggetto in gruppi diversi)."""
    seed = np.char.add(np.char.add(t["seed0"].astype(str), "-"), np.char.add(t["seed1"].astype(str), "-"))
    seed = np.char.add(seed, t["seed2"].astype(str))
    if "key" in t:
        seed = np.where(t["seed0"] < 0, t["key"], seed)
    return np.where(t["domain"] == "famos", np.char.add("famos/", t["person"]), seed)


def summarize(t: dict) -> dict:
    n = len(t["domain"])
    gid = np.char.add(np.char.add(t["seed0"].astype(str), "-"), np.char.add(t["seed1"].astype(str), "-"))
    gid = np.char.add(gid, t["seed2"].astype(str))
    view_id = np.char.add(np.char.add(gid, "#"), t["vi"].astype(str))
    ident = identity_ids(t)
    first = {}
    for i, d, o in zip(ident, t["domain"], t["origin"]):
        first.setdefault(i, (d, o))
    cv = {}
    for d in np.unique(t["domain"]):
        _, k = np.unique(ident[t["domain"] == d], return_index=True)
        S = t["S"][t["domain"] == d][k].astype(np.float64)
        S = S[np.isfinite(S)]
        if len(S) > 1:
            cv[str(d)] = {"n": int(len(S)), "S_mean_mm": float(S.mean()), "S_cv": float(S.std(ddof=1) / S.mean())}
    return {"files": int(t["_files"]), "rows": n, "unique_views": int(len(np.unique(view_id))),
            "groups": int(len(np.unique(gid))), "identities": len(first),
            "identities_by_domain": dict(Counter(d for d, _ in first.values())),
            "identities_by_origin": dict(Counter(o for _, o in first.values())),
            "views_by_domain": dict(Counter(t["domain"].tolist())), "views_by_origin": dict(Counter(t["origin"].tolist())),
            "views_by_label": dict(Counter(t["label"].tolist())), "views_by_expr": dict(Counter(t["expr"].tolist())),
            "views_by_license": dict(Counter(t["license"].tolist())),
            "share_redistributable_views": float(np.mean(t["license_rank"] == 0)),
            "with_seed": float(np.mean(t["seed0"] >= 0)), "rings": dict(Counter(t["ring"].astype(str).tolist())),
            "S_by_domain": cv}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--merge", type=Path)
    a = ap.parse_args()
    t = load(a.run_dir)
    s = summarize(t)
    (a.run_dir / "views_used_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s, indent=1))
    if a.merge:
        np.savez_compressed(a.merge, **{k: v for k, v in t.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
