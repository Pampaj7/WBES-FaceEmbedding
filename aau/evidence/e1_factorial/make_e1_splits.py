#!/usr/bin/env python3
"""E1 (paper/PLAN_MASSIVE.md, sezione 13): split delle celle del fattoriale varieta' x quantita'.

    aau/run.sh aau/evidence/e1_factorial/make_e1_splits.py      (numpy: nel container)

Il trainer (v2_work/fastio/train_steps.py) sceglie i soggetti di training dallo split esplicito
(``--split-json``, ``rebuild_subject_split``): le celle sono SOLO split nuovi, stesso held-out, stesso
eval online e stessi dati (spec) del run su scala 1060130. Nessuna modifica al trainer, nessuna copia
di dati: questo script scrive liste di id.

Celle (training; held-out, ``online_eval`` e ``online_eval_extra`` identici a split_scale_all.json):
  c3m  il run su scala stesso (riferimento, nessuno split nuovo): BFM 392 + ICT 54.008 + GNM 10.000
  c2m  BFM 392 + ICT 54.008
  c2f  BFM 392 + ICT ~1/10: 401 di ICT-5000 + 5.000 nuove (stesso rapporto 4.008/50.000)
  c3f  BFM 392 + ICT + GNM con lo stesso totale non-BFM di c2f (5.401) e il rapporto ICT/GNM di c3m
       (54.008/10.000): 338 ICT-5000 + 4.219 nuove + 844 GNM. Annidata: le sue ICT sono un
       sottoinsieme di quelle di c2f, quindi c3f - c2f = "844 ICT sostituite da 844 GNM"
  g1   GNM 10.000, nient'altro

Sottocampionamento stratificato per (sorgente, insieme di etichette della mesh): ogni strato
mantiene la sua quota (resto maggiore sul totale), quindi restano il rapporto ICT-5000/nuove e la
distribuzione di viste ed espressioni per identita' (ICT nuove: 6 topologie + 8 espressioni; GNM:
1 o 2 espressioni). Una permutazione per strato (seme 1234, la politica del run su scala, piu'
l'indice dello strato): c2f ne prende i primi n, c3f i primi n' <= n dello STESSO ordine.

Uscite in questa cartella: split_<cella>.json (stesso formato di split_scale_all.json) e
subsets.json (strati, conteggi, controlli).
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPLIT_C3M = REPO / "aau/data_scale/split_scale_all.json"
FROZEN = REPO / "aau/data_scale/heldout_frozen.json"
INDEX = REPO / "datasets/SCALE_ALL/shards/index.npz"
ICT_VIEW = REPO / "datasets/ICT/train_ready/npz_withops"
SEED = 1234
FRACTION_F = 0.1          # "ICT sottocampionato a circa 1/10"


def source_of(sid: str) -> str:
    num = int(sid[2:])
    if num < 1000:
        return "bfm"
    if 100000 <= num < 200000:
        return "gnm"
    if 10000 <= num < 15000:
        return "ict5000"
    if 20000 <= num < 70000:
        return "ictnew"
    raise SystemExit(f"{sid}: sorgente sconosciuta")


def labels_by_subject() -> dict[str, tuple[str, ...]]:
    """id -> etichette ordinate delle sue mesh: indice dei tar (ICT nuove, GNM) e vista di ICT-5000."""
    out: dict[str, list[str]] = defaultdict(list)
    with np.load(INDEX) as z:
        names = [str(n) for n in z["names"]]
    for n in names + sorted(os.listdir(ICT_VIEW)):
        if not n.endswith(".npz") or "_GTready_" not in n:
            continue
        sid, lab = n[:-4].split("_GTready_", 1)
        out[sid].append(lab)
    return {s: tuple(sorted(v)) for s, v in out.items()}


def largest_remainder(weights: dict, total: int) -> dict:
    """Quote intere proporzionali a ``weights`` che sommano a ``total`` (metodo del resto maggiore)."""
    w = sum(weights.values())
    q = {k: total * v / w for k, v in weights.items()}
    n = {k: int(np.floor(x)) for k, x in q.items()}
    for k in sorted(q, key=lambda k: (q[k] - n[k], str(k)), reverse=True)[: total - sum(n.values())]:
        n[k] += 1
    return n


def main() -> None:
    c3m = json.loads(SPLIT_C3M.read_text())
    train = list(c3m["train"])
    held = set(c3m["heldout"])
    fz = json.loads(FROZEN.read_text())
    frozen = set(fz["bfm"]) | set(fz["ict_view"]) | set(fz.get("gnm", []))
    labs = labels_by_subject()

    by_src: dict[str, list[str]] = defaultdict(list)
    for s in train:
        by_src[source_of(s)].append(s)
    missing = [s for s in train if source_of(s) != "bfm" and s not in labs]
    if missing:
        raise SystemExit(f"{len(missing)} soggetti di training senza mesh nell'indice/vista (primo {missing[0]})")

    # strati: (sorgente, etichette); ordine deterministico, una permutazione ciascuno
    strata: dict[tuple, list[str]] = defaultdict(list)
    for src in ("ict5000", "ictnew", "gnm"):
        for s in sorted(by_src[src], key=lambda x: int(x[2:])):
            strata[(src, labs[s])].append(s)
    keys = sorted(strata, key=lambda k: (k[0], len(k[1]), k[1]))
    perm = {}
    for i, k in enumerate(keys):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, i]))
        perm[k] = rng.permutation(np.array(strata[k], dtype=object)).tolist()

    n_ict = len(by_src["ict5000"]) + len(by_src["ictnew"])
    n_gnm = len(by_src["gnm"])
    ict_keys = [k for k in keys if k[0] != "gnm"]
    gnm_keys = [k for k in keys if k[0] == "gnm"]
    # c2f: ICT a ~1/10, per strato
    n_c2f = int(round(FRACTION_F * n_ict))
    take_c2f = largest_remainder({k: len(strata[k]) for k in ict_keys}, n_c2f)
    # c3f: stesso totale non-BFM di c2f, rapporto ICT/GNM di c3m, per strato (ICT e GNM insieme)
    take_c3f = largest_remainder({k: len(strata[k]) for k in ict_keys + gnm_keys}, n_c2f)
    for k in ict_keys:
        if take_c3f[k] > take_c2f[k]:
            raise SystemExit(f"{k[0]}: c3f prende {take_c3f[k]} > {take_c2f[k]} di c2f, l'annidamento non regge")

    bfm = sorted(by_src["bfm"], key=lambda x: int(x[2:]))
    ict_all = [s for k in ict_keys for s in strata[k]]
    gnm_all = [s for k in gnm_keys for s in strata[k]]
    cells = {
        "c2m": bfm + ict_all,
        "c2f": bfm + [s for k in ict_keys for s in perm[k][: take_c2f[k]]],
        "c3f": bfm + [s for k in ict_keys + gnm_keys for s in perm[k][: take_c3f[k]]],
        "g1": gnm_all,
    }

    # controlli
    c2f_ict = {s for s in cells["c2f"] if source_of(s) in ("ict5000", "ictnew")}
    c3f_ict = {s for s in cells["c3f"] if source_of(s) in ("ict5000", "ictnew")}
    assert c3f_ict <= c2f_ict, "c3f non annidata in c2f"
    for name, tr in cells.items():
        assert len(set(tr)) == len(tr), name
        assert set(tr) <= set(train), f"{name}: soggetti fuori dal training di c3m"
        assert not set(tr) & held, f"{name}: soggetti held-out nel training"
        assert not set(tr) & frozen, f"{name}: soggetti congelati nel training"

    def counts(tr: list[str]) -> dict:
        c = Counter(source_of(s) for s in tr)
        return {f"train_{k}": c.get(k, 0) for k in ("bfm", "ict5000", "ictnew", "gnm")}

    def strata_counts(tr: list[str]) -> dict:
        st = set(tr)
        return {f"{k[0]}|{len(k[1])} etichette|{','.join(k[1])}": sum(s in st for s in strata[k]) for k in keys}

    base = {k: c3m[k] for k in ("heldout", "online_eval", "online_eval_extra")}
    report = {"seed": SEED, "fraction_f": FRACTION_F, "source_split": str(SPLIT_C3M.relative_to(REPO)),
              "strata": {f"{k[0]}|{len(k[1])} etichette|{','.join(k[1])}": len(strata[k]) for k in keys},
              "cells": {"c3m": {"counts": counts(train), "strata": strata_counts(train)}}}
    for name, tr in cells.items():
        tr = sorted(tr, key=lambda x: int(x[2:]))
        c = counts(tr)
        note = (f"E1 cella {name}: training scelto da {SPLIT_C3M.name} con make_e1_splits.py (strati per sorgente "
                f"ed etichette, seme {SEED}); held-out, online_eval e online_eval_extra identici a {SPLIT_C3M.name}")
        (HERE / f"split_{name}.json").write_text(json.dumps(
            {"source": str(SPLIT_C3M.relative_to(REPO)), "note": note, "train": tr, **base,
             "counts": {**c, "heldout": len(c3m["heldout"])}}, indent=0) + "\n")
        report["cells"][name] = {"counts": c, "strata": strata_counts(tr)}
    nb = {n: report["cells"][n]["counts"] for n in report["cells"]}
    ratio = {n: {"ict5000/ictnew": (v["train_ict5000"] / v["train_ictnew"]) if v["train_ictnew"] else None,
                 "ict/gnm": ((v["train_ict5000"] + v["train_ictnew"]) / v["train_gnm"]) if v["train_gnm"] else None,
                 "non_bfm": v["train_ict5000"] + v["train_ictnew"] + v["train_gnm"]} for n, v in nb.items()}
    report["ratios"] = ratio
    report["checks"] = {"c3f_ict_subset_of_c2f_ict": True, "no_heldout_no_frozen_in_train": True,
                        "c2f_vs_c3f_same_non_bfm_total": ratio["c2f"]["non_bfm"] == ratio["c3f"]["non_bfm"]}
    (HERE / "subsets.json").write_text(json.dumps(report, indent=1) + "\n")
    for n, v in nb.items():
        print(f"[e1-split] {n}: {v} rapporti {ratio[n]}", flush=True)
    print(f"[e1-split] strati: {report['strata']}", flush=True)


if __name__ == "__main__":
    main()
