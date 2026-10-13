#!/usr/bin/env python3
"""Ava-256, prova confermativa: testa adattata di R6 (PROTOCOL_CONFERMATIVO, sez. 10), la procedura di D5 con il blocco
del dominio da spec in coda alle 8 sorgenti di ``joint``. Collaudata su HIFI3D con lo STESSO codice (spec
``aau/ava256/spec/hifi3d_full.json``: blocco = il pool NON valutato di HIFI3D di D3, confrontato col blocco di
``d3_stats.build_sources``).

    v3_work/unified_gt/run.sh aau/ava256/conf_head.py block --spec <json>        (solo il blocco e i controlli)
    v3_work/unified_gt/run.sh aau/ava256/conf_head.py fit --spec <json> --workers 2    (nodo i256-a10-10, conf.sbatch)

Importati in sola lettura: ``d3_stats`` (embedding e GT delle sorgenti, ``make_block``, ``gt_matrix``,
``store_by_names``), ``d3_e2`` (``blocks_for(arm, "joint")``: le 8 sorgenti con FLAME a meta', ``fit_e2``: perdita
riscalata per blocco e ``d3_head.fit``), ``d3_head`` (``calib_c``), ``diag`` (checkpoint, ``dp_per_unit``).

Blocco del dominio (``head`` dello spec): ``subjects`` x le 5 etichette senza crop (``diag.LABELS``), embedding del seme
(``head.embeddings[<braccio>]``, stesso checkpoint della calibrazione: K4), x = u x dp_per_unit, GT SR
(``head.gt_sr``, ``d3_stats.gt_matrix``: D_orig x dP_per_unit del json), coppie di soggetti diversi ed etichette
diverse (``d3_head.Block.pairs``); Ava-256: 56 x 5, 1.540 x 20 = 30.800 coppie.

``fit`` per braccio: configurazione (r, lambda) = ``d3_e2_heads.csv`` riga ``e2`` / ``joint`` del braccio, controllata
contro lo spec (r = 64, lambda = 1e-6). Controllo K0 (nessun dato del dominio): la testa con le sole 8 sorgenti, stessa
configurazione, deve riprodurre c e alpha / alpha0 di quella riga entro ``K0_REL`` relativo; se no si riporta lo scarto e
si prosegue (come K0 di D5). Poi la testa sulle 8 sorgenti + il blocco (pesi EQ: ``d3_head.Problem``, ogni blocco 1/9),
c della testa con ``calib_c`` sulle SOLE 8 sorgenti (la GT del dominio non entra in nessun c). Nessun rifit: si
registrano nodo, job, ``success`` e iterazioni. Uscite ``<work>/head/<braccio>.npz`` (``alpha``, ``W``, ``r``, ``lam``,
``alpha0``, ``nit``, ``success``, ``loss``, ``c``, ``dpu``, ``host``) e ``<evidence>/head/<braccio>.json``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS.parent / "diagnostics"))
sys.path.insert(0, str(THIS))
import conf_common as cc  # noqa: E402
import d3_e2 as E2  # noqa: E402
import d3_head as H  # noqa: E402
import d3_stats as S  # noqa: E402
import diag  # noqa: E402

K0_REL = 1e-6


def info_of(arm: str) -> dict:
    """Checkpoint e dp_per_unit del braccio (quelli della calibrazione, come ``d5_fewshot.main_fit``)."""
    _, _, ck = S.store_by_names(diag.CALIB_EMB / arm / "embeddings.npz", diag.set_names("bfm"))
    return {"ckpt": {arm: ck}, "dpu": {arm: diag.dp_per_unit(ck)}}


def config(arm: str) -> tuple[int, float]:
    for r in csv.DictReader(open(diag.EV / "d3_e2_heads.csv")):
        if r["kind"] == "e2" and r["arm"] == arm and r["setting"] == "joint":
            return int(r["r"]), float(r["lambda"])
    raise SystemExit(f"{arm}: riga e2 / joint assente da d3_e2_heads.csv")


def k0_reference(arm: str) -> dict:
    for r in csv.DictReader(open(diag.EV / "d3_e2_heads.csv")):
        if r["kind"] == "e2" and r["arm"] == arm and r["setting"] == "joint":
            return {"c": float(r["c"]), "alpha_over_alpha0": float(r["alpha_over_alpha0"]), "nit": int(r["nit"])}
    raise SystemExit(f"{arm}: riga e2 / joint assente")


def spec_block(s: dict, arm: str, info: dict) -> H.Block:
    """Il blocco del dominio dello spec (``head``), con i controlli K4 (checkpoint) e di dimensione."""
    h = s["head"]
    names = sorted(f"{x}_GTready_{lab}.npz" for x in h["subjects"] for lab in diag.LABELS)
    names, Z, ck = S.store_by_names(cc.rp(h["embeddings"][arm]), names)
    if ck.resolve() != info["ckpt"][arm].resolve():
        raise SystemExit(f"{arm}: embedding del blocco col checkpoint {ck}, atteso {info['ckpt'][arm]}")
    if not np.isfinite(Z).all():
        raise SystemExit(f"{arm}: embedding del blocco non finiti")
    gn, G = S.gt_matrix(cc.rp(h["gt_sr"]), "sr")
    b = S.make_block(names, Z, gn, G, info["dpu"][arm], s["name"])
    if len(b.G) != len(h["subjects"]) or len(names) != len(h["subjects"]) * len(diag.LABELS):
        raise SystemExit(f"{arm}: blocco di {len(b.G)} soggetti e {len(names)} mesh")
    return b


def block_report(b: H.Block) -> dict:
    i, j = H.pair_index(b)
    return {"subjects": int(len(b.G)), "meshes": int(len(b.X)), "pairs": int(len(i)), "p": int(b.X.shape[1])}


def step_block(spec: Path) -> None:
    """Il blocco per ogni braccio; nei collaudi anche il confronto col blocco di ``d3_stats.build_sources`` della
    sorgente ``head.compare_source`` (stessi X, soggetti, etichette, G)."""
    s = cc.load(spec)
    if s.get("freeze"):
        cc.verify(s)
    out = {}
    for arm in s["head"]["arms"]:
        info = info_of(arm)
        b = spec_block(s, arm, info)
        rep = block_report(b)
        src = s["head"].get("compare_source")
        if src:
            S.build_sources(arm, info)
            ref = S._B[(arm, src)]
            rep["compare_source"] = src
            rep["equal"] = {k: bool(np.array_equal(getattr(b, k), getattr(ref, k))) for k in ("X", "subj", "lab", "G")}
            rep["X_max_abs_diff"] = float(np.abs(b.X - ref.X).max())
        out[arm] = rep
    cc.atomic_json(cc.evidence(s) / "head" / "block.json", {"spec": s["_path"], "arms": out, **cc.stamp()})
    print(f"[conf-head] blocco {s['name']}: {json.dumps(out)}", flush=True)
    bad = [a for a, r in out.items() if "equal" in r and not all(r["equal"].values())]
    if bad:
        raise SystemExit(f"blocco diverso da quello di d3_stats: {bad}")


def step_fit(spec: Path) -> None:
    s = cc.load(spec)
    if s.get("freeze"):
        cc.verify(s)
    for arm in s["head"]["arms"]:
        t0 = time.time()
        info = info_of(arm)
        r, lam = config(arm)
        if (r, lam) != (int(s["head"]["r"]), float(s["head"]["lambda"])):
            raise SystemExit(f"{arm}: configurazione {r, lam} diversa da quella dello spec")
        S.build_sources(arm, info)
        blocks = E2.blocks_for(arm, "joint")
        # K0: le sole 8 sorgenti, nessun dato del dominio
        h0 = E2.fit_e2(blocks, r, lam)
        c0 = H.calib_c(lambda b, i, j: h0.dist(b.X[i], b.X[j]), blocks)
        ref = k0_reference(arm)
        k0 = {"c": c0, "alpha_over_alpha0": h0.alpha / h0.alpha0, "nit": h0.nit, "success": h0.success,
              "reference": ref, "rel_c": abs(c0 / ref["c"] - 1.0),
              "rel_alpha": abs(h0.alpha / h0.alpha0 / ref["alpha_over_alpha0"] - 1.0)}
        k0["pass"] = k0["rel_c"] <= K0_REL and k0["rel_alpha"] <= K0_REL
        if s.get("collaudo") and not s["head"].get("fit_block", False):
            rep = {"arm": arm, "K0": k0, "r": r, "lambda": lam, "host": socket.gethostname(), **cc.stamp()}
            cc.atomic_json(cc.evidence(s) / "head" / f"{arm}.json", rep)
            print(f"[conf-head] {arm}: K0 {json.dumps(k0)} ({time.time() - t0:.0f}s)", flush=True)
            continue
        b = spec_block(s, arm, info)
        head = E2.fit_e2(blocks + [b], r, lam)                     # EQ: ogni blocco 1/9, perdita riscalata
        c = H.calib_c(lambda bb, i, j: head.dist(bb.X[i], bb.X[j]), blocks)   # SOLO le 8 sorgenti
        out = cc.work(s) / "head" / f"{arm}.npz"
        diag.atomic_savez(out, alpha=head.alpha, W=head.W, r=head.r, lam=head.lam, alpha0=head.alpha0, nit=head.nit,
                          success=head.success, loss=head.loss, c=c, dpu=info["dpu"][arm],
                          checkpoint=str(info["ckpt"][arm]), host=socket.gethostname())
        rep = {"arm": arm, "K0": k0, "r": r, "lambda": lam, "alpha_over_alpha0": head.alpha / head.alpha0,
               "nit": head.nit, "success": head.success, "loss": head.loss, "c": c, "block": block_report(b),
               "sources": [x.name for x in blocks], "host": socket.gethostname(), "seconds": time.time() - t0,
               "npz": str(out), **cc.stamp()}
        cc.atomic_json(cc.evidence(s) / "head" / f"{arm}.json", rep)
        print(f"[conf-head] {arm}: success {head.success}, nit {head.nit}, c {c:.6f}, K0 {k0['pass']} "
              f"({time.time() - t0:.0f}s)", flush=True)


def load_head(path: Path) -> tuple[H.Head, float, dict]:
    """(testa, c, meta) di ``fit`` per il valutatore."""
    with np.load(path) as z:
        h = H.Head(float(z["alpha"]), np.asarray(z["W"]), int(z["r"]), float(z["lam"]), float(z["alpha0"]),
                   int(z["nit"]), bool(z["success"]), float(z["loss"]))
        meta = {"c": float(z["c"]), "dpu": float(z["dpu"]), "checkpoint": str(z["checkpoint"]), "host": str(z["host"]),
                "success": bool(z["success"]), "nit": int(z["nit"])}
    return h, meta["c"], meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("block", "fit"))
    p.add_argument("--spec", type=Path, required=True)
    a = p.parse_args()
    step_block(a.spec) if a.step == "block" else step_fit(a.spec)


if __name__ == "__main__":
    main()
