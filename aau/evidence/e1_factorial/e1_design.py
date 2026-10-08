#!/usr/bin/env python3
"""E1: il disegno delle celle, calcolato PRIMA dei numeri con le funzioni del trainer.

    aau/run.sh aau/evidence/e1_factorial/e1_design.py        (nel container: importa train_steps)

Per ogni cella ripete la logica di v2_work/fastio/train_steps.py con le SUE funzioni, importate e non
riscritte: ``partition_blocks`` (blocchi stratificati, BFM residente in tutti, block_seed 1234),
blocco residente all'epoca e = (e-1)*K//E con E = ceil(T/S), ``epoch_subset`` (seme 1234+31+e, quota
BFM 0.0887372, il resto agli altri domini in proporzione ai soggetti del blocco). Ne ricava:
  - se ogni epoca e' realizzabile (nessun soggetto due volte nella stessa epoca: il trainer si
    fermerebbe con "blocco troppo piccolo");
  - passi per dominio per epoca;
  - identita' VISTE entro 10.548 e 21.096 passi (epoche 36 e 72) ed esposizioni per identita';
  - memoria della cache di ogni blocco toccato, ESATTA (stessa formula del trainer, ``block_gib``:
    indice dei tar per le mesh nuove, view_bytes_all.json per le viste), e mesh dai tar del blocco
    successivo (il pre-pass le scrive su /tmp, che conta contro --mem).
Picco di RAM previsto = RSS di base + cache del blocco + /tmp del blocco successivo, con RSS di base e
byte di /tmp per mesh MISURATI sul run su scala 1060130 (``[mem]`` del train.log e mem_job.log).
E' una previsione da componenti misurate: il picco vero lo misura il cgroup di ogni run (mem_job.log).

Scrive design.json e design.md in aau/runs/evidence/e1/.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "v2_work/fastio"))
sys.path.insert(0, str(REPO / "v2_work/train_v2"))
sys.path.insert(0, str(REPO / "aau/data_scale"))

import train_steps as ts  # noqa: E402
from cache_budget import load_index, predicted_bytes  # noqa: E402

OUT = REPO / "aau/runs/evidence/e1"
SCALE_RUN = REPO / "aau/runs/data_scale_runs/scale_bfm_ict_gnm_s1234_nocanon_noaug_20261007_1411"
S, B, SEED, BFM_SHARE = 293, 5, 1234, 0.0887372
CKPT_EPOCHS = (36, 72)
# cella -> (split, blocchi K, passi nominali T, epoca a cui il job si ferma). c2m: T nominale 91.709
# (E = 313) perche' (e-1)*40//313 cambia blocco alle STESSE epoche di (e-1)*46//360 di c3m fino
# all'epoca 72 e oltre; il job si ferma dopo il checkpoint dell'epoca 72 (21.096 passi).
CELLS = {
    "c3m": (REPO / "aau/data_scale/split_scale_all.json", 46, 105480, 72),
    "c2m": (HERE / "split_c2m.json", 40, 91709, 72),
    "c2f": (HERE / "split_c2f.json", 4, 21096, 72),
    "c3f": (HERE / "split_c3f.json", 4, 21096, 72),
    "g1": (HERE / "split_g1.json", 6, 21096, 72),
}


def dom(s: str) -> str:
    return ts.domain_of_name(s + "_GTready_x.npz")


def measured_scale_run() -> dict:
    """RSS senza cache (dopo il rilascio del blocco 0 al cambio 0->1) e picco di /tmp (shmem) del run su scala."""
    log = (SCALE_RUN / "train.log").read_text()
    m = re.search(r"\[mem\] cambio 0->1: blocco 0 liberato.*?RSS processo ([0-9.]+) GiB", log)
    shm = [float(x) for x in re.findall(r"shmem=([0-9.]+)", (SCALE_RUN / "mem_job.log").read_text())]
    nr = [float(r) + float(s) for r, s in re.findall(r"rss=([0-9.]+) shmem=([0-9.]+)", (SCALE_RUN / "mem_job.log").read_text())]
    return {"base_rss_gib": float(m.group(1)), "max_shmem_gib": max(shm), "max_rss_plus_shmem_gib": max(nr)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = load_index(REPO / "datasets/SCALE_ALL/shards/index.npz")
    tar_bytes: dict[str, list[int]] = {}
    for i, n in enumerate(idx["names"]):
        sid = str(n).split("_GTready_")[0]
        tar_bytes.setdefault(sid, []).append(predicted_bytes(int(idx["n"][i]), int(idx["m"][i]), int(idx["E"][i])))
    view_bytes: dict[str, int] = {}
    for n, b in json.loads((REPO / "aau/data_scale/view_bytes_all.json").read_text()).items():
        sid = n.split("_GTready_")[0]
        view_bytes[sid] = view_bytes.get(sid, 0) + int(b)

    def block_mem(block: list[str]) -> dict:
        tb = sum(sum(tar_bytes.get(s, [])) for s in block)
        vb = sum(view_bytes.get(s, 0) for s in block)
        miss = [s for s in block if s not in tar_bytes and s not in view_bytes]
        if miss:
            raise SystemExit(f"{len(miss)} soggetti senza byte noti (primo {miss[0]})")
        return {"subjects": len(block), "cache_gib": (tb + vb) / 2 ** 30, "tar_cache_gib": tb / 2 ** 30,
                "tar_meshes": sum(len(tar_bytes.get(s, [])) for s in block)}

    meas = measured_scale_run()
    report = {"S": S, "batch": B, "seed": SEED, "bfm_share": BFM_SHARE, "measured_scale_run": meas, "cells": {}}
    for cell, (split_path, K, T, stop) in CELLS.items():
        split = json.loads(Path(split_path).read_text())
        train = sorted(split["train"])
        spec = {"pin_domains": ["bfm"], "domain_step_share": {"bfm": BFM_SHARE}, "stratify_blocks": True,
                "n_blocks": K, "block_seed": SEED}
        blocks = ts.partition_blocks(train, spec)
        E = math.ceil(T / S)
        seen, expo, steps_dom, block_of, ok = {}, Counter(), Counter(), {}, True
        seen_at = {}
        for e in range(1, stop + 1):
            k = (e - 1) * K // E
            block_of[e] = k
            try:
                sub = ts.epoch_subset(blocks[k], min(S, T - (e - 1) * S), B, SEED + 31 + e, True,
                                      spec["domain_step_share"])
            except SystemExit as exc:
                ok = False
                report.setdefault("errors", []).append(f"{cell} epoca {e}: {exc}")
                break
            for s in sub:
                seen[s] = seen.get(s, 0) + 1
            expo.update(sub)
            if e == 1:
                steps_dom = Counter({d: n // B for d, n in Counter(dom(s) for s in sub).items()})
            if e in CKPT_EPOCHS:
                c = Counter(dom(s) for s in seen)
                ex = np.asarray(list(seen.values()))
                seen_at[e] = {"steps": e * S, "blocks_touched": sorted(set(block_of.values())),
                              "seen": dict(sorted(c.items())), "seen_total": len(seen),
                              "exposures_median": float(np.median(ex)), "exposures_mean": float(ex.mean()),
                              "seen_non_bfm": int(sum(v for d, v in c.items() if d != "bfm"))}
        touched = sorted(set(block_of.values()))
        nxt = [k + 1 for k in touched if k + 1 < K]
        mem = {k: block_mem(blocks[k]) for k in sorted(set(touched) | set(nxt))}
        # picco previsto: cache del blocco k (o k+1 mentre si carica) + /tmp del blocco k+1
        shm_per_mesh = None
        report["cells"][cell] = {
            "split": str(Path(split_path).relative_to(REPO)), "n_blocks": K, "T_nominal": T, "E_nominal": E,
            "stop_epoch": stop, "train_counts": dict(sorted(Counter(dom(s) for s in train).items())),
            "block_sizes": [len(b) for b in blocks][:min(K, 12)], "block_domains_first": dict(Counter(dom(s) for s in blocks[0])),
            "feasible": ok, "steps_per_epoch_by_domain": dict(sorted(steps_dom.items())),
            "block_change_epochs": [e for e in range(2, stop + 1) if block_of[e] != block_of[e - 1]],
            "at": seen_at, "blocks_mem": {str(k): v for k, v in mem.items()}, "next_blocks": nxt, "_shm": shm_per_mesh}

    # /tmp per mesh: picco di shmem del run su scala diviso per le mesh dai tar del blocco piu' PICCOLO fra
    # quelli preparati nel periodo del log (blocchi 1-20, gia' passati): limite superiore per mesh
    c3m_blocks = ts.partition_blocks(sorted(json.loads(CELLS["c3m"][0].read_text())["train"]),
                                     {"pin_domains": ["bfm"], "stratify_blocks": True, "n_blocks": 46, "block_seed": SEED})
    n_min = min(block_mem(c3m_blocks[k])["tar_meshes"] for k in range(1, 21))
    mib_per_mesh = meas["max_shmem_gib"] * 1024 / n_min
    report["tmp_mib_per_tar_mesh"] = mib_per_mesh
    for cell, r in report["cells"].items():
        r.pop("_shm")
        peaks = []
        for k in map(int, r["blocks_mem"]):
            if str(k + 1) in r["blocks_mem"]:
                tmp = r["blocks_mem"][str(k + 1)]["tar_meshes"] * mib_per_mesh / 1024
                peaks.append(max(r["blocks_mem"][str(k)]["cache_gib"], r["blocks_mem"][str(k + 1)]["cache_gib"]) + tmp)
            else:
                peaks.append(r["blocks_mem"][str(k)]["cache_gib"])
        r["predicted_peak_gib"] = meas["base_rss_gib"] + max(peaks)
        r["cache_max_gib"] = max(v["cache_gib"] for v in r["blocks_mem"].values())
    (OUT / "design.json").write_text(json.dumps(report, indent=1) + "\n")

    L = ["# E1: disegno delle celle (calcolato prima dei numeri)\n",
         f"Generato da `aau/evidence/e1_factorial/e1_design.py` con le funzioni del trainer (`partition_blocks`, "
         f"`epoch_subset`) e la formula della cache del trainer. S = {S} passi per epoca, batch {B}, seme {SEED}, "
         f"quota BFM {BFM_SHARE}. Checkpoint alle epoche {CKPT_EPOCHS} = passi {CKPT_EPOCHS[0] * S} e {CKPT_EPOCHS[1] * S}.\n",
         "| cella | training (per dominio) | blocchi K | E nominale | passi/epoca per dominio | cambi di blocco (epoca) | realizzabile |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell, r in report["cells"].items():
        L.append(f"| {cell} | {r['train_counts']} | {r['n_blocks']} | {r['E_nominal']} | {r['steps_per_epoch_by_domain']} | "
                 f"{r['block_change_epochs']} | {r['feasible']} |")
    L += ["\n## Identita' VISTE entro il checkpoint\n",
          "Un'identita' e' vista se compare in almeno un'epoca fino a quel checkpoint; esposizioni = epoche in cui compare "
          "(una per epoca, fino a 6 mesh ciascuna).\n",
          "| cella | passi | blocchi toccati | viste per dominio | viste totali | non-BFM | esposizioni mediana / media |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for cell, r in report["cells"].items():
        for e, a in r["at"].items():
            L.append(f"| {cell} | {a['steps']} | {len(a['blocks_touched'])} di {r['n_blocks']} | {a['seen']} | {a['seen_total']} | "
                     f"{a['seen_non_bfm']} | {a['exposures_median']:.0f} / {a['exposures_mean']:.1f} |")
    L += ["\n## Memoria\n",
          f"Cache esatta per blocco (formula del trainer) e picco previsto = RSS di base {meas['base_rss_gib']:.1f} GiB "
          f"(run su scala, dopo il rilascio del blocco al cambio 0->1: GT 65.600^2 float64 piu' il resto) + cache + /tmp "
          f"del blocco successivo a {mib_per_mesh:.2f} MiB per mesh dai tar (picco di shmem del run su scala "
          f"{meas['max_shmem_gib']:.1f} GiB / {n_min} mesh del blocco piu' piccolo fra 1 e 20). Riferimento misurato: "
          f"picco rss+shmem del run su scala {meas['max_rss_plus_shmem_gib']:.1f} GiB.\n",
          "| cella | cache max per blocco (GiB) | mesh dai tar per blocco (max) | picco previsto rss+shmem (GiB) |",
          "| --- | --- | --- | --- |"]
    for cell, r in report["cells"].items():
        L.append(f"| {cell} | {r['cache_max_gib']:.1f} | {max(v['tar_meshes'] for v in r['blocks_mem'].values())} | "
                 f"{r['predicted_peak_gib']:.1f} |")
    if report.get("errors"):
        L += ["\n## ERRORI\n", *[f"- {e}" for e in report["errors"]]]
    (OUT / "design.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
