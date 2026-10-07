#!/usr/bin/env python3
"""Memoria al cambio di blocco: la cache vecchia torna davvero al sistema? (OOM del run 1056832)

    aau/run.sh aau/data_scale/switch_memtest.py --mode pop|gc|trim --n-subj 1250

Riproduce il cambio di blocco di train_steps.py senza GPU: carica il blocco A con la stessa
``fast_data.CachedDataset`` (16 thread, cache non pinned), lo libera, carica il blocco B, e misura
l'RSS del processo (VmRSS, /proc/self/status) dopo ogni fase. Blocchi: viste ICT-5000 con operatori
(datasets/ICT/train_ready/npz_withops), soggetti 0..n-1 per A e n..2n-1 per B, 6 mesh ciascuno.
  pop   come il codice del run 1056832: la tupla (cache, indici) esce dalla lista, nient'altro
  gc    piu' ``del`` esplicito e ``gc.collect()``
  trim  piu' ``malloc_trim(0)`` di glibc
L'ambiente (per esempio ``MALLOC_MMAP_THRESHOLD_``) e' riportato nell'output.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "v2_work/fastio"))
import fast_data as fd  # noqa: E402

VIEW = REPO_ROOT / "datasets/ICT/train_ready/npz_withops"


def rss_gib() -> float:
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 2 ** 20
    return float("nan")


def make_block(tmp: Path, subj: range) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    for s in subj:
        sid = f"id{10000 + s:05d}"
        for t in ("original", "remesh", "crop", "noisy", "down8k", "up60k"):
            n = f"{sid}_GTready_{t}.npz"
            (tmp / n).symlink_to((VIEW / n).resolve())
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pop", "gc", "trim"), required=True)
    ap.add_argument("--n-subj", type=int, default=1250)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    a = ap.parse_args()
    orig = fd._to_residency
    fd._to_residency = lambda s, d, pin: orig(s, d, False)          # come train_steps: non pinned
    A = make_block(a.work / "A", range(0, a.n_subj))
    B = make_block(a.work / "B", range(a.n_subj, 2 * a.n_subj))
    rep = {"mode": a.mode, "n_subj": a.n_subj,
           "env": {k: v for k, v in os.environ.items() if k.startswith("MALLOC")},
           "rss_gib": {"start": rss_gib()}}
    parts = []
    t0 = time.time()
    parts.append(fd.CachedDataset(A, workers=16, residency="ram", max_gb=float("inf"), verbose=False))
    rep["rss_gib"]["A_loaded"] = rss_gib()
    parts.pop()
    rep["rss_gib"]["A_popped"] = rss_gib()
    if a.mode in ("gc", "trim"):
        gc.collect()
        rep["rss_gib"]["A_gc"] = rss_gib()
    if a.mode == "trim":
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        rep["rss_gib"]["A_trim"] = rss_gib()
    parts.append(fd.CachedDataset(B, workers=16, residency="ram", max_gb=float("inf"), verbose=False))
    rep["rss_gib"]["B_loaded"] = rss_gib()
    rep["seconds"] = time.time() - t0
    a_size = rep["rss_gib"]["A_loaded"] - rep["rss_gib"]["start"]
    rep["block_gib_by_rss"] = a_size
    rep["peak_ratio_B_over_A"] = (rep["rss_gib"]["B_loaded"] - rep["rss_gib"]["start"]) / a_size
    print(json.dumps(rep, indent=1), flush=True)
    a.out_json.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
