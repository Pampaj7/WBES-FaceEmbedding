#!/usr/bin/env python
"""Quanto si legge dall'anello condiviso (CephFS) con la copia locale dei consumatori, senza GPU.

    aau/run.sh v3_work/stream/bench_mirror.py --ring datasets/STREAM/shared_ring --ranks 0-5 --world 12 \
        --threads 2 --seconds 600 --out mirror.json

Un StreamConsumer per rank simulato (``--ranks`` del nodo, ``--world`` rank in tutto) con ``extra_mirror`` su /tmp
e ``--threads`` copie parallele per rank, come i rank di un nodo del training: a regime le copie seguono gli shard
nuovi del rank. Misura MB/s e shard/s copiati in totale, shard persi (usciti dall'anello prima della copia) e, dagli
header, le viste/s che arrivano ai rank. Le copie si cancellano alla fine.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "trainer"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--ranks", default="0-5")
    ap.add_argument("--world", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from consumer import StreamConsumer
    from ring import Ring, ShardReader
    lo, _, hi = a.ranks.partition("-")
    ranks = list(range(int(lo), int(hi or lo) + 1))
    tmp = Path(tempfile.mkdtemp(prefix="mirror_bench_"))
    try:
        empty = tmp / "local"
        Ring(empty)
        cs = [StreamConsumer(empty, rank=r, world=a.world, extra_rings=[(a.ring, r, a.world)],
                             extra_mirror=tmp / "m", mirror_threads=a.threads, extra_refresh_s=5) for r in ranks]
        t0 = time.time()
        warm = None
        while time.time() - t0 < a.seconds:
            time.sleep(10)
            if warm is None and time.time() - t0 > 60:       # il primo minuto copia l'arretrato: fuori dal regime
                warm = (time.time(), sum(c.mstat["bytes"] for c in cs), sum(c.mstat["copied"] for c in cs))
        t1 = time.time()
        b = sum(c.mstat["bytes"] for c in cs)
        n = sum(c.mstat["copied"] for c in cs)
        sample = []
        for c in cs:
            for p in list((tmp / "m").glob(f"rank{c.rank:03d}_x1/shards/*.shard"))[:4]:
                try:
                    sample.append(sum(len(g["views"]) for g in ShardReader(p).groups))
                except (OSError, ValueError):
                    continue
        per_shard = sum(sample) / len(sample) if sample else 16.0
        out = {"ranks": ranks, "world": a.world, "threads": a.threads, "seconds": t1 - t0,
               "copied_shards": n, "copied_gib": b / 2 ** 30, "missed": sum(c.mstat["missed"] for c in cs),
               "MB_per_s_all": b / 1e6 / (t1 - t0)}
        if warm is not None:
            dt = t1 - warm[0]
            out.update(MB_per_s_steady=(b - warm[1]) / 1e6 / dt, shards_per_s_steady=(n - warm[2]) / dt,
                       views_per_s_steady_est=(n - warm[2]) / dt * per_shard, views_per_shard_sample=per_shard)
        print("[mirror-bench] " + json.dumps(out), flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
