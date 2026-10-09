#!/usr/bin/env python3
"""Banda di CephFS per l'anello condiviso: scrittura e lettura di file grandi come gli shard, con P processi.

    python3 v3_work/stream/bench_ceph.py write --dir D --files 64 --mb 64 --procs 8 --out w.json   (solo stdlib)
    python3 v3_work/stream/bench_ceph.py read  --dir D --procs 1,8,16 --out r.json                  (altro nodo)

write: P processi scrivono ``--files`` file da ``--mb`` MB (byte casuali, tmp + rename come ring.Ring.write).
read: per ogni P, P processi leggono i file a pezzi da ``--chunk-mb`` (come il consumatore copia le viste dalla
mappa), ognuno la sua parte; la lettura va fatta da un nodo che non li ha scritti (page cache) e una sola volta per
file (al secondo P i file sono gia' in cache: si legge un sottoinsieme diverso per ogni P).
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _write(task) -> float:
    path, mb = task
    t0 = time.time()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for _ in range(mb):
            fh.write(os.urandom(1 << 20))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return time.time() - t0


def _read(task) -> int:
    path, chunk = task
    n = 0
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for off in range(0, len(mm), chunk):
            n += len(bytes(mm[off:off + chunk]))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("write", "read"))
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--files", type=int, default=64)
    ap.add_argument("--mb", type=int, default=64)
    ap.add_argument("--procs", default="8")
    ap.add_argument("--chunk-mb", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"mode": a.mode, "host": os.uname().nodename, "dir": str(a.dir), "rows": []}
    if a.mode == "write":
        a.dir.mkdir(parents=True, exist_ok=True)
        P = int(a.procs.split(",")[0])
        t0 = time.time()
        with ProcessPoolExecutor(P) as ex:
            list(ex.map(_write, [(a.dir / f"f{i:05d}.bin", a.mb) for i in range(a.files)]))
        dt = time.time() - t0
        out["rows"].append({"procs": P, "files": a.files, "mb_each": a.mb, "seconds": dt,
                            "MB_per_s": a.files * a.mb / dt})
    else:
        files = sorted(a.dir.glob("f*.bin"))
        ps = [int(x) for x in a.procs.split(",")]
        per = len(files) // len(ps)
        for k, P in enumerate(ps):
            sub = files[k * per:(k + 1) * per]
            t0 = time.time()
            with ProcessPoolExecutor(P) as ex:
                n = sum(ex.map(_read, [(f, int(a.chunk_mb * (1 << 20))) for f in sub]))
            dt = time.time() - t0
            out["rows"].append({"procs": P, "files": len(sub), "MB": n / 2 ** 20, "seconds": dt,
                                "MB_per_s": n / 2 ** 20 / dt})
    print(json.dumps(out), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
