#!/usr/bin/env python3
"""E9 (c): all_reduce NCCL fra due GPU L40S, sullo stesso nodo o su due nodi.

    srun aau/run.sh aau/evidence/e9_bench/nccl_bench.py --out-json aau/runs/evidence/e9/nccl_<tag>.json

Un processo per GPU (rank = SLURM_PROCID, GPU = SLURM_LOCALID), rendezvous TCP sul primo nodo del job.
Messaggi float32 da 8 B (latenza a vuoto), 10, 50 e 100 MB (10^6 byte). Per taglia: 5 giri di
riscaldamento, poi n giri cronometrati a muro con sincronizzazione CUDA; tempo per all_reduce = media
dei giri, il massimo fra i rank. algbw = byte / tempo; busbw = algbw * 2(n-1)/n, la banda che il
collettivo chiede a ogni collegamento (con 2 rank coincide con algbw). L'interfaccia scelta da NCCL
si legge nel log con NCCL_DEBUG=INFO (lo sbatch lo imposta).

``--backend gloo``: stesso protocollo su tensori CPU, senza GPU. E' il ripiego per misurare la rete
fra due nodi L40S quando non ci sono GPU libere su due nodi: gloo usa TCP sull'interfaccia
dell'hostname (bond0), quindi misura il trasporto Socket, non RoCE.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist

SIZES = {"8B": 8, "10MB": 10_000_000, "50MB": 50_000_000, "100MB": 100_000_000}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    a = ap.parse_args()
    gpu = a.backend == "nccl"
    rank, world = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    local = int(os.environ.get("SLURM_LOCALID", 0))
    if gpu:
        torch.cuda.set_device(local % torch.cuda.device_count())
    sync = torch.cuda.synchronize if gpu else (lambda: None)
    dist.init_process_group(a.backend, init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                            rank=rank, world_size=world)
    dev = torch.device("cuda" if gpu else "cpu")
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())

    res = {}
    for tag, nbytes in SIZES.items():
        x = torch.ones(max(1, nbytes // 4), device=dev)
        n = a.iters * (10 if nbytes < 1_000_000 else 1)
        for _ in range(5):
            dist.all_reduce(x)
        sync()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n):
            dist.all_reduce(x)
        sync()
        t = torch.tensor([(time.perf_counter() - t0) / n], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        t = float(t.item())
        algbw = x.numel() * 4 / t / 1e9
        res[tag] = {"bytes": x.numel() * 4, "iters": n, "time_ms": t * 1e3, "algbw_GBps": algbw,
                    "busbw_GBps": algbw * 2 * (world - 1) / world}
        if rank == 0:
            print(f"[{a.backend}] {tag:>6}: {t * 1e3:9.3f} ms  algbw {algbw:7.2f} GB/s", flush=True)
        del x
    if rank == 0:
        rep = {"backend": a.backend, "world": world, "hosts": hosts,
               "nccl": ".".join(map(str, torch.cuda.nccl.version())) if gpu else None,
               "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if gpu else None,
               "env": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}, "results": res}
        a.out_json.parent.mkdir(parents=True, exist_ok=True)
        a.out_json.write_text(json.dumps(rep, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
