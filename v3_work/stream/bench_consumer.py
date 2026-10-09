#!/usr/bin/env python
"""Capacita' del consumatore senza GPU: piani e campioni serviti al secondo da un rank, sull'anello vivo.

    aau/run.sh v3_work/stream/bench_consumer.py --ring /tmp/$SLURM_JOB_ID/stream --seconds 60 \
        --batch 16 --views 4 --out consumer.json

Fa quello che fa il trainer per ogni passo meno la GPU: ``plan`` (scelta dei gruppi, GT, prenotazione) e la
lettura di tutte le entries (materialize in thread: copia dalla mappa, autovettori fp16 -> fp32, sparsi,
rotazione e scala). mesh/s qui e' il tetto del lato dati per rank: se supera di molto le mesh/s che la GPU
consuma, il collo di bottiglia non e' lo stream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "trainer"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--reuse", type=float, default=4.0)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--fast-data", action="store_true", help="sparsi senza riordino, come il trainer con --fast-data")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    import common  # noqa: F401
    import data_v3 as dv
    import sources as S
    dv.FAST["on"] = bool(a.fast_data)
    from consumer import StreamConsumer, StreamGT
    from sampler_v3 import DrawCfg
    uni = S.Unified()
    c = StreamConsumer(a.ring, reuse=a.reuse, prefetch=a.prefetch, gt=StreamGT(uni.A, S.gt_scale_mm()),
                       min_groups=a.batch)
    cfg = DrawCfg(p_noise=0.6, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation", "rotation", "jitter"],
                  noise_mode_probs=[4 / 7, 2 / 7, 1 / 7], max_meshes=a.views)
    rng = np.random.default_rng(0)
    c.plan(a.batch, a.views, cfg, rng)        # attesa iniziale e import fuori dal cronometro
    c.c.update(uses=0, unique_views_used=0, serve_s=0.0, plan_s=0.0, plans=0)
    t0 = time.time()
    n_plans = n_mesh = n_vert = 0
    nxt = c.plan(a.batch, a.views, cfg, rng)
    while time.time() - t0 < a.seconds:
        cur, nxt = nxt, c.plan(a.batch, a.views, cfg, rng)
        for e in cur.entries:
            s = c[e[1]]
            n_vert += int(s["verts"].shape[0])
            n_mesh += 1
        n_plans += 1
    dt = time.time() - t0
    st = c.stats()
    out = {"seconds": dt, "fast_data": bool(a.fast_data), "cpus": len(os.sched_getaffinity(0)), "plans": n_plans,
           "plans_per_s": n_plans / dt, "meshes_per_s": n_mesh / dt,
           "mean_verts": n_vert / max(n_mesh, 1), "batch": a.batch, "views": a.views, "prefetch": a.prefetch,
           "serve_s_per_plan": st["serve_s"] / max(n_plans, 1), "plan_s_per_plan": st["plan_s"] / max(n_plans, 1),
           "reuse_factor": st["reuse_factor"], "over_reuse_groups": st["over_reuse_groups"],
           "pool_groups": st["pool_groups"], "pool_shards": st["pool_shards"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("[bench-consumer] " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
