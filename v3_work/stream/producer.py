#!/usr/bin/env python3
"""Produttori CPU della pipeline P1: viste fresche, operatori e shard nel buffer ad anello.

    aau/run.sh v3_work/stream/producer.py --ring /tmp/$SLURM_JOB_ID/stream --ring-gb 40 --n-proc 96 \
        [--k-eig 64|128] [--alpha 0] [--domains bfm2019,ict,gnm,flame2020,famos] [--duration 0]

Ogni processo (fork del padre, che carica modelli e basi una volta: le pagine restano condivise) ripete:
  1. dominio d ~ p_d proporzionale a n_d^alpha (``sources.domain_probs``; alpha 0 = uniforme);
  2. un'identita' fresca (3DMM: coefficienti a code larghe; FaMoS: una persona TRAIN) e il suo s_i dalla
     forma neutra (GT unificata);
  3. ``--views`` viste: espressione con probabilita' ``--p-expr`` (prior del modello; FaMoS: un fotogramma
     registrato), discretizzazione estratta coi pesi ``--label-weights`` (up60k a bassa frequenza),
     operatori con ``--k-eig`` autovettori (views.make_view);
  4. ``--groups-per-shard`` identita' in uno shard, scritto nell'anello (ring.Ring, FIFO a ``--ring-gb``).
Tutti i numeri casuali di un processo vengono da SeedSequence([seme, processo]).

Statistiche: ``<ring>/stats/w<NNN>.json`` per processo (viste, gruppi, byte, secondi per fase, fallimenti) e,
dal padre, una riga ogni ``--stats-every`` s con viste/s nell'intervallo, shard e GiB nell'anello;
``--summary`` scrive il riepilogo finale in JSON (regime = dopo il primo shard di ogni processo).

Thread BLAS a 1 per processo (imposti qui, prima di numpy): la parallelizzazione e' per processo.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import sources as S  # noqa: E402
from ring import Ring  # noqa: E402


def parse_weights(txt: str, keys) -> dict:
    out = {}
    for item in txt.split(","):
        k, v = item.split("=")
        if k not in keys:
            raise SystemExit(f"{k!r} non in {keys}")
        out[k] = float(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    import views as VW
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ring", required=True, help="directory dell'anello (su /tmp: conta contro --mem)")
    p.add_argument("--ring-gb", type=float, default=32.0, help="budget dell'anello in GiB (FIFO)")
    p.add_argument("--n-proc", type=int, default=8)
    p.add_argument("--k-eig", type=int, default=128, choices=[64, 128])
    p.add_argument("--evecs-dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--domains", default=",".join(S.DOMAINS))
    p.add_argument("--alpha", type=float, default=0.0, help="p_d ~ n_d^alpha (0 = uniforme fra domini)")
    p.add_argument("--domain-sizes", default="", help="n_d, es. 'famos=80,ict=100' (default: modi / persone)")
    p.add_argument("--views", type=int, default=4, help="viste per identita'")
    p.add_argument("--groups-per-shard", type=int, default=2)
    p.add_argument("--p-expr", type=float, default=0.6)
    p.add_argument("--label-weights", default=",".join(f"{k}={v:g}" for k, v in VW.LABEL_WEIGHTS.items()))
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--duration", type=float, default=0.0, help="secondi (0 = finche' non viene fermato)")
    p.add_argument("--stats-every", type=float, default=30.0)
    p.add_argument("--summary", default="", help="JSON del riepilogo finale")
    p.add_argument("--cpus", default="", help="CPU logiche dei produttori, es. '8-31,40-63' (vuoto = tutte): le altre "
                                              "restano al trainer (la contesa rallenta il passo, vedi README)")
    return p


def cpu_list(txt: str) -> set:
    out: set = set()
    for item in txt.split(","):
        a, _, b = item.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


# --- un gruppo = un'identita' con le sue viste -----------------------------------------------------------

def make_group(src, domain: str, rng: np.random.Generator, cfg, uni: S.Unified, key: str, st: dict) -> dict:
    import views as VW
    t0 = time.perf_counter()
    ident = src.identity(rng)
    s = uni.svec(src.neutral_points(ident))
    if src.kind == "famos":
        key = f"famos/{ident}"
    st["t_ident"] += time.perf_counter() - t0
    out = []
    for _ in range(cfg.views):
        t1 = time.perf_counter()
        V, F, tag = src.view_mesh(ident, rng, bool(rng.random() < cfg.p_expr))
        label = cfg.labels[int(rng.choice(len(cfg.labels), p=cfg.label_p))]
        st["t_mesh"] += time.perf_counter() - t1
        try:
            arr, meta = VW.make_view(V, F, label, cfg.k_eig, cfg.evecs_dtype, int(rng.integers(2 ** 31)))
        except Exception as exc:  # noqa: BLE001  (una vista rotta non ferma il produttore)
            st["failures"] += 1
            st["last_failure"] = f"{domain}/{label}: {type(exc).__name__}: {exc}"
            continue
        meta["expr"] = tag
        for ph in ("gen", "ops", "pack"):
            st[f"t_{ph}"] += meta.pop(f"t_{ph}")
        st["views"] += 1
        st["by_label"][label] = st["by_label"].get(label, 0) + 1
        st["verts"] += meta["n"]
        out.append((arr, meta))
    st["by_domain"][domain] = st["by_domain"].get(domain, 0) + len(out)
    return {"key": key, "domain": domain, "s": s, "views": out}


def worker(wid: int, cfg, srcs: dict, uni: S.Unified, stop_at: float) -> None:
    import torch
    import views as VW
    torch.set_num_threads(1)
    if cfg.cpus:
        os.sched_setaffinity(0, cpu_list(cfg.cpus))
    VW.install_grad_vec()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, wid]))
    ring = Ring(cfg.ring, int(cfg.ring_gb * 2 ** 30))
    doms = list(cfg.probs)
    p = np.asarray([cfg.probs[d] for d in doms])
    st = {"wid": wid, "pid": os.getpid(), "t_start": time.time(), "views": 0, "groups": 0, "shards": 0, "bytes": 0,
          "verts": 0, "failures": 0, "evicted": 0, "by_domain": {}, "by_label": {}, "t_first_shard": None,
          "views_at_first_shard": 0, **{f"t_{k}": 0.0 for k in ("ident", "mesh", "gen", "ops", "pack", "write")}}
    stats_path = ring.root / "stats" / f"w{wid:03d}.json"
    last_dump, n = 0.0, 0
    try:
        while not stop_at or time.time() < stop_at:
            groups = []
            for _ in range(cfg.groups_per_shard):
                d = doms[int(rng.choice(len(doms), p=p))]
                g = make_group(srcs[d], d, rng, cfg, uni, f"{d}/{cfg.seed}-{wid}-{n}", st)
                n += 1
                if g["views"]:
                    groups.append(g)
            if not groups:
                continue
            t0 = time.perf_counter()
            _, nb = ring.write(groups, wid)
            st["t_write"] += time.perf_counter() - t0
            st["shards"] += 1
            st["groups"] += len(groups)
            st["bytes"] += nb
            if st["t_first_shard"] is None:
                st["t_first_shard"], st["views_at_first_shard"] = time.time(), st["views"]
            if time.time() - last_dump > cfg.stats_every / 2:
                _dump(stats_path, st)
                last_dump = time.time()
    finally:
        st["t_end"] = time.time()
        _dump(stats_path, st)


def _dump(path: Path, st: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, path)


# --- padre ---------------------------------------------------------------------------------------------

def read_stats(ring: Ring) -> list[dict]:
    out = []
    for f in sorted((ring.root / "stats").glob("w*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    return out


def summarize(ring: Ring, t0: float, cfg) -> dict:
    st = read_stats(ring)
    now = time.time()
    views = sum(s["views"] for s in st)
    steady = [s for s in st if s.get("t_first_shard")]
    # regime: viste dopo il primo shard di ogni processo, sul tempo dal primo shard alla fine (o ad adesso)
    v_ss = sum(s["views"] - s["views_at_first_shard"] for s in steady)
    t_ss = [(s.get("t_end") or now) - s["t_first_shard"] for s in steady]
    rate_ss = sum((s["views"] - s["views_at_first_shard"]) / max(t, 1e-9) for s, t in zip(steady, t_ss))
    lst = ring.listing()
    phases = {k: sum(s[f"t_{k}"] for s in st) for k in ("ident", "mesh", "gen", "ops", "pack", "write")}
    by_dom, by_lab = {}, {}
    for s in st:
        for k, v in s["by_domain"].items():
            by_dom[k] = by_dom.get(k, 0) + v
        for k, v in s["by_label"].items():
            by_lab[k] = by_lab.get(k, 0) + v
    return {"n_proc": cfg.n_proc, "k_eig": cfg.k_eig, "evecs_dtype": cfg.evecs_dtype, "alpha": cfg.alpha,
            "probs": cfg.probs, "views_per_group": cfg.views, "elapsed_s": now - t0, "views": views,
            "views_per_s_wall": views / max(now - t0, 1e-9), "views_steady": v_ss,
            "views_per_s_steady": rate_ss, "groups": sum(s["groups"] for s in st),
            "shards_written": sum(s["shards"] for s in st), "bytes_written": sum(s["bytes"] for s in st),
            "mean_verts_per_view": sum(s["verts"] for s in st) / max(views, 1),
            "bytes_per_view": sum(s["bytes"] for s in st) / max(views, 1),
            "failures": sum(s["failures"] for s in st),
            "last_failure": next((s["last_failure"] for s in st if s.get("last_failure")), ""),
            "cpu_s_per_view": {k: v / max(views, 1) for k, v in phases.items()},
            "views_by_domain": by_dom, "views_by_label": by_lab,
            "ring": {"shards": len(lst), "gib": sum(b for _, b, _ in lst) / 2 ** 30,
                     "seq_min": lst[0][0] if lst else None, "seq_max": lst[-1][0] if lst else None}}


def main() -> None:
    import views as VW
    cfg = build_parser().parse_args()
    cfg.labels = list(VW.LABELS)
    lw = parse_weights(cfg.label_weights, VW.LABELS)
    w = np.asarray([lw.get(k, 0.0) for k in cfg.labels])
    cfg.label_p = w / w.sum()
    domains = [d for d in cfg.domains.split(",") if d]
    t_load = time.time()
    uni = S.Unified()
    srcs = S.build_sources(domains, uni)
    sizes = {d: srcs[d].n_id for d in domains}
    if cfg.domain_sizes:
        sizes.update({k: int(v) for k, v in parse_weights(cfg.domain_sizes, domains).items()})
    cfg.probs = S.domain_probs(sizes, cfg.alpha)
    print(f"[producer] sorgenti in {time.time() - t_load:.0f}s: " + "; ".join(
        f"{d} n_d={sizes[d]} p={cfg.probs[d]:.3f} {srcs[d].describe()}" for d in domains), flush=True)
    ring = Ring(cfg.ring, int(cfg.ring_gb * 2 ** 30))
    for f in (ring.root / "stats").glob("w*.json"):
        f.unlink()
    t0 = time.time()
    stop_at = t0 + cfg.duration if cfg.duration > 0 else 0.0
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i, cfg, srcs, uni, stop_at), daemon=True) for i in range(cfg.n_proc)]
    for pr in procs:
        pr.start()

    def stop(*_):
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
    signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(0)))
    print(f"[producer] {cfg.n_proc} processi, k_eig={cfg.k_eig} evecs={cfg.evecs_dtype}, anello {ring.root} "
          f"({cfg.ring_gb:g} GiB), alpha={cfg.alpha} p={cfg.probs}, CPU {cfg.cpus or 'tutte'}", flush=True)
    last_v, last_t = 0, t0
    try:
        while any(pr.is_alive() for pr in procs):
            for pr in procs:
                pr.join(timeout=cfg.stats_every / len(procs))
            if time.time() - last_t >= cfg.stats_every:
                s = summarize(ring, t0, cfg)
                now = time.time()
                print(f"[producer] {now - t0:6.0f}s viste {s['views']} ({(s['views'] - last_v) / (now - last_t):.1f}/s "
                      f"nell'intervallo, regime {s['views_per_s_steady']:.1f}/s) gruppi {s['groups']} fallite "
                      f"{s['failures']} | anello {s['ring']['shards']} shard {s['ring']['gib']:.2f} GiB "
                      f"seq {s['ring']['seq_min']}-{s['ring']['seq_max']}", flush=True)
                last_v, last_t = s["views"], now
    finally:
        stop()
        s = summarize(ring, t0, cfg)
        print("[producer] riepilogo " + json.dumps(s), flush=True)
        if cfg.summary:
            Path(cfg.summary).parent.mkdir(parents=True, exist_ok=True)
            Path(cfg.summary).write_text(json.dumps(s, indent=1) + "\n")


if __name__ == "__main__":
    main()
