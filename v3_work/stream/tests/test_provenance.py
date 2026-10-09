#!/usr/bin/env python
"""Correttezza (d): provenienza, rigenerazione dal seme, registro delle viste usate, filtro delle fonti.

    aau/run.sh v3_work/stream/tests/test_provenance.py --out aau/runs/evidence/stream/provenance_check.json

1. ``producer.py --provenance --canonical-gt --expr-frac 0.5 --sources massive --mm-aug ...`` per ``--seconds`` su
   un anello temporaneo (processi veri, come nel run): gruppi puri, ibridi, trasferimenti d'espressione, bump.
2. Ogni shard (fino a ``--max-shards``) rigenerato da regen.check_shard: tipo del gruppo, facce, etichette,
   espressioni identiche, vertici serviti entro 1e-6, coefficienti (o persona FaMoS) identici.
3. Licenze: ogni gruppo eredita la piu' restrittiva fra le sue fonti (sources.LICENSES); nessun FLAME 2020.
4. Consumatore con ``domains`` = open_core e ``log_views``: solo gruppi con TUTTE le fonti in gnm, ict, flame2023;
   il registro ha una riga per vista usata, con seme e licenza; views_log.summarize lo legge.
5. ``--mm-aug`` con un tipo sconosciuto ferma il produttore.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--n-proc", type=int, default=6)
    ap.add_argument("--max-shards", type=int, default=40)
    ap.add_argument("--mm-aug", default="hybrid=0.3,expr_transfer=0.15,rbf=0.15")
    a = ap.parse_args()
    import sources as S
    import regen
    import views_log
    from consumer import StreamConsumer, StreamGT
    from ring import Ring
    from sampler_v3 import DrawCfg
    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        ring_dir = Path(tmp) / "ring"
        cmd = [sys.executable, str(STREAM / "producer.py"), "--ring", str(ring_dir), "--ring-gb", "8", "--n-proc",
               str(a.n_proc), "--k-eig", "64", "--sources", "massive", "--provenance", "--canonical-gt",
               "--expr-frac", "0.5", "--duration", str(a.seconds), "--stats-every", "30", "--mm-aug", a.mm_aug,
               "--summary", str(Path(tmp) / "producers.json")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"producer: {r.stderr[-2000:]}")
        prod = json.loads((Path(tmp) / "producers.json").read_text())
        out["producer"] = {k: prod[k] for k in ("views", "views_per_s_steady", "views_by_domain", "expr_fraction",
                                                "failures", "bytes_per_view", "views_by_origin", "mm_aug")}
        print(f"[prov] produttori: {out['producer']}", flush=True)
        lst = Ring(ring_dir).listing()
        rows, srcs, lic_ok, doms = [], {}, True, set()
        for seq, _, p in lst[:a.max_shards]:
            res = regen.check_shard(p, srcs=srcs)
            rows += res
            from ring import ShardReader
            for hg in ShardReader(p).groups:
                doms |= set(hg["prov"]["sources"])
                lic_ok &= tuple([hg["prov"]["license"], hg["prov"]["license_rank"]]) == \
                    S.inherited_license(hg["prov"]["sources"])
        out["regen"] = {"groups": len(rows), "views": sum(len(x["views"]) for x in rows),
                        "groups_by_origin": {o: sum(x["origin"] == o for x in rows) for o in {x["origin"] for x in rows}},
                        "pass": all(x["pass"] for x in rows), "domains": sorted(doms),
                        "max_verts_abs": max(v["verts_max_abs"] for x in rows for v in x["views"]),
                        "failed": [x for x in rows if not x["pass"]][:3]}
        out["licenses"] = {"pass": bool(lic_ok and "flame2020" not in doms), "table": S.LICENSES}
        print(f"[prov] rigenerazione: {out['regen']}", flush=True)
        # consumatore: nucleo aperto + registro
        uni = S.Unified()
        c = StreamConsumer(ring_dir, reuse=2, prefetch=0, gt=StreamGT(uni.A, 1.0), gt_kind="sr",
                           domains=S.parse_sources("open_core"), log_views=True, wait_s=60)
        drawcfg = DrawCfg(p_noise=0.0, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation"],
                          noise_mode_probs=[1.0], max_meshes=4)
        rng = np.random.default_rng(0)
        for _ in range(6):
            c.plan(4, 4, drawcfg, rng)
        pool_doms = sorted({d for (seq, g) in c.pool for d in c.readers[seq].groups[g]["prov"]["sources"]})
        n_unique = c.c["unique_views_used"]
        run_dir = Path(tmp) / "run"
        n_rows = c.flush_log(run_dir / "views_used" / "rank000_e00001_0.npz")
        summ = views_log.summarize(views_log.load(run_dir))
        out["consumer"] = {"pool_domains": pool_doms, "unique_views_used": n_unique, "log_rows": n_rows,
                           "summary": summ,
                           "pass": bool(set(pool_doms) <= {"gnm", "ict", "flame2023"} and n_rows == n_unique
                                        and summ["with_seed"] == 1.0 and summ["share_redistributable_views"] == 1.0)}
        print(f"[prov] consumatore open_core: {out['consumer']}", flush=True)
        r = subprocess.run([sys.executable, str(STREAM / "producer.py"), "--ring", str(Path(tmp) / "x"), "--mm-aug",
                            "chimera=0.2"], capture_output=True, text=True)
        out["mm_aug"] = {"returncode": r.returncode, "message": (r.stderr.strip().splitlines() or [""])[-1],
                         "pass": r.returncode != 0}
    out["pass"] = bool(out["regen"]["pass"] and out["licenses"]["pass"] and out["consumer"]["pass"]
                       and out["mm_aug"]["pass"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"[prov] ESITO {'PASSA' if out['pass'] else 'FALLISCE'} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
