#!/usr/bin/env python
"""Correttezza (e): la ricetta c3m del preset validated (massive_node.sh, STREAM_RECIPE c3m) e' quella del C3M.

    aau/run.sh v3_work/stream/tests/test_recipe_c3m.py --out aau/runs/evidence/stream/recipe_c3m_check.json

1. Ricetta senza operatori (``producer.view_recipe`` su ``--groups`` gruppi per dominio, gli argomenti che passa
   massive_node.sh): viste con espressione per dominio contro --expr-frac (C3M: bfm2019 0, ict 0.2315, gnm 0.1968),
   6 discretizzazioni distinte per identita' (``--label-draw perm``).
2. Produttori veri per ``--seconds`` (``--sources validated``, niente --mm-aug): ogni gruppo puro con una sola fonte,
   discretizzazioni distinte, ricetta nello header; regen.check_shard rigenera i primi shard.
3. Consumatore con la rotazione di massive_node.sh (``rot_deg`` "0" -> 0, 0, 0) e batch a dominio singolo
   (StreamPlans come train_stream.py): 0 batch misti per template e per FONTI, rotazione 0 in ogni vista prenotata.
4. Controllo positivo del contatore per fonti: un anello con ``--mm-aug hybrid=0.5,expr_transfer=0.5`` deve dare
   batch misti per fonti (> 0) anche coi batch a dominio singolo per template.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
for _p in (STREAM, REPO / "v3_work" / "trainer"):
    sys.path.insert(0, str(_p))

EXPR_C3M = "bfm2019=0,ict=0.2315,gnm=0.1968"     # massive_node.sh, STREAM_RECIPE c3m


def producer_args(ring: Path, a, extra=()) -> list:
    return [sys.executable, str(STREAM / "producer.py"), "--ring", str(ring), "--ring-gb", "6", "--n-proc",
            str(a.n_proc), "--k-eig", "128", "--evecs-dtype", "fp32", "--sources", "validated", "--provenance",
            "--canonical-gt", "--expr-frac", EXPR_C3M, "--views", "6", "--label-draw", "perm", "--seed", "20261010",
            "--duration", str(a.seconds), "--stats-every", "30", *extra]


def consume(ring: Path, steps: int, rot: str) -> tuple:
    """StreamPlans a dominio singolo sui domini di validated, B 5 x <= 6 viste (la ricetta per rank)."""
    import sources as S
    from consumer import StreamConsumer, StreamGT, StreamPlans
    from sampler_v3 import DrawCfg
    uni = S.Unified()
    c = StreamConsumer(ring, reuse=4, prefetch=0, gt=StreamGT(uni.A, 1.0), gt_kind="sr", scale=0.0,
                       rot_deg=[float(x) for x in rot.split(",")], domains=S.parse_sources("validated"), wait_s=120)
    drawcfg = DrawCfg(p_noise=0.6, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation", "rotation", "jitter"],
                      noise_mode_probs=[4 / 7, 2 / 7, 1 / 7], max_meshes=6)
    plans = StreamPlans(c, steps, 5, 6, drawcfg, np.random.default_rng(0), 1,
                        batch_domains=sorted(S.parse_sources("validated")))
    rots, labels_per_id = [], []
    for plan in plans:
        by_id: dict = {}
        for key, h, topo, _ in plan.entries:
            seq, args = c.handles[h]
            rots.append(args[3])
            by_id.setdefault(key, []).append(topo.split("+")[0])
        labels_per_id += [len(set(v)) for v in by_id.values()]
        for _, h, _, _ in plan.entries:
            c[h]
    return {**c.stats(), "rot_deg": list(c.rot_deg)}, rots, labels_per_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", type=int, default=3000, help="gruppi per dominio della ricetta senza operatori")
    ap.add_argument("--seconds", type=float, default=240)
    ap.add_argument("--n-proc", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--max-shards", type=int, default=6)
    a = ap.parse_args()
    import producer as PR
    import regen
    import sources as S
    import views as VW
    from ring import Ring, ShardReader
    out: dict = {}
    # 1. ricetta senza operatori
    cfg = PR.build_parser().parse_args(["--ring", "x", "--sources", "validated", "--expr-frac", EXPR_C3M, "--views", "6",
                                        "--label-draw", "perm"])
    doms = S.parse_sources(cfg.sources)
    cfg.expr_frac = PR.parse_expr_frac(cfg.expr_frac, doms)
    cfg.labels = list(VW.LABELS)
    lw = PR.parse_weights(cfg.label_weights, VW.LABELS)
    cfg.label_p = np.asarray([lw[k] for k in cfg.labels]) / sum(lw.values())
    srcs = S.build_sources(doms)
    rec = {}
    for d in doms:
        n_expr, n_views, distinct, per_group = 0, 0, Counter(), Counter()
        for gi in range(a.groups):
            rng = np.random.default_rng(np.random.SeedSequence([1, gi]))
            ident = srcs[d].identity(rng)
            vs = list(PR.view_recipe(srcs[d], ident, rng, cfg, d))
            e = sum(t == "expr" for _, _, _, t, _, _, _ in vs)
            n_expr, n_views = n_expr + e, n_views + len(vs)
            distinct[len({v[4] for v in vs})] += 1                 # (i, V, F, tag, label, seme, fotogramma)
            per_group[e] += 1
        f, want = n_expr / n_views, cfg.expr_frac[d]
        se = float(np.sqrt(max(want * (1 - want), 1e-12) / (a.groups * 6)))
        rec[d] = {"expr_fraction": f, "expected": want, "se": se, "expr_views_per_group": dict(sorted(per_group.items())),
                  "distinct_labels_per_group": dict(distinct),
                  "pass": bool(abs(f - want) <= 4 * se + 1e-12 and set(distinct) == {6})}
    out["recipe"] = {"groups_per_domain": a.groups, "by_domain": rec, "pass": all(r["pass"] for r in rec.values())}
    print(f"[c3m] ricetta: {json.dumps(out['recipe'])}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        # 2. produttori veri, validated, niente mm_aug
        ring = Path(tmp) / "ring"
        r = subprocess.run(producer_args(ring, a, ["--summary", str(Path(tmp) / "p.json")]), capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SystemExit(f"producer: {r.stderr[-2000:]}")
        prod = json.loads((Path(tmp) / "p.json").read_text())
        out["producer"] = {k: prod.get(k) for k in ("views", "views_per_s_steady", "views_by_domain", "expr_fraction",
                                                    "expr_fraction_by_domain", "views_by_origin", "label_draw",
                                                    "expr_frac", "failures")}
        groups, multi_src, bad_labels, heads = 0, 0, 0, set()
        for _, _, p in Ring(ring).listing():
            rd = ShardReader(p)
            heads.add(json.dumps({k: rd.head["recipe"][k] for k in ("label_draw", "expr_frac", "mm_aug_probs")},
                                 sort_keys=True))
            for hg in rd.groups:
                groups += 1
                multi_src += int(hg["prov"]["sources"] != [hg["domain"]] or hg.get("origin") != "pure")
                bad_labels += int(len({m["label"] for m in hg["views"]}) != len(hg["views"]))
        res = []
        for _, _, p in Ring(ring).listing()[:a.max_shards]:
            res += regen.check_shard(p)
        out["shards"] = {"groups": groups, "groups_not_pure_single_source": multi_src,
                         "groups_with_repeated_labels": bad_labels, "recipe_headers": sorted(heads),
                         "regen_groups": len(res), "regen_pass": all(x["pass"] for x in res),
                         "pass": bool(groups > 0 and multi_src == 0 and bad_labels == 0 and res
                                      and all(x["pass"] for x in res))}
        print(f"[c3m] shard: {out['shards']}", flush=True)
        # 3. consumatore con la rotazione e i batch di massive_node.sh
        st, rots, lpi = consume(ring, a.steps, "0")
        out["consumer"] = {"plans": st["plans"], "batches_mixed": st["batches_mixed"],
                           "batches_mixed_sources": st["batches_mixed_sources"],
                           "batches_by_sources": st["batches_by_sources"], "rot_deg": st["rot_deg"],
                           "views_booked": len(rots), "views_with_rotation": sum(any(x != 0 for x in r) for r in rots),
                           "distinct_labels_per_id": dict(Counter(lpi)), "reuse_factor": st["reuse_factor"]}
        out["consumer"]["pass"] = bool(st["plans"] == a.steps and st["batches_mixed"] == 0
                                       and st["batches_mixed_sources"] == 0 and len(rots) > 0
                                       and out["consumer"]["views_with_rotation"] == 0 and set(lpi) == {6})
        print(f"[c3m] consumatore: {out['consumer']}", flush=True)
        # 4. controllo positivo: mm_aug -> batch misti per fonti
        ring2 = Path(tmp) / "ring_aug"
        b = argparse.Namespace(**{**vars(a), "seconds": min(a.seconds, 180)})
        r = subprocess.run(producer_args(ring2, b, ["--mm-aug", "hybrid=0.5,expr_transfer=0.5"]), capture_output=True,
                           text=True)
        if r.returncode != 0:
            raise SystemExit(f"producer mm_aug: {r.stderr[-2000:]}")
        st2, rots2, _ = consume(ring2, min(a.steps, 30), "30,15,10")
        out["positive_control"] = {"plans": st2["plans"], "batches_mixed": st2["batches_mixed"],
                                   "batches_mixed_sources": st2["batches_mixed_sources"],
                                   "batches_by_sources": st2["batches_by_sources"],
                                   "views_with_rotation": sum(any(x != 0 for x in r) for r in rots2),
                                   "views_booked": len(rots2)}
        out["positive_control"]["pass"] = bool(st2["batches_mixed"] == 0 and st2["batches_mixed_sources"] > 0
                                               and out["positive_control"]["views_with_rotation"] > 0)
        print(f"[c3m] controllo positivo: {out['positive_control']}", flush=True)
    out["pass"] = all(out[k]["pass"] for k in ("recipe", "shards", "consumer", "positive_control"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"[c3m] ESITO {'PASSA' if out['pass'] else 'FALLISCE'} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
