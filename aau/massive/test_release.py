#!/usr/bin/env python
"""Controlli del codice del modello da rilasciare (PLAN_MASSIVE sez. 23), prima di ogni run.

    AAU_NV= aau/run.sh aau/massive/test_release.py --out aau/runs/massive_release/checks/test_release.json

1. Invarianza del comportamento di default (le modifiche a v3_work valgono solo con --release, STREAM_RELEASE, le
   variabili nuove): gruppi di producer.make_group coi semi di provenienza, scritti dal codice di ``--base`` (git
   archive, default HEAD) e da quello di adesso SENZA i flag nuovi, con gli argomenti di massive_node.sh: ricetta
   c3m su ``validated`` e ricetta max su bfm2019, ict, gnm, flame2023, famos (FaMoS col template FLAME 2020, il
   default). Ricetta, metadati, header e byte dello shard (a parte t_created) e array identici bit per bit
   (v3_work/stream/tests/test_partial.py: dump_groups, compare_dumps, importati). Run dir di train_v3.make_run_dir
   sulla riga di lancio di smoke_validated_1067645 uguale fra base e adesso, anche con WBES_STEP_JSONL.
2. Divieti: senza release i 3DMM di test/sviluppo sono un errore (sources.build_sources ValueError, libreria RoleError).
3. Fonti ``release``: per ogni dominio ``--groups`` gruppi coi semi di provenienza e la ricetta del run (quota 0.5,
   6 discretizzazioni senza reinserimento): viste finite, GT (fr, sr, S) finita, FaMoS su FLAME 2023 senza le persone
   escluse; uno shard scritto e rigenerato con regen.check_shard (facce, vertici, etichette, espressioni).
Uscita 1 se un controllo non passa.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
STREAM = REPO / "v3_work" / "stream"
sys.path.insert(0, str(STREAM / "tests"))

import test_partial as TP  # noqa: E402  (dump_groups, compare_dumps, base_tree, run_dir_name)

EXTRA = {"validated_c3m": "",
         "max_famos": "--sources bfm2019,ict,gnm,flame2023,famos --expr-frac 0.5 --label-draw replace"}


def dumps(base: Path, tmp: Path, n: int) -> dict:
    out = {}
    for k, ex in EXTRA.items():
        procs = {t: subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--dump", str(root),
                                      str(tmp / f"{k}_{t}.npz"), "--groups", str(n), "--dump-extra", ex])
                 for t, root in (("base", base), ("now", REPO))}
        for t, p in procs.items():
            if p.wait():
                raise SystemExit(f"dump {k}/{t} fallito")
        r = TP.compare_dumps(tmp / f"{k}_base.npz", tmp / f"{k}_now.npz")
        r["pass"] = bool(r["recipe_equal"] and r["same_array_keys"] and not r["fields_differing"]
                         and r["groups_meta_equal"] and r["views_meta_equal"] and r["shard_head_equal"]
                         and r["shard_data_equal"])
        out[k] = r
        print(f"[test] invarianza {k}: {json.dumps(r)}", flush=True)
    return out


def guards() -> dict:
    sys.path[:0] = [str(STREAM), str(REPO / "v3_work" / "trainer")]
    import sources as S
    from v3_work.mm import RoleError, load_for_training
    out = {}
    for d in S.RELEASE_ONLY:
        try:
            S.build_sources([d])
            out[f"{d}_sources"] = False
        except ValueError:
            out[f"{d}_sources"] = True
        try:
            load_for_training(d)
            out[f"{d}_library"] = False
        except RoleError:
            out[f"{d}_library"] = True
    try:
        S.build_sources(["famos"], famos_exclude=["x"])
        out["famos_exclude_without_release"] = False
    except ValueError:
        out["famos_exclude_without_release"] = True
    out["pass"] = all(out.values())
    return out


def release_groups(n: int, tmp: Path) -> dict:
    import producer as PR
    import regen
    import sources as S
    import views as VW
    from ring import Ring
    from targets import CanonTargets
    VW.install_grad_vec()
    split = json.loads(S.FAMOS_SPLIT.read_text())
    excl = sorted(split["train"])[:3]
    cfg = PR.build_parser().parse_args(TP.PROD_ARGS + ["--sources", "release", "--expr-frac", "0.5", "--release",
                                                       "--famos-exclude", ",".join(excl), "--seed", "20261013"])
    cfg.mm_aug_probs, cfg.labels = {}, list(VW.LABELS)
    w = np.asarray([VW.LABEL_WEIGHTS[k] for k in cfg.labels])
    cfg.label_p = w / w.sum()
    cfg.famos_exclude_list = excl
    cfg.subdiv, cfg.partial, cfg.code_version = {}, None, ""
    domains = S.parse_sources(cfg.sources)
    cfg.expr_frac = PR.parse_expr_frac(cfg.expr_frac, domains)
    uni = S.Unified()
    t0 = time.time()
    srcs = S.build_sources(domains, uni, release=True, famos_exclude=excl)
    tg = CanonTargets(srcs)
    cfg.mm_factor = {d: tg.mm_factor(d, srcs[d]) for d in domains}
    rec = PR.recipe(cfg)
    res = {"domains": domains, "load_s": time.time() - t0, "mm_factor": cfg.mm_factor,
           "famos": srcs["famos"].describe(), "recipe_release": rec.get("release"),
           "recipe_famos_exclude": rec.get("famos_exclude"), "by_domain": {}}
    res["famos_excluded_absent"] = not (set(excl) & set(srcs["famos"].persons))
    res["famos_template_2023"] = srcs["famos"].template == "flame2023"
    gs = []
    for di, d in enumerate(domains):
        st = {"views": 0, "verts": 0, "failures": 0, "by_domain": {}, "by_label": {},
              **{f"t_{k}": 0.0 for k in ("ident", "mesh", "gen", "ops", "pack", "write")}}
        rows = []
        for i in range(n):
            seed = (cfg.seed, di, i)
            rg = np.random.default_rng(np.random.SeedSequence(list(seed)))
            t1 = time.time()
            g = PR.make_group(srcs[d], d, rg, cfg, uni, f"{d}/t-{di}-{i}", st, tg, seed)
            gs.append(g)
            fin = all(np.isfinite(a[f]).all() for a, _ in g["views"] for f in ("verts", "mass", "evals", "evecs"))
            rows.append({"views": len(g["views"]), "labels": sorted(m["label"] for _, m in g["views"]),
                         "expr": sum(m["expr"] == "expr" for _, m in g["views"]),
                         "verts": [int(m["n"]) for _, m in g["views"]], "finite": bool(fin),
                         "gt_finite": bool(np.isfinite(g["fr"]).all() and np.isfinite(g["sr"]).all()
                                           and np.isfinite(g["S"])), "S_mm": float(g["S"]),
                         "person": g["prov"].get("person"), "seconds": round(time.time() - t1, 1)})
        res["by_domain"][d] = {"groups": rows, "failures": st["failures"], "last_failure": st.get("last_failure", "")}
        print(f"[test] {d}: {json.dumps(res['by_domain'][d])}", flush=True)
    ring = Ring(tmp / "release_ring", 2 ** 34)
    ring.write(gs, 0, rec)
    path = sorted(ring.seqs().items())[0][1]
    chk = regen.check_shard(path)
    res["regen"] = [{"key": r["key"], "pass": r["pass"]} for r in chk]
    res["pass"] = bool(res["famos_excluded_absent"] and res["famos_template_2023"] and res["recipe_release"]
                       and all(r["pass"] for r in chk)
                       and all(g["finite"] and g["gt_finite"] and g["views"] == 6 and len(set(g["labels"])) == 6
                               for v in res["by_domain"].values() for g in v["groups"])
                       and all(v["failures"] == 0 for v in res["by_domain"].values()))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="json dei risultati (obbligatorio salvo --dump)")
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--groups", type=int, default=6, help="gruppi per l'invarianza")
    ap.add_argument("--release-groups", type=int, default=2, help="gruppi per dominio delle fonti release")
    ap.add_argument("--dump", nargs=2, metavar=("ALBERO", "NPZ"), help="(interno)")
    ap.add_argument("--dump-extra", default="")
    a = ap.parse_args()
    if a.dump:
        TP.dump_groups(Path(a.dump[0]), Path(a.dump[1]), a.groups, shlex.split(a.dump_extra))
        return
    if a.out is None:
        ap.error("--out obbligatorio")
    res = {"base": subprocess.run(["git", "-C", str(REPO), "rev-parse", a.base], capture_output=True, text=True,
                                  check=True).stdout.strip(), "t": time.time()}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", "/tmp")) as td:
        tmp = Path(td)
        base = TP.base_tree(tmp, a.base)
        res["invariance"] = dumps(base, tmp, a.groups)
        names = {"base": TP.run_dir_name(base, False), "now": TP.run_dir_name(REPO, False)}
        os.environ["WBES_STEP_JSONL"] = "1"
        names["now_jsonl"] = TP.run_dir_name(REPO, False)
        os.environ.pop("WBES_STEP_JSONL")
        names["pass"] = len({names["base"], names["now"], names["now_jsonl"]}) == 1
        res["run_dir"] = names
        print(f"[test] run dir: {json.dumps(names)}", flush=True)
        res["guards"] = guards()
        print(f"[test] divieti: {json.dumps(res['guards'])}", flush=True)
        res["release"] = release_groups(a.release_groups, tmp)
    res["pass"] = bool(all(v["pass"] for v in res["invariance"].values()) and res["run_dir"]["pass"]
                       and res["guards"]["pass"] and res["release"]["pass"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[test] PASS={res['pass']} -> {a.out}", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
