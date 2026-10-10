#!/usr/bin/env python
"""Correttezza (f): parzialita' variabile (partial_aug.py) e invarianza a parzialita' spenta.

    aau/run.sh v3_work/stream/tests/test_partial.py --out aau/runs/evidence/stream/partial_aug/partial_check.json

1. Geometria sulle mesh vere (preset validated: bfm2019, ict, gnm, ``--ids`` identita' neutre per dominio, tutte le
   discretizzazioni, ``--seeds`` semi per mesh, p = 1): perdita d'area per modo, log del rapporto sqrt(area) contro
   quello del crop di valutazione sulle stesse mesh (make_crop su original), validita' (una componente di facce,
   nessun vertice isolato, nessuna faccia degenere, >= min_verts, perdita <= max_loss), determinismo (due chiamate
   identiche), operatori DiffusionNet (views.operators + serve_like_loader, valori finiti) su ``--ops`` viste.
2. Invarianza (sottoprocessi freschi, stessa sequenza di chiamate): i gruppi di producer.make_group coi semi di
   provenienza e gli argomenti di massive_node.sh (ricetta c3m) scritti dal codice di HEAD (``git archive``) e da
   quello di adesso senza --partial-p: ricetta, metadati, header e byte dello shard (ring.Ring.write, a parte
   t_created) e array identici bit per bit. Poi --partial-p 0.5 sugli
   stessi semi: stesse identita', etichette, espressioni, semi del rumore; le viste non parziali identiche.
3. Run dir: train_v3.make_run_dir sulla riga di lancio di smoke_validated_1067645 con il codice di HEAD e di adesso
   (con e senza WBES_PREEMPT_SAVE): stesso nome, uguale alla directory del run.
4. Produttori veri con --partial-p 0.5 per ``--seconds``: quota di viste parziali, perdite, fallimenti; regen.check_shard
   rigenera i primi shard (facce, vertici, parametri della parzialita').
5. StreamPlans.skip: con skip k i primi k piani sono None e il consumatore non ne estrae nessuno.
"""
from __future__ import annotations

import argparse
import hashlib
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
STREAM = THIS.parent
REPO = STREAM.parents[1]
LAUNCH = REPO / "aau/runs/evidence/stream/smoke_validated_1067645/node0/launch.txt"
# producer.py come in massive_node.sh con STREAM_RECIPE c3m (senza anello, CPU, semi e statistiche)
PROD_ARGS = ["--ring", "/nonexistent", "--k-eig", "128", "--evecs-dtype", "fp32", "--sources", "validated",
             "--provenance", "--canonical-gt", "--expr-frac", "bfm2019=0,ict=0.2315,gnm=0.1968", "--views", "6",
             "--label-draw", "perm"]


def use_tree(root: Path) -> None:
    for p in (root / "v3_work" / "stream", root / "v3_work" / "trainer"):
        sys.path.insert(0, str(p))


# --- 2. invarianza: un sottoprocesso per albero ---------------------------------------------------------------

def dump_groups(root: Path, out: Path, n_groups: int, extra: list) -> None:
    """Gruppi di make_group (provenienza: SeedSequence([seme, 0, n])), domini a turno, con il setup di producer.main."""
    use_tree(root)
    import producer as PR
    import sources as S
    import views as VW
    from targets import CanonTargets
    VW.install_grad_vec()
    cfg = PR.build_parser().parse_args(PROD_ARGS + ["--seed", "20261011"] + extra)
    cfg.mm_aug_probs = {}
    cfg.labels = list(VW.LABELS)
    lw = PR.parse_weights(cfg.label_weights, VW.LABELS)
    w = np.asarray([lw.get(k, 0.0) for k in cfg.labels])
    cfg.label_p = w / w.sum()
    domains = S.parse_sources(cfg.sources)
    cfg.expr_frac = PR.parse_expr_frac(cfg.expr_frac, domains)
    cfg.code_version = ""
    if "--partial-p" in extra:
        import partial_aug
        cfg.partial = partial_aug.config(cfg.partial_p, cfg.partial_area)
    uni = S.Unified()
    srcs = S.build_sources(domains, uni, v_max=cfg.v_max, v_work=cfg.v_work)
    tg = CanonTargets(srcs)
    cfg.mm_factor = {d: tg.mm_factor(d, srcs[d]) for d in domains}
    rec = PR.recipe(cfg)
    st = {"views": 0, "verts": 0, "failures": 0, "by_domain": {}, "by_label": {},
          **{f"t_{k}": 0.0 for k in ("ident", "mesh", "gen", "ops", "pack", "write")}}
    arrays, groups, gs = {}, [], []
    for n in range(n_groups):
        d = domains[n % len(domains)]
        seed = (cfg.seed, 0, n)
        rg = np.random.default_rng(np.random.SeedSequence(list(seed)))
        assert PR.group_kind(rg, cfg, d) == "pure"
        g = PR.make_group(srcs[d], d, rg, cfg, uni, f"{d}/{cfg.seed}-0-{n}", st, tg, seed)
        gs.append(g)
        for k in ("s", "fr", "sr", "zid"):
            if k in g:
                arrays[f"g{n}_{k}"] = np.asarray(g[k])
        views = []
        for vi, (arr, meta) in enumerate(g["views"]):
            for f, a in arr.items():
                arrays[f"g{n}_v{vi}_{f}"] = a
            views.append(meta)
        groups.append({"key": g["key"], "domain": d, "S": g.get("S"), "prov": g.get("prov"), "views": views})
    np.savez(out, **arrays)
    from ring import Ring, ShardReader
    ring = Ring(out.with_suffix(".ring"), 2 ** 34)
    _, nb = ring.write([g for g in gs if g["views"]], 0, rec)
    path = sorted(ring.seqs().items())[0][1]
    head = dict(ShardReader(path).head)
    head.pop("t_created")
    body = path.read_bytes()
    body = body[int.from_bytes(body[8:16], "little"):]        # i dati: dall'inizio scritto nei byte 8-16
    out.with_suffix(".json").write_text(json.dumps({"recipe": rec, "groups": groups, "failures": st["failures"],
                                                    "shard_head": head, "shard_bytes": nb,
                                                    "shard_data_sha256": hashlib.sha256(body).hexdigest()},
                                                   default=str))


def head_tree(tmp: Path) -> Path:
    """v3_work di HEAD (git archive) con il resto del repo in link simbolici."""
    root = tmp / "head"
    root.mkdir()
    arc = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD", "v3_work"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(root)], input=arc, check=True)
    for p in REPO.iterdir():
        if p.name not in ("v3_work", ".git"):
            (root / p.name).symlink_to(p)
    return root


def compare_dumps(a: Path, b: Path) -> dict:
    ja, jb = json.loads(a.with_suffix(".json").read_text()), json.loads(b.with_suffix(".json").read_text())
    za, zb = np.load(a), np.load(b)
    strip = lambda m: {k: v for k, v in m.items() if not k.startswith("t_")}  # noqa: E731
    diff_fields = sorted({k.split("_", 2)[-1] for k in za.files
                          if k not in zb.files or za[k].dtype != zb[k].dtype or za[k].shape != zb[k].shape
                          or za[k].tobytes() != zb[k].tobytes()})
    meta_eq = all(strip(x) == strip(y) for ga, gb in zip(ja["groups"], jb["groups"])
                  for x, y in zip(ga["views"], gb["views"]))
    return {"recipe_equal": ja["recipe"] == jb["recipe"], "n_arrays": len(za.files),
            "same_array_keys": sorted(za.files) == sorted(zb.files), "fields_differing": diff_fields,
            "groups_meta_equal": [{k: v for k, v in g.items() if k != "views"} for g in ja["groups"]]
            == [{k: v for k, v in g.items() if k != "views"} for g in jb["groups"]],
            "views_meta_equal": meta_eq, "views": sum(len(g["views"]) for g in ja["groups"]),
            "shard_head_equal": ja["shard_head"] == jb["shard_head"], "shard_bytes": [ja["shard_bytes"], jb["shard_bytes"]],
            "shard_data_equal": ja["shard_data_sha256"] == jb["shard_data_sha256"]}


def compare_partial(off: Path, on: Path) -> dict:
    """Stessi semi con e senza --partial-p: tutto uguale tranne le viste parziali (e la ricetta, che ha ``partial``)."""
    jo, jp = json.loads(off.with_suffix(".json").read_text()), json.loads(on.with_suffix(".json").read_text())
    zo, zp = np.load(off), np.load(on)
    same_draws, untouched_equal, n_on, n_off, losses = True, True, 0, 0, []
    for n, (go, gp) in enumerate(zip(jo["groups"], jp["groups"])):
        same_draws &= go["key"] == gp["key"] and go["S"] == gp["S"] and go["prov"] == gp["prov"]
        for k in ("s", "fr", "sr", "zid"):
            if f"g{n}_{k}" in zo.files:
                same_draws &= zo[f"g{n}_{k}"].tobytes() == zp[f"g{n}_{k}"].tobytes()
        for vi, (mo, mp) in enumerate(zip(go["views"], gp["views"])):
            same_draws &= all(mo[k] == mp[k] for k in ("label", "expr", "vi", "noise_seed", "frame"))
            if mp["partial"]["on"]:
                n_on += 1
                losses.append(mp["partial"]["loss"])
            else:
                n_off += 1
                untouched_equal &= all(zo[f"g{n}_v{vi}_{f}"].tobytes() == zp[f"g{n}_v{vi}_{f}"].tobytes()
                                       for f in ("verts", "faces", "mass"))
    return {"recipe_partial": jp["recipe"].get("partial"), "same_identities_labels_expr_noise": bool(same_draws),
            "untouched_views_identical": bool(untouched_equal), "views_partial": n_on, "views_whole": n_off,
            "loss_mean": float(np.mean(losses)) if losses else None, "failures": jp["failures"]}


def run_dir_name(root: Path, preempt: bool) -> str:
    env = {**os.environ, "WBES_PREEMPT_SAVE": "1" if preempt else "0"}
    code = ("import sys, shlex; sys.path[:0] = [sys.argv[1] + '/v3_work/stream', sys.argv[1] + '/v3_work/trainer']\n"
            "import train_stream as T\n"
            "a = T.build_parser().parse_args(shlex.split(open(sys.argv[2]).read())[1:]); T.tv.check_args(a)\n"
            "print(T.tv.make_run_dir(a).name)")
    r = subprocess.run([sys.executable, "-c", code, str(root), str(LAUNCH)], capture_output=True, text=True, env=env,
                       cwd=str(REPO))
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    return r.stdout.strip().splitlines()[-1]


# --- 1. geometria -------------------------------------------------------------------------------------

def valid(V: np.ndarray, F: np.ndarray, cfg: dict) -> dict:
    import igl
    import partial_aug as PA
    n_comp = int(igl.facet_components(np.asarray(F, dtype=np.int64))[0])
    used = np.zeros(len(V), dtype=bool)
    used[F.reshape(-1)] = True
    degen = int(((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2])).sum()
                + (PA.face_areas(V, F) <= 0).sum())
    return {"components": n_comp, "isolated": int((~used).sum()), "degenerate": degen,
            "ok": n_comp == 1 and bool(used.all()) and degen == 0 and len(V) >= cfg["min_verts"]}


def geometry(a) -> dict:
    use_tree(REPO)
    import partial_aug as PA
    import sources as S
    import views as VW
    VW.install_grad_vec()
    cfg = PA.config(1.0)
    uni = S.Unified()
    domains = S.parse_sources("validated")
    srcs = S.build_sources(domains, uni)
    rows, crop_lr, ops, n_ops, t_part = [], [], {"ok": 0, "fail": 0, "errors": []}, 0, []
    for d in domains:
        rng = np.random.default_rng(np.random.SeedSequence([20261011, domains.index(d)]))
        for i in range(a.ids):
            ident = srcs[d].identity(rng)
            V, F, _ = srcs[d].view_mesh(ident, rng, False)
            Vo, Fo = VW.discretize(V, F, "original", 0)
            Vc, Fc = VW.discretize(V, F, "crop", 0)
            crop_lr.append(0.5 * np.log(PA.face_areas(Vc, Fc).sum() / PA.face_areas(Vo, Fo).sum()))
            for label in VW.LABELS:
                ns = int(rng.integers(2 ** 31))
                Vd, Fd = VW.discretize(V, F, label, ns)
                A0 = PA.face_areas(Vd, Fd).sum()
                for s in range(a.seeds):
                    seed = int(rng.integers(2 ** 31))
                    t0 = time.perf_counter()
                    Vp, Fp, info = PA.apply(Vd, Fd, cfg, seed)
                    t_part.append(time.perf_counter() - t0)
                    Vq, Fq, info2 = PA.apply(Vd, Fd, cfg, seed)
                    det = info == info2 and np.array_equal(Vp, Vq) and np.array_equal(Fp, Fq)
                    lr = 0.5 * np.log(PA.face_areas(Vp, Fp).sum() / A0)
                    rows.append({"domain": d, "label": label, "on": info["on"], "mode": info.get("mode", "-"),
                                 "family": info.get("family", "-"), "holes": info.get("holes", 0),
                                 "target": info.get("target", np.nan), "loss": info.get("loss", 0.0),
                                 "log_sqrt_ratio": float(lr), "n_in": len(Vd), "n_out": len(Vp),
                                 "fallback": bool(info.get("fallback")), "deterministic": bool(det),
                                 **valid(Vp, Fp, cfg)})
                    if n_ops < a.ops and info["on"]:
                        n_ops += 1
                        try:
                            VW.serve_like_loader(VW.operators(Vp, Fp, 128))
                            ops["ok"] += 1
                        except Exception as exc:  # noqa: BLE001
                            ops["fail"] += 1
                            ops["errors"].append(f"{d}/{label}: {type(exc).__name__}: {exc}")
    on = [r for r in rows if r["on"]]
    loss = np.asarray([r["loss"] for r in on])
    lr = np.asarray([r["log_sqrt_ratio"] for r in on])
    q = lambda x: {f"p{p}": float(np.percentile(x, p)) for p in (0, 5, 25, 50, 75, 95, 100)}  # noqa: E731
    by_mode = {}
    for m in sorted({r["mode"] for r in on}):
        x = np.asarray([r["loss"] for r in on if r["mode"] == m])
        by_mode[m] = {"n": len(x), "loss": q(x)}
    tgt = np.asarray([r["loss"] - r["target"] for r in on if r["mode"] in ("band", "plane") and r["holes"] == 0])
    return {"views": len(rows), "partial": len(on), "fallback": sum(r["fallback"] for r in rows),
            "valid": sum(r["ok"] for r in rows), "invalid_examples": [r for r in rows if not r["ok"]][:5],
            "deterministic": sum(r["deterministic"] for r in rows),
            "loss": q(loss), "log_sqrt_ratio": q(lr), "loss_by_mode": by_mode,
            "loss_minus_target_band_plane_noholes": q(tgt) if len(tgt) else None,
            "eval_crop_log_sqrt_ratio": q(np.asarray(crop_lr)),
            "frac_partial_beyond_eval_crop": float((lr < np.min(crop_lr)).mean()),
            "frac_partial_within_eval_crop_range": float(((lr >= np.min(crop_lr)) & (lr <= np.max(crop_lr))).mean()),
            "holes_made": {int(k): int(v) for k, v in zip(*np.unique([r["holes"] for r in on], return_counts=True))},
            "band_family": {k: sum(r["family"] == k for r in on) for k in ("radius", "arc")},
            "seconds_per_apply": {"mean": float(np.mean(t_part)), "p95": float(np.percentile(t_part, 95))},
            "operators": ops}, rows


# --- 4.-5. produttori veri, rigenerazione, skip ------------------------------------------------------------------

def producers(a, tmp: Path) -> dict:
    use_tree(REPO)
    import regen
    import sources as S
    from consumer import StreamConsumer, StreamGT, StreamPlans
    from ring import Ring, ShardReader
    from sampler_v3 import DrawCfg
    ring = tmp / "ring"
    cmd = [sys.executable, str(STREAM / "producer.py"), *PROD_ARGS, "--ring", str(ring), "--ring-gb", "6",
           "--n-proc", str(a.n_proc), "--seed", "20261011", "--partial-p", "0.5", "--duration", str(a.seconds),
           "--stats-every", "30", "--summary", str(tmp / "producers.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"producer: {r.stderr[-3000:]}")
    prod = json.loads((tmp / "producers.json").read_text())
    out = {"producer": {k: prod.get(k) for k in ("views", "views_per_s_steady", "failures", "last_failure",
                                                  "partial_fraction", "partial_views", "partial_fallback",
                                                  "partial_mean_loss", "partial_by_mode", "views_by_label",
                                                  "cpu_s_per_view")}}
    shards = sorted(Ring(ring).seqs().items())
    losses, on, n = [], 0, 0
    for _, p in shards:
        rd = ShardReader(p)
        assert rd.head["recipe"]["partial"]["p"] == 0.5
        for g in rd.groups:
            for m in g["views"]:
                n += 1
                if m["partial"]["on"]:
                    on += 1
                    losses.append(m["partial"]["loss"])
    out["shards"] = {"n": len(shards), "views": n, "partial": on, "fraction": on / max(n, 1),
                     "loss_mean": float(np.mean(losses)) if losses else None}
    srcs: dict = {}
    res = []
    for _, p in shards[:a.max_shards]:
        res += regen.check_shard(p, srcs=srcs)
    out["regen"] = {"groups": len(res), "pass": sum(r["pass"] for r in res),
                    "views": sum(len(r["views"]) for r in res),
                    "partial_ok": sum(v["partial_ok"] for r in res for v in r["views"]),
                    "max_verts_abs": max(v["verts_max_abs"] for r in res for v in r["views"]),
                    "failed": [r for r in res if not r["pass"]][:3]}
    c = StreamConsumer(ring, reuse=4, prefetch=0, gt=StreamGT(S.Unified().A, 1.0), gt_kind="sr", scale=0.0,
                       rot_deg=[0.0], domains=S.parse_sources("validated"), wait_s=60)
    drawcfg = DrawCfg(p_noise=0.6, sigma_min=5e-4, sigma_max=2e-2, noise_modes=["translation"], noise_mode_probs=[1.0],
                      max_meshes=6)
    plans = StreamPlans(c, 6, 2, 6, drawcfg, np.random.default_rng(0), 1, batch_domains=sorted(S.parse_sources("validated")))
    plans.skip = 4
    got = [None if p is None else len(p.entries) for p in plans]
    out["skip"] = {"steps": 6, "skip": 4, "plans": got, "consumer_plans_drawn": c.c["plans"],
                   "pass": got[:4] == [None] * 4 and all(x for x in got[4:]) and c.c["plans"] == 2}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ids", type=int, default=4, help="identita' per dominio (geometria)")
    ap.add_argument("--seeds", type=int, default=6, help="semi per mesh (geometria)")
    ap.add_argument("--ops", type=int, default=40, help="viste parziali con gli operatori DiffusionNet")
    ap.add_argument("--groups", type=int, default=6, help="gruppi del confronto bit per bit")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--dump", nargs=2, metavar=("ALBERO", "NPZ"), help="(interno) gruppi del confronto")
    ap.add_argument("--dump-extra", default="")
    ap.add_argument("--only", default="geometry,invariance,rundir,producers")
    a = ap.parse_args()
    if a.dump:
        dump_groups(Path(a.dump[0]), Path(a.dump[1]), a.groups, shlex.split(a.dump_extra))
        return
    only = set(a.only.split(","))
    res: dict = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        if "invariance" in only or "rundir" in only:
            head = head_tree(tmp)
            res["head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip()
        if "invariance" in only:
            jobs = {"head": (head, ""), "now_off": (REPO, ""), "now_on": (REPO, "--partial-p 0.5")}
            procs = {k: subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--dump", str(root),
                                          str(tmp / f"{k}.npz"), "--groups", str(a.groups), "--dump-extra", ex])
                     for k, (root, ex) in jobs.items()}
            for k, p in procs.items():
                if p.wait() != 0:
                    raise SystemExit(f"dump {k} fallito")
            res["invariance_off_vs_head"] = compare_dumps(tmp / "head.npz", tmp / "now_off.npz")
            res["partial_on_vs_off"] = compare_partial(tmp / "now_off.npz", tmp / "now_on.npz")
            print(json.dumps({k: res[k] for k in ("invariance_off_vs_head", "partial_on_vs_off")}), flush=True)
        if "rundir" in only:
            names = {"head": run_dir_name(head, False), "now": run_dir_name(REPO, False),
                     "now_preempt": run_dir_name(REPO, True)}
            existing = sorted(p.name for p in (LAUNCH.parent / "runs").iterdir())
            res["run_dir"] = {**names, "existing": existing,
                              "pass": len(set(names.values())) == 1 and names["head"] in existing}
            print(json.dumps(res["run_dir"]), flush=True)
        if "geometry" in only:
            res["geometry"], rows = geometry(a)
            print(json.dumps(res["geometry"]), flush=True)
            if a.out:
                import csv
                a.out.parent.mkdir(parents=True, exist_ok=True)
                with open(a.out.with_name("partial_views.csv"), "w", newline="") as fh:
                    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    wr.writeheader()
                    wr.writerows(rows)
        if "producers" in only:
            res["producers"] = producers(a, tmp)
            print(json.dumps(res["producers"], default=str), flush=True)
    g, inv = res.get("geometry"), res.get("invariance_off_vs_head")
    res["pass"] = {
        "geometry": None if g is None else g["valid"] == g["views"] and g["deterministic"] == g["views"]
        and g["operators"]["fail"] == 0 and g["fallback"] == 0,
        "invariance": None if inv is None else inv["recipe_equal"] and inv["same_array_keys"]
        and not inv["fields_differing"] and inv["views_meta_equal"] and inv["groups_meta_equal"]
        and inv["shard_head_equal"] and inv["shard_data_equal"],
        "partial_on_vs_off": None if "partial_on_vs_off" not in res else
        res["partial_on_vs_off"]["same_identities_labels_expr_noise"] and res["partial_on_vs_off"]["untouched_views_identical"],
        "run_dir": None if "run_dir" not in res else res["run_dir"]["pass"],
        "regen": None if "producers" not in res else res["producers"]["regen"]["pass"] == res["producers"]["regen"]["groups"],
        "skip": None if "producers" not in res else res["producers"]["skip"]["pass"]}
    print(json.dumps(res["pass"]), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
