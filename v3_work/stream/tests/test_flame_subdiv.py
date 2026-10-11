#!/usr/bin/env python
"""Correttezza (g): suddivisione di FLAME 2023 nello stream (``producer.py --subdiv flame2023=1``, cella B'
dell'ablazione 2x2, emendamento 1) e invarianza a opzione spenta.

    aau/run.sh v3_work/stream/tests/test_flame_subdiv.py --out aau/runs/evidence/stream/ablation_2x2/flame_subdiv/check.json

1. Invarianza (sottoprocessi freschi, ``test_partial.dump_groups``, importato): i gruppi di producer.make_group coi semi
   di provenienza e gli argomenti di massive_node.sh (ricetta c3m) con le fonti di B (``validated,flame2023``, quota
   d'espressioni flame2023 0.2142) scritti dal codice della revisione di base (``--base``, default 4e3114a: quello delle
   celle A-D in corso, ``git archive``) e da quello di adesso senza --subdiv: ricetta, metadati, header e byte dello
   shard, array identici bit per bit.
2. --subdiv flame2023=1 sugli stessi semi: stesse identita', etichette, espressioni, semi del rumore; i gruppi non FLAME
   identici bit per bit; le viste FLAME = la procedura di D1 (``aau/diagnostics/d1_gen.work_mesh``: mesh della vista
   suddivisa con igl.upsample, poi views.discretize), rifatta qui per un'altra strada (sorgente nativa,
   producer.view_recipe, igl.upsample sulla vista canonica: i punti medi commutano con la similarita'): facce identiche,
   vertici serviti entro 1e-6; vertici di original 6.986 (la flame2023_s1 di D1); regen.check_shard sullo shard.
3. Produttori veri con --subdiv flame2023=1 per ``--seconds``: viste per dominio, vertici delle viste FLAME per
   etichetta, fallimenti; regen.check_shard sui primi shard.
La run dir (train_v3.make_run_dir) non dipende dagli argomenti dei produttori: test_partial.py --only rundir.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
STREAM = THIS.parent
REPO = STREAM.parents[1]
sys.path.insert(0, str(THIS))

import test_partial as TP  # noqa: E402  (sola lettura: albero di base, confronto bit per bit)

BASE = "4e3114a"         # copia congelata delle celle A, B, C, D (commit di lancio)
SOURCES_B = ["--sources", "validated,flame2023", "--expr-frac", "bfm2019=0,ict=0.2315,gnm=0.1968,flame2023=0.2142"]
SUBDIV = "flame2023=1"


def dump_on(out: Path, n_groups: int) -> None:
    """test_partial.dump_groups con --subdiv (sorgenti suddivise come in producer.main)."""
    TP.use_tree(REPO)
    import producer as PR
    import sources as S
    real = S.build_sources

    def build(domains, unified=None, v_max=S.V_MAX, v_work=S.V_WORK, subdiv=None):
        return real(domains, unified, v_max=v_max, v_work=v_work, subdiv={"flame2023": 1})
    S.build_sources = build             # dump_groups costruisce le sorgenti senza subdiv
    orig = PR.recipe

    def recipe(cfg, acfg=None):
        cfg.subdiv = {"flame2023": 1}   # come producer.main: la ricetta dello shard porta ``subdiv``
        return orig(cfg, acfg)
    PR.recipe = recipe
    TP.dump_groups(REPO, out, n_groups, SOURCES_B + ["--subdiv", SUBDIV])


def d1_procedure(dump: Path, n_groups: int) -> dict:
    """Viste FLAME neutre del dump con --subdiv contro la procedura di D1 (d1_gen.work_mesh sulla sorgente nativa, poi
    il frame canonico della vista e views.discretize); tutte: vertici per etichetta; regen.check_shard sullo shard."""
    TP.use_tree(REPO)
    import producer as PR
    import regen
    import sources as S
    import views as VW
    from ring import Ring
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "aau/diagnostics"))
    import d1_gen                       # sola lettura: work_mesh, la mesh di lavoro degli insiemi FLAME di D1
    VW.install_grad_vec()
    j = json.loads(dump.with_suffix(".json").read_text())
    z = np.load(dump)
    cfg = PR.build_parser().parse_args(TP.PROD_ARGS + ["--seed", "20261011"] + SOURCES_B)
    cfg.labels = list(VW.LABELS)
    lw = PR.parse_weights(cfg.label_weights, VW.LABELS)
    w = np.asarray([lw.get(k, 0.0) for k in cfg.labels])
    cfg.label_p = w / w.sum()
    domains = S.parse_sources(cfg.sources)
    cfg.expr_frac = PR.parse_expr_frac(cfg.expr_frac, domains)
    src = S.build_sources(["flame2023"])["flame2023"]            # nativa: 1.787 vertici
    c = src.canon
    out = {"views_neutral": 0, "views_expr": 0, "faces_equal": 0, "max_abs_served": 0.0,
           "verts_by_label": defaultdict(set)}
    for n, g in enumerate(j["groups"]):
        if g["domain"] != "flame2023":
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in g["prov"]["seed"]]))
        assert PR.group_kind(rng, cfg, "flame2023") == "pure"
        ident = src.identity(rng)
        if not np.array_equal(np.asarray(ident), z[f"g{n}_zid"]):
            raise SystemExit(f"gruppo {n}: identita' diversa")
        Vw, Fw = d1_gen.work_mesh(src, ident, 1)                 # D1: media + base x z, igl.upsample 1-a-4
        Vw = c["scale"] * (Vw @ np.asarray(c["R"]).T) + np.asarray(c["t"])
        for i, V, F, tag, label, ns, _ in PR.view_recipe(src, ident, rng, cfg, "flame2023"):
            vi = next(k for k, m in enumerate(g["views"]) if m["vi"] == i)
            out["verts_by_label"][label].add(int(len(z[f"g{n}_v{vi}_verts"])))
            if tag != "neutral":                                  # D1 non ha espressioni: solo vertici e regen
                out["views_expr"] += 1
                continue
            Vd, Fd = VW.discretize(Vw, Fw, label, ns)
            same = np.array_equal(np.asarray(z[f"g{n}_v{vi}_faces"], dtype=np.int64), np.asarray(Fd, dtype=np.int64))
            out["views_neutral"] += 1
            out["faces_equal"] += int(same)
            if same:
                d = float(np.abs(regen.served(Vd) - np.asarray(z[f"g{n}_v{vi}_verts"], dtype=np.float64)).max())
                out["max_abs_served"] = max(out["max_abs_served"], d)
    out["verts_by_label"] = {k: sorted(v) for k, v in sorted(out["verts_by_label"].items())}
    path = sorted(Ring(dump.with_suffix(".ring")).seqs().items())[0][1]
    res = regen.check_shard(path, srcs={})
    out["regen"] = {"groups": len(res), "pass": sum(r["pass"] for r in res),
                    "max_verts_abs": max(v["verts_max_abs"] for r in res for v in r["views"])}
    out["pass"] = bool(out["views_neutral"] > 0 and out["faces_equal"] == out["views_neutral"]
                       and out["max_abs_served"] < 1e-6 and 6986 in out["verts_by_label"].get("original", [])
                       and out["regen"]["pass"] == len(res))
    return out


def compare_on_off(off: Path, on: Path) -> dict:
    """Stessi semi con e senza --subdiv: stesse estrazioni; i gruppi non FLAME identici; ricetta con ``subdiv``."""
    jo, jp = json.loads(off.with_suffix(".json").read_text()), json.loads(on.with_suffix(".json").read_text())
    zo, zp = np.load(off), np.load(on)
    same_draws, others_equal, n_flame = True, True, 0
    for n, (go, gp) in enumerate(zip(jo["groups"], jp["groups"])):
        same_draws &= go["key"] == gp["key"] and go["S"] == gp["S"] and go["prov"] == gp["prov"]
        for k in ("s", "fr", "sr", "zid"):
            if f"g{n}_{k}" in zo.files:
                same_draws &= zo[f"g{n}_{k}"].tobytes() == zp[f"g{n}_{k}"].tobytes()
        for vi, (mo, mp) in enumerate(zip(go["views"], gp["views"])):
            same_draws &= all(mo[k] == mp[k] for k in ("label", "expr", "vi", "noise_seed", "frame"))
            if go["domain"] == "flame2023":
                n_flame += 1
            else:
                others_equal &= all(zo[f"g{n}_v{vi}_{f}"].tobytes() == zp[f"g{n}_v{vi}_{f}"].tobytes()
                                    for f in ("verts", "faces", "mass"))
    return {"recipe_subdiv_on": jp["recipe"].get("subdiv"), "recipe_subdiv_off": jo["recipe"].get("subdiv"),
            "recipe_equal_but_subdiv": {k: v for k, v in jp["recipe"].items() if k != "subdiv"} == jo["recipe"],
            "same_identities_labels_expr_noise": bool(same_draws), "non_flame_views_identical": bool(others_equal),
            "flame_views": n_flame, "failures": jp["failures"]}


def producers(a, tmp: Path) -> dict:
    TP.use_tree(REPO)
    import regen
    from ring import Ring, ShardReader
    ring = tmp / "ring"
    cmd = [sys.executable, str(STREAM / "producer.py"), *TP.PROD_ARGS, *SOURCES_B, "--subdiv", SUBDIV,
           "--ring", str(ring), "--ring-gb", "6", "--n-proc", str(a.n_proc), "--seed", "20261011",
           "--duration", str(a.seconds), "--stats-every", "30", "--summary", str(tmp / "producers.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"producer: {r.stderr[-3000:]}")
    prod = json.loads((tmp / "producers.json").read_text())
    shards = sorted(Ring(ring).seqs().items())
    vb = defaultdict(set)
    for _, p in shards:
        rd = ShardReader(p)
        assert rd.head["recipe"]["subdiv"] == {"flame2023": 1}
        for g in rd.groups:
            if g["domain"] == "flame2023":
                for m in g["views"]:
                    vb[m["label"]].add(int(m["n"]))
    res = []
    for _, p in shards[:a.max_shards]:
        res += regen.check_shard(p, srcs={})
    out = {"producer": {k: prod.get(k) for k in ("views", "views_per_s_steady", "failures", "last_failure", "subdiv",
                                                  "views_by_domain", "mean_verts_per_view", "expr_fraction_by_domain")},
           "flame_verts_by_label": {k: [min(v), max(v)] for k, v in sorted(vb.items())},
           "regen": {"groups": len(res), "pass": sum(r["pass"] for r in res),
                     "max_verts_abs": max((v["verts_max_abs"] for r in res for v in r["views"]), default=None)}}
    out["pass"] = bool(prod.get("failures") == 0 and vb and min(vb.get("original", {0})) == 6986
                       and out["regen"]["pass"] == out["regen"]["groups"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--groups", type=int, default=8, help="gruppi del confronto (domini a turno: 2 per dominio)")
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--n-proc", type=int, default=12)
    ap.add_argument("--max-shards", type=int, default=6)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--dump-on", default="", help="(interno) npz del dump con --subdiv")
    a = ap.parse_args()
    if a.dump_on:
        dump_on(Path(a.dump_on), a.groups)
        return
    res: dict = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        base = TP.base_tree(tmp, a.base)
        res["base"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", a.base], capture_output=True,
                                     text=True).stdout.strip()
        res["head"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
        extra = " ".join(shlex.quote(x) for x in SOURCES_B)
        procs = {k: subprocess.Popen([sys.executable, str(THIS / "test_partial.py"), "--dump", str(root),
                                      str(tmp / f"{k}.npz"), "--groups", str(a.groups), "--dump-extra", extra])
                 for k, root in (("base", base), ("now_off", REPO))}
        procs["now_on"] = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--dump-on",
                                            str(tmp / "now_on.npz"), "--groups", str(a.groups)])
        for k, p in procs.items():
            if p.wait() != 0:
                raise SystemExit(f"dump {k} fallito")
        res["invariance_off_vs_base"] = TP.compare_dumps(tmp / "base.npz", tmp / "now_off.npz")
        res["subdiv_on_vs_off"] = compare_on_off(tmp / "now_off.npz", tmp / "now_on.npz")
        res["d1_procedure"] = d1_procedure(tmp / "now_on.npz", a.groups)
        print(json.dumps({k: res[k] for k in ("invariance_off_vs_base", "subdiv_on_vs_off", "d1_procedure")},
                         default=str), flush=True)
        if a.seconds > 0:
            res["producers"] = producers(a, tmp)
            print(json.dumps(res["producers"], default=str), flush=True)
    inv, oo = res["invariance_off_vs_base"], res["subdiv_on_vs_off"]
    res["pass"] = {
        "invariance": inv["recipe_equal"] and inv["same_array_keys"] and not inv["fields_differing"]
        and inv["views_meta_equal"] and inv["groups_meta_equal"] and inv["shard_head_equal"] and inv["shard_data_equal"],
        "subdiv_on_vs_off": oo["recipe_equal_but_subdiv"] and oo["recipe_subdiv_on"] == {"flame2023": 1}
        and oo["recipe_subdiv_off"] is None and oo["same_identities_labels_expr_noise"] and oo["non_flame_views_identical"]
        and oo["flame_views"] > 0 and oo["failures"] == 0,
        "d1_procedure": res["d1_procedure"]["pass"],
        "producers": None if "producers" not in res else res["producers"]["pass"]}
    print(json.dumps(res["pass"]), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
