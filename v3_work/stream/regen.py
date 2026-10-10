#!/usr/bin/env python3
"""Rigenerazione deterministica di un gruppo dello stream dal suo seme ("ricetta piu' semi", per il rilascio).

    aau/run.sh v3_work/stream/regen.py --shard <anello>/shards/<seq>.shard [--group g] [--out dir]
    aau/run.sh v3_work/stream/regen.py --domain gnm --seed 20261009,3,117 --recipe recipe.json --out dir

Un gruppo scritto con ``producer.py --provenance`` ha il seme SeedSequence([seme, processo, gruppo]); con la ricetta
dello shard (viste, quota d'espressioni anche per dominio, pesi e modo delle discretizzazioni, topologia di
lavoro, moltiplicatori) ``producer.group_kind`` e poi ``producer.view_recipe`` (puri) o ``producer.aug_group_spec``
(mm_aug: ibridi, trasferimenti d'espressione, bump) ripetono le stesse estrazioni nello stesso ordine: identita', espressioni (FaMoS:
il fotogramma), discretizzazione e seme del rumore; ``views.discretize`` da' la mesh. I gruppi mm_aug hanno anche i
parametri espliciti nello header (``prov.mm_aug``: ``v3_work/mm_aug.assemble`` li ricostruisce senza generatore). Gli operatori NON si rigenerano bit per bit (eigsh), la mesh si'.
Con ``partial`` nella ricetta (producer.py --partial-p) la parte tolta si rigenera da ``partial_aug.apply`` col seme
del rumore della vista; ``--shard`` confronta anche i parametri realizzati (``partial`` dei metadati).
Le mesh sono nel frame canonico della libreria (mm a scala s_d); x ``mm_factor`` (targets.py) nei mm veri.

Con ``--shard`` si VERIFICA contro le viste dello shard: facce identiche e vertici serviti (centro e maxabs)
uguali entro 1e-6. ``--out``: un npz per vista (V, F, etichetta, espressione, mm_factor) e il json di provenienza.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import sources as S  # noqa: E402


_AUG: dict = {}


def aug_library(recipe: dict):
    """(AugLibrary, AugConfig) della ricetta di uno shard con --mm-aug (una volta per configurazione)."""
    from v3_work.mm_aug import aug as AG
    key = json.dumps(recipe["mm_aug_config"], sort_keys=True)
    if key not in _AUG:
        acfg = AG.config_from_dict(recipe["mm_aug_config"])
        _AUG[key] = (AG.AugLibrary(acfg), acfg)
    return _AUG[key]


def discretize(V, F, label: str, noise_seed: int, recipe: dict) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """views.discretize e, con ``partial`` nella ricetta, la parzialita' della vista (info dei metadati)."""
    import views as VW
    Vd, Fd = VW.discretize(V, F, label, noise_seed)
    if recipe.get("partial") is None:
        return Vd, Fd, None
    import partial_aug
    return partial_aug.apply(Vd, Fd, recipe["partial"], noise_seed)


def regenerate(src, seed, recipe: dict, domain: str = "") -> tuple[object, list[dict]]:
    """(identita', [{vi, V, F, label, expr, noise_seed, frame, partial}]) del gruppo col seme ``seed``: lo stesso
    ordine di estrazioni del produttore (tipo del gruppo con --mm-aug, poi view_recipe o aug_group_spec)."""
    import producer as PR
    cfg = argparse.Namespace(**{**recipe, "label_p": np.asarray(recipe["label_p"], dtype=np.float64)})
    cfg.mm_aug_probs = recipe.get("mm_aug_probs") or {}
    cfg.label_draw = recipe.get("label_draw", "replace")          # ricette di prima: con reinserimento
    rng = np.random.default_rng(np.random.SeedSequence([int(x) for x in seed]))
    d = domain or (src.name if src.kind == "mm" else "famos")
    kind = PR.group_kind(rng, cfg, d)
    out = []
    if kind != "pure":
        lib, acfg = aug_library(recipe)
        G = PR.aug_group_spec(lib, acfg, d, kind, rng, cfg)
        for v in G["views"]:
            Vd, Fd, pi = discretize(v["V"], v["F"], v["label"], v["noise_seed"], recipe)
            out.append({"vi": v["i"], "V": Vd, "F": Fd, "label": v["label"], "expr": v["tag"],
                        "noise_seed": v["noise_seed"], "frame": int((v["expression"] or {}).get("frame", -1)),
                        "kind": G["kind"], "partial": pi})
        return np.asarray(G["identity"]["z_A"], dtype=np.float64), out
    ident = src.identity(rng)
    for i, V, F, tag, label, noise_seed, frame in PR.view_recipe(src, ident, rng, cfg, d):
        Vd, Fd, pi = discretize(V, F, label, noise_seed, recipe)
        out.append({"vi": i, "V": Vd, "F": Fd, "label": label, "expr": tag, "noise_seed": noise_seed, "frame": frame,
                    "kind": "pure", "partial": pi})
    return ident, out


def served(V: np.ndarray) -> np.ndarray:
    """Vertici come li serve il loader (centro e maxabs), per il confronto con lo shard."""
    V = np.asarray(V, dtype=np.float64)
    V = V - V.mean(0)
    return V / np.abs(V).max()


def check_shard(path: Path, groups=None, srcs: dict | None = None) -> list[dict]:
    """Rigenera i gruppi dello shard e li confronta con le viste scritte."""
    from ring import ShardReader
    rd = ShardReader(path)
    rec = rd.head.get("recipe")
    if rec is None:
        raise SystemExit(f"{path}: shard senza ricetta (producer.py --provenance)")
    srcs = srcs if srcs is not None else {}
    res = []
    for g in (range(len(rd.groups)) if groups is None else groups):
        hg = rd.groups[g]
        d = hg["domain"]
        if d not in srcs:
            srcs[d] = S.build_sources([d], v_max=rec["v_max"], v_work=rec["v_work"])[d]
        ident, views = regenerate(srcs[d], hg["prov"]["seed"], rec, d)
        by_vi = {v["vi"]: v for v in views}
        row = {"key": hg["key"], "domain": d, "seed": hg["prov"]["seed"], "license": hg["prov"]["license"],
               "origin": hg.get("origin", "pure"), "views": []}
        z = rd.zid(g)
        if z is not None:
            row["zid_max_abs"] = float(np.abs(np.asarray(ident) - z).max())
        elif hg["prov"].get("person") is not None:
            row["person_ok"] = ident == hg["prov"]["person"]
        for vi, meta in enumerate(hg["views"]):
            r = by_vi[meta["vi"]]
            a = rd.view(g, vi)
            ok_f = np.array_equal(np.asarray(a["faces"]), r["F"])
            dv = float(np.abs(served(r["V"]) - np.asarray(a["verts"], dtype=np.float64)).max()) if ok_f else float("inf")
            row["views"].append({"vi": meta["vi"], "label": meta["label"], "expr": meta["expr"],
                                 "label_ok": r["label"] == meta["label"], "expr_ok": r["expr"] == meta["expr"],
                                 "partial_ok": r["partial"] == meta.get("partial"),
                                 "faces_equal": bool(ok_f), "verts_max_abs": dv})
        row["pass"] = bool(all(v["faces_equal"] and v["label_ok"] and v["expr_ok"] and v["partial_ok"]
                               and v["verts_max_abs"] < 1e-6 for v in row["views"]) and row.get("zid_max_abs", 0.0) == 0.0
                           and row.get("person_ok", True) and all(v["kind"] == row["origin"] for v in views))
        res.append(row)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", type=Path)
    ap.add_argument("--group", type=int, default=-1)
    ap.add_argument("--domain", default="")
    ap.add_argument("--seed", default="", help="seme,processo,gruppo")
    ap.add_argument("--recipe", type=Path, help="json della ricetta (lo header 'recipe' di uno shard)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    import views as VW
    VW.install_grad_vec()
    if a.shard:
        res = check_shard(a.shard, None if a.group < 0 else [a.group])
        print(json.dumps(res, indent=1, default=str))
        return
    rec = json.loads(a.recipe.read_text())
    src = S.build_sources([a.domain], v_max=rec["v_max"], v_work=rec["v_work"])[a.domain]
    seed = [int(x) for x in a.seed.split(",")]
    ident, views = regenerate(src, seed, rec, a.domain)
    if a.out:
        from targets import CanonTargets
        f = CanonTargets({a.domain: src}).mm_factor(a.domain, src)
        a.out.mkdir(parents=True, exist_ok=True)
        for v in views:
            np.savez_compressed(a.out / f"{a.domain}_{'-'.join(map(str, seed))}_v{v['vi']}.npz", V=v["V"], F=v["F"],
                                label=v["label"], expr=v["expr"], mm_factor=f)
        lic = S.inherited_license([a.domain])
        (a.out / "provenance.json").write_text(json.dumps({"domain": a.domain, "seed": seed, "recipe": rec,
                                                           "license": lic[0], "mm_factor": f}, indent=1) + "\n")
    print(f"[regen] {a.domain} {seed}: {len(views)} viste " + ", ".join(f"{v['label']}/{v['expr']}" for v in views))


if __name__ == "__main__":
    main()
