#!/usr/bin/env python3
"""Ablazione 2x2 dello stream (FLAME 2023 x parzialita' variabile): insiemi FLAME di D1 rigenerati e letture del
protocollo ``aau/runs/evidence/stream/ablation_2x2/PROTOCOL.md`` (scritto prima dei numeri, commit 27f4893).

    AAU_NV="" aau/run.sh v3_work/stream/eval_ablation.py flame-gen --workers 32       (slurm/ablation_2x2_flame.sbatch)
    v3_work/unified_gt/run.sh v3_work/stream/eval_ablation.py summary --workers 32    (slurm/ablation_2x2_summary.sbatch)

``flame-gen`` (sez. 4.4): gli insiemi ``flame2023_s1`` e ``flame2023`` di D1 col codice di D1 (``aau/diagnostics/
d1_gen.py``, importato in sola lettura, uscite reindirizzate): mesh grezze in ``datasets/ABL2X2_D1/in``; tabella di
scala, GT, ``gt_all_sr.npz`` e ``gen.json`` in ``ablation_2x2/d1flame``. Controllo (``d1flame/check.json``, uscita 1 se
non passa): nomi, d_FR, d_P e S identici a ``diagnostics/d1/gt_<insieme>.npz``, vertici per etichetta come
``d1/gen.json``. Gli operatori li calcola la sbatch.

``summary`` (sez. 4-5): per cella (A, B, C, D) il checkpoint ``runs/v3_*/checkpoints/epoch100_ema.pth``, c di
``eval/calib/calib.json`` (sha256 del checkpoint controllato), embedding [s, u] dei test (passo ``form``,
``eval/form_*``, uno per famiglia), della calibrazione (held-out) e, per A e C, di FLAME (``eval/flame``); ogni
embedding deve venire dal checkpoint della cella. Righe:
  * famiglie mai viste: ``nocrop`` = ``fact_paired.rows_for`` (HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse
    ``mesh_pair_nocrop``; seme del gruppo di E12); ``all_cross`` e righe col crop (``crop``) = ``bp_paired_e2.
    all_cross_rows`` (copiata), seme di ``all_cross``, crop su almeno un lato; ``nocrop_all`` = le righe senza crop di
    all_cross con le repliche di all_cross (solo per il crollo). Controllo: all_cross senza crop = rows_for (chiavi e GT);
  * held-out ``bfm``, ``ict``, ``gnm`` e FLAME di D1: ``diag.set_names`` / ``diag.rows``, GT ``d1_stats.gt_static`` /
    ``gt_new``, repliche ``diag.boot_counts`` (``d1_stats.boot``).
Distanze ``fact_paired.distances`` / ``diag.arm_distances`` con la c della cella: ``form_cal`` (letta con FR),
``shape`` = d_P (letta con SR), ``form`` = d_F non calibrata (descrittiva). Spearman per replica ``fact_paired._rep``
(righe ripetute c_a c_b volte). Maschera comune: GT e distanze delle quattro celle finite, soggetti diversi.
Effetti per (insieme, gruppo, GT): E_F = [(B - A) + (D - C)] / 2, E_P = [(C - A) + (D - B)] / 2, I = (D - C) - (B - A),
semplici B - A, D - C, C - A, D - B; crollo = rho(crop) - rho(nocrop_all) e i suoi effetti; medie sulle famiglie
(replica per replica). Letture R_F, R_P1, R_P2, R_I con le soglie del protocollo (``DELTA``).
Uscite in ``ablation_2x2/`` (``--out``): ``spearman.csv``, ``effects.csv``, ``readings.json``, ``controls.json``,
``results.md`` e ``reps.npz`` (repliche, fuori da git). Mai Ava-256, mai FaMoS TEST (nessuna ``famos_frame``).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
DIAG = REPO / "aau/diagnostics"                          # D1: diag, d1_gen, d1_stats (sola lettura)
ABL = REPO / "aau/runs/evidence/stream/ablation_2x2"
D1_RAW = REPO / "datasets/ABL2X2_D1/in"                  # mesh grezze FLAME rigenerate (fuori da git)
CELLS = ("A", "B", "C", "D")
FLAME_CELLS = ("A", "C")                                 # FLAME 2023 mai visto (sez. 4.4)
EPOCH = 100                                              # epoch100_ema.pth, 60.000 passi (sez. 3)
FAMILIES = ("hifi3d", "facescape", "faceverse")
FORM_DIR = {"hifi3d": "form_hifi", "facescape": "form_devfs", "faceverse": "form_fv_expr"}   # fact_paired.FORM_DIR
FV_STAGE = REPO / "aau/runs/ws_faceverse_expr/data_736f96956a/scale_e108_flip_topology"   # bp_paired_e2.FV_STAGE
SEEN = ("bfm", "ict", "gnm")
FLAME_SETS = ("flame2023_s1", "flame2023")
GROUPS = ("nocrop", "all_cross", "crop")
# soglie (sez. 5): 2.5 x la deviazione standard da seme attesa (sigma_run 0.011 / 0.018 / 0.042), al centesimo
DELTA = {"nocrop": {"M": 0.03, "S": 0.04, "I": 0.06}, "all_cross": {"M": 0.05, "S": 0.06, "I": 0.09},
         "crop": {"M": 0.10, "S": 0.15, "I": 0.21}}
PRIMARY = {"fr": "form_cal", "sr": "shape"}              # la distanza letta con ciascuna GT
DISTS = ("form_cal", "shape", "form")
EFFECTS = ("E_F", "E_P", "I", "B-A", "D-C", "C-A", "D-B")
KIND = {"E_F": "M", "E_P": "M", "I": "I", "B-A": "S", "D-C": "S", "C-A": "S", "D-B": "S"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def ci(v: np.ndarray) -> dict:
    """Stima puntuale (replica 0), IC 95% percentile e P(<= 0) sulle repliche 1.. finite (diag.ci)."""
    b = v[1:][np.isfinite(v[1:])]
    lo, hi = np.percentile(b, [2.5, 97.5]) if len(b) else (np.nan, np.nan)
    return {"point": float(v[0]), "ci_low": float(lo), "ci_high": float(hi),
            "p_le0": float((b <= 0).mean()) if len(b) else float("nan"), "n_boot": int(len(b))}


def d1_modules():
    """diag e d1_stats di D1, importati senza scrivere .pyc in aau/diagnostics."""
    sys.dont_write_bytecode = True
    if str(DIAG) not in sys.path:
        sys.path.insert(0, str(DIAG))
    import d1_stats
    import diag
    return diag, d1_stats


# ------------------------------------------------------------------------------------------ flame-gen

def flame_gen(workers: int) -> None:
    """Gli insiemi FLAME di D1 rigenerati con d1_gen (diag.RAW_IN e diag.D1 reindirizzati) e il controllo."""
    diag, _ = d1_modules()
    import d1_gen
    ref = diag.D1
    out = ABL / "d1flame"
    out.mkdir(parents=True, exist_ok=True)
    diag.RAW_IN, diag.D1 = D1_RAW, out              # d1_gen li legge a ogni scrittura (anche nei worker, fork)
    sys.argv = [str(DIAG / "d1_gen.py"), "--workers", str(workers), "--sets", ",".join(FLAME_SETS)]
    d1_gen.main()
    gref = json.loads((ref / "gen.json").read_text())["sets"]
    gnew = json.loads((out / "gen.json").read_text())["sets"]
    chk = {"definition": "eval_ablation.py flame-gen: GT e vertici per etichetta degli insiemi rigenerati contro D1",
           "reference": str(ref), "raw_dir": str(D1_RAW), "sets": {}}
    for s in FLAME_SETS:
        with np.load(ref / f"gt_{s}.npz", allow_pickle=True) as a, np.load(out / f"gt_{s}.npz", allow_pickle=True) as b:
            names = [str(x) for x in a["names"]] == [str(x) for x in b["names"]]
            diff = {k: float(np.max(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64))
                                    / np.maximum(np.abs(np.asarray(a[k], np.float64)), 1e-12)))
                    for k in ("D_fr", "D_sr", "S")} if names else {}
        verts = gref[s]["verts_by_label"] == gnew[s]["verts_by_label"]
        chk["sets"][s] = {"names_equal": names, "max_rel_diff": diff, "verts_by_label_equal": verts,
                          "verts_by_label": gnew[s]["verts_by_label"], "verts_by_label_d1": gref[s]["verts_by_label"],
                          "pass": bool(names and max(diff.values()) <= 1e-6 and verts)}
    chk["pass"] = all(v["pass"] for v in chk["sets"].values())
    atomic_text(out / "check.json", json.dumps(chk, indent=1) + "\n")
    print(f"[abl-flame] controllo contro D1: {json.dumps(chk['sets'])}", flush=True)
    if not chk["pass"]:
        raise SystemExit(f"[abl-flame] ERRORE: insiemi FLAME diversi da D1 ({out / 'check.json'}): FLAME non si valuta")


# ------------------------------------------------------------------------------------------ celle

def load_cells(root: Path, epoch: int) -> dict:
    """Checkpoint, c (calib.json, sha256 controllato) ed embedding di ogni cella."""
    out = {}
    for x in CELLS:
        ck = sorted((root / x / "runs").glob(f"v3_*/checkpoints/epoch{epoch:03d}_ema.pth"))
        if len(ck) != 1:
            raise SystemExit(f"cella {x}: {len(ck)} checkpoint epoch{epoch:03d}_ema.pth in {root / x / 'runs'}")
        ck = ck[0].resolve()
        cal = json.loads((root / x / "eval/calib/calib.json").read_text())
        sha = sha256(ck)
        if cal["sha256_checkpoint"] != sha or Path(cal["checkpoint"]).resolve() != ck:
            raise SystemExit(f"cella {x}: calib.json di {cal['checkpoint']} ({cal['sha256_checkpoint'][:12]}), "
                             f"atteso {ck} ({sha[:12]})")
        emb = {"heldout": root / x / "eval/calib/embeddings.npz"}
        for f in FAMILIES:
            hits = sorted((root / x / "eval" / FORM_DIR[f]).glob(f"data_*/scale_v3*fulle{epoch:03d}*/zs_zeroshot/embeddings.npz"))
            if len(hits) != 1:
                raise SystemExit(f"cella {x}, {f}: {len(hits)} embedding in {root / x / 'eval' / FORM_DIR[f]}")
            emb[f] = hits[0]
        if x in FLAME_CELLS:
            emb["flame"] = root / x / "eval/flame/embeddings.npz"
        for k, p in emb.items():
            with np.load(p, allow_pickle=True) as z:
                e = Path(str(z["checkpoint"])).resolve()
            if e != ck:
                raise SystemExit(f"cella {x}, {k}: embedding di {e}, atteso {ck}")
        out[x] = {"ckpt": ck, "sha256": sha, "c": float(cal["c_median"]), "c_ls": float(cal["c_ls"]),
                  "c_by_domain": {d: cal.get(f"c_median_{d}") for d in SEEN}, "emb": emb,
                  "sha256_embeddings": {k: sha256(p) for k, p in emb.items()}}
    return out


# ------------------------------------------------------------------------------------------ famiglie di test

def all_cross_rows(view: str, fp, be):
    """bp_paired_e2.all_cross_rows (copiata): (df con GT e taglia oracolo, idx, seme) del gruppo all_cross."""
    if view == "hifi3d":
        fr = be.e12m.hifi_frames()[0]
        df, seed = fr["all_cross"][0], int(fr["all_cross"][2])
    elif view == "facescape":
        fr = be.e12m.fs_frames()
        df, seed = fr["all_cross"][0], int(fr["all_cross"][2])
    else:
        pm = be.base.read_pair_metrics(FV_STAGE / be.zsum.STAGE)
        df = pm.assign(subject_a=pm["subject_a"].astype(str), subject_b=pm["subject_b"].astype(str))
        seed = int(be.base.stable_seed(1234, "expr_sec", "scale_e108@bfm", "all_cross", "latent_distance", "raw_chamfer"))
    df = df[be.zsum.PAIR_KEYS + ["gt_distance"]].assign(subject_a=lambda d: d["subject_a"].astype(str),
                                                         subject_b=lambda d: d["subject_b"].astype(str))
    idx = be.zes.Index(be.blmm.subjects(view))
    orc = fp.oracle_S(be.blmm.VIEWS[view]["gt"])
    D = {"oracle_size": be.log_abs([orc[s] for s, _ in idx.keys])}
    df = be.add_gts(be.add_columns(df, D, idx), be.GT_SET[view]).reset_index(drop=True)
    return df, idx, seed


def check_rows(df, ref) -> dict:
    """bp_paired_e2.check_rows (copiata): tolto il crop, chiavi e GT = fact_paired.rows_for."""
    nc = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    key = ["subject_a", "topology_a", "subject_b", "topology_b"]
    a = nc[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    b = ref[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    same = len(a) == len(b) and bool((a[key].to_numpy() == b[key].to_numpy()).all())
    diff = float(max(np.nanmax(np.abs(a[g].to_numpy() - b[g].to_numpy())) for g in ("gt_fr", "gt_sr"))) if same else None
    return {"rows_nocrop": int(len(a)), "rows_fact_paired": int(len(b)), "keys_equal": same, "max_abs_diff_gt": diff,
            "pass": bool(same and diff is not None and diff <= 1e-9)}


def cell_columns(view: str, idx, cells: dict, fp) -> dict:
    """{'<cella>|<distanza>': D (n, n) sulle chiavi di idx}: fact_paired.distances con la c della cella."""
    out = {}
    for x, info in cells.items():
        with np.load(info["emb"][view], allow_pickle=True) as z:
            Z = np.asarray(z["Z"], np.float64)
            keys = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
        pos = {k: r for r, k in enumerate(keys)}
        Z = Z[[pos[k] for k in idx.keys]]
        n = len(Z)
        i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
        for k, d in fp.distances(Z, i, j, info["ckpt"], (info["c"], info["c_ls"])).items():
            if k in DISTS:
                out[f"{x}|{k}"] = d.reshape(n, n)
    return out


def replicates(df, methods: list, seed: int, n_boot: int, workers: int, fp, sub=None) -> tuple[dict, dict]:
    """{(gt, misura, metodo): (1 + n_boot,)} con fact_paired._rep; conteggi come fact_paired.main sui soggetti di df
    (``sub``: sottoinsieme delle righe, stesse repliche)."""
    cols = {m: df[m].to_numpy(np.float64) for m in methods + ["oracle_size"]}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    mask = (sa != sb) & np.all([np.isfinite(v) for v in cols.values()], axis=0)
    if sub is not None:
        mask &= sub
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    o = cols["oracle_size"]
    fp._J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
             "methods": methods, "low": o[mask] <= np.quantile(o[mask], 0.2)}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(fp._rep, range(len(counts)), chunksize=4)
    V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
    return V, {"seed": int(seed), "n_rows": int(mask.sum()), "n_rows_out": int(((sa != sb) & (sub if sub is not None
               else True)).sum() - mask.sum()), "n_subjects": int(len(subjects))}


def families(cells: dict, n_boot: int, workers: int) -> tuple[dict, dict]:
    """{(famiglia, gruppo): {(gt, cella|distanza): repliche di rho}} e i controlli."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "v3_work/trainer/tools"))
    import fact_paired as fp       # sola lettura; importa blmm_eval (righe, GT e semi di E12)
    be = fp.be
    V, ctrl = {}, {}
    for view in FAMILIES:
        t0 = time.time()
        ref, _, seed_nc = fp.rows_for(view)
        df, idx, seed_all = all_cross_rows(view, fp, be)
        chk = check_rows(df, ref)
        if not chk["pass"]:
            raise SystemExit(f"{view}: righe di all_cross senza crop diverse da fact_paired.rows_for: {chk}")
        D = cell_columns(view, idx, cells, fp)
        M = sorted(D)
        nc, al = be.add_columns(ref, D, idx), be.add_columns(df, D, idx)
        crop = (al["topology_a"].eq("crop") | al["topology_b"].eq("crop")).to_numpy()
        ctrl[view] = {"rows_check": chk, "groups": {}}
        for g, frame, seed, sub in (("nocrop", nc, seed_nc, None), ("all_cross", al, seed_all, None),
                                    ("crop", al, seed_all, crop), ("nocrop_all", al, seed_all, ~crop)):
            reps, info = replicates(frame, M, seed, n_boot, workers, fp, sub)
            V[(view, g)] = {(gt, m): v for (gt, kind, m), v in reps.items() if kind == "rho"}
            ctrl[view]["groups"][g] = info
        print(f"[abl-summary] {view}: righe {json.dumps(ctrl[view]['groups'])}, {time.time() - t0:.0f}s", flush=True)
    return V, ctrl


# ------------------------------------------------------------------------------------------ held-out e FLAME (D1)

def d1_sets(cells: dict, workers: int) -> tuple[dict, dict]:
    """{(insieme, 'd1'): {(gt, cella|distanza): repliche}} per gli held-out visti (4 celle) e FLAME (A e C)."""
    diag, d1_stats = d1_modules()
    V, ctrl = {}, {}
    for name in SEEN + FLAME_SETS:
        t0 = time.time()
        who = CELLS if name in SEEN else FLAME_CELLS
        names = diag.set_names(name)
        i, j = diag.rows(names)
        subj_of, subj = diag.subject_index(names)
        G = d1_stats.gt_static(name, names) if diag.SETS[name]["static"] else d1_stats.gt_new(name, names)
        cols = {f"gt_{k}": G[k][i, j] for k in ("sr", "fr")}
        for x in who:
            Z, ck = diag.load_embeddings(cells[x]["emb"]["heldout" if name in SEEN else "flame"], names)
            if ck.resolve() != cells[x]["ckpt"]:
                raise SystemExit(f"{name}, cella {x}: embedding di {ck}")
            for k, v in diag.arm_distances(Z, i, j, diag.dp_per_unit(ck), cells[x]["c"]).items():
                cols[f"{x}|{k}"] = v
        mask = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        reps = d1_stats.boot(name, {k: v[mask] for k, v in cols.items()}, subj_of[i][mask], subj_of[j][mask],
                             len(subj), workers)
        V[(name, "d1")] = reps
        ctrl[name] = {"n_meshes": len(names), "n_rows": int(mask.sum()), "n_rows_out": int((~mask).sum()),
                      "n_subjects": len(subj), "boot": f"diag.boot_counts({len(subj)}, {diag.SETS[name]['group']})",
                      "cells": list(who)}
        print(f"[abl-summary] {name}: {ctrl[name]}, {time.time() - t0:.0f}s", flush=True)
    return V, ctrl


# ------------------------------------------------------------------------------------------ effetti e letture

def effects_of(R: dict) -> dict:
    """Effetti dalle repliche di rho delle quattro celle (stesse righe e repliche)."""
    return {"E_F": ((R["B"] - R["A"]) + (R["D"] - R["C"])) / 2, "E_P": ((R["C"] - R["A"]) + (R["D"] - R["B"])) / 2,
            "I": (R["D"] - R["C"]) - (R["B"] - R["A"]), "B-A": R["B"] - R["A"], "D-C": R["D"] - R["C"],
            "C-A": R["C"] - R["A"], "D-B": R["D"] - R["B"]}


def flag(e: dict, delta: float) -> str:
    if e["point"] >= delta and e["ci_low"] > 0:
        return "beneficio"
    if e["point"] <= -delta and e["ci_high"] < 0:
        return "danno"
    return "-"


def verdict(cells: list, delta: float) -> dict:
    """Sez. 5.1: SI' (beneficio in >= 3 celle su 6, in >= 2 famiglie, nessun danno), NO (nessun beneficio), PARZIALE."""
    ben = [c for c in cells if c["point"] >= delta and c["ci_low"] > 0]
    dan = [c for c in cells if c["point"] <= -delta and c["ci_high"] < 0]
    v = "SI'" if len(ben) >= 3 and len({c["set"] for c in ben}) >= 2 and not dan else ("NO" if not ben else "PARZIALE")
    tag = lambda c: f"{c['set']} {c['gt']} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"  # noqa: E731
    return {"verdict": v, "delta": delta, "benefit": [tag(c) for c in ben], "harm": [tag(c) for c in dan],
            "cells": [tag(c) for c in cells]}


def summarize_effects(V: dict) -> tuple[list, list, dict]:
    """Righe di spearman.csv e di effects.csv; repliche degli effetti per le letture."""
    sp, ef, E = [], [], {}
    for (st, g), reps in V.items():
        if g == "nocrop_all":          # stesse righe di nocrop con le repliche di all_cross: solo per il crollo
            continue
        who = sorted({m.split("|")[0] for _, m in reps})
        for gt in ("fr", "sr"):
            for dist in DISTS:
                if dist == "form" and gt == "sr":
                    continue
                R = {x: reps[(gt, f"{x}|{dist}")] for x in who if (gt, f"{x}|{dist}") in reps}
                for x, v in R.items():
                    sp.append({"set": st, "group": g, "gt": gt, "distance": dist, "cell": x, **ci(v)})
                if len(R) == 4:
                    for k, v in effects_of(R).items():
                        E[(st, g, gt, dist, k)] = v
                elif set(R) == set(FLAME_CELLS):
                    E[(st, g, gt, dist, "C-A")] = R["C"] - R["A"]
    # crollo = rho(righe col crop) - rho(righe senza crop di all_cross), stesse repliche (seme di all_cross)
    for view in FAMILIES:
        if (view, "crop") not in V:
            continue
        for gt in ("fr", "sr"):
            dist = PRIMARY[gt]
            R = {x: V[(view, "crop")][(gt, f"{x}|{dist}")] - V[(view, "nocrop_all")][(gt, f"{x}|{dist}")] for x in CELLS}
            for x, v in R.items():
                sp.append({"set": view, "group": "crollo", "gt": gt, "distance": dist, "cell": x, **ci(v)})
            for k, v in effects_of(R).items():
                E[(view, "crollo", gt, dist, k)] = v
    # medie sulle famiglie (stime indipendenti, replica per replica)
    for g in GROUPS + ("crollo",):
        for gt in ("fr", "sr"):
            for dist in DISTS:
                for k in EFFECTS:
                    vs = [E[(f, g, gt, dist, k)] for f in FAMILIES if (f, g, gt, dist, k) in E]
                    if len(vs) == len(FAMILIES):
                        E[("media famiglie", g, gt, dist, k)] = np.mean(vs, axis=0)
    for (st, g, gt, dist, k), v in E.items():
        # soglie del gruppo; il crollo con quelle delle righe col crop; held-out e FLAME (d1) con quelle di nocrop
        d = DELTA["crop" if g == "crollo" else (g if g in DELTA else "nocrop")][KIND[k]]
        r = {"set": st, "group": g, "gt": gt, "distance": dist, "effect": k, **ci(v), "threshold": d}
        r["flag"] = flag(r, d)
        ef.append(r)
    return sp, ef, E


def readings(ef: list) -> dict:
    """R_F, R_P1, R_P2, R_I (sez. 5)."""
    get = {(r["set"], r["group"], r["gt"], r["distance"], r["effect"]): r for r in ef}

    def cells_of(group: str, effect: str) -> list:
        return [get[(f, group, gt, PRIMARY[gt], effect)] for f in FAMILIES for gt in ("fr", "sr")
                if (f, group, gt, PRIMARY[gt], effect) in get]

    out = {"R_F": verdict(cells_of("nocrop", "E_F"), DELTA["nocrop"]["M"]),
           "R_P1": verdict(cells_of("crop", "E_P"), DELTA["crop"]["M"])}
    nc = cells_of("nocrop", "E_P")
    d = DELTA["nocrop"]["M"]
    harm = [c for c in nc if c["point"] <= -d and c["ci_high"] < 0]
    small = [c for c in nc if c["ci_high"] < 0 and c["point"] > -d]
    tag = lambda c: f"{c['set']} {c['gt']} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"  # noqa: E731
    out["R_P2"] = {"verdict": "costa" if harm else "non costa", "delta": d, "harm": [tag(c) for c in harm],
                   "below_threshold": [tag(c) for c in small], "cells": [tag(c) for c in nc]}
    p1, p2 = out["R_P1"]["verdict"], out["R_P2"]["verdict"]
    out["R_P"] = ("riduce il crollo senza costo" if p1 == "SI'" and p2 == "non costa" else
                  f"R_P1 {p1}, R_P2 {p2}")
    inter = []
    for group in ("nocrop", "crop"):
        dI, dS = DELTA[group]["I"], DELTA[group]["S"]
        for c in cells_of(group, "I"):
            if abs(c["point"]) >= dI and (c["ci_low"] > 0 or c["ci_high"] < 0):
                simple = {k: get[(c["set"], group, c["gt"], c["distance"], k)] for k in ("B-A", "D-C", "C-A", "D-B")}
                inter.append({"group": group, "set": c["set"], "gt": c["gt"], "I": tag(c), "delta_I": dI,
                              "simple": {k: f"{s['point']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}] "
                                            f"{flag(s, dS)}" for k, s in simple.items()}, "delta_S": dS})
    out["R_I"] = {"cells": inter, "note": "nessuna interazione da leggere" if not inter else
                  "effetti principali da leggere per strato nelle celle elencate"}
    return out


# ------------------------------------------------------------------------------------------ training

def training_stats(root: Path) -> dict:
    """summary.json di summarize_run.py (fine del run) e producers.json di ogni cella; differenze > 20% segnalate."""
    out = {}
    for x in CELLS:
        s = root / x / "node0" / "summary.json"
        p = root / x / "node0" / "producers.json"
        if not s.exists():
            continue
        d = json.loads(s.read_text())
        sp = d.get("s_per_step_by_epoch") or [float("nan")]
        prod = json.loads(p.read_text()) if p.exists() else {}
        out[x] = {"steps": d.get("steps"), "s_per_step_mean": float(np.mean(sp)), "s_per_step_last": sp[-1],
                  "fresh_views_per_s": d.get("producers", {}).get("views_per_s_steady"),
                  "reuse_steady": d.get("reuse", {}).get("steady_last_epochs"),
                  "reuse_cumulative": d.get("reuse", {}).get("cumulative"),
                  "identities_produced_last_segment": prod.get("groups"), "loss_last": (d.get("loss_by_epoch") or [None])[-1],
                  "gpu_util": d.get("gpu_util_mean_while_training"), "batches_by_domain": d.get("batches_by_domain"),
                  "partial_fraction": prod.get("partial_fraction"), "expr_fraction_by_domain": prod.get("expr_fraction_by_domain")}
    flags = []
    for a, b in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B")):
        for k in ("fresh_views_per_s", "reuse_steady"):
            va, vb = (out.get(a) or {}).get(k), (out.get(b) or {}).get(k)
            if va and vb and abs(va / vb - 1) > 0.20:
                flags.append(f"{k} {a} {va:.2f} contro {b} {vb:.2f} ({100 * (va / vb - 1):+.0f}%)")
    return {"cells": out, "flags_20pct": flags}


# ------------------------------------------------------------------------------------------ uscite

def fmt(r: dict) -> str:
    return f"{r['point']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"


def fmt_rho(r: dict) -> str:
    return f"{r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def results_md(out: Path, sp: list, ef: list, rd: dict, cells: dict, ctrl: dict, tr: dict) -> None:
    S = {(r["set"], r["group"], r["gt"], r["distance"], r["cell"]): r for r in sp}
    F = {(r["set"], r["group"], r["gt"], r["distance"], r["effect"]): r for r in ef}
    prot = (ABL / "PROTOCOL.sha256").read_text().split()[0] if (ABL / "PROTOCOL.sha256").exists() else "?"
    md = ["# Ablazione 2x2 dello stream: risultati", "",
          f"Protocollo `PROTOCOL.md` (sha256 {prot[:12]}..., commit 27f4893), scritto prima dei numeri. Generato da "
          "`v3_work/stream/eval_ablation.py summary`; numeri in `spearman.csv`, `effects.csv`, letture in `readings.json`, "
          "controlli in `controls.json`. Un seme per cella: gli IC coprono i soggetti di test, non la variabilita' da run "
          "a run (soglie della sez. 5 dal rumore fra semi di C3F).", "",
          "## Letture (sez. 5)", "",
          f"- **R_F, effetto FLAME (nocrop, delta {rd['R_F']['delta']:.2f}): {rd['R_F']['verdict']}**. Benefici: "
          f"{'; '.join(rd['R_F']['benefit']) or 'nessuno'}. Danni: {'; '.join(rd['R_F']['harm']) or 'nessuno'}.",
          f"- **R_P1, parzialita' sulle righe col crop (delta {rd['R_P1']['delta']:.2f}): {rd['R_P1']['verdict']}**. "
          f"Benefici: {'; '.join(rd['R_P1']['benefit']) or 'nessuno'}. Danni: {'; '.join(rd['R_P1']['harm']) or 'nessuno'}.",
          f"- **R_P2, costo senza crop (delta {rd['R_P2']['delta']:.2f}): {rd['R_P2']['verdict']}**. Danni: "
          f"{'; '.join(rd['R_P2']['harm']) or 'nessuno'}; sotto soglia: {'; '.join(rd['R_P2']['below_threshold']) or 'nessuno'}.",
          f"- **R_P: {rd['R_P']}**.",
          f"- **R_I: {rd['R_I']['note']}**" + ("." if not rd["R_I"]["cells"] else ": " + "; ".join(
              f"{c['group']} {c['set']} {c['gt']} I {c['I']} (semplici: " + ", ".join(f"{k} {v}" for k, v in c["simple"].items()) + ")"
              for c in rd["R_I"]["cells"])), ""]
    md += ["## Celle", "", "| cella | checkpoint (run dir ...__hash) | sha256 | c | c_LS | c bfm / ict / gnm |",
           "|---|---|---|---|---|---|"]
    for x, i in cells.items():
        cb = " / ".join(f"{i['c_by_domain'][d]:.3f}" for d in SEEN)
        md.append(f"| {x} | `...__{i['ckpt'].parents[1].name.split('__')[-1]}/{i['ckpt'].name}` | {i['sha256'][:12]} | "
                  f"{i['c']:.4f} | {i['c_ls']:.4f} | {cb} |")
    if tr["cells"]:
        md += ["", "## Training (descrittivo)", "", "| cella | passi | s/passo medio | viste fresche/s | riuso a regime | "
               "riuso cumulato | identita' prodotte (ultimo segmento) | GPU % | quota parziale |", "|---|---|---|---|---|---|---|---|---|"]
        for x, t in tr["cells"].items():
            md.append(f"| {x} | {t['steps']} | {t['s_per_step_mean']:.3f} | {t['fresh_views_per_s'] or float('nan'):.2f} | "
                      f"{t['reuse_steady'] or float('nan'):.1f} | {t['reuse_cumulative'] or float('nan'):.1f} | "
                      f"{t['identities_produced_last_segment']} | {t['gpu_util'] or float('nan'):.0f} | "
                      f"{t['partial_fraction'] if t['partial_fraction'] is not None else '-'} |")
        md += ["", f"Differenze oltre il 20% fra celle confrontate: {'; '.join(tr['flags_20pct']) or 'nessuna'}."]
    for title, gt in (("GT FR, d_F calibrata", "fr"), ("GT SR, d_P", "sr")):
        dist = PRIMARY[gt]
        md += ["", f"## Famiglie mai viste, {title}", "", "| famiglia | gruppo | A | B | C | D | E_F | E_P | I |",
               "|---|---|---|---|---|---|---|---|---|"]
        for f in FAMILIES + ("media famiglie",):
            for g in GROUPS + ("crollo",):
                if (f, g, gt, dist, "E_F") not in F:
                    continue
                rho = [fmt_rho(S[(f, g, gt, dist, x)]) if (f, g, gt, dist, x) in S else "-" for x in CELLS]
                eff = [f"{fmt(F[(f, g, gt, dist, k)])} {F[(f, g, gt, dist, k)]['flag'] if F[(f, g, gt, dist, k)]['flag'] != '-' else ''}".strip()
                       for k in ("E_F", "E_P", "I")]
                md.append(f"| {f} | {g} | " + " | ".join(rho) + " | " + " | ".join(eff) + " |")
        md += ["", "Effetti semplici:", "", "| famiglia | gruppo | B - A | D - C | C - A | D - B |", "|---|---|---|---|---|---|"]
        for f in FAMILIES + ("media famiglie",):
            for g in GROUPS + ("crollo",):
                if (f, g, gt, dist, "B-A") in F:
                    md.append(f"| {f} | {g} | " + " | ".join(fmt(F[(f, g, gt, dist, k)]) for k in ("B-A", "D-C", "C-A", "D-B")) + " |")
    md += ["", "## Famiglie mai viste, GT FR con d_F non calibrata (descrittiva)", "",
           "| famiglia | gruppo | A | B | C | D | E_F | E_P | I |", "|---|---|---|---|---|---|---|---|---|"]
    for f in FAMILIES:
        for g in GROUPS:
            if (f, g, "fr", "form", "E_F") in F:
                md.append(f"| {f} | {g} | " + " | ".join(fmt_rho(S[(f, g, 'fr', 'form', x)]) for x in CELLS) + " | "
                          + " | ".join(fmt(F[(f, g, "fr", "form", k)]) for k in ("E_F", "E_P", "I")) + " |")
    md += ["", "## Held-out sintetici (in distribuzione; si legge SR, FR circolare per c)", "",
           "| insieme | GT | A | B | C | D | E_F | E_P | I |", "|---|---|---|---|---|---|---|---|---|"]
    for st in SEEN:
        for gt in ("sr", "fr"):
            dist = PRIMARY[gt]
            if (st, "d1", gt, dist, "E_F") in F:
                md.append(f"| {st} | {gt} | " + " | ".join(fmt_rho(S[(st, 'd1', gt, dist, x)]) for x in CELLS) + " | "
                          + " | ".join(f"{fmt(F[(st, 'd1', gt, dist, k)])} {F[(st, 'd1', gt, dist, k)]['flag'] if F[(st, 'd1', gt, dist, k)]['flag'] != '-' else ''}".strip()
                                       for k in ("E_F", "E_P", "I")) + " |")
    md += ["", "## FLAME 2023 di D1 (solo A e C, mai visto)", "", "| insieme | GT | A | C | C - A |", "|---|---|---|---|---|"]
    for st in FLAME_SETS:
        for gt in ("sr", "fr"):
            dist = PRIMARY[gt]
            if (st, "d1", gt, dist, "C-A") in F:
                md.append(f"| {st} | {gt} | {fmt_rho(S[(st, 'd1', gt, dist, 'A')])} | {fmt_rho(S[(st, 'd1', gt, dist, 'C')])} | "
                          f"{fmt(F[(st, 'd1', gt, dist, 'C-A')])} |")
    md += ["", "## Controlli (tutti in `controls.json`)", ""]
    for f, c in ctrl["families"].items():
        rc = c["rows_check"]
        md.append(f"- {f}: all_cross senza crop = `fact_paired.rows_for` {rc['pass']} ({rc['rows_nocrop']} righe, scarto GT "
                  f"{rc['max_abs_diff_gt']}); " + "; ".join(f"{g} {i['n_rows']} righe (fuori maschera {i['n_rows_out']}), "
                                                           f"{i['n_subjects']} soggetti, seme {i['seed']}"
                                                           for g, i in c["groups"].items()))
    for s, c in ctrl.get("d1", {}).items():
        md.append(f"- {s}: {c['n_rows']} righe (fuori maschera {c['n_rows_out']}), {c['n_subjects']} soggetti, "
                  f"{c['boot']}, celle {', '.join(c['cells'])}")
    md.append(f"- repliche bootstrap: {ctrl['n_boot']}; checkpoint all'epoca {ctrl['epoch']}; sha256 di checkpoint ed "
              "embedding per cella in `controls.json`")
    atomic_text(out / "results.md", "\n".join(md) + "\n")


def summary(root: Path, out: Path, epoch: int, n_boot: int, workers: int, skip_d1: bool) -> None:
    t0 = time.time()
    cells = load_cells(root, epoch)
    print(f"[abl-summary] celle: " + "; ".join(f"{x} c {i['c']:.4f} {i['ckpt']}" for x, i in cells.items()), flush=True)
    V, ctrl_f = families(cells, n_boot, workers)
    ctrl = {"families": ctrl_f}
    if not skip_d1:
        V1, ctrl["d1"] = d1_sets(cells, workers)
        V.update(V1)
    sp, ef, E = summarize_effects(V)
    rd = readings(ef)
    tr = training_stats(root)
    ctrl["cells"] = {x: {"checkpoint": str(i["ckpt"]), "sha256_checkpoint": i["sha256"], "c": i["c"], "c_ls": i["c_ls"],
                         "c_by_domain": i["c_by_domain"], "embeddings": {k: str(p) for k, p in i["emb"].items()},
                         "sha256_embeddings": i["sha256_embeddings"]} for x, i in cells.items()}
    ctrl["n_boot"], ctrl["epoch"], ctrl["root"] = n_boot, epoch, str(root)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "spearman.csv", sp)
    write_csv(out / "effects.csv", ef)
    atomic_text(out / "readings.json", json.dumps({**rd, "training": tr}, indent=1) + "\n")
    atomic_text(out / "controls.json", json.dumps(ctrl, indent=1, default=str) + "\n")
    tmp = out / f".reps.{os.getpid()}.tmp.npz"
    np.savez(tmp, **{"|".join(map(str, k)): v for k, v in E.items()})
    os.replace(tmp, out / "reps.npz")
    results_md(out, sp, ef, rd, cells, ctrl, tr)
    print(f"[abl-summary] R_F {rd['R_F']['verdict']}, R_P1 {rd['R_P1']['verdict']}, R_P2 {rd['R_P2']['verdict']}, "
          f"R_I {len(rd['R_I']['cells'])} celle; {time.time() - t0:.0f}s -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("flame-gen")
    g.add_argument("--workers", type=int, default=32)
    s = sub.add_parser("summary")
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--root", type=Path, default=ABL, help="directory delle celle A, B, C, D (prova: un'altra)")
    s.add_argument("--out", type=Path, default=None, help="uscite (default: --root)")
    s.add_argument("--epoch", type=int, default=EPOCH)
    s.add_argument("--skip-d1", action="store_true", help="solo famiglie di test (prova)")
    a = ap.parse_args()
    if a.cmd == "flame-gen":
        flame_gen(a.workers)
    else:
        summary(a.root.resolve(), (a.out or a.root).resolve(), a.epoch, a.n_boot, a.workers, a.skip_d1)


if __name__ == "__main__":
    main()
