#!/usr/bin/env python3
"""Ablazione 2x2 dello stream, letture dell'emendamento 1 (``aau/runs/evidence/stream/ablation_2x2/
PROTOCOL_emendamento_1.md``, scritto prima dei numeri): le righe, le repliche e le distanze di ``eval_ablation.py``
(importato in sola lettura) con le celle in piu' A' (secondo seme) e B' (FLAME suddiviso), le soglie per cella da A',
le regole con conferma su due celle, il controllo che B, D e B' abbiano imparato FLAME, le righe con down8k, il riuso
simmetrico, c al 10% del run e il codice d'analisi congelato.

    v3_work/unified_gt/run.sh v3_work/stream/eval_ablation_em1.py --workers 32   (slurm/ablation_2x2_summary_em1.sbatch)

Codice congelato (sez. 7 dell'emendamento): ``analysis_freeze.json`` (commit e sha256 di eval_ablation.py, di questo
file, fact_paired.py, blmm_eval.py, diag.py, d1_stats.py); prima di ogni calcolo lo sha256 dei file e dei moduli
importati deve coincidere, altrimenti ci si ferma (``--no-freeze``: solo prove, uscite marcate). Commit e stato git
(scritti fuori dal container dalla sbatch, ``--code-state``) in ``controls.json``.

Celle: A, B, C, D obbligatorie; A' (``Ap``) e B' (``Bp``) se il loro checkpoint e la loro valutazione ci sono, altrimenti
le letture che le usano sono "provvisorie" (A': soglie della sola tabella) o "in attesa" (B'), e il riepilogo va
rieseguito quando ci sono. FLAME 2023 di D1 per ogni cella con ``eval/flame/embeddings.npz``.
Gruppi delle famiglie: ``nocrop`` (fact_paired.rows_for), ``nocrop_down8k`` / ``nocrop_altre`` (righe nocrop con
down8k su almeno un lato / le altre, stesse repliche: descrittive), ``all_cross``, ``crop``, ``nocrop_all`` (solo per il
crollo); ``down8k_meno_altre`` = effetto sulle righe con down8k meno effetto sulle altre, replica per replica.
Effetti: E_F, E_P, I e semplici (eval_ablation.effects_of) con A-D; ``Bp-A``, ``Bp-B``, ``Ap-A``; sugli insiemi
FLAME ogni cella contro A e D - C. Soglia per (insieme, gruppo, GT, distanza) = max(soglia della tabella, 2.5 x m x
|rho_A' - rho_A| / sqrt(2)), m = 1 (principale), sqrt(2) (semplice e contrasti), 2 (interazione).
Uscite in ``ablation_2x2/em1`` (``--out``): spearman.csv, effects.csv, readings.json, controls.json, results.md, reps.npz.
Mai Ava-256, mai FaMoS TEST.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
sys.dont_write_bytecode = True                           # niente .pyc in aau/diagnostics
sys.path.insert(0, str(THIS))

import eval_ablation as EA  # noqa: E402  (sola lettura: righe, repliche, distanze, soglie della tabella)

ABL = EA.ABL
FREEZE = ABL / "analysis_freeze.json"                    # commit e sha256 del codice d'analisi (emendamento 1, sez. 7)
MAIN = ("A", "B", "C", "D")
EXTRA = ("Ap", "Bp")                                     # A' = A col seme 2345, B' = B con FLAME suddiviso 1-a-4
LABEL = {"Ap": "A'", "Bp": "B'"}
MULT = {"M": 1.0, "S": float(np.sqrt(2.0)), "I": 2.0}    # deviazione da seme per tipo di effetto, in unita' di sigma
KIND = {**EA.KIND, "Bp-A": "S", "Bp-B": "S", "Ap-A": "S"}
LN12 = float(np.log(1.2))                                # riuso: |ln(va / vb)| > ln 1.2 (sez. 3 dell'emendamento)
GROUPS = ("nocrop", "nocrop_down8k", "nocrop_altre", "all_cross", "crop")


def lab(x: str) -> str:
    return LABEL.get(x, x)


# ------------------------------------------------------------------------------------------ codice congelato

def check_freeze(required: bool) -> dict:
    """sha256 dei file di analysis_freeze.json contro i file e i moduli importati; SystemExit se diversi."""
    fz = json.loads(FREEZE.read_text()) if FREEZE.exists() else None
    if fz is None:
        if required:
            raise SystemExit(f"{FREEZE} assente: il codice d'analisi non e' congelato (emendamento 1, sez. 7)")
        return {"freeze": None, "pass": False, "note": "PROVA: codice non congelato"}
    import blmm_eval
    import d1_stats
    import diag
    import fact_paired
    mods = {"v3_work/stream/eval_ablation.py": EA, "v3_work/stream/eval_ablation_em1.py": sys.modules[__name__],
            "v3_work/trainer/tools/fact_paired.py": fact_paired, "aau/baselines_mm/blmm_eval.py": blmm_eval,
            "aau/diagnostics/diag.py": diag, "aau/diagnostics/d1_stats.py": d1_stats}
    rows, bad = {}, []
    for rel, want in fz["sha256"].items():
        got = EA.sha256(REPO / rel)
        m = mods.get(rel)
        loaded = None if m is None else str(Path(m.__file__).resolve())
        same_file = m is None or Path(loaded) == (REPO / rel).resolve()
        rows[rel] = {"sha256": got, "frozen": want, "module_file": loaded, "equal": got == want and same_file}
        if not rows[rel]["equal"]:
            bad.append(rel)
    missing = sorted(set(mods) - set(fz["sha256"]))
    out = {"commit": fz["commit"], "files": rows, "missing_from_freeze": missing, "pass": not bad and not missing}
    if required and not out["pass"]:
        raise SystemExit(f"codice d'analisi diverso dal commit {fz['commit']} congelato: {bad or missing}")
    return out


# ------------------------------------------------------------------------------------------ celle

def load_cells(root: Path, epoch: int) -> tuple[dict, dict]:
    """eval_ablation.load_cells generalizzata: A-D obbligatorie, A' e B' se pronte; FLAME dove c'e'."""
    out, missing = {}, {}
    for x in MAIN + EXTRA:
        ck = sorted((root / x / "runs").glob(f"v3_*/checkpoints/epoch{epoch:03d}_ema.pth"))
        cal = root / x / "eval/calib/calib.json"
        emb = {"heldout": root / x / "eval/calib/embeddings.npz"}
        for f in EA.FAMILIES:
            hits = sorted((root / x / "eval" / EA.FORM_DIR[f]).glob(
                f"data_*/scale_v3*fulle{epoch:03d}*/zs_zeroshot/embeddings.npz"))
            emb[f] = hits[0] if len(hits) == 1 else None
        ready = len(ck) == 1 and cal.exists() and all(p is not None and p.exists() for p in emb.values())
        if not ready:
            if x in MAIN:
                raise SystemExit(f"cella {x}: checkpoint, calibrazione o embedding mancanti in {root / x}")
            missing[x] = "checkpoint finale o valutazione assenti"
            continue
        ck = ck[0].resolve()
        c = json.loads(cal.read_text())
        sha = EA.sha256(ck)
        if c["sha256_checkpoint"] != sha or Path(c["checkpoint"]).resolve() != ck:
            raise SystemExit(f"cella {x}: calib.json di {c['checkpoint']}, atteso {ck}")
        fl = root / x / "eval/flame/embeddings.npz"
        if fl.exists():
            emb["flame"] = fl
        for k, p in emb.items():
            with np.load(p, allow_pickle=True) as z:
                e = Path(str(z["checkpoint"])).resolve()
            if e != ck:
                raise SystemExit(f"cella {x}, {k}: embedding di {e}, atteso {ck}")
        c010 = root / x / "eval/calib_e010/calib.json"
        out[x] = {"ckpt": ck, "sha256": sha, "c": float(c["c_median"]), "c_ls": float(c["c_ls"]),
                  "c_by_domain": {d: c.get(f"c_median_{d}") for d in EA.SEEN}, "emb": emb,
                  "sha256_embeddings": {k: EA.sha256(p) for k, p in emb.items()},
                  "calib_e010": json.loads(c010.read_text()) if c010.exists() else None}
    return out, missing


# ------------------------------------------------------------------------------------------ righe e repliche

def families(cells: dict, n_boot: int, workers: int) -> tuple[dict, dict]:
    """eval_ablation.families con i sottoinsiemi down8k delle righe nocrop (stesse repliche)."""
    sys.path.insert(0, str(REPO / "v3_work/trainer/tools"))
    import fact_paired as fp
    be = fp.be
    V, ctrl = {}, {}
    for view in EA.FAMILIES:
        t0 = time.time()
        ref, _, seed_nc = fp.rows_for(view)
        df, idx, seed_all = EA.all_cross_rows(view, fp, be)
        chk = EA.check_rows(df, ref)
        if not chk["pass"]:
            raise SystemExit(f"{view}: righe di all_cross senza crop diverse da fact_paired.rows_for: {chk}")
        D = EA.cell_columns(view, idx, cells, fp)
        M = sorted(D)
        nc, al = be.add_columns(ref, D, idx), be.add_columns(df, D, idx)
        crop = (al["topology_a"].eq("crop") | al["topology_b"].eq("crop")).to_numpy()
        d8 = (nc["topology_a"].eq("down8k") | nc["topology_b"].eq("down8k")).to_numpy()
        ctrl[view] = {"rows_check": chk, "groups": {}}
        for g, frame, seed, sub in (("nocrop", nc, seed_nc, None), ("nocrop_down8k", nc, seed_nc, d8),
                                    ("nocrop_altre", nc, seed_nc, ~d8), ("all_cross", al, seed_all, None),
                                    ("crop", al, seed_all, crop), ("nocrop_all", al, seed_all, ~crop)):
            reps, info = EA.replicates(frame, M, seed, n_boot, workers, fp, sub)
            V[(view, g)] = {(gt, m): v for (gt, kind, m), v in reps.items() if kind == "rho"}
            ctrl[view]["groups"][g] = info
        print(f"[abl-em1] {view}: righe {json.dumps(ctrl[view]['groups'])}, {time.time() - t0:.0f}s", flush=True)
    return V, ctrl


def d1_sets(cells: dict, workers: int) -> tuple[dict, dict]:
    """eval_ablation.d1_sets con le celle che hanno gli embedding (held-out: tutte; FLAME: dove c'e')."""
    diag, d1_stats = EA.d1_modules()
    V, ctrl = {}, {}
    for name in EA.SEEN + EA.FLAME_SETS:
        t0 = time.time()
        key = "heldout" if name in EA.SEEN else "flame"
        who = [x for x in cells if key in cells[x]["emb"]]
        if not who:
            continue
        names = diag.set_names(name)
        i, j = diag.rows(names)
        subj_of, subj = diag.subject_index(names)
        G = d1_stats.gt_static(name, names) if diag.SETS[name]["static"] else d1_stats.gt_new(name, names)
        cols = {f"gt_{k}": G[k][i, j] for k in ("sr", "fr")}
        for x in who:
            Z, ck = diag.load_embeddings(cells[x]["emb"][key], names)
            if ck.resolve() != cells[x]["ckpt"]:
                raise SystemExit(f"{name}, cella {x}: embedding di {ck}")
            for k, v in diag.arm_distances(Z, i, j, diag.dp_per_unit(ck), cells[x]["c"]).items():
                cols[f"{x}|{k}"] = v
        mask = np.all([np.isfinite(v) for v in cols.values()], axis=0)
        V[(name, "d1")] = d1_stats.boot(name, {k: v[mask] for k, v in cols.items()}, subj_of[i][mask],
                                        subj_of[j][mask], len(subj), workers)
        ctrl[name] = {"n_rows": int(mask.sum()), "n_rows_out": int((~mask).sum()), "n_subjects": len(subj),
                      "boot": f"diag.boot_counts({len(subj)}, {diag.SETS[name]['group']})", "cells": who}
        print(f"[abl-em1] {name}: {ctrl[name]}, {time.time() - t0:.0f}s", flush=True)
    return V, ctrl


# ------------------------------------------------------------------------------------------ effetti e soglie

def contrasts(R: dict, flame_set: bool) -> dict:
    """Effetti disponibili dalle repliche di rho per cella."""
    out = {}
    if all(x in R for x in MAIN) and not flame_set:
        out.update(EA.effects_of(R))
    if flame_set:
        out.update({f"{x}-A": R[x] - R["A"] for x in R if x != "A" and "A" in R})
        if "D" in R and "C" in R:
            out["D-C"] = R["D"] - R["C"]
    else:
        if "Bp" in R and "A" in R:
            out["Bp-A"] = R["Bp"] - R["A"]
        if "Bp" in R and "B" in R:
            out["Bp-B"] = R["Bp"] - R["B"]
        if "Ap" in R and "A" in R:
            out["Ap-A"] = R["Ap"] - R["A"]
    return out


def group_key(g: str) -> str:
    return "crop" if g in ("crop", "crollo") else ("all_cross" if g == "all_cross" else "nocrop")


def summarize(V: dict) -> tuple[list, list, dict]:
    """spearman.csv, effects.csv (con soglia per cella e flag), repliche degli effetti."""
    sp, E, rho = [], {}, {}
    for (st, g), reps in V.items():
        if g == "nocrop_all":
            continue
        cells = sorted({m.split("|")[0] for _, m in reps})
        for gt in ("fr", "sr"):
            for dist in EA.DISTS:
                if dist == "form" and gt == "sr":
                    continue
                R = {x: reps[(gt, f"{x}|{dist}")] for x in cells if (gt, f"{x}|{dist}") in reps}
                for x, v in R.items():
                    sp.append({"set": st, "group": g, "gt": gt, "distance": dist, "cell": x, **EA.ci(v)})
                    rho[(st, g, gt, dist, x)] = v
                for k, v in contrasts(R, st in EA.FLAME_SETS).items():
                    E[(st, g, gt, dist, k)] = v
    for view in EA.FAMILIES:                 # crollo = rho(crop) - rho(senza crop di all_cross), repliche di all_cross
        if (view, "crop") not in V:
            continue
        for gt in ("fr", "sr"):
            dist = EA.PRIMARY[gt]
            R = {}
            for x in sorted({m.split("|")[0] for _, m in V[(view, "crop")]}):
                v = V[(view, "crop")][(gt, f"{x}|{dist}")] - V[(view, "nocrop_all")][(gt, f"{x}|{dist}")]
                R[x] = v
                sp.append({"set": view, "group": "crollo", "gt": gt, "distance": dist, "cell": x, **EA.ci(v)})
                rho[(view, "crollo", gt, dist, x)] = v
            for k, v in contrasts(R, False).items():
                E[(view, "crollo", gt, dist, k)] = v
    for (st, g, gt, dist, k), v in list(E.items()):   # down8k contro le altre righe nocrop (stesse repliche)
        if g == "nocrop_down8k" and (st, "nocrop_altre", gt, dist, k) in E:
            E[(st, "down8k_meno_altre", gt, dist, k)] = v - E[(st, "nocrop_altre", gt, dist, k)]
    for g in GROUPS + ("down8k_meno_altre", "crollo"):   # medie sulle famiglie (stime indipendenti, replica per replica)
        for gt in ("fr", "sr"):
            for dist in EA.DISTS:
                for k in set(k for (_, gg, _, _, k) in E if gg == g):
                    vs = [E[(f, g, gt, dist, k)] for f in EA.FAMILIES if (f, g, gt, dist, k) in E]
                    if len(vs) == len(EA.FAMILIES):
                        E[("media famiglie", g, gt, dist, k)] = np.mean(vs, axis=0)
    ef = []
    for (st, g, gt, dist, k), v in E.items():
        kind = KIND.get(k, "S")
        d_tab = EA.DELTA[group_key(g)][kind]
        a, ap = rho.get((st, g, gt, dist, "A")), rho.get((st, g, gt, dist, "Ap"))
        sigma = abs(float(ap[0]) - float(a[0])) / np.sqrt(2.0) if a is not None and ap is not None else None
        d = d_tab if sigma is None or st == "media famiglie" else max(d_tab, 2.5 * MULT[kind] * sigma)
        r = {"set": st, "group": g, "gt": gt, "distance": dist, "effect": k, **EA.ci(v), "threshold_table": d_tab,
             "sigma_seed_cell": sigma, "threshold": float(d)}
        r["flag"] = EA.flag(r, r["threshold"])
        ef.append(r)
    return sp, ef, E


# ------------------------------------------------------------------------------------------ letture

def tag(c: dict) -> str:
    return f"{c['set']} {c['gt']} {c['point']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] (soglia {c['threshold']:.3f})"


def verdict(cells: list) -> dict:
    """Sez. 2a dell'emendamento: SI' (beneficio in >= 3 celle su 6, in >= 2 famiglie, nessun danno); PARZIALE (>= 2 celle
    con beneficio); "non confermato" (una sola cella); NO (nessuna cella con beneficio >= soglia, con un seme)."""
    ben = [c for c in cells if c["point"] >= c["threshold"] and c["ci_low"] > 0]
    dan = [c for c in cells if c["point"] <= -c["threshold"] and c["ci_high"] < 0]
    if len(ben) >= 3 and len({c["set"] for c in ben}) >= 2 and not dan:
        v = "SI'"
    elif len(ben) >= 2:
        v = "PARZIALE"
    elif len(ben) == 1:
        v = "non confermato"
    else:
        v = "NO"
    return {"verdict": v, "harm": "danno confermato" if len(dan) >= 2 else
            ("danno non confermato" if dan else "nessun danno"), "benefit_cells": [tag(c) for c in ben],
            "harm_cells": [tag(c) for c in dan], "cells": [tag(c) for c in cells]}


def readings(ef: list, have: set) -> dict:
    get = {(r["set"], r["group"], r["gt"], r["distance"], r["effect"]): r for r in ef}

    def cells_of(group: str, effect: str) -> list:
        return [get[(f, group, gt, EA.PRIMARY[gt], effect)] for f in EA.FAMILIES for gt in ("fr", "sr")
                if (f, group, gt, EA.PRIMARY[gt], effect) in get]

    def learned(x: str, ref: str) -> dict:
        r = get.get(("flame2023_s1", "d1", "sr", "shape", f"{x}-{ref}"))
        if r is None:
            return {"pass": None, "cell": None, "note": f"controllo assente (embedding FLAME di {lab(x)} o {lab(ref)})"}
        return {"pass": bool(r["point"] > 0 and r["ci_low"] > 0), "cell": tag(r)}

    out = {"R_F": verdict(cells_of("nocrop", "E_F")), "R_P1": verdict(cells_of("crop", "E_P"))}
    out["flame_learned"] = {"B-A": learned("B", "A"), "D-C": learned("D", "C"), "Bp-A": learned("Bp", "A")}
    fa = out["flame_learned"]["B-A"]["pass"]
    out["R_F"]["interpretable"] = "si'" if fa else ("non interpretabile" if fa is False else
                                                    "non interpretabile (controllo FLAME assente)")
    nc = cells_of("nocrop", "E_P")
    harm = [c for c in nc if c["point"] <= -c["threshold"] and c["ci_high"] < 0]
    small = [c for c in nc if c["ci_high"] < 0 and c["point"] > -c["threshold"]]
    out["R_P2"] = {"verdict": "costa" if len(harm) >= 2 else ("costo non confermato" if harm else "non costa"),
                   "harm_cells": [tag(c) for c in harm], "below_threshold": [tag(c) for c in small],
                   "cells": [tag(c) for c in nc]}
    p1, p2 = out["R_P1"]["verdict"], out["R_P2"]["verdict"]
    out["R_P"] = "riduce il crollo senza costo" if p1 == "SI'" and p2 == "non costa" else f"R_P1 {p1}, R_P2 {p2}"
    inter = []
    for group in ("nocrop", "crop"):
        for c in cells_of(group, "I"):
            if abs(c["point"]) >= c["threshold"] and (c["ci_low"] > 0 or c["ci_high"] < 0):
                simple = {k: get[(c["set"], group, c["gt"], c["distance"], k)] for k in ("B-A", "D-C", "C-A", "D-B")}
                inter.append({"group": group, "set": c["set"], "gt": c["gt"], "I": tag(c),
                              "simple": {k: f"{tag(s)} {s['flag']}" for k, s in simple.items()}})
    out["R_I"] = {"cells": inter, "note": "nessuna interazione da leggere" if not inter else
                  "effetti principali da leggere per strato nelle celle elencate"}
    if "Bp" in have:
        out["R_F_Bp"] = verdict(cells_of("nocrop", "Bp-A"))
        bl = out["flame_learned"]["Bp-A"]["pass"]
        out["R_F_Bp"]["interpretable"] = "si'" if bl else ("non interpretabile" if bl is False else
                                                         "non interpretabile (controllo FLAME assente)")
    else:
        out["R_F_Bp"] = {"verdict": "in attesa di B'"}
    out["status"] = "definitivo" if {"Ap", "Bp"} <= have else (
        "PROVVISORIO: " + ", ".join(f"manca {lab(x)}" for x in EXTRA if x not in have)
        + " (rieseguire quando c'e')")
    out["thresholds"] = "per cella da A'" if "Ap" in have else "solo tabella (A' assente)"
    return out


# ------------------------------------------------------------------------------------------ riuso e c

def diversity(root: Path, cells: list) -> dict:
    """Per cella: viste fresche/s (producers.json), riuso a regime (summary.json), viste uniche per epoca (stream_stats
    dei due rank, media dalla seconda epoca); coppie confrontate con |ln(va / vb)| > ln 1.2; verso della distorsione."""
    out = {}
    for x in cells:
        n0 = root / x / "node0"
        st = [sorted((root / x / "runs").glob(f"v3_*/stream_stats_rank{r}.jsonl")) for r in (0, 1)]
        if not all(st):
            continue
        rows = [[json.loads(line) for line in p[0].read_text().splitlines() if line.strip()] for p in st]
        by_ep = {}
        for rr in rows:
            for r in rr:
                by_ep.setdefault(int(r["epoch"]), []).append(int(r["d_unique_views_used"]))
        u = [sum(v) for e, v in sorted(by_ep.items()) if e >= 2 and len(v) == 2]
        summ = json.loads((n0 / "summary.json").read_text()) if (n0 / "summary.json").exists() else {}
        prod = json.loads((n0 / "producers.json").read_text()) if (n0 / "producers.json").exists() else {}
        out[x] = {"unique_views_per_epoch": float(np.mean(u)) if u else None, "epochs": len(u),
                  "fresh_views_per_s": prod.get("views_per_s_steady"),
                  "reuse_steady": (summ.get("reuse") or {}).get("steady_last_epochs"),
                  "s_per_step_mean": float(np.mean(summ["s_per_step_by_epoch"])) if summ.get("s_per_step_by_epoch") else None,
                  "steps": summ.get("steps")}
    flags = []
    for a, b in (("B", "A"), ("D", "C"), ("C", "A"), ("D", "B"), ("Bp", "A"), ("Bp", "B"), ("Ap", "A")):
        for k in ("fresh_views_per_s", "reuse_steady", "unique_views_per_epoch"):
            va, vb = (out.get(a) or {}).get(k), (out.get(b) or {}).get(k)
            if va and vb and abs(np.log(va / vb)) > LN12:
                flags.append(f"{k}: {lab(a)} {va:.2f} contro {lab(b)} {vb:.2f} (ln {np.log(va / vb):+.3f})")
    u = {x: (out.get(x) or {}).get("unique_views_per_epoch") for x in MAIN + EXTRA}

    def bias(pairs) -> dict | None:
        if not all(u.get(p) and u.get(q) for p, q in pairs):
            return None
        r = float(np.mean([np.log(u[p] / u[q]) for p, q in pairs]))
        if r > 0:
            d = "a favore dell'effetto (le celle trattate vedono piu' viste distinte: possibile sovrastima)"
        elif r < 0:
            d = "contro l'effetto (le celle trattate vedono meno viste distinte: possibile sottostima)"
        else:
            d = "nessuna (stesse viste distinte per epoca)"
        return {"ln_ratio_unique_views": r, "beyond_ln_1.2": abs(r) > LN12, "direction": d}
    return {"cells": out, "flags_ln_1.2": flags,
            "bias": {"E_F": bias((("B", "A"), ("D", "C"))), "E_P": bias((("C", "A"), ("D", "B"))),
                     "Bp-A": bias((("Bp", "A"),))}}


def calib_check(cells: dict) -> dict:
    """PLAN_MASSIVE sez. 22.5: c e scarto fra domini al 10% del run (epoch010_ema) e alla fine; descrittivo."""
    out = {}
    for x, i in cells.items():
        row = {"c_e100": i["c"], "c_by_domain_e100": i["c_by_domain"]}
        vals = [v for v in i["c_by_domain"].values() if v]
        row["max_over_min_e100"] = max(vals) / min(vals) if vals else None
        e = i.get("calib_e010")
        if e:
            bd = {d: e.get(f"c_median_{d}") for d in EA.SEEN}
            vv = [v for v in bd.values() if v]
            row.update(c_e010=e["c_median"], c_by_domain_e010=bd, max_over_min_e010=max(vv) / min(vv) if vv else None)
        out[x] = row
    return out


# ------------------------------------------------------------------------------------------ uscite

def results_md(out: Path, sp: list, ef: list, rd: dict, cells: dict, ctrl: dict, div: dict, cal: dict) -> None:
    S = {(r["set"], r["group"], r["gt"], r["distance"], r["cell"]): r for r in sp}
    F = {(r["set"], r["group"], r["gt"], r["distance"], r["effect"]): r for r in ef}
    cols = [x for x in MAIN + EXTRA if x in cells]
    fz = ctrl.get("freeze") or {}
    md = ["# Ablazione 2x2 dello stream: letture dell'emendamento 1", "",
          f"**Stato: {rd['status']}.** Soglie: {rd['thresholds']}. Codice d'analisi congelato: commit "
          f"{fz.get('commit', '-')} ({'verificato' if fz.get('pass') else 'NON verificato'}). Protocollo `PROTOCOL.md` "
          "con `PROTOCOL_emendamento_1.md`; numeri in `spearman.csv`, `effects.csv`; letture in `readings.json`. Un seme "
          "per cella (A' stima il rumore fra semi). NO = nessun beneficio >= soglia rilevato con un seme, non \"nessun "
          "effetto\"; le tre famiglie di test sono 3DMM est-asiatici: \">= 2 famiglie\" non e' una replica indipendente; "
          "un solo generatore (FLAME 2023) non risponde a \"un generatore in piu' in generale\".", "",
          "## Letture", ""]
    fl = rd["flame_learned"]
    md += [f"- **Controllo FLAME** (flame2023_s1, SR, d_P; positivo con IC sopra 0): B - A "
           f"{fl['B-A'].get('cell') or fl['B-A'].get('note')} -> {fl['B-A']['pass']}; D - C "
           f"{fl['D-C'].get('cell') or fl['D-C'].get('note')} -> {fl['D-C']['pass']}; B' - A "
           f"{fl['Bp-A'].get('cell') or fl['Bp-A'].get('note')} -> {fl['Bp-A']['pass']}.",
           f"- **R_F, FLAME 2023 nativo al posto di 1/4 dei passi (nocrop): {rd['R_F']['verdict']}** "
           f"({rd['R_F']['interpretable']}); {rd['R_F']['harm']}. Benefici: {'; '.join(rd['R_F']['benefit_cells']) or 'nessuno'}. "
           f"Danni: {'; '.join(rd['R_F']['harm_cells']) or 'nessuno'}.",
           f"- **R_F', FLAME 2023 suddiviso al posto di 1/4 dei passi (B' - A, nocrop): {rd['R_F_Bp']['verdict']}**"
           + (f" ({rd['R_F_Bp']['interpretable']}); {rd['R_F_Bp']['harm']}. Benefici: "
              f"{'; '.join(rd['R_F_Bp']['benefit_cells']) or 'nessuno'}. Danni: {'; '.join(rd['R_F_Bp']['harm_cells']) or 'nessuno'}."
              if "interpretable" in rd["R_F_Bp"] else "."),
           f"- **R_P1, parzialita' sulle righe col crop: {rd['R_P1']['verdict']}**; {rd['R_P1']['harm']}. Benefici: "
           f"{'; '.join(rd['R_P1']['benefit_cells']) or 'nessuno'}. Danni: {'; '.join(rd['R_P1']['harm_cells']) or 'nessuno'}.",
           f"- **R_P2, costo senza crop: {rd['R_P2']['verdict']}**. Danni: {'; '.join(rd['R_P2']['harm_cells']) or 'nessuno'}; "
           f"sotto soglia: {'; '.join(rd['R_P2']['below_threshold']) or 'nessuno'}.",
           f"- **R_P: {rd['R_P']}**.",
           f"- **R_I: {rd['R_I']['note']}**" + ("." if not rd["R_I"]["cells"] else ": " + "; ".join(
               f"{c['group']} {c['set']} {c['gt']} I {c['I']}" for c in rd["R_I"]["cells"]))]
    b = div["bias"]
    md += [f"- **Riuso, verso della distorsione** (ln del rapporto delle viste uniche per epoca, trattate su controlli): "
           + "; ".join(f"{k} " + (f"{v['ln_ratio_unique_views']:+.3f} {v['direction']}" + (" (oltre ln 1.2)" if v["beyond_ln_1.2"] else "")
                                   if v else "n.d.") for k, v in b.items()) + ".", ""]
    md += ["## Celle", "", "| cella | checkpoint | sha256 | c | c_LS | c bfm / ict / gnm | FLAME |", "|---|---|---|---|---|---|---|"]
    for x, i in cells.items():
        cb = " / ".join(f"{i['c_by_domain'][d]:.3f}" for d in EA.SEEN)
        md.append(f"| {lab(x)} | `...__{i['ckpt'].parents[1].name.split('__')[-1]}/{i['ckpt'].name}` | {i['sha256'][:12]} | "
                  f"{i['c']:.4f} | {i['c_ls']:.4f} | {cb} | {'si' if 'flame' in i['emb'] else 'no'} |")
    if ctrl.get("missing_cells"):
        md.append(f"\nCelle assenti: {ctrl['missing_cells']}.")
    md += ["", "## Riuso e viste (descrittivo)", "", "| cella | viste uniche per epoca | viste fresche/s | riuso a regime | "
           "s/passo | passi |", "|---|---|---|---|---|---|"]
    for x, d in div["cells"].items():
        f = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
        md.append(f"| {lab(x)} | {f(d['unique_views_per_epoch'], 0)} | {f(d['fresh_views_per_s'])} | {f(d['reuse_steady'], 1)} | "
                  f"{f(d['s_per_step_mean'], 3)} | {d['steps'] if d['steps'] is not None else '-'} |")
    md += ["", f"Coppie oltre |ln| > ln 1.2: {'; '.join(div['flags_ln_1.2']) or 'nessuna'}.", ""]
    md += ["## c al 10% del run e alla fine (PLAN_MASSIVE sez. 22.5, descrittivo)", "",
           "| cella | c e010 | max/min fra domini e010 | c e100 | max/min fra domini e100 |", "|---|---|---|---|---|"]
    for x, r in cal.items():
        g = lambda k: "-" if r.get(k) is None else f"{r[k]:.3f}"  # noqa: E731
        md.append(f"| {lab(x)} | {g('c_e010')} | {g('max_over_min_e010')} | {g('c_e100')} | {g('max_over_min_e100')} |")
    for title, gt in (("GT FR, d_F calibrata", "fr"), ("GT SR, d_P", "sr")):
        dist = EA.PRIMARY[gt]
        md += ["", f"## Famiglie mai viste, {title}", "",
               "| famiglia | gruppo | " + " | ".join(lab(x) for x in cols) + " | E_F | E_P | I | B' - A | A' - A |",
               "|---|---|" + "---|" * (len(cols) + 5)]
        for f in EA.FAMILIES + ("media famiglie",):
            for g in GROUPS + ("down8k_meno_altre", "crollo"):
                if (f, g, gt, dist, "E_F") not in F:
                    continue
                rho = [EA.fmt_rho(S[(f, g, gt, dist, x)]) if (f, g, gt, dist, x) in S else "-" for x in cols]
                eff = []
                for k in ("E_F", "E_P", "I", "Bp-A", "Ap-A"):
                    r = F.get((f, g, gt, dist, k))
                    eff.append("-" if r is None else f"{EA.fmt(r)} s{r['threshold']:.2f}" + (f" {r['flag']}" if r["flag"] != "-" else ""))
                md.append(f"| {f} | {g} | " + " | ".join(rho) + " | " + " | ".join(eff) + " |")
    md += ["", "## Held-out sintetici e FLAME 2023 di D1 (SR con d_P; FR descrittiva)", "",
           "| insieme | GT | " + " | ".join(lab(x) for x in cols) + " | contrasti |", "|---|---|" + "---|" * (len(cols) + 1)]
    for st in EA.SEEN + EA.FLAME_SETS:
        for gt in ("sr", "fr"):
            dist = EA.PRIMARY[gt]
            rho = [EA.fmt_rho(S[(st, "d1", gt, dist, x)]) if (st, "d1", gt, dist, x) in S else "-" for x in cols]
            if all(r == "-" for r in rho):
                continue
            con = [f"{k} {EA.fmt(r)}" + (f" {r['flag']}" if r["flag"] != "-" else "") for (s2, g2, gt2, d2, k), r in F.items()
                   if s2 == st and g2 == "d1" and gt2 == gt and d2 == dist]
            md.append(f"| {st} | {gt} | " + " | ".join(rho) + " | " + "; ".join(con) + " |")
    md += ["", "## Controlli (tutti in `controls.json`)", ""]
    for f, c in ctrl["families"].items():
        rc = c["rows_check"]
        md.append(f"- {f}: all_cross senza crop = `fact_paired.rows_for` {rc['pass']}; " + "; ".join(
            f"{g} {i['n_rows']} righe, seme {i['seed']}" for g, i in c["groups"].items()))
    for s, c in ctrl.get("d1", {}).items():
        md.append(f"- {s}: {c['n_rows']} righe, {c['boot']}, celle {', '.join(lab(x) for x in c['cells'])}")
    md.append(f"- repliche: {ctrl['n_boot']}; codice: {json.dumps({k: v.get('equal') for k, v in (fz.get('files') or {}).items()})}")
    EA.atomic_text(out / "results.md", "\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--root", type=Path, default=ABL)
    ap.add_argument("--out", type=Path, default=None, help="uscite (default: <root>/em1)")
    ap.add_argument("--epoch", type=int, default=EA.EPOCH)
    ap.add_argument("--code-state", type=Path, default=None, help="json della sbatch: commit e stato git")
    ap.add_argument("--no-freeze", action="store_true", help="solo prove: nessun controllo del codice congelato")
    a = ap.parse_args()
    t0 = time.time()
    root = a.root.resolve()
    out = (a.out or root / "em1").resolve()
    EA.d1_modules()
    sys.path.insert(0, str(REPO / "v3_work/trainer/tools"))
    import fact_paired  # noqa: F401  (per il controllo del codice congelato)
    freeze = check_freeze(not a.no_freeze)
    cells, missing = load_cells(root, a.epoch)
    print(f"[abl-em1] celle {[lab(x) for x in cells]}, assenti {missing}; codice {freeze.get('commit')} "
          f"pass={freeze['pass']}", flush=True)
    V, ctrl_f = families(cells, a.n_boot, a.workers)
    V1, ctrl_d1 = d1_sets(cells, a.workers)
    V.update(V1)
    sp, ef, E = summarize(V)
    rd = readings(ef, set(cells))
    div = diversity(root, list(cells))
    cal = calib_check(cells)
    ctrl = {"freeze": freeze, "code_state": json.loads(a.code_state.read_text()) if a.code_state else None,
            "families": ctrl_f, "d1": ctrl_d1, "missing_cells": missing, "n_boot": a.n_boot, "epoch": a.epoch,
            "root": str(root), "cells": {x: {"checkpoint": str(i["ckpt"]), "sha256_checkpoint": i["sha256"], "c": i["c"],
                                             "c_ls": i["c_ls"], "c_by_domain": i["c_by_domain"],
                                             "embeddings": {k: str(p) for k, p in i["emb"].items()},
                                             "sha256_embeddings": i["sha256_embeddings"]} for x, i in cells.items()}}
    out.mkdir(parents=True, exist_ok=True)
    EA.write_csv(out / "spearman.csv", sp)
    EA.write_csv(out / "effects.csv", ef)
    EA.atomic_text(out / "readings.json", json.dumps({**rd, "diversity": div, "calib": cal}, indent=1, default=str) + "\n")
    EA.atomic_text(out / "controls.json", json.dumps(ctrl, indent=1, default=str) + "\n")
    tmp = out / f".reps.{os.getpid()}.tmp.npz"
    np.savez(tmp, **{"|".join(map(str, k)): v for k, v in E.items()})
    os.replace(tmp, out / "reps.npz")
    results_md(out, sp, ef, rd, cells, ctrl, div, cal)
    print(f"[abl-em1] {rd['status']}; R_F {rd['R_F']['verdict']} ({rd['R_F']['interpretable']}), R_F' "
          f"{rd['R_F_Bp']['verdict']}, R_P1 {rd['R_P1']['verdict']}, R_P2 {rd['R_P2']['verdict']}, R_I "
          f"{len(rd['R_I']['cells'])} celle; {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
