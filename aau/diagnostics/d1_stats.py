#!/usr/bin/env python3
"""D1 (PROTOCOL_D.md sez. 3.4-3.6): Spearman per insieme col bootstrap per soggetto, delta, controllo C1, letture R1, R2.

    AAU_NV="" aau/run.sh aau/diagnostics/d1_stats.py --workers 32          (diag.sbatch, passo d1)

Per insieme (``diag.SETS``): mesh nei nomi ordinati, righe ``diag.rows``; colonne = d_P (``shape``) e d_F calibrata
(``form_cal``) dei bracci (``diag.ARMS``; embedding della calibrazione per i visti, ``d1/emb/<chiave>`` per i nuovi),
``vb_sr`` / ``vb_fr`` di B-GNM e B-FLAME (``d1/bp``: visti e FLAME; niente B sui rigenerati), GT ``sr`` e ``fr``.
Maschera comune per insieme (righe con tutte le colonne finite: un fit B fallito toglie le sue righe a tutti).
Repliche: ``diag.boot_counts`` per gruppo d'identita' (stessi conteggi per ict / regen_ict, gnm / regen_gnm, FLAME
suddiviso / nativo), Spearman pesato c_a c_b.

Uscite: ``d1_spearman.csv`` (rho per insieme, GT, metodo), ``d1_delta.csv`` (delta dentro un insieme e fra insiemi:
bracci - B, rigenerato - statico, suddiviso - nativo, Delta_gen, Delta_read, controllo di difficolta'),
``d1/controls.json`` (C1, maschere, scala della GT), ``d1/readings.json`` (R1, R2), ``d1/reps.npz`` (tutte le repliche).
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time

import numpy as np

import diag

GTS = ("sr", "fr")
ARM_DIST = {"sr": "shape", "fr": "form_cal"}       # la distanza del braccio che si legge con ciascuna GT
_J: dict = {}


def gt_static(name: str, names: list[str]) -> dict:
    """{sr, fr}: matrici GT (mesh x mesh) di training di C3F per un insieme visto o rigenerato (per soggetto)."""
    sid = [diag.split_name(n)[0] for n in names]
    out = {}
    for kind, path in (("sr", diag.GT_SR), ("fr", diag.GT_FR)):
        with np.load(path, allow_pickle=True) as z:
            pos = {str(x): k for k, x in enumerate(z["names"])}
            ii = np.asarray([pos[s] for s in sid])
            D = np.asarray(z["D_orig"], np.float64)[np.ix_(ii, ii)]
        if kind == "sr":
            D = D * float(json.loads(path.with_suffix(".json").read_text())["dP_per_unit"])
        out[kind] = D
    return out


def gt_new(name: str, names: list[str]) -> dict:
    with np.load(diag.D1 / f"gt_{name}.npz", allow_pickle=True) as z:
        pos = {str(x): k for k, x in enumerate(z["names"])}
        ii = np.asarray([pos[diag.split_name(n)[0]] for n in names])
        return {kind: np.asarray(z[f"D_{kind}"], np.float64)[np.ix_(ii, ii)] for kind in GTS}


def bp_columns(name: str, names: list[str]) -> dict:
    """{(modello, gt): matrice (mesh x mesh)} di B; vuoto per i rigenerati."""
    out = {}
    for model in diag.BP_MODELS:
        if name in diag.SEEN:
            p = diag.D1 / "bp" / f"static_{model}.npz"
            keys = (f"{name}_names", f"{name}_D_sr", f"{name}_D_fr")
        elif name in diag.FLAME_SETS:
            p = diag.D1 / "bp" / f"{name}_{model}.npz"
            keys = ("names", "D_sr", "D_fr")
        else:
            return {}
        with np.load(p, allow_pickle=True) as z:
            if [str(x) for x in z[keys[0]]] != names:
                raise SystemExit(f"{p}: mesh in ordine diverso dall'insieme {name}")
            out[(model, "sr")], out[(model, "fr")] = np.asarray(z[keys[1]]), np.asarray(z[keys[2]])
    return out


def columns(name: str) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """(colonne sulle righe, sa, sb, informazioni) di un insieme."""
    names = diag.set_names(name)
    i, j = diag.rows(names)
    subj_of, subj = diag.subject_index(names)
    G = gt_static(name, names) if diag.SETS[name]["static"] else gt_new(name, names)
    cal = diag.calibration()
    cols = {f"gt_{k}": G[k][i, j] for k in GTS}
    info = {"n_meshes": len(names), "n_subjects": len(subj), "n_rows": int(len(i)), "ckpt": {}}
    for arm in diag.ARMS:
        emb = (diag.CALIB_EMB / arm if diag.SETS[name]["static"] else diag.D1 / "emb" / arm) / "embeddings.npz"
        Z, ckpt = diag.load_embeddings(emb, names)
        info["ckpt"][arm] = str(ckpt)
        for k, v in diag.arm_distances(Z, i, j, diag.dp_per_unit(ckpt), cal[arm]).items():
            cols[f"{arm}|{k}"] = v
        if name in ("regen_ict", "regen_gnm"):      # C1-emb: stesse mesh deterministiche, embedding statico
            Zs, _ = diag.load_embeddings(diag.CALIB_EMB / arm / "embeddings.npz", names)
            lab = np.asarray([diag.split_name(n)[1] for n in names])
            info.setdefault("c1_emb_max_abs", {})[arm] = {
                lb: float(np.abs(Z[lab == lb] - Zs[lab == lb]).max()) for lb in diag.LABELS}
            info.setdefault("c1_emb_scale", {})[arm] = float(np.abs(Zs).max())
    for (model, kind), D in bp_columns(name, names).items():
        cols[f"B-{model}|vb_{kind}"] = D[i, j]
    mask = np.all([np.isfinite(v) for v in cols.values()], axis=0)
    info["n_rows_masked_out"] = int((~mask).sum())
    iu = np.triu_indices(len(subj), 1)
    first = [k for k, n in enumerate(names) if n.endswith("_GTready_original.npz")]   # una mesh per soggetto
    info["dP_gt_subject_pairs"] = {"median": float(np.median(G["sr"][np.ix_(first, first)][iu])),
                                   "iqr": [float(x) for x in np.percentile(G["sr"][np.ix_(first, first)][iu], [25, 75])]}
    return {k: v[mask] for k, v in cols.items()}, subj_of[i][mask], subj_of[j][mask], info


def _rep(task):
    """Una replica: ranghi pesati di ogni colonna una volta sola, poi Pearson dei ranghi con ciascuna GT
    (= ``diag.wspearman`` colonna per colonna)."""
    name, b = task
    J = _J[name]
    c = J["counts"][b]
    w = c[J["sa"]] * c[J["sb"]]
    k = w > 0
    w = w[k]
    R = {m: diag.wranks(x[k], w) for m, x in J["cols"].items()}
    return {(g, m): diag.wpearson(r, R[f"gt_{g}"], w) for g in GTS for m, r in R.items() if not m.startswith("gt_")}


def boot(name: str, cols: dict, sa: np.ndarray, sb: np.ndarray, n_subj: int, workers: int) -> dict:
    """{(gt, metodo): (1 + N_BOOT,) rho}."""
    _J[name] = {"cols": cols, "sa": sa, "sb": sb, "counts": diag.boot_counts(n_subj, diag.SETS[name]["group"])}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_rep, [(name, b) for b in range(diag.N_BOOT + 1)], chunksize=8)
    del _J[name]
    return {k: np.asarray([r[k] for r in reps]) for k in reps[0]}


def label(name: str, method: str) -> str:
    """Etichetta di B nel contesto dell'insieme (sez. 3.5)."""
    if not method.startswith("B-"):
        return method
    model = method[2:].split("|")[0]
    exact = (model == "gnm" and name in ("gnm",)) or (model == "flame2023" and name in diag.FLAME_SETS)
    return f"{method} ({'prior esatto' if exact else 'incrociato'})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    V, info = {}, {}
    for name in diag.SETS:
        cols, sa, sb, inf = columns(name)
        info[name] = inf
        V[name] = boot(name, cols, sa, sb, inf["n_subjects"], a.workers)
        print(f"[d1] {name}: {inf['n_rows']} righe ({inf['n_rows_masked_out']} fuori maschera), "
              f"{len(V[name])} colonne x GT, {time.time() - t0:.0f}s", flush=True)
    # ------------------------------------------------------------------ rho per insieme
    recs = []
    for name, Vn in V.items():
        for (g, m), v in sorted(Vn.items()):
            p, lo, hi, _ = diag.ci(v)
            recs.append({"set": name, "gt": g, "method": label(name, m), "rho": p, "ci_low": lo, "ci_high": hi,
                         "n_rows": info[name]["n_rows"] - info[name]["n_rows_masked_out"],
                         "n_subjects": info[name]["n_subjects"], "boot_group": diag.SETS[name]["group"]})
    with open(diag.EV / "d1_spearman.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    # ------------------------------------------------------------------ delta
    deltas = []

    def add(kind, arm, g, what, v, extra=None):
        p, lo, hi, ple = diag.ci(v)
        deltas.append({"kind": kind, "arm": arm, "gt": g, "what": what, "delta": p, "ci_low": lo, "ci_high": hi,
                       "p_le0": ple, **(extra or {})})
        return {"delta": p, "ci_low": lo, "ci_high": hi, "p_le0": ple}

    readings = {"R1": {}, "R1_native": {}, "R2": {}, "difficulty": {}, "C1": {}}
    for arm in diag.ARMS:
        for g in GTS:
            col = f"{arm}|{ARM_DIST[g]}"
            # dentro un insieme: braccio - B
            for name in diag.SEEN + diag.FLAME_SETS:
                for model in diag.BP_MODELS:
                    key = (g, f"B-{model}|vb_{g}")
                    if key in V[name]:
                        add("within", arm, g, f"{name}: {col} - {label(name, key[1])}", V[name][(g, col)] - V[name][key])
            # C1: rigenerato - statico (stessi soggetti, conteggi, righe)
            for d in ("ict", "gnm"):
                r = add("C1", arm, g, f"regen_{d} - {d}: {col}", V[f"regen_{d}"][(g, col)] - V[d][(g, col)])
                readings["C1"].setdefault(arm, {})[f"{d}|{g}"] = {**r, "pass": abs(r["delta"]) <= 0.02}
            # risoluzione: suddiviso - nativo (stesse identita')
            add("resolution", arm, g, f"flame2023_s1 - flame2023: {col}",
                V["flame2023_s1"][(g, col)] - V["flame2023"][(g, col)])
            seen = np.mean([V[s][(g, col)] for s in diag.SEEN], axis=0)
            for fl, store in (("flame2023_s1", "R1"), ("flame2023", "R1_native")):
                r = add("gen", arm, g, f"media visti - {fl}: {col}", seen - V[fl][(g, col)])
                if g == "sr":
                    readings[store][arm] = {**r, "strong": bool(r["delta"] >= 0.10 and r["ci_low"] > 0)}
            # R2: media sui visti di braccio - B-FLAME (incrociato)
            key = (g, f"B-flame2023|vb_{g}")
            r = add("read", arm, g, f"media visti: {col} - B-flame2023|vb_{g} (incrociato)",
                    np.mean([V[s][(g, col)] - V[s][key] for s in diag.SEEN], axis=0))
            if g == "sr":
                readings["R2"][arm] = {**r, "reading_limit": bool(r["ci_high"] < 0)}
            # controllo di difficolta': B-GNM incrociato su bfm, ict, FLAME suddiviso
            kb = (g, f"B-gnm|vb_{g}")
            d_arm = np.mean([V[s][(g, col)] for s in ("bfm", "ict")], axis=0) - V["flame2023_s1"][(g, col)]
            d_b = np.mean([V[s][kb] for s in ("bfm", "ict")], axis=0) - V["flame2023_s1"][kb]
            if arm == diag.ARMS[0]:
                add("difficulty", "B-gnm", g, "media {bfm, ict} - flame2023_s1: B-gnm|vb (incrociato)", d_b)
            add("difficulty", arm, g, f"media {{bfm, ict}} - flame2023_s1: {col}", d_arm)
            r = add("difficulty", arm, g, f"DiD: ({col}) - (B-gnm|vb_{g})", d_arm - d_b)
            if g == "sr":
                readings["difficulty"][arm] = r
    with open(diag.EV / "d1_delta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deltas[0]))
        w.writeheader()
        w.writerows(deltas)
    # ------------------------------------------------------------------ letture
    def both(store, key):
        v = [readings[store][a][key] for a in diag.RULE_ARMS]
        return "si" if all(v) else ("no" if not any(v) else f"non risolto (si solo {diag.RULE_ARMS[v.index(True)]})")

    c1_gt = json.loads((diag.D1 / "gen.json").read_text())["sets"]
    c1_pass = all(r["pass"] for a in diag.ARMS for r in readings["C1"][a].values()) and \
        all(c1_gt[s]["c1_gt"]["pass"] for s in ("regen_ict", "regen_gnm"))
    readings["verdict"] = {"C1_compatible": c1_pass, "R1_variety_strong": both("R1", "strong"),
                           "R1_native_variety_strong": both("R1_native", "strong"),
                           "R2_reading_limit": both("R2", "reading_limit"),
                           "rule_arms": list(diag.RULE_ARMS)}
    readings["note"] = ("R1/R2 sui visti statici" if c1_pass else
                        "C1 NON passa: vedi PROTOCOL_D sez. 3.3 (R1/R2 da rifare coi rigenerati)")
    diag.atomic_json(diag.D1 / "readings.json", readings)
    diag.atomic_json(diag.D1 / "controls.json", {"sets": info, "c1_gt": {s: c1_gt[s]["c1_gt"]
                                                                           for s in ("regen_ict", "regen_gnm")},
                                                 "seconds": time.time() - t0})
    diag.atomic_savez(diag.D1 / "reps.npz", **{f"{n}|{g}|{m}": v for n, Vn in V.items() for (g, m), v in Vn.items()})
    print(f"[d1] verdetto {readings['verdict']}; {len(recs)} rho, {len(deltas)} delta, {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
