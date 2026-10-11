#!/usr/bin/env python3
"""D4 (``aau/runs/evidence/diagnostics/PROTOCOL_D4.md``): Spearman e delta appaiati canonico - originale, controlli,
lettura preregistrata e ``results_d4.md``.

    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/d4_stats.py --workers 64   (d4.sbatch, passo stats)
    UGT_THREADS=1 v3_work/unified_gt/run.sh aau/diagnostics/d4_stats.py --summary-only (solo results_d4.md)

Righe di all_cross con GT FR / SR (``bp_paired_e2.all_cross_rows``, importato in sola lettura) e, sulle stesse coppie di
soggetti, le righe di stessa topologia (5 etichette senza crop, come ``bp_paired_e3.run_topo``). Gruppi (sez. 3):
nocrop_cross (primario), stessa topologia (anche per etichetta), all_cross, righe col crop. Colonne: per braccio
(``ARMS``) l'ingresso originale (store ufficiali, ``bp_e3.store``) e quello canonico (``d4/emb``), distanze di
``fact_paired.distances`` con la c di ``fact_paired.calibration``. Spearman pesato c_a c_b (``diag.wranks`` /
``diag.wpearson``), 1000 repliche per soggetto coi conteggi di ``fact_paired.main`` e il seme del gruppo primario
ufficiale della vista; maschera comune. Uscite: ``d4_spearman.csv``, ``d4_delta.csv``, ``d4/controls.json``,
``d4/readings.json``, ``results_d4.md``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import diag  # noqa: E402
import d4_remesh as R  # noqa: E402

sys.path.insert(0, str(diag.REPO / "aau/baselines_param"))
import bp_paired_e2 as bpe2  # noqa: E402  (sola lettura, attraverso bp_paired: fact_paired, blmm_eval)

fp, be = bpe2.bpp.fp, bpe2.bpp.be
EV = diag.EV
D4 = R.D4
VIEWS = tuple(R.VIEWS)                                       # facescape, hifi3d, faceverse_neutral
NOCROP = tuple(t for t in R.TOPOLOGIES if t != "crop")
SEEDS_D3 = {"facescape": 796786, "hifi3d": 990708, "faceverse_neutral": 271049}
ROWS_D3 = {"facescape": 88725, "hifi3d": 98224, "faceverse_neutral": 99000}
# distanze per braccio, GT dichiarata di ognuna: (distanza, gt, ruolo)
DECLARED = {"factorized_s1234": (("shape", "sr", "primaria"), ("form_cal", "fr", "vincolo")),
            "factorized_s2345": (("shape", "sr", "primaria"), ("form_cal", "fr", "vincolo")),
            "ctrlfr_s1234": (("z", "fr", "secondaria"), ("z", "sr", "secondaria"))}
DISTS = {"factorized_s1234": ("shape", "form_cal"), "factorized_s2345": ("shape", "form_cal"), "ctrlfr_s1234": ("z",)}
RULE_ARMS = ("factorized_s1234", "factorized_s2345")
GROUPS = ("nocrop_cross", "stessa topologia", "all_cross", "righe col crop") + tuple(f"stessa: {t}" for t in NOCROP)
# descrittivo, NON nel protocollo: nocrop_cross ristretto alle righe di D3 / fact_paired (la loro maschera toglie le righe
# con baseline geometriche non finite: FaceScape e HIFI3D); K1 si legge qui (righe comuni)
D3ROWS = "nocrop_cross, righe di D3"
SR_MIN, FR_MIN, N_DOM, HARM = 0.02, -0.02, 2, -0.02            # lettura (sez. 5)
LABEL = {"shape": "d_P", "form_cal": "d_F cal.", "z": "||z||"}
_J: dict = {}


def primary_seed(view: str) -> int:
    """Seme del gruppo primario ufficiale (``fact_paired.rows_for``: E12 nocrop_cross, FaceVerse mesh_pair_nocrop)."""
    src = "faceverse" if view == "faceverse_neutral" else view
    fr = {"hifi3d": lambda: be.e12m.hifi_frames()[0], "facescape": be.e12m.fs_frames, "faceverse": be.e12m.fv_frames}[src]()
    return int(fr[fp.VIEWS[src][1]][2])


def arm_matrices(view: str, idx) -> tuple[dict, dict]:
    """{colonna: D (n, n)} sulle chiavi di idx: ``<braccio>|<distanza>|orig`` (store) e ``|canon`` (d4/emb)."""
    cal = fp.calibration()
    n = len(idx.keys)
    i, j = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    out, ctrl = {}, {}
    for pre, v, e in R.ARMS:
        st = R.bp_e3.store(view, v, e)
        for kind, path in (("orig", st), ("canon", D4 / "emb" / view / pre / "embeddings.npz")):
            with np.load(path, allow_pickle=True) as z:
                Zr = np.asarray(z["Z"], np.float64)
                keys = list(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]))
                ckpt = Path(str(z["checkpoint"]))
            if kind == "canon":
                ck_store = Path(str(np.load(st, allow_pickle=True)["checkpoint"]))
                if ckpt.resolve() != ck_store.resolve():
                    raise SystemExit(f"{path}: checkpoint {ckpt}, store {ck_store}")
            pos = {k: r for r, k in enumerate(keys)}
            have = np.asarray([k in pos for k in idx.keys])
            Z = np.full((n, Zr.shape[1]), np.nan)
            Z[have] = Zr[[pos[k] for k, h in zip(idx.keys, have) if h]]
            rel = str(path.relative_to(diag.REPO)) if path.is_relative_to(diag.REPO) else str(path)
            ctrl[f"{view}|{pre}|{kind}"] = {"meshes": int(have.sum()), "of": n, "embeddings": rel}
            for d, M in fp.distances(Z, i, j, ckpt, cal.get(pre)).items():
                if d in DISTS[pre]:
                    out[f"{pre}|{d}|{kind}"] = M.reshape(n, n)
    return out, ctrl


def view_rows(view: str) -> dict:
    """Righe (all_cross + stessa topologia), colonne, GT, gruppi, conteggi delle repliche e controlli di una vista."""
    df, idx, seed_all = bpe2.all_cross_rows(view)
    ref, _, seed_ref = fp.rows_for("faceverse" if view == "faceverse_neutral" else view)
    ref = ref.astype({"subject_a": str, "subject_b": str})
    k5 = rows_check(df, ref)
    ok = (ref["subject_a"] != ref["subject_b"]).to_numpy()
    for b in [b for b in fp.BASELINES if b in ref]:          # maschera di fact_paired.main (e di D3): baseline finite
        ok &= np.isfinite(ref[b].to_numpy(np.float64))
    keys_ref = set(zip(ref["subject_a"][ok], ref["topology_a"][ok], ref["subject_b"][ok], ref["topology_b"][ok]))
    key = ["subject_a", "subject_b"]
    nocrop = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    g = nocrop.groupby(key)[["gt_fr", "gt_sr"]]
    spread = float((g.max() - g.min()).to_numpy().max())       # GT costante dentro la coppia di soggetti
    pairs = g.first().reset_index()
    same = pd.concat([pairs.assign(topology_a=t, topology_b=t) for t in NOCROP], ignore_index=True)
    c4 = key + ["topology_a", "topology_b", "gt_fr", "gt_sr"]
    rows = pd.concat([df[c4].assign(src="all_cross"), same[c4].assign(src="same")], ignore_index=True)
    M, ctrl = arm_matrices(view, idx)
    rows = be.add_columns(rows, M, idx)
    cols = {m: rows[m].to_numpy(np.float64) for m in M}
    gt = {x: rows[f"gt_{x}"].to_numpy(np.float64) for x in fp.GTS}
    subjects = np.array(sorted(set(nocrop["subject_a"]) | set(nocrop["subject_b"])))
    s2i = {s: k for k, s in enumerate(subjects)}
    sa, sb = rows["subject_a"].map(s2i).to_numpy(), rows["subject_b"].map(s2i).to_numpy()
    fin = np.all([np.isfinite(v) for v in list(cols.values()) + list(gt.values())], axis=0)
    mask = (sa != sb) & fin
    ta, tb, src = rows["topology_a"].to_numpy(str), rows["topology_b"].to_numpy(str), rows["src"].to_numpy(str)
    ac = src == "all_cross"
    crop = (ta == "crop") | (tb == "crop")
    groups = {"nocrop_cross": ac & ~crop, "stessa topologia": src == "same", "all_cross": ac, "righe col crop": ac & crop}
    groups.update({f"stessa: {t}": (src == "same") & (ta == t) for t in NOCROP})
    in_ref = np.asarray([k in keys_ref for k in zip(rows["subject_a"].astype(str), ta, rows["subject_b"].astype(str), tb)])
    groups[D3ROWS] = ac & ~crop & in_ref
    seed = primary_seed(view)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.ones(len(subjects), dtype=np.int64)] +
                      [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects))
                       for _ in range(diag.N_BOOT)])
    info = {"seed": seed, "seed_fact_paired": int(seed_ref), "seed_all_cross": int(seed_all), "subjects": int(len(subjects)),
            "K5": k5, "rows_fact_paired_masked": int(ok.sum()),
            "rows": {k: int(v.sum()) for k, v in groups.items()},
            "rows_masked": {k: int((v & mask).sum()) for k, v in groups.items()},
            "dropped_by_mask": {k: int((v & (sa != sb) & ~fin).sum()) for k, v in groups.items()},
            "nan_rows_by_column": {c: int((~np.isfinite(x) & (sa != sb)).sum()) for c, x in cols.items()
                                   if not np.isfinite(x[sa != sb]).all()},
            "gt_spread_within_subject_pair": spread, "embeddings": ctrl}
    return {"cols": {k: v[mask] for k, v in cols.items()}, "gt": {k: v[mask] for k, v in gt.items()},
            "sa": sa[mask], "sb": sb[mask], "groups": {k: np.flatnonzero(v[mask]) for k, v in groups.items()},
            "counts": counts, "info": info, "df_all": df}


def rows_check(df: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """K5 (``bp_paired_e2.check_rows``, sulle righe di ``fact_paired.rows_for`` gia' calcolate): tolto il crop, chiavi e
    GT di all_cross = quelle di fact_paired."""
    nc = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    key = ["subject_a", "topology_a", "subject_b", "topology_b"]
    a = nc[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    b = ref[key + ["gt_fr", "gt_sr"]].astype({"subject_a": str, "subject_b": str}).sort_values(key).reset_index(drop=True)
    same = len(a) == len(b) and bool((a[key].to_numpy() == b[key].to_numpy()).all())
    gd = float(max(np.abs(a[g] - b[g]).max() for g in ("gt_fr", "gt_sr"))) if same else None
    return {"rows_nocrop": int(len(a)), "rows_fact_paired": int(len(b)), "keys_equal": same, "max_abs_diff_gt": gd,
            "pass": bool(same and gd == 0.0)}


def _rep(k: int) -> dict:
    """Spearman pesato c_a c_b di ogni colonna con ogni GT, per gruppo, alla replica k."""
    J = _J
    c = J["counts"][k]
    wt = (c[J["sa"]] * c[J["sb"]]).astype(np.float64)
    out = {}
    for g, rows in J["groups"].items():
        w = wt[rows]
        sel = rows[w > 0]
        w = w[w > 0]
        rg = {x: diag.wranks(J["gt"][x][sel], w) for x in fp.GTS}
        for m, x in J["cols"].items():
            r = diag.wranks(x[sel], w)
            for gt in fp.GTS:
                out[(g, gt, m)] = diag.wpearson(r, rg[gt], w)
    return out


def ci(x: np.ndarray) -> tuple[float, float, float, float]:
    return diag.ci(x)


def run(workers: int) -> None:
    global _J
    t0 = time.time()
    recs, drecs, controls = [], [], {"views": {}}
    for view in VIEWS:
        T = view_rows(view)
        _J = T
        with mp.get_context("fork").Pool(workers) as pool:
            reps = pool.map(_rep, range(len(T["counts"])), chunksize=4)
        V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
        controls["views"][view] = T["info"]
        print(f"[d4-stats] {view}: seme {T['info']['seed']}, righe {T['info']['rows_masked']} "
              f"({time.time() - t0:.0f}s)", flush=True)
        for (g, gt, m), a in V.items():
            pre, d, kind = m.split("|")
            p, lo, hi, _ = ci(a)
            recs.append({"domain": view, "group": g, "gt": gt, "arm": pre, "distance": d, "input": kind, "column": m,
                         "rho": p, "ci_low": lo, "ci_high": hi, "n_rows": int(len(T["groups"][g])),
                         "seed": T["info"]["seed"]})
        for pre, dists in DISTS.items():
            for d in dists:
                o, c = f"{pre}|{d}|orig", f"{pre}|{d}|canon"
                for gt in fp.GTS:
                    for g in GROUPS + (D3ROWS,):             # canonico - originale
                        drecs.append(_drec(view, g, gt, pre, d, "canonico - originale", V[(g, gt, c)] - V[(g, gt, o)],
                                           V[(g, gt, c)], V[(g, gt, o)], len(T["groups"][g]), T["info"]["seed"]))
                    for kind, m in (("originale", o), ("canonico", c)):     # residuo stessa topologia - nocrop_cross
                        drecs.append(_drec(view, "stessa topologia - nocrop_cross", gt, pre, d, f"residuo, {kind}",
                                           V[("stessa topologia", gt, m)] - V[("nocrop_cross", gt, m)],
                                           V[("stessa topologia", gt, m)], V[("nocrop_cross", gt, m)],
                                           len(T["groups"]["nocrop_cross"]), T["info"]["seed"]))
                    ro = V[("stessa topologia", gt, o)] - V[("nocrop_cross", gt, o)]
                    rc = V[("stessa topologia", gt, c)] - V[("nocrop_cross", gt, c)]
                    drecs.append(_drec(view, "stessa topologia - nocrop_cross", gt, pre, d,
                                       "residuo canonico - residuo originale", rc - ro, rc, ro,
                                       len(T["groups"]["nocrop_cross"]), T["info"]["seed"]))
        np.savez(D4 / f"reps_{view}.npz", keys=np.asarray(["|".join(k) for k in V]), V=np.stack(list(V.values())))
    S, Dl = pd.DataFrame(recs), pd.DataFrame(drecs)
    S.to_csv(EV / "d4_spearman.csv", index=False)
    Dl.to_csv(EV / "d4_delta.csv", index=False)
    controls.update(checks(S, controls["views"]))
    controls["seconds"] = time.time() - t0
    diag.atomic_json(D4 / "controls.json", controls)
    diag.atomic_json(D4 / "readings.json", readings(Dl))
    print(f"[d4-stats] {len(S)} valori, {len(Dl)} delta in {time.time() - t0:.0f}s", flush=True)


def _drec(view, g, gt, pre, d, what, x, a, b, n, seed) -> dict:
    p, lo, hi, ple = ci(x)
    return {"domain": view, "group": g, "gt": gt, "arm": pre, "distance": d, "contrast": what, "delta": p,
            "ci_low": lo, "ci_high": hi, "p_le0": ple, "a_point": float(a[0]), "b_point": float(b[0]), "n_rows": int(n),
            "seed": seed}


# ------------------------------------------------------------------------------------------ controlli

def checks(S: pd.DataFrame, views: dict) -> dict:
    out = {}
    d3 = pd.read_csv(EV / "d3_spearman.csv")
    k1 = []
    for view in VIEWS:
        for pre in RULE_ARMS:
            r = d3[(d3["test"] == view) & (d3["rows"] == "cross") & (d3["column"] == f"{pre}|dP|shape") & (d3["gt"] == "sr")]
            m = S[(S["domain"] == view) & (S["group"] == D3ROWS) & (S["gt"] == "sr") &
                  (S["column"] == f"{pre}|shape|orig")]
            if len(r) and len(m):
                k1.append({"view": view, "arm": pre, "d3": float(r.iloc[0]["rho"]), "d4": float(m.iloc[0]["rho"]),
                           "abs_diff": abs(float(r.iloc[0]["rho"]) - float(m.iloc[0]["rho"])),
                           "rows_d3": int(r.iloc[0]["n_rows"]), "rows_d4": int(m.iloc[0]["n_rows"]),
                           "seed_d3": int(r.iloc[0]["seed"]), "seed_d4": int(m.iloc[0]["seed"])})
    out["K1"] = {"cells": k1, "max_abs_diff": max((x["abs_diff"] for x in k1), default=float("nan")),
                 "rows_equal": all(x["rows_d3"] == x["rows_d4"] for x in k1),
                 "pass": bool(k1 and all(x["abs_diff"] <= 1e-6 and x["rows_d3"] == x["rows_d4"] for x in k1))}
    p = D4 / "emb_check" / "hifi3d_factorized_s1234" / "embeddings.npz"
    if p.exists():
        with np.load(p, allow_pickle=True) as z, np.load(R.bp_e3.store("hifi3d", "factorized", "072"), allow_pickle=True) as r:
            a = dict(zip(zip([str(x) for x in z["subjects"]], [str(x) for x in z["topologies"]]), np.asarray(z["Z"], np.float64)))
            b = dict(zip(zip([str(x) for x in r["subjects"]], [str(x) for x in r["topologies"]]), np.asarray(r["Z"], np.float64)))
        diff = float(max(np.abs(a[k] - b[k]).max() for k in a)) if sorted(a) == sorted(b) else float("nan")
        out["K2"] = {"n": len(a), "keys_equal": sorted(a) == sorted(b), "max_abs_diff": diff, "pass": bool(diff <= 1e-3)}
    else:
        out["K2"] = {"missing": str(p), "pass": False}
    rs = json.loads((D4 / "remesh_stats.json").read_text())
    out["K3"] = rs.get("K3_determinism", {"pass": False})
    out["K4"] = rs.get("K4_validity", {})
    out["K5"] = {v: views[v]["K5"] for v in VIEWS}
    out["seeds"] = {v: {"d4": views[v]["seed"], "fact_paired": views[v]["seed_fact_paired"], "d3": SEEDS_D3[v],
                        "pass": views[v]["seed"] == views[v]["seed_fact_paired"] == SEEDS_D3[v]} for v in VIEWS}
    return out


# ------------------------------------------------------------------------------------------ lettura

def readings(Dl: pd.DataFrame) -> dict:
    """Sez. 5: P(seme), verdetto, DANNOSA; domini validi da K4 (> 1% non valide: fuori dalla regola)."""
    rs = json.loads((D4 / "remesh_stats.json").read_text()).get("K4_validity", {})
    valid = [v for v in VIEWS if rs.get(v, {}).get("fraction_invalid", 1.0) <= 0.01]
    x = Dl[(Dl["group"] == "nocrop_cross") & (Dl["contrast"] == "canonico - originale")]
    out = {"valid_domains": valid, "arms": {}}
    for pre in RULE_ARMS:
        sr = x[(x["arm"] == pre) & (x["distance"] == "shape") & (x["gt"] == "sr")].set_index("domain")
        fr = x[(x["arm"] == pre) & (x["distance"] == "form_cal") & (x["gt"] == "fr")].set_index("domain")
        up = [v for v in valid if v in sr.index and sr.loc[v, "delta"] >= SR_MIN and sr.loc[v, "ci_low"] > 0]
        down = [v for v in valid if v in sr.index and sr.loc[v, "delta"] <= HARM and sr.loc[v, "ci_high"] < 0]
        fr_ok = all(v in fr.index and fr.loc[v, "delta"] >= FR_MIN for v in valid)
        out["arms"][pre] = {"sr_up_domains": up, "fr_ok_all": bool(fr_ok), "P": bool(len(up) >= N_DOM and fr_ok),
                            "harm_domains": down, "harm": bool(len(down) >= N_DOM),
                            "sr": {v: [float(sr.loc[v, c]) for c in ("delta", "ci_low", "ci_high")] for v in sr.index},
                            "fr": {v: [float(fr.loc[v, c]) for c in ("delta", "ci_low", "ci_high")] for v in fr.index}}
    complete = len(valid) == len(VIEWS) or len(valid) >= N_DOM
    n_p = sum(out["arms"][a]["P"] for a in RULE_ARMS)
    out["verdict"] = ("non valutabile" if not complete else
                      "UTILE" if n_p == 2 else "NON CONFERMATO" if n_p == 1 else "NO")
    out["harmful"] = bool(all(out["arms"][a]["harm"] for a in RULE_ARMS))
    return out


# ------------------------------------------------------------------------------------------- markdown

def f3(p, lo, hi, sign=True) -> str:
    s = "+" if sign else ""
    return f"{p:{s}.3f} [{lo:{s}.3f}, {hi:{s}.3f}]"


def embed_times() -> dict:
    """{uscita relativa: secondi} dal log piu' recente di d4_embed.sbatch (righe ``gpu X: N s <uscita>``)."""
    logs = sorted((diag.REPO / "aau/logs").glob("wbes-d4-embed-*.out"), key=lambda q: q.stat().st_mtime)
    out = {}
    for ln in (logs[-1].read_text().splitlines() if logs else []):
        m = re.search(r"gpu \S+: (\d+) s (\S+)$", ln)
        if m:
            out[str(Path(m.group(2)).parent).split("/diagnostics/d4/", 1)[-1]] = int(m.group(1))
    return out


def summary() -> None:
    S = pd.read_csv(EV / "d4_spearman.csv")
    Dl = pd.read_csv(EV / "d4_delta.csv")
    C = json.loads((D4 / "controls.json").read_text())
    Rd = json.loads((D4 / "readings.json").read_text())
    rs = json.loads((D4 / "remesh_stats.json").read_text())
    cost = json.loads((D4 / "cost.json").read_text()) if (D4 / "cost.json").exists() else None
    sha = (EV / "PROTOCOL_D4.sha256").read_text().split()[0]
    md = ["# Diagnostica D4: ridiscretizzazione canonica al test, risultati", "",
          f"Protocollo `PROTOCOL_D4.md` (sha256 {sha[:8]}..., commit 5fd3061), scritto prima dei numeri. Remesh isotropico "
          f"esplicito di pymeshlab {rs['definition']['pymeshlab']} a lato {rs['definition']['L_mm']} mm su tutte le 6 "
          "topologie dei 100 soggetti valutati; tabella di scala con l'area della mesh canonica, operatori areanorm k 128, "
          "embedding degli store. Generato da `aau/diagnostics/d4_stats.py`; numeri in `d4_spearman.csv`, `d4_delta.csv`, "
          "`d4/controls.json`, `d4/readings.json`, `d4/remesh_stats.json`, `d4/cost.json`. IC 95% percentile, 1000 "
          "repliche per soggetto, semi del gruppo primario ufficiale (" +
          ", ".join(f"{v} {C['views'][v]['seed']}" for v in VIEWS) + ").", "", "## Lettura preregistrata (sez. 5)", ""]
    md.append(f"- **Verdetto: {Rd['verdict']}**" + (" e DANNOSA" if Rd["harmful"] else "") +
              f" (domini validi per la regola: {', '.join(Rd['valid_domains'])}).")
    for pre in RULE_ARMS:
        a = Rd["arms"][pre]
        md.append(f"  - {pre}: P = {a['P']} (d_P con SR >= +0.02 e IC > 0 in {len(a['sr_up_domains'])} domini "
                  f"{a['sr_up_domains']}; d_F cal. con FR >= -0.02 ovunque: {a['fr_ok_all']}); dannosa in "
                  f"{len(a['harm_domains'])} domini {a['harm_domains']}; " +
                  "; ".join(f"{v} SR {f3(*a['sr'][v])}, FR {f3(*a['fr'][v])}" for v in VIEWS if v in a["sr"]))
    md += ["", "## Delta canonico - originale (stesse righe e repliche)", "",
           "| dominio | braccio | distanza | GT | nocrop_cross (primario) | stessa topologia | all_cross | righe col crop |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    x = Dl[Dl["contrast"] == "canonico - originale"]
    for v in VIEWS:
        for pre, decl in DECLARED.items():
            for d, gt, role in decl:
                cells = []
                for g in ("nocrop_cross", "stessa topologia", "all_cross", "righe col crop"):
                    r = x[(x["domain"] == v) & (x["arm"] == pre) & (x["distance"] == d) & (x["gt"] == gt) & (x["group"] == g)]
                    cells.append(f3(r.iloc[0].delta, r.iloc[0].ci_low, r.iloc[0].ci_high) if len(r) else "-")
                b = "**" if role == "primaria" else ""
                md.append(f"| {v} | {pre} | {LABEL[d]} ({role}) | {gt.upper()} | {b}{cells[0]}{b} | " + " | ".join(cells[1:]) + " |")
    md += ["", f"Sensibilita' (descrittiva, non nel protocollo): nocrop_cross ristretto alle righe di D3 e di fact_paired "
               "(la loro maschera toglie le righe con baseline geometriche non finite), delta canonico - originale:", "",
           "| dominio | braccio | distanza | GT | righe | delta |", "| --- | --- | --- | --- | --- | --- |"]
    for v in VIEWS:
        for pre, decl in DECLARED.items():
            for d, gt, _ in decl:
                r = x[(x["domain"] == v) & (x["arm"] == pre) & (x["distance"] == d) & (x["gt"] == gt) & (x["group"] == D3ROWS)]
                if len(r):
                    md.append(f"| {v} | {pre} | {LABEL[d]} | {gt.upper()} | {int(r.iloc[0].n_rows)} | "
                              f"{f3(r.iloc[0].delta, r.iloc[0].ci_low, r.iloc[0].ci_high)} |")
    md += ["", "Valori (rho, originale -> canonico):", "",
           "| dominio | braccio | distanza | GT | nocrop_cross | stessa topologia | all_cross | righe col crop |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in VIEWS:
        for pre, decl in DECLARED.items():
            for d, gt, _ in decl:
                cells = []
                for g in ("nocrop_cross", "stessa topologia", "all_cross", "righe col crop"):
                    y = S[(S["domain"] == v) & (S["arm"] == pre) & (S["distance"] == d) & (S["gt"] == gt) & (S["group"] == g)]
                    o, c = y[y["input"] == "orig"], y[y["input"] == "canon"]
                    cells.append(f"{o.iloc[0].rho:.3f} -> {c.iloc[0].rho:.3f}" if len(o) and len(c) else "-")
                md.append(f"| {v} | {pre} | {LABEL[d]} | {gt.upper()} | " + " | ".join(cells) + " |")
    md += ["", "## Residuo stessa topologia - nocrop_cross, prima e dopo", "",
           "Frazione recuperata = delta canonico - originale su nocrop_cross / residuo originale (solo dove il residuo "
           "originale ha IC sopra 0; descrittiva).", "",
           "| dominio | braccio | distanza | GT | residuo originale | residuo canonico | canonico - originale | frazione recuperata |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    y = Dl[Dl["group"] == "stessa topologia - nocrop_cross"]
    for v in VIEWS:
        for pre, decl in DECLARED.items():
            for d, gt, _ in decl:
                q = y[(y["domain"] == v) & (y["arm"] == pre) & (y["distance"] == d) & (y["gt"] == gt)].set_index("contrast")
                ro, rc, dd = (q.loc[k] for k in ("residuo, originale", "residuo, canonico", "residuo canonico - residuo originale"))
                dn = x[(x["domain"] == v) & (x["arm"] == pre) & (x["distance"] == d) & (x["gt"] == gt) &
                       (x["group"] == "nocrop_cross")].iloc[0]
                frac = f"{dn.delta / ro.delta:.2f}" if ro.ci_low > 0 else "-"
                md.append(f"| {v} | {pre} | {LABEL[d]} | {gt.upper()} | {f3(ro.delta, ro.ci_low, ro.ci_high)} | "
                          f"{f3(rc.delta, rc.ci_low, rc.ci_high)} | {f3(dd.delta, dd.ci_low, dd.ci_high)} | {frac} |")
    md += ["", "Stessa topologia per etichetta (rho originale -> canonico):", "",
           "| dominio | braccio | distanza | GT | " + " | ".join(NOCROP) + " |", "| --- | --- | --- | --- |" + " --- |" * len(NOCROP)]
    for v in VIEWS:
        for pre, decl in DECLARED.items():
            for d, gt, _ in decl:
                cells = []
                for t in NOCROP:
                    q = S[(S["domain"] == v) & (S["arm"] == pre) & (S["distance"] == d) & (S["gt"] == gt) &
                          (S["group"] == f"stessa: {t}")]
                    o, c = q[q["input"] == "orig"], q[q["input"] == "canon"]
                    cells.append(f"{o.iloc[0].rho:.3f} -> {c.iloc[0].rho:.3f}")
                md.append(f"| {v} | {pre} | {LABEL[d]} | {gt.upper()} | " + " | ".join(cells) + " |")
    md += ["", "## Remesh: validita' e geometria (sez. 1)", "",
           "| vista | topologia | vertici prima (mediana) | dopo (mediana, min-max) | lato mm (mediana) | CV del lato | "
           "area dopo / prima (mediana, min) | Hausdorff ingresso -> canonica mm (media mediana, max p90) | piu' componenti | "
           "con spigoli non-manifold | s per mesh (mediana, p90) |", "| --- |" + " --- |" * 10]
    for v in VIEWS:
        for t, b in rs[v]["by_topology"].items():
            md.append(f"| {v} | {t} | {b['vertices_in_median']:.0f} | {b['vertices_out_median']:.0f} "
                      f"({b['vertices_out_min']:.0f}-{b['vertices_out_max']:.0f}) | {b['edge_median_mm']:.2f} | "
                      f"{b['edge_cv_median']:.2f} | {b['area_out_over_in_median']:.3f} ({b['area_out_over_in_min']:.3f}) | "
                      f"{b['hausdorff_mean_mm_median']:.3f} ({b['hausdorff_max_mm_p90']:.2f}) | {b['multi_component']} | "
                      f"{b['with_nonmanifold_edges']} | {b['seconds_median']:.1f} ({b['seconds_p90']:.1f}) |")
    if cost:
        s = cost["sample"]
        md += ["", "## Costo per mesh (sez. 6)", "",
               f"Campione di {s['ops_original_s']['n']} mesh (primi {cost['n_subjects_per_view']} soggetti valutati x 6 "
               f"topologie x 3 viste), un thread, nodo {cost['host']}; remesh su tutte le 1.800: mediana "
               f"{cost['remesh_all_1800']['median']:.2f} s, p90 {cost['remesh_all_1800']['p90']:.2f} s.", "",
               "| | mediana s | p90 s | media s |", "| --- | --- | --- | --- |"]
        for k, lab in (("ops_original_s", "operatori sull'originale (pipeline attuale)"), ("remesh_s", "remesh"),
                       ("ops_canonical_s", "operatori sulla canonica"), ("canonical_total_s", "remesh + operatori canonici")):
            md.append(f"| {lab} | {s[k]['median']:.2f} | {s[k]['p90']:.2f} | {s[k]['mean']:.2f} |")
        r = s["ratio_canonical_total_over_original_ops"]
        md += ["", f"Rapporto (remesh + operatori canonici) / operatori originali per mesh: mediana {r['median']:.2f} "
                   f"(p10 {r['p10']:.2f}, p90 {r['p90']:.2f}).", "",
               "| vista | operatori originali (mediana s) | remesh + operatori canonici (mediana s) |", "| --- | --- | --- |"]
        for v, b in cost["sample_by_view"].items():
            md.append(f"| {v} | {b['ops_original_s']['median']:.2f} | {b['canonical_total_s']['median']:.2f} |")
        md += ["", "| topologia | operatori originali (mediana s) | remesh + operatori canonici (mediana s) |",
               "| --- | --- | --- |"]
        for t, b in cost["sample_by_topology"].items():
            md.append(f"| {t} | {b['ops_original_s']['median']:.2f} | {b['canonical_total_s']['median']:.2f} |")
    et = embed_times()
    if et:
        md += ["", "Embedding, tempo a parete per 600 mesh nello stesso job (2 code su 2 A100, preload compreso): " +
               "; ".join(f"`{k}` {v} s" for k, v in sorted(et.items())) + ". Confronto dichiarato: "
               f"HIFI3D canonica {et.get('emb/hifi3d/factorized_s1234', '-')} s contro originale (controllo della catena) "
               f"{et.get('emb_check/hifi3d_factorized_s1234', '-')} s."]
    md += ["", "## Controlli", ""]
    k1 = C["K1"]
    md.append(f"- K1 (originale = D3, d_P con SR, sulle righe comuni = righe di D3): scarto massimo {k1['max_abs_diff']:.1e}, righe uguali "
              f"{k1['rows_equal']}: **{'passa' if k1['pass'] else 'NON passa'}** (" +
              "; ".join(f"{c['view']} {c['arm']} {c['d4']:.4f} contro {c['d3']:.4f}, righe {c['rows_d4']} / {c['rows_d3']}"
                        for c in k1["cells"]) + ").")
    k2 = C["K2"]
    md.append(f"- K2 (catena: factorized s1234 sulla cache dello store di HIFI3D): " +
              (f"{k2.get('n')} mesh, scarto massimo {k2.get('max_abs_diff', float('nan')):.1e}" if "n" in k2 else k2.get("missing", "")) +
              f": **{'passa' if k2['pass'] else 'NON passa'}**.")
    md.append(f"- K3 (determinismo del remesher, {len(C['K3'].get('meshes', []))} mesh): "
              f"**{'passa' if C['K3'].get('pass') else 'NON passa'}**.")
    md.append("- K4 (validita'): " + "; ".join(f"{v} {b['valid']} / {b['expected']} valide" for v, b in C["K4"].items()) +
              f": **{'passa' if all(b['pass'] for b in C['K4'].values()) else 'NON passa'}**.")
    md.append("- K5 (righe nocrop = fact_paired.rows_for): " + "; ".join(
        f"{v} {b['rows_nocrop']} righe, {'chiavi uguali' if b.get('keys_equal') else 'chiavi DIVERSE'} (scarto GT "
        f"{b.get('max_abs_diff_gt')})" for v, b in C["K5"].items()) +
        f": **{'passa' if all(b['pass'] for b in C['K5'].values()) else 'NON passa'}**. Semi: " +
        "; ".join(f"{v} {b['d4']} (fact_paired {b['fact_paired']}, D3 {b['d3']})" for v, b in C["seeds"].items()) + ".")
    md.append("- Righe per gruppo dopo la maschera: " + "; ".join(
        f"{v}: " + ", ".join(f"{g} {n}" for g, n in C["views"][v]["rows_masked"].items() if not g.startswith("stessa: "))
        for v in VIEWS) + "; tolte dalla maschera: " + "; ".join(
        f"{v} {sum(C['views'][v]['dropped_by_mask'][g] for g in ('all_cross', 'stessa topologia'))}" for v in VIEWS) + ".")
    (EV / "results_d4.md").write_text("\n".join(md) + "\n")
    print(f"[d4-stats] {EV / 'results_d4.md'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    if not a.summary_only:
        run(a.workers)
    summary()


if __name__ == "__main__":
    main()
