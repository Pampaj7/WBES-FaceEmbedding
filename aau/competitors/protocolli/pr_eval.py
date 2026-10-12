#!/usr/bin/env python3
"""Protocolli della comunita': Spearman con GT FR e SR (IC) e confronti appaiati con ICP rigido in mm e il nostro.

    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_eval.py spearman --workers 32    (LEGGE LE GT di test)
    v3_work/unified_gt/run.sh aau/competitors/protocolli/pr_eval.py results                  (solo results.md)
    (pr.sbatch, passo eval)

Righe: quelle della matrice unificata (``bp_paired_e2.all_cross_rows``: HIFI3D, FaceScape dev, FaceVerse con
espressioni, FaceVerse neutra con le righe e la GT di FaceVerse) e FaMoS TEST (``fact_paired.famos_frame``, blocco
scan gallery -> scan, 105 righe). Insiemi come ``aau/full_matrix/fm_eval.groups_of``: ``nocrop`` (topologie diverse,
nessuna crop), ``crop`` (una crop), ``all_cross``, ``con_perturbazione`` P (P su un lato, l'altro una topologia
diversa senza crop; per P = crop e' ``crop``). Repliche: 1000 per soggetto col seme della T1 del dominio (quello di
``fm_eval``: HIFI3D 990708, FaceScape 796786, FaceVerse e neutra 271049; FaMoS ``famos_eval.bootstrap_counts(15, n,
1234)``), Spearman pesato c_a c_b con ``fact_paired._rep`` (attraverso ``bp_paired_e3.replicas``: rho con GT FR e SR,
parziale dato l'oracolo, quintile basso). Maschera comune fra le colonne di QUESTO confronto (righe con tutte finite).

Colonne: i 4 protocolli (``competitors_new/<metodo>/<vista>.npz``, ``D_mm``, media dei due versi) e, per i confronti,
ICP + Chamfer in mm per coppia (``blmm``, colonna ``mm_rigid_icp_chamfer`` della matrice) e il nostro factorized
(semi 1234 e 2345: d_F non calibrata con FR, d_P con SR; ``fact_paired.model_columns``; vista neutra come
``fm_eval.main``). Delta appaiati (replica per replica, IC 95% percentile, ``p_le0``): NoW originale - NoW metrical,
protocollo - ICP mm, protocollo - nostro (negativo = il nostro ordina meglio). Uscite in
``aau/runs/evidence/protocolli/``: ``spearman.csv``, ``deltas.csv``, ``controls.json``, ``results.md``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_common as pc  # noqa: E402

SEEDS = {"hifi3d": 990708, "facescape": 796786, "faceverse": 271049, "faceverse_neutral": 271049}
OURS = {"fr": ("factorized_s1234|form", "factorized_s2345|form"), "sr": ("factorized_s1234|shape", "factorized_s2345|shape")}
ICP = "mm_rigid_icp_chamfer"
CLABEL = {**pc.LABEL, ICP: "ICP + Chamfer in mm (per coppia)",
          "factorized_s1234|form": "factorized s1234, d_F non cal.", "factorized_s2345|form": "factorized s2345, d_F non cal.",
          "factorized_s1234|shape": "factorized s1234, d_P", "factorized_s2345|shape": "factorized s2345, d_P"}
SETS = [("nocrop", "-"), ("crop", "crop"), ("all_cross", "-")] + \
    [("con_perturbazione", t) for t in ("original", "remesh", "down8k", "up60k", "noisy")]


def t1_seed(view: str) -> int:
    """Seme della T1 dalle frame di E12 (``fact_paired.rows_for``: ``fr[gruppo][2]``), controllato contro SEEDS."""
    import fact_paired as fp
    src = "faceverse" if view == "faceverse_neutral" else view
    e = fp.be.e12m
    s = {"hifi3d": lambda: e.hifi_frames()[0]["nocrop_cross"][2], "facescape": lambda: e.fs_frames()["nocrop_cross"][2],
         "faceverse": lambda: e.fv_frames()["mesh_pair_nocrop"][2]}[src]()
    if int(s) != SEEDS[view]:
        raise SystemExit(f"{view}: seme della T1 {s} diverso da {SEEDS[view]}")
    return int(s)


def plugin_matrix(method: str, view: str, keys: list) -> np.ndarray:
    with np.load(pc.PLUGIN_ROOT / method / f"{view}.npz") as z:
        mine = list(zip(z["subjects"].astype(str), z["topologies"].astype(str)))
        D = np.asarray(z["D_mm"], np.float64)
    pos = {k: r for r, k in enumerate(mine)}
    if sorted(mine) != sorted(keys):
        raise SystemExit(f"{method} {view}: mesh diverse dalle chiavi delle righe")
    o = np.asarray([pos[k] for k in keys])
    return D[np.ix_(o, o)]


def methods_of(view: str) -> list:
    return [m for m in pc.METHODS if (pc.PLUGIN_ROOT / m / f"{view}.npz").exists()]


def ref_columns(view: str, idx) -> dict:
    """ICP mm per coppia (blmm, come ``fm_eval.geo_columns``) e il nostro (``fact_paired.model_columns``)."""
    import bp_paired as bpp
    fp, be, blmm = bpp.fp, bpp.be, bpp.blmm
    if view == "faceverse_neutral":                       # come fm_eval.main / bp_paired.main
        fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
        fp.FORM_DIR["fvn"] = os.path.relpath(bpp.NEUTRAL_EMB, fp.EVAL)
    out = {ICP: be.zes.facebench_distances(blmm.view_root(view) / "mm", "rigid_icp_chamfer", idx)}
    arms = fp.model_columns(view, idx)
    out.update({c: arms[c] for g in OURS.values() for c in g})
    return out


def groups(ta: np.ndarray, tb: np.ndarray, mask: np.ndarray) -> dict:
    cross = (ta != tb) & mask
    crop = (ta == "crop") | (tb == "crop")
    G = {("nocrop", "-"): cross & ~crop, ("crop", "crop"): cross & crop, ("all_cross", "-"): cross}
    for t in ("original", "remesh", "down8k", "up60k", "noisy"):
        G[("con_perturbazione", t)] = cross & ((ta == t) | (tb == t)) & ~crop
    return G


def records(V: dict, view: str, g: tuple, n_rows: int, seed: int, cols: list) -> tuple[list, list]:
    S, Dl = [], []

    def ci(x):
        b = x[1:][np.isfinite(x[1:])]
        return (tuple(np.percentile(b, [2.5, 97.5])) if len(b) else (np.nan, np.nan)), b

    base = {"dominio": view, "insieme_righe": g[0], "perturbazione": g[1], "n_righe": n_rows, "seme": seed}
    for (gt, kind, m), a in V.items():
        (lo, hi), b = ci(a)
        S.append({**base, "gt": gt, "kind": kind, "colonna": m, "metodo": CLABEL.get(m, m), "rho": a[0], "ic_lo": lo,
                  "ic_hi": hi, "n_boot": len(b)})
    prot = [m for m in pc.METHODS if m in cols]
    for gt in ("fr", "sr"):
        comp = []
        if "now_original" in prot and "now_metrical" in prot:
            comp.append(("now_original", "now_metrical", "NoW originale - NoW metrical"))
        comp += [(m, ICP, "protocollo - ICP mm") for m in prot]
        comp += [(m, o, "protocollo - nostro") for m in prot for o in OURS[gt]]
        for a_, b_, tipo in comp:
            if (gt, "rho", a_) not in V or (gt, "rho", b_) not in V:
                continue
            d = V[(gt, "rho", a_)] - V[(gt, "rho", b_)]
            (lo, hi), b = ci(d)
            Dl.append({**base, "gt": gt, "tipo": tipo, "colonna_a": a_, "metodo_a": CLABEL[a_], "colonna_b": b_,
                       "metodo_b": CLABEL[b_], "rho_a": V[(gt, "rho", a_)][0], "rho_b": V[(gt, "rho", b_)][0],
                       "delta": d[0], "ic_lo": lo, "ic_hi": hi, "p_le0": float((b <= 0).mean()) if len(b) else np.nan})
    return S, Dl


def run_view(view: str, n_boot: int, workers: int) -> tuple[list, list, dict]:
    import bp_paired_e2 as bpe2
    import bp_paired_e3 as bpe3
    df, idx, _ = bpe2.all_cross_rows(view)
    seed = t1_seed(view)
    keys = list(idx.keys)
    D = {m: plugin_matrix(m, view, keys) for m in methods_of(view)}
    D.update(ref_columns(view, idx))
    df = bpe3.be.add_columns(df, D, idx)
    cols = list(D)
    X = {m: df[m].to_numpy(np.float64) for m in cols + ["oracle_size"]}
    X.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in ("fr", "sr")})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    mask = (sa != sb) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    ta, tb = df["topology_a"].to_numpy(str), df["topology_b"].to_numpy(str)
    S, Dl = [], []
    for g, gm in groups(ta, tb, mask).items():
        V = bpe3.replicas(X, sa, sb, gm, cols, counts, workers)
        s, d = records(V, view, g, int(gm.sum()), seed, cols)
        S += s
        Dl += d
    info = {"seed": seed, "rows": int(len(df)), "rows_mask": int(mask.sum()),
            "rows_nan_by_column": {m: int((~np.isfinite(X[m]) & (sa != sb)).sum()) for m in cols}}
    print(f"[pr-eval] {view}: {int(mask.sum())}/{len(df)} righe, colonne {cols}, seme {seed}", flush=True)
    return S, Dl, info


def run_famos(n_boot: int, workers: int) -> tuple[list, list, dict]:
    import bp_paired_e3 as bpe3
    fp = bpe3.fp
    import famos_common as fc
    cols, G, sa, sb, counts, _ = fp.famos_frame(n_boot)
    with np.load(fc.VIEW_DIR / "gt_matrix.npz") as z:
        gt_names = [str(x) for x in z["names"]]
    keys = [(s, "scan_gallery") for s in gt_names]
    X = {m: plugin_matrix(m, "famos", keys)[sa, sb] for m in methods_of("famos")}
    X.update({c: cols[c] for c in [ICP] + [c for g in OURS.values() for c in g] + ["oracle_size"]})
    X.update({f"gt_{g}": G[g] for g in ("fr", "sr")})
    use = [m for m in X if m != "oracle_size" and not m.startswith("gt_")]
    mask = (sa != sb) & np.all([np.isfinite(v) for v in X.values()], axis=0)
    V = bpe3.replicas(X, sa, sb, mask, use, counts, workers)
    S, Dl = records(V, "famos", ("nocrop", "-"), int(mask.sum()), 1234, use)
    print(f"[pr-eval] famos: {int(mask.sum())}/{len(sa)} righe, colonne {use}", flush=True)
    return S, Dl, {"seed": 1234, "rows": int(len(sa)), "rows_mask": int(mask.sum())}


def spearman(n_boot: int, workers: int, views: tuple = pc.VIEWS) -> None:
    sys.path.insert(0, str(pc.AAU / "famos"))
    t0 = time.time()
    S, Dl, info = [], [], {}
    for view in views:
        if not methods_of(view):
            info[view] = "nessun protocollo calcolato"
            continue
        s, d, inf = run_famos(n_boot, workers) if view == "famos" else run_view(view, n_boot, workers)
        S += s
        Dl += d
        info[view] = inf
    pc.EVID.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(S).to_csv(pc.EVID / "spearman.csv", index=False)
    pd.DataFrame(Dl).to_csv(pc.EVID / "deltas.csv", index=False)
    info["wall_s"] = time.time() - t0
    info["n_boot"] = n_boot
    pc.atomic_json(pc.EVID / "controls.json", {**info, **pc.stamp()})
    write_results()


# ------------------------------------------------------------------------------------------ risultati

def f3(r, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(r)} [{f.format(lo)}, {f.format(hi)}]"


def write_results() -> None:
    S = pd.read_csv(pc.EVID / "spearman.csv")
    Dl = pd.read_csv(pc.EVID / "deltas.csv")
    S = S[S["kind"] == "rho"]
    views = [v for v in pc.VIEWS if v in set(S["dominio"])]
    L = ["# Protocolli della comunita' sul nostro benchmark: Spearman con GT FR / SR e confronti appaiati", "",
         "Generato da `aau/competitors/protocolli/pr_eval.py` (non modificare a mano; definizioni nel docstring e nel "
         "README di questa cartella). Righe della matrice unificata, 1000 repliche per soggetto col seme della T1, IC "
         "95% percentile, maschera comune fra le colonne di questo confronto. Distanza di un protocollo = media dei due "
         "versi (A predetta, B riferimento e viceversa).", ""]
    cols = list(pc.METHODS) + [ICP] + [c for g in OURS.values() for c in g]
    for gt in ("fr", "sr"):
        for s, p in [("nocrop", "-"), ("crop", "crop"), ("all_cross", "-")]:
            X = S[(S["gt"] == gt) & (S["insieme_righe"] == s) & (S["perturbazione"] == p)]
            if X.empty:
                continue
            vs = [v for v in views if v in set(X["dominio"])]
            L += [f"## GT {gt.upper()}, {s}", "", "| metodo | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
            for c in cols:
                if c.endswith("|form") and gt == "sr" or c.endswith("|shape") and gt == "fr":
                    continue
                cells = []
                for v in vs:
                    r = X[(X["colonna"] == c) & (X["dominio"] == v)]
                    cells.append(f3(r.iloc[0].rho, r.iloc[0].ic_lo, r.iloc[0].ic_hi) if len(r) else "-")
                if any(x != "-" for x in cells):
                    L.append(f"| {CLABEL.get(c, c)} | " + " | ".join(cells) + " |")
            n = {v: int(X[X["dominio"] == v]["n_righe"].iloc[0]) for v in vs}
            L += ["", "Righe: " + ", ".join(f"{v} {n[v]}" for v in vs), ""]
    for gt in ("fr", "sr"):
        X = Dl[(Dl["gt"] == gt) & Dl["insieme_righe"].isin(["nocrop", "crop", "all_cross"])]
        if X.empty:
            continue
        L += [f"## Delta appaiati, GT {gt.upper()} (a - b; negativo = b ordina meglio)", "",
              "| dominio | confronto | nocrop | crop | all_cross |", "|---|---|---|---|---|"]
        for v in views:
            Y = X[X["dominio"] == v]
            for a_, b_ in dict.fromkeys(zip(Y["colonna_a"], Y["colonna_b"])):
                cells = []
                for s in ("nocrop", "crop", "all_cross"):
                    r = Y[(Y["colonna_a"] == a_) & (Y["colonna_b"] == b_) & (Y["insieme_righe"] == s)]
                    cells.append(f3(r.iloc[0].delta, r.iloc[0].ic_lo, r.iloc[0].ic_hi, True) if len(r) else "-")
                L.append(f"| {v} | {CLABEL[a_]} - {CLABEL[b_]} | " + " | ".join(cells) + " |")
        L.append("")
    X = S[(S["insieme_righe"] == "con_perturbazione")]
    if not X.empty:
        L += ["## Per perturbazione (P su un lato, l'altro una topologia diversa senza crop)", ""]
        for gt in ("fr", "sr"):
            ps = ["original", "remesh", "down8k", "up60k", "noisy"]
            L += [f"GT {gt.upper()}:", "", "| dominio | metodo | " + " | ".join(ps) + " |", "|---|---|" + "---|" * len(ps)]
            for v in views:
                for c in list(pc.METHODS) + [ICP, OURS[gt][0]]:
                    Y = X[(X["gt"] == gt) & (X["dominio"] == v) & (X["colonna"] == c)]
                    if Y.empty:
                        continue
                    cells = [f3(*Y[Y["perturbazione"] == p][["rho", "ic_lo", "ic_hi"]].iloc[0]) if
                             len(Y[Y["perturbazione"] == p]) else "-" for p in ps]
                    L.append(f"| {v} | {CLABEL.get(c, c)} | " + " | ".join(cells) + " |")
            L.append("")
    (pc.EVID / "results.md").write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=("spearman", "results"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--views", default=",".join(pc.VIEWS), help="prove: un sottoinsieme delle viste")
    a = p.parse_args()
    if a.step == "results":
        write_results()
    else:
        spearman(a.n_boot, a.workers, tuple(a.views.split(",")))


if __name__ == "__main__":
    main()
