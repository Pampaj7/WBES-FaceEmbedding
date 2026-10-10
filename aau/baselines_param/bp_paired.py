#!/usr/bin/env python3
"""Spearman e delta appaiati (braccio - concorrente parametrico / varifold) sulle righe e repliche di fact_paired.

    v3_work/unified_gt/run.sh aau/baselines_param/bp_paired.py --workers 32
    (bp.sbatch, passo paired; dopo bp_fit.py e bp_varifold.py; protocollo PROTOCOL.md)

``v3_work/trainer/tools/fact_paired.py`` e' IMPORTATO e NON modificato (lo usa il job 1066515): righe, GT, seme e
colonne dei bracci per dominio sono quelli di ``fact_paired.rows_for`` / ``model_columns`` / ``famos_frame``
(HIFI3D e dev FaceScape ``nocrop_cross``, FaceVerse con espressioni ``mesh_pair_nocrop``, FaMoS TEST blocco scan
gallery -> scan), le repliche quelle del suo ``main`` (per soggetto, ``default_rng(seme)``, 1000; FaMoS
``famos_eval.bootstrap_counts(15, n, 1234)``), Spearman e delta con ``fact_paired._rep`` e ``fact_paired.summarize``
(rho, parziale dato l'oracolo, quintile basso). FaceVerse neutra: le stesse righe e GT di FaceVerse (stessi soggetti
e topologie, ``faceverse_neutral/PROTOCOL.md``), bracci dagli embedding della vista neutra
(``aau/runs/evidence/faceverse_neutral/embed``), mesh neutre per le colonne nuove; delle baseline di fact_paired si
tiene solo la taglia oracolo (serve alla parziale ed e' per identita').

Colonne nuove: ``<modello>_{coef,fr,sr}`` dai ``fit.npz`` di bp_fit.py e ``varifold`` dalle Gram di
bp_varifold.py. Maschera COMUNE per dominio come fact_paired (righe con tutte le distanze finite, colonne nuove
comprese): se nessun fit fallisce coincide con la sua, e i valori dei bracci devono ridare factorized_paired.csv
(controllo). Delta = braccio - colonna nuova per ogni colonna dei bracci. Uscite in ``aau/runs/evidence/
baselines_param/``: ``paired.csv`` (tutte le righe di summarize), ``spearman.csv`` (solo i valori dei metodi),
``controls.json``, ``results.md``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS = Path(__file__).resolve().parent
REPO = THIS.parents[1]
sys.path.insert(0, str(REPO / "v3_work/trainer/tools"))
sys.path.insert(0, str(THIS))

import fact_paired as fp  # noqa: E402  (sola lettura)
import bp  # noqa: E402

be, blmm = fp.be, fp.blmm
NEW = [f"{m}_{d}" for m in bp.MODELS for d in ("coef", "fr", "sr")] + ["varifold"]
LABEL = {"gnm_coef": "GNM (visto), coefficienti", "gnm_fr": "GNM (visto), mesh d'identita' FR",
         "gnm_sr": "GNM (visto), mesh d'identita' SR", "flame2023_coef": "FLAME 2023 Open, coefficienti",
         "flame2023_fr": "FLAME 2023 Open, mesh d'identita' FR", "flame2023_sr": "FLAME 2023 Open, mesh d'identita' SR",
         "varifold": "varifold in mm (massa unitaria)"}
NEUTRAL_EMB = REPO / "aau/runs/evidence/faceverse_neutral/embed"
VIEW_ORDER = ("hifi3d", "facescape", "faceverse", "faceverse_neutral", "famos")
# riferimenti delle tabelle: (colonna, etichetta)
REF_ARMS = (("factorized_s1234|form_cal", "factorized s1234, d_F cal."), ("factorized_s2345|form_cal", "factorized s2345, d_F cal."),
            ("factorized_s1234|shape", "factorized s1234, d_P"), ("factorized_s2345|shape", "factorized s2345, d_P"),
            ("ctrlfr_s1234|z", "ctrlfr s1234"), ("ctrlfr_s2345|z", "ctrlfr s2345"))
REF_BL = ("mm_nicp_template", "mm_rigid_icp_chamfer", "cs_nicp_p2tri", "oracle_size")


# ------------------------------------------------------------------------------------------- colonne

def fit_files(view: str) -> dict:
    return {m: bp.out_dir(view, m) / "fit.npz" for m in bp.MODELS}


def varifold_gram(view: str) -> tuple[np.ndarray, list]:
    """(G (S, n, n) dagli shard, chiavi (soggetto, topologia)); errore se manca una cella."""
    parts = sorted(bp.out_dir(view).glob("varifold_*of*.npz"))
    if not parts:
        raise FileNotFoundError(f"{view}: varifold assente")
    G, keys = None, None
    for p in parts:
        with np.load(p) as z:
            k = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
            if keys is not None and k != keys:
                raise SystemExit(f"{p}: mesh diverse dagli altri shard")
            keys = k
            G = z["G"] if G is None else np.where(np.isfinite(G), G, z["G"])
    if not np.isfinite(G).all():
        raise FileNotFoundError(f"{view}: varifold incompleto ({int((~np.isfinite(G)).sum())} celle)")
    return G, keys


def varifold_distances(G: np.ndarray) -> np.ndarray:
    """d_ij = sqrt(sum_s (G_ii + G_jj - 2 G_ij)) (``geometric_kernel.distances``)."""
    d = np.diagonal(G, axis1=1, axis2=2)
    return np.sqrt(np.clip(d[:, :, None] + d[:, None, :] - 2.0 * G, 0.0, None).sum(0))


def new_matrices(view: str) -> tuple[dict, list]:
    """({colonna: D (n, n)}, chiavi (soggetto, topologia) comuni) delle colonne nuove di una vista."""
    out, keys = {}, None
    for m, p in fit_files(view).items():
        with np.load(p) as z:
            k = list(zip([str(s) for s in z["subjects"]], [str(t) for t in z["topologies"]]))
            for d in ("coef", "fr", "sr"):
                out[f"{m}_{d}"] = np.asarray(z[f"D_{d}"], np.float64)
        if keys is not None and k != keys:
            raise SystemExit(f"{view}: mesh dei fit in ordine diverso")
        keys = k
    G, k = varifold_gram(view)
    if k != keys:
        raise SystemExit(f"{view}: mesh del varifold diverse da quelle dei fit")
    out["varifold"] = varifold_distances(G)
    return out, keys


def on_index(D: dict, keys: list, idx) -> dict:
    """Matrici sulle chiavi di ``idx`` (600, crop compreso): NaN dove manca la mesh (il crop)."""
    pos = np.asarray([idx.pos[k] for k in keys])
    out = {}
    for c, M in D.items():
        X = np.full((len(idx.keys), len(idx.keys)), np.nan)
        X[np.ix_(pos, pos)] = M
        out[c] = X
    return out


# ------------------------------------------------------------------------------------------- righe

def frame(view: str, n_boot: int):
    """(cols, M, bl, sa, sb, counts, seed, ctrl) di un dominio, con le colonne nuove."""
    if view == "famos":
        cols, G, sa, sb, counts, ctrl = fp.famos_frame(n_boot)
        D, keys = new_matrices("famos")
        with np.load(blmm.DATASETS / "FAMOS" / "test_view" / "gt_matrix.npz") as z:
            gt_names = [str(x) for x in z["names"]]
        order = np.asarray([[s for s, _ in keys].index(n) for n in gt_names])
        for c, Mx in D.items():
            cols[c] = Mx[np.ix_(order, order)][sa, sb]
        cols.update({f"gt_{g}": G[g] for g in fp.GTS})
        M = [m for m in cols if "|" in m]
        bl = [b for b in fp.BASELINES if b in cols and b not in NEW]
        return cols, M, bl, sa, sb, counts, 1234, ctrl
    src = "faceverse" if view == "faceverse_neutral" else view
    df, idx, seed = fp.rows_for(src)
    Mc = fp.model_columns(view, idx)
    D, keys = new_matrices(view)
    df = be.add_columns(df, {**Mc, **on_index(D, keys, idx)}, idx)
    bl = ["oracle_size"] if view == "faceverse_neutral" else [b for b in fp.BASELINES if b in df and b not in NEW]
    cols = {m: df[m].to_numpy(np.float64) for m in list(Mc) + bl + NEW}
    cols.update({f"gt_{g}": df[f"gt_{g}"].to_numpy(np.float64) for g in fp.GTS})
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa, sb = df["subject_a"].map(s2i).to_numpy(), df["subject_b"].map(s2i).to_numpy()
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)] + \
        [np.bincount(rng.integers(0, len(subjects), len(subjects)), minlength=len(subjects)) for _ in range(n_boot)]
    return cols, [m for m in Mc if "|form_at_c" not in m], bl, sa, sb, counts, seed, []


def run_view(view: str, n_boot: int, workers: int) -> tuple[list, dict]:
    cols, M, bl, sa, sb, counts, seed, ctrl = frame(view, n_boot)
    finite = {c: np.isfinite(v) for c, v in cols.items()}
    mask = (sa != sb) & np.all(list(finite.values()), axis=0)
    o = cols["oracle_size"]
    fp._J = {"counts": counts, "sa": sa[mask], "sb": sb[mask], "cols": {k: v[mask] for k, v in cols.items()},
             "methods": M + bl + NEW, "low": o[mask] <= np.quantile(o[mask], 0.2)}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(fp._rep, range(len(counts)), chunksize=4)
    V = {key: np.array([r[key] for r in reps]) for key in reps[0]}
    recs = fp.summarize(V, view, M, NEW, int(mask.sum()), seed)
    info = {"rows_total": int(len(sa)), "rows_mask": int(mask.sum()),
            "nan_rows_by_new_column": {c: int((~finite[c] & (sa != sb)).sum()) for c in NEW},
            "n_arm_columns": len(M), "baselines": bl, "famos_controls": [f"{k}: {d:.1e}" for k, d in ctrl]}
    print(f"[bp-paired] {view}: {len(M)} colonne dei bracci, {len(bl)} baseline, {int(mask.sum())}/{len(sa)} righe",
          flush=True)
    return recs, info


# ------------------------------------------------------------------------------------------ controlli

def controls(P: pd.DataFrame) -> dict:
    """Valori dei bracci contro aau/runs/evidence/trainer_v3/factorized_paired.csv (stesse chiavi)."""
    ref_path = fp.EV / "factorized_paired.csv"
    ref = pd.read_csv(ref_path)
    key = ["domain", "gt", "kind", "arm", "distance"]
    a = P[(P["baseline"] == "-") & (P["arm"] != "baseline")][key + ["arm_point", "n_rows"]]
    b = ref[(ref["baseline"] == "-") & (ref["arm"] != "baseline")][key + ["arm_point", "n_rows"]]
    j = a.merge(b, on=key, suffixes=("", "_ref"))
    out = {"reference": str(ref_path.relative_to(REPO)), "reference_mtime": os.path.getmtime(ref_path),
           "n_compared": int(len(j)), "max_abs_diff_arm_point": float((j["arm_point"] - j["arm_point_ref"]).abs().max())
           if len(j) else None, "rows_equal": bool((j["n_rows"] == j["n_rows_ref"]).all()) if len(j) else None,
           "by_domain": {}}
    for d, g in j.groupby("domain"):
        out["by_domain"][d] = {"n": int(len(g)), "max_abs_diff": float((g["arm_point"] - g["arm_point_ref"]).abs().max()),
                               "n_rows": sorted({int(x) for x in g["n_rows"]}), "n_rows_ref": sorted({int(x) for x in g["n_rows_ref"]})}
    return out


# ------------------------------------------------------------------------------------------ markdown

def fmt(p, lo, hi, sign=False) -> str:
    f = "{:+.3f}" if sign else "{:.3f}"
    return f"{f.format(p)} [{f.format(lo)}, {f.format(hi)}]"


def fit_table() -> list[str]:
    md = ["| vista | modello | mesh | fallite | RMS fit mm, mediana (max) | resid. NICP mm, mediana | s/mesh, mediana | "
          "vertici della regione |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for v in VIEW_ORDER:
        for m, p in fit_files(v).items():
            if not p.exists():
                continue
            with np.load(p) as z:
                n, nf = len(z["subjects"]), len(z["failed"])
                md.append(f"| {v} | {m} | {n} | {nf} ({nf / n:.1%}) | {np.nanmedian(z['rms_mm']):.2f} "
                          f"({np.nanmax(z['rms_mm']):.2f}) | {np.nanmedian(z['nicp_resid_mm']):.2f} | "
                          f"{np.median(z['seconds']):.1f} | {len(z['region_vertices'])} |")
    return md


def write_results(P: pd.DataFrame, info: dict, ctrl: dict) -> None:
    out = bp.OUT_ROOT
    sha = (out / "PROTOCOL.sha256").read_text().split()[0] if (out / "PROTOCOL.sha256").exists() else "?"
    md = ["# Concorrenti parametrici (GNM, FLAME 2023 Open) e varifold: risultati", "",
          f"Protocollo `PROTOCOL.md` (sha256 `{sha}`), codice `aau/baselines_param/`. Righe, GT e repliche di "
          "`v3_work/trainer/tools/fact_paired.py` (importato, non modificato). IC 95% percentile, 1000 repliche per "
          "soggetto. Tutti i numeri in `spearman.csv` (valori) e `paired.csv` (valori e delta, anche parziale e "
          "quintile basso).", ""]
    concl = out / "conclusions.md"
    if concl.exists():                                   # scritte a mano sui numeri qui sotto
        md += [concl.read_text().strip(), ""]
    md += ["## Fit", ""] + fit_table() + [""]
    md += ["## Controlli", "", "```", json.dumps(ctrl, indent=1), json.dumps(info, indent=1), "```", ""]
    R = P[(P["kind"] == "rho") & (P["baseline"] == "-")]
    for v in VIEW_ORDER:
        x = R[R["domain"] == v]
        if x.empty:
            continue
        md += [f"## {v} ({x['group'].iloc[0]}, {int(x['n_rows'].iloc[0])} righe)", "",
               "Spearman con la GT (rho, IC 95%):", "", "| metodo | FR | SR |", "| --- | --- | --- |"]
        rows = [(c, LABEL[c], "baseline", LABEL[c]) for c in NEW] + \
            [(c, lab, c.split("|")[0], c.split("|")[1]) for c, lab in REF_ARMS] + \
            [(c, fp.BASELINES[c], "baseline", fp.BASELINES[c]) for c in REF_BL]
        for c, lab, arm, dist in rows:
            cells = []
            for g in fp.GTS:
                r = x[(x["arm"] == arm) & (x["distance"] == dist) & (x["gt"] == g)]
                cells.append(fmt(r.iloc[0].arm_point, r.iloc[0].arm_ci_low, r.iloc[0].arm_ci_high) if len(r) else "-")
            if any(cl != "-" for cl in cells):
                md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md += ["", "Delta appaiati, braccio - concorrente (rho, stesse righe e repliche; IC 95%, P(delta <= 0)):", ""]
        D = P[(P["domain"] == v) & (P["kind"] == "rho") & P["baseline"].isin([LABEL[c] for c in NEW])]
        for g, arms in (("fr", ("factorized_s1234|form_cal", "factorized_s2345|form_cal", "ctrlfr_s1234|z", "ctrlfr_s2345|z")),
                        ("sr", ("factorized_s1234|shape", "factorized_s2345|shape", "ctrlfr_s1234|z", "ctrlfr_s2345|z"))):
            hdr = [dict(REF_ARMS)[a] for a in arms]
            md += [f"GT {g.upper()}:", "", "| concorrente | " + " | ".join(hdr) + " |", "| --- |" + " --- |" * len(arms)]
            for c in NEW:
                cells = []
                for a in arms:
                    arm, dist = a.split("|")
                    r = D[(D["arm"] == arm) & (D["distance"] == dist) & (D["gt"] == g) & (D["baseline"] == LABEL[c])]
                    cells.append(fmt(r.iloc[0].delta, r.iloc[0].ci_low, r.iloc[0].ci_high, True) +
                                 f", P {r.iloc[0].p_le0:.3f}" if len(r) else "-")
                md.append(f"| {LABEL[c]} | " + " | ".join(cells) + " |")
            md.append("")
    (out / "results.md").write_text("\n".join(md) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--views", default=",".join(VIEW_ORDER))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--summary-only", action="store_true", help="solo results.md dai csv gia' scritti")
    a = p.parse_args()
    out = bp.OUT_ROOT
    # vista neutra: gli embedding dei bracci stanno fuori da c3f_eval (in memoria, fact_paired non si tocca)
    fp.VIEWS["faceverse_neutral"] = ("fvn", "mesh_pair_nocrop")
    fp.FORM_DIR["fvn"] = os.path.relpath(NEUTRAL_EMB, fp.EVAL)
    fp.BASELINES.update(LABEL)
    if a.summary_only:
        write_results(pd.read_csv(out / "paired.csv"), *json.loads((out / "controls.json").read_text()).values())
        return
    recs, info = [], {}
    for view in a.views.split(","):
        r, info[view] = run_view(view, a.n_boot, a.workers)
        recs += r
    P = pd.DataFrame(recs)
    P.to_csv(out / "paired.csv", index=False)
    P[P["baseline"] == "-"].to_csv(out / "spearman.csv", index=False)
    ctrl = controls(P)
    (out / "controls.json").write_text(json.dumps({"info": info, "factorized_paired": ctrl}, indent=1) + "\n")
    print(json.dumps(ctrl, indent=1), flush=True)
    write_results(P, info, ctrl)


if __name__ == "__main__":
    main()
