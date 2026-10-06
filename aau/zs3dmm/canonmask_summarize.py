#!/usr/bin/env python3
"""Maschera canonica: tabelle (a)-(d) di aau/runs/canonmask/PROTOCOL.md, con IC bootstrap per soggetto appaiati.

    aau/run.sh aau/zs3dmm/canonmask_summarize.py --tag hifi --label HIFI3D --runs <data_fp> --gt <gt> \\
        --root datasets/HIFI3D --base-baselines <dir> --out-dir aau/runs/canonmask
    (canonmask_summarize.sbatch, tutti i domini)

Nessuna distanza calcolata qui: pair_metrics del breakdown (``<arm>`` = base, ``<arm>_canonmask``,
``<arm>_canonmask_offcenter``, ``<arm>_canonmask_identity``) e matrici faceBench (``baselines`` o
``outlineB``, ``baselines_canonmask*``). Ogni condizione e' unita alla base per (soggetti, topologie): stesse
righe, stessa GT.

Contrasti: somma pesata di Spearman (GT, colonna) su gruppi di righe diversi, tutti ricalcolati sulle STESSE
repliche bootstrap (soggetti con reinserimento, peso della coppia = prodotto dei conteggi, come
``weighted_bootstrap_spearman``). Cosi' "dopo − prima" (stesse righe, due colonne), il gap "crop<->X −
original<->X" (stessa colonna, righe diverse ma stessi soggetti) e la variazione del gap (quattro termini)
hanno IC appaiati per soggetto.

``--heldout-only joint`` (BFM, ICT): il latente del congiunto solo sui soggetti fuori dal suo training (split di
WS2); le altre metriche su tutti i soggetti.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import zs_summarize as zs  # noqa: E402

base = zs.base
common = zs.common
KEYS = zs.PAIR_KEYS
TOPOS = zs.TOPOLOGIES
OTH4 = ("noisy", "remesh", "down8k", "up60k")
CONDS = {"canon": "canonmask", "off": "canonmask_offcenter", "ident": "canonmask_identity"}
COND_LABEL = {"canon": "canonica", "off": "offcenter (controllo negativo)", "ident": "identita'"}
_BM = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="$ZS_ROOT: le viste <nome>_view col manifest")
    p.add_argument("--arms", nargs="+", default=["joint", "bfm_only"])
    p.add_argument("--base-baselines", type=Path, required=True)
    p.add_argument("--train-split", type=Path, default=None)
    p.add_argument("--train-name-offset", type=int, default=0)
    p.add_argument("--heldout-only", nargs="*", default=[], help="bracci da restringere ai soggetti fuori training")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def has_crop(df: pd.DataFrame) -> pd.Series:
    return df["topology_a"].eq("crop") | df["topology_b"].eq("crop")


def pair_sel(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    return ((df["topology_a"].eq(x) & df["topology_b"].eq(y)) | (df["topology_a"].eq(y) & df["topology_b"].eq(x)))


# ---------------------------------------------------------------------------- dati

def read_cond(runs: Path, root: Path, name: str, alt: str | None) -> pd.DataFrame:
    src = zs.arm_sources(runs, name)
    if alt is not None:
        man = src["topology"].parent / f"{alt}_manifest.json"
        cur = root / f"{alt}_view" / "manifest.json"
        if not man.exists() or man.read_text() != cur.read_text():
            raise SystemExit(f"{name}: {alt}_manifest.json assente o diverso dalla vista attuale {cur}")
    pm = base.read_pair_metrics(src["topology"])
    if pm.duplicated(KEYS).any():
        raise SystemExit(f"{name}: righe duplicate")
    return pm


def model_frames(args) -> tuple[dict, list]:
    """metrica -> frame con before e after_<cond>; piu' i controlli riga per riga dell'identita'."""
    frames, ident_rows = {}, []
    cham = None
    for arm in args.arms:
        b = read_cond(args.runs, args.root, arm, None)
        merged = b[KEYS + ["gt_distance", "latent_distance", "raw_chamfer"]].rename(
            columns={"latent_distance": "lat_before", "raw_chamfer": "ch_before"})
        for c, alt in CONDS.items():
            e = read_cond(args.runs, args.root, f"{arm}_{alt}", alt)
            m = merged.merge(e[KEYS + ["gt_distance", "latent_distance", "raw_chamfer"]], on=KEYS,
                             suffixes=("", "_e"), validate="one_to_one")
            if len(m) != len(merged) or len(m) != len(e):
                raise SystemExit(f"{arm}_{alt}: righe non allineate ({len(merged)}, {len(e)}, {len(m)})")
            if float((m["gt_distance"] - m["gt_distance_e"]).abs().max()) > 0:
                raise SystemExit(f"{arm}_{alt}: GT diversa")
            merged = m.drop(columns=["gt_distance_e"]).rename(
                columns={"latent_distance": f"lat_{c}", "raw_chamfer": f"ch_{c}"})
        for col, lab in (("lat", "latente"), ("ch", "Chamfer eval")):
            d = (merged[f"{col}_ident"] - merged[f"{col}_before"]).abs()
            rel = d / merged[f"{col}_before"].abs().clip(lower=1e-12)
            ident_rows.append({"metrica": f"{lab} ({arm})", "n_righe": len(merged), "max_abs_diff": float(d.max()),
                               "max_rel_diff": float(rel.max()), "righe_rel_>1e-4": int((rel > 1e-4).sum())})
        lat = merged[KEYS + ["gt_distance"]].copy()
        lat["before"] = merged["lat_before"]
        for c in CONDS:
            lat[f"after_{c}"] = merged[f"lat_{c}"]
        frames[f"latente {zs.ARM_LABEL[arm]}"] = (arm, lat)
        if cham is None:
            cham = merged[KEYS + ["gt_distance"]].copy()
            cham["before"] = merged["ch_before"]
            for c in CONDS:
                cham[f"after_{c}"] = merged[f"ch_{c}"]
    frames["Chamfer eval (repo)"] = (None, cham)
    return frames, ident_rows


def bl_rows(root: Path, metric: str, gt, pairs) -> pd.DataFrame:
    D_gt, idx = gt
    parts = []
    for ta, tb in pairs:
        D, subjects, _, _, _ = common.load_matrix(common.matrix_path(metric, ta, tb, root))
        i, j = common.subject_pair_indices(len(subjects))
        sa, sb = np.asarray(subjects)[i], np.asarray(subjects)[j]
        parts.append(pd.DataFrame({"subject_a": sa, "subject_b": sb, "topology_a": ta, "topology_b": tb,
                                   "gt_distance": D_gt[[idx[s] for s in sa], [idx[s] for s in sb]], "v": D[i, j]}))
    return pd.concat(parts, ignore_index=True)


def baseline_frames(args, gt) -> tuple[dict, list]:
    cross = [(a, b) for a in TOPOS for b in TOPOS if a != b]
    crop_pairs = [p for p in cross if "crop" in p]
    frames, ident_rows = {}, []
    for metric in zs.BL_METRICS:
        f = bl_rows(args.base_baselines, metric, gt, cross).rename(columns={"v": "before"})
        for c, alt in CONDS.items():
            root = args.runs / f"baselines_{alt}"
            man = root / f"{alt}_manifest.json"
            if not man.exists() or man.read_text() != (args.root / f"{alt}_view" / "manifest.json").read_text():
                raise SystemExit(f"{root}: {alt}_manifest.json assente o diverso dalla vista attuale")
            e = bl_rows(root, metric, gt, crop_pairs if c == "ident" else cross)
            f = f.merge(e[KEYS + ["v"]].rename(columns={"v": f"after_{c}"}), on=KEYS, how="left", validate="one_to_one")
        sel = has_crop(f)
        d = (f.loc[sel, "after_ident"] - f.loc[sel, "before"]).abs()
        both = np.isfinite(f.loc[sel, "after_ident"]) & np.isfinite(f.loc[sel, "before"])
        ident_rows.append({"metrica": zs.BL_LABEL[metric] + " (solo coppie con crop)", "n_righe": int(sel.sum()),
                           "max_abs_diff": float(d[both].max()), "max_rel_diff": float(
                               (d[both] / f.loc[sel, "before"][both].abs().clip(lower=1e-12)).max()),
                           "righe_rel_>1e-4": int(((d / f.loc[sel, "before"].abs().clip(lower=1e-12)) > 1e-4)[both].sum()),
                           "NaN_solo_da_un_lato": int((np.isfinite(f.loc[sel, "after_ident"]) != np.isfinite(f.loc[sel, "before"])).sum())})
        frames[zs.BL_LABEL[metric]] = (None, f)
    return frames, ident_rows


# ---------------------------------------------------------------------- bootstrap

def term(df: pd.DataFrame, sel: pd.Series, col: str, coef: float, s2i: dict) -> tuple:
    w = df.loc[sel, ["subject_a", "subject_b", "gt_distance", col]]
    w = w[np.isfinite(w["gt_distance"]) & np.isfinite(w[col]) & (w["subject_a"] != w["subject_b"])]
    return (w["subject_a"].map(s2i).to_numpy(np.int32), w["subject_b"].map(s2i).to_numpy(np.int32),
            w["gt_distance"].to_numpy(np.float64), w[col].to_numpy(np.float64), float(coef))


def contrast_task(task):
    global _BM
    if _BM is None:
        _BM = base.load_bootstrap_module()
    key, terms, n_subj, n_boot, seed = task
    point = sum(c * _BM.finite_spearman(g, v) for _, _, g, v, c in terms)
    parts = [float(_BM.finite_spearman(g, v)) for _, _, g, v, c in terms]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, n_subj, size=n_subj), minlength=n_subj)
        tot, ok = 0.0, True
        for sa, sb, g, v, c in terms:
            wt = cnt[sa].astype(np.int64) * cnt[sb].astype(np.int64)
            k = wt > 0
            if k.sum() < 3:
                ok = False
                break
            r = _BM.finite_spearman(np.repeat(g[k], wt[k]), np.repeat(v[k], wt[k]))
            if not np.isfinite(r):
                ok = False
                break
            tot += c * r
        if ok:
            reps.append(tot)
    reps = np.asarray(reps)
    lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
    return key, {"point": float(point), "lo": float(lo), "hi": float(hi), "parts": parts,
                 "n_rows": [len(t[0]) for t in terms], "n_boot": len(reps)}


def build_tasks(args, frames: dict, heldout: dict) -> list:
    tasks = []
    for metric, (arm, df) in frames.items():
        if arm is not None and arm in heldout:
            keep = set(heldout[arm])
            df = df[df["subject_a"].isin(keep) & df["subject_b"].isin(keep)]
        # Coppie fallite in faceBench (NaN) fuori da TUTTI i contrasti, non solo dal termine: prima, canonica
        # e offcenter guardano le stesse righe. L'identita' (solo crop per le baseline) filtra da se'.
        df = df[np.isfinite(df["before"]) & np.isfinite(df["after_canon"]) & np.isfinite(df["after_off"])]
        subs = sorted(set(df["subject_a"]) | set(df["subject_b"]))
        s2i = {s: i for i, s in enumerate(subs)}
        crop, nocrop = has_crop(df), ~has_crop(df)
        allr = pd.Series(True, index=df.index)

        def add(name, spec):
            key = (args.tag, metric, name)
            terms = [term(df, sel, col, coef, s2i) for sel, col, coef in spec]
            tasks.append((key, terms, len(subs), args.n_bootstrap, base.stable_seed(args.seed, *key)))

        for c in ("canon", "off"):
            a = f"after_{c}"
            if df[a].isna().all():
                continue
            for x in OTH4:
                cx, ox = pair_sel(df, "crop", x), pair_sel(df, "original", x)
                add(f"{c}|crop<->{x}|delta", [(cx, a, 1), (cx, "before", -1)])
                add(f"{c}|original<->{x}|delta", [(ox, a, 1), (ox, "before", -1)])
                add(f"{c}|{x}|gap_before", [(cx, "before", 1), (ox, "before", -1)])
                add(f"{c}|{x}|gap_after", [(cx, a, 1), (ox, a, -1)])
                add(f"{c}|{x}|gap_change", [(cx, a, 1), (ox, a, -1), (cx, "before", -1), (ox, "before", 1)])
            co = pair_sel(df, "crop", "original")
            add(f"{c}|crop<->original|delta", [(co, a, 1), (co, "before", -1)])
            add(f"{c}|agg|crop_delta", [(crop, a, 1), (crop, "before", -1)])
            add(f"{c}|agg|nocrop_delta", [(nocrop, a, 1), (nocrop, "before", -1)])
            add(f"{c}|agg|gap_before", [(crop, "before", 1), (nocrop, "before", -1)])
            add(f"{c}|agg|gap_after", [(crop, a, 1), (nocrop, a, -1)])
            add(f"{c}|agg|gap_change", [(crop, a, 1), (nocrop, a, -1), (crop, "before", -1), (nocrop, "before", 1)])
            add(f"{c}|agg|all_delta", [(allr, a, 1), (allr, "before", -1)])
        if not df["after_off"].isna().all():
            add("canon-off|agg|crop", [(crop, "after_canon", 1), (crop, "after_off", -1)])
            add("canon-off|agg|nocrop", [(nocrop, "after_canon", 1), (nocrop, "after_off", -1)])
            add("canon-off|agg|gap", [(crop, "after_canon", 1), (nocrop, "after_canon", -1),
                                      (crop, "after_off", -1), (nocrop, "after_off", 1)])
        idsel = crop if df.loc[nocrop, "after_ident"].isna().all() else allr
        add("ident|" + ("crop" if idsel is crop else "all") + "|delta",
            [(idsel, "after_ident", 1), (idsel, "before", -1)])
    return tasks


# --------------------------------------------------------------------------- output

def ci(r: dict) -> str:
    if not np.isfinite(r["point"]):
        return "n/d"
    return f"{r['point']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"


def pr(x: float) -> str:
    return "n/d" if not np.isfinite(x) else f"{x:.3f}"


def main() -> None:
    args = parse_args()
    global _BM
    _BM = base.load_bootstrap_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = zs.load_gt(args.gt)
    frames, ident_m = model_frames(args)
    blf, ident_b = baseline_frames(args, gt)
    frames.update(blf)

    heldout = {}
    if args.train_split is not None:
        sp = json.loads(args.train_split.read_text())["models"]
        for arm in args.heldout_only:
            train = set(sp[arm]["train"])
            subs = sorted(set(frames[f"latente {zs.ARM_LABEL[arm]}"][1]["subject_a"]) |
                          set(frames[f"latente {zs.ARM_LABEL[arm]}"][1]["subject_b"]))
            heldout[arm] = [s for s in subs if f"id{int(s[2:]) + args.train_name_offset:04d}" not in train]
            print(f"[canon-sum] {arm}: {len(heldout[arm])}/{len(subs)} soggetti fuori dal training", flush=True)

    tasks = build_tasks(args, frames, heldout)
    print(f"[canon-sum] {len(tasks)} contrasti bootstrap su {args.workers} processi", flush=True)
    res = {}
    with mp.get_context("fork").Pool(args.workers) as pool:
        for key, r in pool.imap_unordered(contrast_task, tasks, chunksize=1):
            res[(key[1], key[2])] = r
    rows = [{"dominio": args.label, "metrica": m, "contrasto": n, "stima": r["point"], "ci_low": r["lo"],
             "ci_high": r["hi"], "termini": json.dumps([round(p, 4) for p in r["parts"]]),
             "righe_per_termine": json.dumps(r["n_rows"]), "n_boot": r["n_boot"]} for (m, n), r in res.items()]
    pd.DataFrame(rows).sort_values(["metrica", "contrasto"]).to_csv(args.out_dir / f"{args.tag}_contrasts.csv", index=False)

    def R(m, n):
        return res.get((m, n), {"point": np.nan, "lo": np.nan, "hi": np.nan, "parts": [np.nan] * 4})

    sub_note = {m: (f"{len(heldout[arm])} soggetti fuori dal training" if arm in heldout else "tutti i soggetti")
                for m, (arm, _) in frames.items()}
    md = [f"## {args.label}", ""]
    md.append("Spearman mesh-pair con la GT del dominio, clean. IC 95% bootstrap per soggetto, appaiati sulle stesse "
              f"repliche ({args.n_bootstrap}). Prima = base, dopo = maschera canonica. gap = ρ(crop<->X) − ρ(original<->X).")
    md.append("")
    md.append("### (a) Per tipo di coppia, maschera canonica")
    for m in frames:
        md += ["", f"**{m}** ({sub_note[m]})", "",
               "| X | crop<->X prima → dopo | Δ crop<->X | original<->X prima → dopo | Δ original<->X | gap prima | gap dopo | Δ gap |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for x in OTH4:
            dc, do = R(m, f"canon|crop<->{x}|delta"), R(m, f"canon|original<->{x}|delta")
            md.append(f"| {x} | {pr(dc['parts'][1])} → {pr(dc['parts'][0])} | {ci(dc)} | "
                      f"{pr(do['parts'][1])} → {pr(do['parts'][0])} | {ci(do)} | {ci(R(m, f'canon|{x}|gap_before'))} | "
                      f"{ci(R(m, f'canon|{x}|gap_after'))} | {ci(R(m, f'canon|{x}|gap_change'))} |")
        dc, dn = R(m, "canon|agg|crop_delta"), R(m, "canon|agg|nocrop_delta")
        md.append(f"| tutte crop / tutte senza crop | {pr(dc['parts'][1])} → {pr(dc['parts'][0])} | {ci(dc)} | "
                  f"{pr(dn['parts'][1])} → {pr(dn['parts'][0])} | {ci(dn)} | {ci(R(m, 'canon|agg|gap_before'))} | "
                  f"{ci(R(m, 'canon|agg|gap_after'))} | {ci(R(m, 'canon|agg|gap_change'))} |")
        co = R(m, "canon|crop<->original|delta")
        md.append(f"| original (solo crop<->original) | {pr(co['parts'][1])} → {pr(co['parts'][0])} | {ci(co)} | - | - | - | - | - |")
    md += ["", "### (b) Costo della maschera sulle coppie senza crop", "",
           "| metrica | senza crop prima → dopo | Δ [IC] | tutte le cross prima → dopo | Δ [IC] |", "| --- | --- | --- | --- | --- |"]
    for m in frames:
        dn, da = R(m, "canon|agg|nocrop_delta"), R(m, "canon|agg|all_delta")
        md.append(f"| {m} | {pr(dn['parts'][1])} → {pr(dn['parts'][0])} | {ci(dn)} | {pr(da['parts'][1])} → "
                  f"{pr(da['parts'][0])} | {ci(da)} |")
    md += ["", "### (c) Controllo negativo: maschera della stessa area, centro spostato di R in direzione casuale per mesh", "",
           "| metrica | Δ crop (offcenter) | Δ senza crop (offcenter) | gap dopo (offcenter) | canonica − offcenter, crop | canonica − offcenter, senza crop | canonica − offcenter, gap |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for m in frames:
        md.append(f"| {m} | {ci(R(m, 'off|agg|crop_delta'))} | {ci(R(m, 'off|agg|nocrop_delta'))} | "
                  f"{ci(R(m, 'off|agg|gap_after'))} | {ci(R(m, 'canon-off|agg|crop'))} | "
                  f"{ci(R(m, 'canon-off|agg|nocrop'))} | {ci(R(m, 'canon-off|agg|gap'))} |")
    md += ["", "### (d) Controllo d'identita' (regione = mesh intera)", "", "Riga per riga contro la base:", "",
           base.md_table(pd.DataFrame(ident_m + ident_b), list(pd.DataFrame(ident_m + ident_b).columns), floats=8), "",
           "Δ Spearman identita' − base:", "", "| metrica | righe | Δ [IC] |", "| --- | --- | --- |"]
    for m in frames:
        k = "ident|all|delta" if (m, "ident|all|delta") in res else "ident|crop|delta"
        md.append(f"| {m} | {k.split('|')[1]} | {ci(R(m, k))} |")

    stats = pd.read_csv(args.root / "canonmask_view" / "mask_stats.csv")
    st = stats.groupby("topology").agg(facce_mediane=("n_faces", "median"), frazione_area=("area_frac", "median"))
    o = stats[stats.topology == "original"].set_index("subject")
    stats["tip_disp_rel"] = [float(np.linalg.norm(r[["tip_x", "tip_y", "tip_z"]].to_numpy(float) -
                                                  o.loc[r.subject, ["tip_x", "tip_y", "tip_z"]].to_numpy(float)) / r["scale"])
                             for _, r in stats.iterrows()]
    st["punta_vs_original_p99_su_S_dom"] = stats.groupby("topology")["tip_disp_rel"].quantile(0.99)
    stats["facce_su_original"] = stats["n_faces"].to_numpy() / o.loc[stats["subject"], "n_faces"].to_numpy()
    st["facce_su_original_mediana"] = stats.groupby("topology")["facce_su_original"].median()
    md += ["", "### Maschera canonica: dimensioni e stabilita' della punta (500 soggetti della vista)", "",
           base.md_table(st.reset_index(), ["topology"] + list(st.columns), floats=4), ""]
    (args.out_dir / f"{args.tag}.md").write_text("\n".join(md) + "\n")
    print(f"[canon-sum] scritto {args.out_dir / f'{args.tag}.md'}", flush=True)


if __name__ == "__main__":
    main()
