#!/usr/bin/env python3
"""E12, passo 4: tutti i metodi esistenti rivalutati con la GT canonica (HIFI3D, FaceVerse, FaceScape dev).

    v3_work/unified_gt/run.sh v3_work/canonical_gt/methods.py          (run.sbatch, passo ``methods``; dopo gt.py)

Protocollo: ``aau/runs/evidence/e12/protocol.md``, sezione 4. Come ``v3_work/unified_gt/eval_methods.py``
(E8), da cui vengono le righe di FaceVerse (``fv_frames``, importata) e i controlli pubblicati (``controls``,
importata); ``evaluate`` e ``_replicate`` sono copiate da li' con le GT di E12. Nessuna distanza ricalcolata,
cambia solo la colonna della GT (``zs_summarize.with_gt`` per nome di soggetto).

  - HIFI3D: righe di ``zs_arcface_vs_scale.frame_a`` (e036, e072, e108, congiunto 1019532, Chamfer eval,
    ArcFace ombreggiato e normal map), faceBench (Chamfer 4096 pt, ICP + Chamfer, ICP + NICP P2Tri) e i
    competitori di ``aau/runs/competitors_hifi3d`` (``comp_summarize.competitor_distances``: NICP su template,
    Uni3D, OpenShape, ShapeDNA, HKS, WKS e le ablazioni del frame). Gruppi ``nocrop_cross`` (primario),
    ``all_cross``, ``subject_pair_mean``; seme per gruppo = quello di e108 - Chamfer eval di ``zs_summarize``.
  - FaceVerse con espressioni (secondario): ``eval_methods.fv_frames`` di E8 (ora con e072).
  - FaceScape dev, vista neutra: ``dev_fs_summarize.pair_frame`` (e108 dagli embedding, Chamfer eval dal
    breakdown, Chamfer intera e su regione stabile); seme per gruppo = quello di ``devfs_paired`` (neutral,
    maxabs, gruppo, scale_e108, raw_chamfer).

GT (protocollo, Emendamento 1): ``maxabs`` (quella delle righe), ``unified`` (i file di E8 / E11; = Procrustes
completo), e da ``datasets/CANONICAL_GT`` (gt.py) ``F``, ``F_centered``, ``F_rig_ls``, ``F_rig_rob``, ``S``,
``EDM``, ``EDM_s``.

Uscite in ``aau/runs/evidence/e12/``: ``methods_spearman.csv``, ``methods_paired.csv``, ``methods_gt_diff.csv``,
``methods_controls.csv``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cgt
from cgt import C

sys.path.insert(0, str(C.REPO_ROOT / "aau" / "competitors"))
os.chdir(C.REPO_ROOT)  # i default dei summarizer sono percorsi relativi alla radice

import comp_summarize as comp  # noqa: E402
import dev_fs_summarize as dfs  # noqa: E402
import eval_methods as e8m  # noqa: E402  (E8, v3_work/unified_gt)
import zs_arcface_summarize as zas  # noqa: E402
import zs_arcface_vs_scale as zavs  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import select_subjects  # noqa: E402

base = zsum.base
RUNS = C.REPO_ROOT / "aau" / "runs"
FB = e8m.FB
GTS = ("maxabs", "unified", "F", "F_centered", "F_rig_ls", "F_rig_rob", "S", "EDM", "EDM_s")
GT_DIFFS = (("F", "maxabs"), ("F", "unified"), ("S", "F"), ("EDM", "F"), ("EDM_s", "EDM"), ("S", "unified"),
            ("F_centered", "F"), ("F_rig_ls", "F"), ("F_rig_rob", "F"), ("unified", "maxabs"))
COMP_DIR = RUNS / "competitors_hifi3d"
UNIFIED = {"hifi3d": C.DATA_ROOT / "gt" / "hifi3d_unified.npz", "faceverse": C.DATA_ROOT / "gt" / "faceverse_unified.npz",
           "facescape": C.DATA_ROOT / "eval" / "facescape_gt_matrix.npz"}
LABEL = {**e8m.LABEL, "chamfer_full": "Chamfer intera (4096 pt)", "chamfer_stable": "Chamfer regione stabile"}

_BM = None
_JOB = {}


def gt_matrices(dom: str) -> dict:
    out = {"unified": zsum.load_gt(UNIFIED[dom])}
    for k in GTS[2:]:
        out[k] = zsum.load_gt(cgt.OUT_DIR / f"{dom}_{k}.npz")
    return out


def add_gts(df: pd.DataFrame, gts: dict) -> pd.DataFrame:
    out = df.copy()
    if "gt_maxabs" not in out:
        out["gt_maxabs"] = out["gt_distance"]
    for k, g in gts.items():
        out[f"gt_{k}"] = zsum.with_gt(df, g)["gt_distance"].to_numpy()
    return out


# ------------------------------------------------------------------------------ righe

def hifi_frames() -> tuple[dict, dict]:
    """Come ``eval_methods.hifi_frames`` di E8 (copiata), piu' ``all_cross`` e i competitori."""
    args = argparse.Namespace(runs=Path("aau/runs/ws_hifi3d/data_328f2bfc1a"),
                              view_dir=Path("datasets/HIFI3D/eval_view/npz"),
                              arcface_root=Path("aau/runs/arcface_render_zs/hifi3d"),
                              joint_stage=Path("aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot"))
    subjects = select_subjects(args.view_dir, 1234)
    idx = zes.Index(subjects)
    D_arc = {name: zas.arcface_distances(args.arcface_root / mode / "arcface_views.npz", idx, zas.VIEWS["3v"])
             for mode, name in (("shaded", "arcface_shaded_3v"), ("normals", "arcface_normals_3v"))}
    df, _ = zavs.frame_a(args, idx, D_arc)
    D_other = {f"fb_{m}": zes.facebench_distances(args.runs / "baselines", m, idx) for m in FB}
    D_comp, comp_labels, _ = comp.competitor_distances(COMP_DIR, idx)
    D_other.update(D_comp)
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"], df["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"], df["topology_b"])])
    for m, D in D_other.items():
        df[m] = D[ia, ib]
    df = df.rename(columns={"chamfer": "chamfer_eval"})
    methods = ["arcface_shaded_3v", "arcface_normals_3v", "scale_e036", "scale_e072", "scale_e108", "joint",
               "chamfer_eval"] + [f"fb_{m}" for m in FB] + list(D_comp)
    df = add_gts(df, gt_matrices("hifi3d"))
    nocrop = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    spm = df.groupby(["subject_a", "subject_b"], as_index=False)[[f"gt_{g}" for g in GTS] + methods].mean()
    seed = lambda g: base.stable_seed(1234, "paired", "", "maxabs", f"{zsum.ARM_LABEL['scale']}, e108 - Chamfer eval", g)  # noqa: E731
    frames = {"nocrop_cross": (nocrop, methods, seed("nocrop_cross")),
              "all_cross": (df, methods, seed("all_cross")),
              "subject_pair_mean": (spm, methods, seed("subject_pair_mean"))}
    return frames, comp_labels


def fv_frames() -> dict:
    """Le righe di E8 (``eval_methods.fv_frames``) con le GT canoniche aggiunte per nome di soggetto."""
    frames, _ = e8m.fv_frames()
    gts = gt_matrices("faceverse")
    gts.pop("unified")                                   # gia' nelle righe di E8, dallo stesso file
    out = {}
    for group, (df, methods, seed) in frames.items():
        out[group] = (add_gts(df, gts), methods, seed)
    return out


def fs_frames() -> dict:
    runs = RUNS / "ws_dev_facescape" / "data_aca84a16c6"
    view = cgt.DATASETS / "DEV_FACESCAPE" / "eval_view"
    subjects = select_subjects(view / "npz", 1234)
    idx = zes.Index(subjects)
    arms = dfs.arms_of(runs)
    if "scale_e108" not in arms or arms["scale_e108"]["breakdown"] is None:
        raise SystemExit(f"{runs}: manca scale_e108_topology")
    lat = {"scale_e108": zes.model_distances(arms["scale_e108"]["stage"], idx)}
    reg = zes.region_distances(runs / "baselines" / "region_chamfer.npz", idx)
    df = dfs.pair_frame(idx, zsum.load_gt(view / "gt_matrix.npz"), lat, reg, arms["scale_e108"]["breakdown"])
    for c, tol in (("_gt_bd", 1e-6), ("_lat_bd", 1e-4)):
        ref = df["gt_distance"] if c == "_gt_bd" else df["lat:scale_e108"]
        if float(np.abs(df[c] - ref).max()) > tol:
            raise SystemExit(f"FaceScape dev: {c} diverso dalla ricostruzione")
    df = df.rename(columns={"lat:scale_e108": "scale_e108", "raw_chamfer": "chamfer_eval"})
    methods = ["scale_e108", "chamfer_eval", "chamfer_full", "chamfer_stable"]
    df = add_gts(df[zsum.PAIR_KEYS + ["gt_distance"] + methods], gt_matrices("facescape"))
    nocrop = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    spm = df.groupby(["subject_a", "subject_b"], as_index=False)[[f"gt_{g}" for g in GTS] + methods].mean()
    seed = lambda g: base.stable_seed(1234, "devfs_paired", "neutral", "maxabs", g, "scale_e108", "raw_chamfer")  # noqa: E731
    return {"nocrop_cross": (nocrop, methods, seed("nocrop_cross")), "all_cross": (df, methods, seed("all_cross")),
            "subject_pair_mean": (spm, methods, seed("subject_pair_mean"))}


# -------------------------------------------------------------------------- bootstrap
# Copia di eval_methods._replicate / evaluate (E8) con le GT e le differenze fra GT di E12.

def _replicate(k: int) -> dict:
    """Spearman di ogni (GT, metodo) e delle differenze appaiate, per la replica k (0 = punto)."""
    global _BM
    if _BM is None:
        _BM = base.load_bootstrap_module()
    J = _JOB
    counts = J["counts"][k]
    wt = counts[J["sa"]].astype(np.int64) * counts[J["sb"]]
    out = {}
    cache = {}

    def sp(g, m, mask):
        mask = J["masks"]["_canon"][mask]
        key = (g, m, mask)
        if key not in cache:
            keep = J["masks"][mask] & (wt > 0)
            x = np.repeat(J["cols"][f"gt_{g}"][keep], wt[keep])
            y = np.repeat(J["cols"][m][keep], wt[keep])
            cache[key] = _BM.finite_spearman(x, y)
        return cache[key]

    for g in GTS:
        for m in J["methods"]:
            out[("row", g, m)] = sp(g, m, m)
    for g, a, b in J["pairs"]:
        out[("pair", g, a, b)] = sp(g, a, f"{a}&{b}") - sp(g, b, f"{a}&{b}")
    for m in J["methods"]:
        for g1, g2 in GT_DIFFS:
            out[("gtdiff", g1, g2, m)] = sp(g1, m, m) - sp(g2, m, m)
    return out


def evaluate(dom: str, group: str, df: pd.DataFrame, methods: list, seed: int, n_boot: int, workers: int):
    global _JOB
    subjects = np.array(sorted(set(df["subject_a"]) | set(df["subject_b"])))
    s2i = {s: i for i, s in enumerate(subjects)}
    sa = df["subject_a"].map(s2i).to_numpy(np.int32)
    sb = df["subject_b"].map(s2i).to_numpy(np.int32)
    cols = {c: df[c].to_numpy(np.float64) for c in [f"gt_{g}" for g in GTS] + methods}
    gt_ok = np.all([np.isfinite(cols[f"gt_{g}"]) for g in GTS], axis=0) & (sa != sb)
    masks = {m: gt_ok & np.isfinite(cols[m]) for m in methods}
    ref = [m for m in ("chamfer_eval", "scale_e108") if m in methods]
    pairs = [(g, a, b) for g in GTS for b in ref for a in methods if a != b]
    for g, a, b in pairs:
        masks[f"{a}&{b}"] = masks[a] & masks[b]
    # maschere uguali -> stesso nome, cosi' ogni Spearman si calcola una volta per replica
    canon, uniq = {}, []
    for name, mk in masks.items():
        hit = next((u for u in uniq if np.array_equal(masks[u], mk)), None)
        if hit is None:
            uniq.append(name)
            hit = name
        canon[name] = hit
    masks = {**masks, "_canon": canon}
    rng = np.random.default_rng(seed)
    counts = [np.ones(len(subjects), dtype=np.int64)]
    for _ in range(n_boot):
        counts.append(np.bincount(rng.integers(0, len(subjects), size=len(subjects)), minlength=len(subjects)))
    _JOB = {"counts": counts, "sa": sa, "sb": sb, "cols": cols, "masks": masks, "methods": methods, "pairs": pairs}
    with mp.get_context("fork").Pool(workers) as pool:
        reps = pool.map(_replicate, range(n_boot + 1), chunksize=8)
    rows, prs, gtd = [], [], []
    for key in reps[0]:
        v = np.array([r[key] for r in reps])
        b = v[1:][np.isfinite(v[1:])]
        lo, hi = np.percentile(b, [2.5, 97.5])
        rec = {"domain": dom, "group": group, "point": float(v[0]), "ci_low": float(lo), "ci_high": float(hi),
               "p_le0": float((b <= 0).mean()), "n_boot": int(len(b))}
        if key[0] == "row":
            rows.append({**rec, "gt": key[1], "method": key[2], "n_rows": int(masks[key[2]].sum()),
                         "n_nan": int((~masks[key[2]] & gt_ok).sum())})
        elif key[0] == "pair":
            prs.append({**rec, "gt": key[1], "a": key[2], "b": key[3]})
        else:
            gtd.append({**rec, "gt_a": key[1], "gt_b": key[2], "method": key[3]})
    return pd.DataFrame(rows), pd.DataFrame(prs), pd.DataFrame(gtd), len(subjects)


# ---------------------------------------------------------------------------- controlli

def controls(R: pd.DataFrame, P: pd.DataFrame) -> pd.DataFrame:
    """Numeri esistenti che devono tornare. ``eval_methods.controls`` di E8 (pubblicati con la GT maxabs), piu':
    E8 (maxabs e unificata, punto e IC: stesso seme), competitori (punti, maxabs e unificata), all_cross di
    HIFI3D (``data_scale_ood/hifi``), FaceScape dev (``evidence/dev_facescape/graded.csv``)."""
    out = [e8m.controls(R, P).assign(source="E8 eval_methods.controls (pubblicati)")]
    rows = []

    def new(dom, group, gt, m):
        x = R[(R["domain"] == dom) & (R["group"] == group) & (R["gt"] == gt) & (R["method"] == m)]
        return x.iloc[0] if len(x) else None

    e8 = pd.read_csv(RUNS / "evidence" / "e8" / "methods_spearman.csv")
    for r in e8[e8["gt"].isin(["maxabs", "unified"])].itertuples():
        n = new(r.domain, r.group, r.gt, r.method)
        if n is not None:
            for c in ("point", "ci_low", "ci_high"):
                rows.append({"domain": r.domain, "what": f"{r.method} {r.group} GT {r.gt} ({c})",
                             "published": getattr(r, c), "recomputed": float(n[c]), "source": "E8 methods_spearman.csv"})
    cs = pd.read_csv(COMP_DIR / "spearman.csv")
    for r in cs.itertuples():
        m = {"ref_latent": "scale_e108", "ref_chamfer": "chamfer_eval"}.get(r.method, r.method)
        n = new("hifi3d", r.scenario, r.gt, m)
        if n is not None:
            rows.append({"domain": "hifi3d", "what": f"{m} {r.scenario} GT {r.gt} (punto)", "published": r.spearman,
                         "recomputed": float(n["point"]), "source": "competitors_hifi3d/spearman.csv"})
    pr = pd.read_csv(RUNS / "data_scale_ood" / "hifi" / "paired.csv")
    for r in pr[(pr["gt"] == "maxabs") & (pr["comparison"] == f"{zsum.ARM_LABEL['scale']}, e108 - Chamfer eval")].itertuples():
        n = P[(P["domain"] == "hifi3d") & (P["group"] == r.scenario) & (P["gt"] == "maxabs") & (P["a"] == "scale_e108")
              & (P["b"] == "chamfer_eval")]
        if len(n):
            for c, c2 in (("point", "diff"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("p_le0", "p_boot_le0")):
                rows.append({"domain": "hifi3d", "what": f"e108 - Chamfer eval {r.scenario} ({c})",
                             "published": getattr(r, c2), "recomputed": float(n[c].iloc[0]),
                             "source": "data_scale_ood/hifi/paired.csv"})
    tc = pd.read_csv(RUNS / "data_scale_ood" / "hifi" / "table_cells.csv")
    tc = tc[(tc["gt"] == "maxabs") & (tc["scenario"] == "clean") & (tc["protocol"] == "mesh_pair_all_cross")]
    for r in tc.itertuples():
        m = r.model
        n = new("hifi3d", "all_cross", "maxabs", m)
        if n is not None:
            pub = r.latent_point_check if np.isfinite(r.latent_point_check) else r.latent_spearman
            rows.append({"domain": "hifi3d", "what": f"{m} all_cross (punto)", "published": pub,
                         "recomputed": float(n["point"]), "source": "data_scale_ood/hifi/table_cells.csv"})
    g = pd.read_csv(RUNS / "evidence" / "dev_facescape" / "graded.csv")
    g = g[(g["view"] == "neutral") & (g["gt"] == "maxabs")]
    mname = {"scale_e108": "scale_e108", "raw_chamfer": "chamfer_eval", "chamfer_full": "chamfer_full",
             "chamfer_stable": "chamfer_stable"}
    for r in g.itertuples():
        if r.method not in mname:
            continue
        if isinstance(r.vs, str):
            if r.method != "scale_e108" or r.vs != "raw_chamfer":
                continue
            n = P[(P["domain"] == "facescape") & (P["group"] == r.scenario) & (P["gt"] == "maxabs")
                  & (P["a"] == "scale_e108") & (P["b"] == "chamfer_eval")]
            for c, c2 in (("point", "diff"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("p_le0", "p_boot_le0")):
                rows.append({"domain": "facescape", "what": f"e108 - Chamfer eval {r.scenario} ({c})",
                             "published": getattr(r, c2), "recomputed": float(n[c].iloc[0]),
                             "source": "evidence/dev_facescape/graded.csv"})
        else:
            n = new("facescape", r.scenario, "maxabs", mname[r.method])
            rows.append({"domain": "facescape", "what": f"{mname[r.method]} {r.scenario} (punto)", "published": r.spearman,
                         "recomputed": float(n["point"]), "source": "evidence/dev_facescape/graded.csv"})
    out.append(pd.DataFrame(rows))
    df = pd.concat(out, ignore_index=True)
    df["abs_diff"] = (df["published"] - df["recomputed"]).abs()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--domains", default="hifi3d,faceverse,facescape")
    a = p.parse_args()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    R, P, G = [], [], []
    for dom in a.domains.split(","):
        frames = {"hifi3d": lambda: hifi_frames()[0], "faceverse": fv_frames, "facescape": fs_frames}[dom]()
        for group, (df, methods, seed) in frames.items():
            r, pr, gd, n_s = evaluate(dom, group, df, methods, seed, a.n_bootstrap, workers)
            print(f"[e12-methods] {dom} {group}: {len(df)} righe, {n_s} soggetti, {len(methods)} metodi", flush=True)
            R.append(r)
            P.append(pr)
            G.append(gd)
    R, P, G = pd.concat(R), pd.concat(P), pd.concat(G)
    R.to_csv(cgt.EVID_DIR / "methods_spearman.csv", index=False)
    P.to_csv(cgt.EVID_DIR / "methods_paired.csv", index=False)
    G.to_csv(cgt.EVID_DIR / "methods_gt_diff.csv", index=False)
    ctrl = controls(R, P)
    ctrl.to_csv(cgt.EVID_DIR / "methods_controls.csv", index=False)
    print(ctrl.groupby("source")["abs_diff"].agg(["count", "max"]).to_string(), flush=True)
    piv = R.pivot_table(index=["domain", "group", "method"], columns="gt", values="point")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
