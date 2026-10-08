#!/usr/bin/env python3
"""Tutti i metodi esistenti rivalutati con la GT unificata: HIFI3D e FaceVerse con espressioni.

    v3_work/unified_gt/run.sh v3_work/unified_gt/eval_methods.py      (dopo shapes.py)

Richiesta del critic (8 ottobre): stessi soggetti, stesse coppie, stesse repliche dei summary
esistenti; cambia SOLO la GT. Nessuna distanza ricalcolata: le righe e le distanze vengono dalle
funzioni dei summarizer del repo, importate:
  - HIFI3D (``aau/runs/data_scale_ood/hifi/summary.md`` e ``arcface_vs_scale_hifi3d.md``):
    ``zs_arcface_vs_scale.frame_a`` (pair_metrics del breakdown di e036/e072/e108 e del congiunto
    1019532, Chamfer eval, ArcFace ombreggiato e normal map sulle stesse righe), piu' le matrici
    faceBench (``zs_expr_summarize.facebench_distances``: Chamfer 4096 pt, ICP + Chamfer, NICP P2Tri)
    lette sulle stesse righe. Gruppi: ``nocrop_cross`` (20 coppie ordinate di topologie) e
    ``subject_pair_mean`` (media per coppia di soggetti sulle 30, clean), come nel summary. In piu'
    ``original_to_original`` (solo la topologia original: la riga che il critic cita, 0.876 / 0.401),
    con i modelli dagli embedding (``scale_eNNN_embed``; il congiunto solo in convenzione BFM, gli
    embedding nativi non esistono).
  - FaceVerse con espressioni (``aau/runs/ws_faceverse_expr``, secondario del protocollo: Spearman con la
    GT d'identita' NEUTRA): righe ``zs_expr_summarize.secondary_frame`` (pair_metrics di
    scale_e108_flip, senza crop), distanze dei modelli dagli embedding, faceBench, ArcFace
    (``arcface_render_zs/fv_expr``). Gruppi: ``mesh_pair_nocrop`` e ``subject_pair_mean_nocrop``.
Repliche: UN seme per (dominio, gruppo), quello della differenza pubblicata ``e108 - Chamfer eval``
(HIFI3D: ``zs_summarize.pboot``; FaceVerse: ``expr_sec``), quindi quella differenza con la GT maxabs
deve tornare identica (controllo, nel csv); tutte le righe, le differenze fra metodi e quelle fra GT
stanno sulle stesse repliche. Ricampionamento di ``zs_summarize.paired_bootstrap`` (soggetti con
reinserimento, coppia pesata per il prodotto dei conteggi, ``finite_spearman``).

GT: ``maxabs`` (quella delle righe), ``unified`` (``datasets/UNIFIED_GT/gt/<dominio>_unified.npz``),
``unified_pairwise`` (variante con Procrustes per coppia), ``coef``, e due GT sulla patch NATIVA
(native_gt.py: ``native_sim`` = Procrustes di similarita' e RMS a vertici uniformi, ``native_sim_area`` =
la definizione dell'unificata senza la mappa FLAME), sostituite per nome di soggetto con
``zs_summarize.with_gt``.

Uscite: ``aau/runs/evidence/e8/methods_spearman.csv``, ``methods_paired.csv``, ``methods_controls.csv``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ugt as C

ZS = C.REPO_ROOT / "aau" / "zs3dmm"
sys.path.insert(0, str(ZS))
os.chdir(C.REPO_ROOT)  # i default dei summarizer sono percorsi relativi alla radice

import zs_arcface_summarize as zas  # noqa: E402
import zs_arcface_vs_scale as zavs  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
import zs_summarize as zsum  # noqa: E402
from zs_stage import select_subjects  # noqa: E402

base = zsum.base
RUNS = C.REPO_ROOT / "aau" / "runs"
GT_DIR = C.DATA_ROOT / "gt"
FB = ("chamfer", "rigid_icp_chamfer", "nicp_p2tri")
GTS = ("maxabs", "unified", "unified_pairwise", "coef", "native_sim", "native_sim_area", "native_sim_area_region")
LABEL = {"arcface_shaded_3v": "ArcFace ombreggiato (3 viste)", "arcface_normals_3v": "ArcFace normal map (3 viste)",
         "scale_e036": "BFM+ICT+GNM e036", "scale_e072": "BFM+ICT+GNM e072", "scale_e108": "BFM+ICT+GNM e108",
         "joint": "BFM+ICT congiunto 1019532", "joint@bfm": "BFM+ICT congiunto 1019532 (conv. BFM)",
         "joint@ict": "BFM+ICT congiunto 1019532 (conv. ICT)",
         "chamfer_eval": "Chamfer eval", "fb_chamfer": "Chamfer faceBench 4096 pt",
         "fb_rigid_icp_chamfer": "ICP rigido + Chamfer", "fb_nicp_p2tri": "ICP + NICP P2Tri"}

_BM = None
_JOB = {}


def gt_matrices(dom: str) -> dict:
    view = {"hifi3d": "datasets/HIFI3D/eval_view", "faceverse": "datasets/FACEVERSE_ZS/eval_view"}[dom]
    paths = {"unified": GT_DIR / f"{dom}_unified.npz", "unified_pairwise": GT_DIR / f"{dom}_unified_pairwise.npz",
             "coef": C.REPO_ROOT / view / "gt_coef_matrix.npz",
             "native_sim": GT_DIR / f"{dom}_native_sim.npz", "native_sim_area": GT_DIR / f"{dom}_native_sim_area.npz",
             "native_sim_area_region": GT_DIR / f"{dom}_native_sim_area_region.npz"}
    return {k: zsum.load_gt(p) for k, p in paths.items()}


def add_gts(df: pd.DataFrame, gts: dict) -> pd.DataFrame:
    out = df.copy()
    out["gt_maxabs"] = out["gt_distance"]
    for k, g in gts.items():
        out[f"gt_{k}"] = zsum.with_gt(df, g)["gt_distance"].to_numpy()
    return out


# ------------------------------------------------------------------------------ righe

def hifi_frames() -> tuple[dict, list]:
    args = argparse.Namespace(runs=Path("aau/runs/ws_hifi3d/data_328f2bfc1a"),
                              view_dir=Path("datasets/HIFI3D/eval_view/npz"),
                              arcface_root=Path("aau/runs/arcface_render_zs/hifi3d"),
                              joint_stage=Path("aau/runs/ws_hifi3d/data_328f2bfc1a/joint_frame-xmymz_flip_ranking/zs_zeroshot"))
    subjects = select_subjects(args.view_dir, 1234)
    idx = zes.Index(subjects)
    D_arc = {name: zas.arcface_distances(args.arcface_root / mode / "arcface_views.npz", idx, zas.VIEWS["3v"])
             for mode, name in (("shaded", "arcface_shaded_3v"), ("normals", "arcface_normals_3v"))}
    df, _ = zavs.frame_a(args, idx, D_arc)
    D_fb = {m: zes.facebench_distances(args.runs / "baselines", m, idx) for m in FB}
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"], df["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"], df["topology_b"])])
    for m, D in D_fb.items():
        df[f"fb_{m}"] = D[ia, ib]
    df = df.rename(columns={"chamfer": "chamfer_eval"})
    methods = ["arcface_shaded_3v", "arcface_normals_3v", "scale_e036", "scale_e072", "scale_e108", "joint",
               "chamfer_eval"] + [f"fb_{m}" for m in FB]
    gts = gt_matrices("hifi3d")
    df = add_gts(df, gts)
    nocrop = df[df["topology_a"].ne("crop") & df["topology_b"].ne("crop")]
    spm_cols = [f"gt_{g}" for g in GTS] + methods
    spm = df.groupby(["subject_a", "subject_b"], as_index=False)[spm_cols].mean()
    # original -> original: coppie i<j, modelli dagli embedding
    D_mod = {f"scale_{t}": zes.model_distances(args.runs / f"scale_{t}_embed" / zsum.STAGE, idx)
             for t in zavs.TAGS}
    D_mod["joint@bfm"] = zes.model_distances(args.joint_stage, idx)
    ro = idx.rows("original")
    iu, ju = np.triu_indices(len(subjects), 1)
    oo = pd.DataFrame({"subject_a": np.array(subjects)[iu], "subject_b": np.array(subjects)[ju],
                       "topology_a": "original", "topology_b": "original"})
    M_oo = {}
    for m in FB:
        M, subj, _, _, _ = zes.common.load_matrix(zes.common.matrix_path(m, "original", "original", args.runs / "baselines"))
        if subj != subjects:
            raise SystemExit(f"faceBench {m} original->original: soggetti diversi")
        M_oo[f"fb_{m}"] = M[iu, ju]
    for name, D in {**D_arc, **D_mod}.items():
        M_oo[name] = D[ro[iu], ro[ju]]
    for k, v in M_oo.items():
        oo[k] = v
    gt_max = zsum.load_gt(Path("datasets/HIFI3D/eval_view/gt_matrix.npz"))
    oo["gt_distance"] = zsum.with_gt(oo, gt_max)["gt_distance"]
    oo = add_gts(oo, gts)
    seed = lambda g: base.stable_seed(1234, "paired", "", "maxabs", f"{zsum.ARM_LABEL['scale']}, e108 - Chamfer eval", g)  # noqa: E731
    frames = {"nocrop_cross": (nocrop, methods, seed("nocrop_cross")),
              "subject_pair_mean": (spm, methods, seed("subject_pair_mean")),
              "original_to_original": (oo, list(M_oo), base.stable_seed(1234, "e8_unified", "original_to_original"))}
    return frames, subjects


def fv_frames() -> tuple[dict, list]:
    runs = Path("aau/runs/ws_faceverse_expr/data_736f96956a")
    view = Path("datasets/FACEVERSE_ZS/expr_view/npz")
    subjects = select_subjects(view, 1234)
    idx = zes.Index(subjects)
    D = {}
    arms = {"joint@bfm": "joint_flip", "joint@ict": "joint_frame-xmymz", "scale_e036": "scale_e036_flip",
            "scale_e072": "scale_e072_flip", "scale_e108": "scale_e108_flip"}
    missing = []
    for name, arm in arms.items():
        stage = runs / f"{arm}_topology" / zsum.STAGE
        if not (stage / "embeddings.npz").exists():
            missing.append(name)                     # e072: job 1061482 ancora in coda l'8 ottobre
            continue
        D[name] = zes.model_distances(stage, idx)
    for m in FB:
        D[f"fb_{m}"] = zes.facebench_distances(runs / "baselines", m, idx)
    root = Path("aau/runs/arcface_render_zs/fv_expr")
    for mode, name in (("shaded", "arcface_shaded_3v"), ("normals", "arcface_normals_3v")):
        D[name] = zas.arcface_distances(root / mode / "arcface_views.npz", idx, zas.VIEWS["3v"])
    stage = runs / "scale_e108_flip_topology" / zsum.STAGE
    df = zes.secondary_frame(stage, D, idx, "scale_e108")
    df["subject_a"], df["subject_b"] = df["subject_a"].astype(str), df["subject_b"].astype(str)
    df = df.rename(columns={"raw_chamfer": "chamfer_eval"})
    methods = [m for m in ["arcface_shaded_3v", "arcface_normals_3v", "scale_e036", "scale_e072", "scale_e108",
                           "joint@bfm", "joint@ict", "chamfer_eval"] + [f"fb_{m}" for m in FB] if m not in missing]
    if missing:
        print(f"[e8-methods] faceverse: embedding assenti per {missing}, righe saltate", flush=True)
    df = add_gts(df, gt_matrices("faceverse"))
    spm = df.groupby(["subject_a", "subject_b"], as_index=False)[[f"gt_{g}" for g in GTS] + methods].mean()
    key = ("scale_e108@bfm", "mesh_pair_nocrop", "latent_distance", "raw_chamfer")
    key2 = ("scale_e108@bfm", "subject_pair_mean_nocrop", "latent_distance", "raw_chamfer")
    frames = {"mesh_pair_nocrop": (df, methods, base.stable_seed(1234, "expr_sec", *key)),
              "subject_pair_mean_nocrop": (spm, methods, base.stable_seed(1234, "expr_sec", *key2))}
    return frames, subjects


# -------------------------------------------------------------------------- bootstrap

def _replicate(k: int) -> dict:
    """Spearman di ogni (GT, metodo) e delle differenze appaiate richieste, per la replica k
    (k = 0: punto, nessun ricampionamento)."""
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
        mk = f"{a}&{b}"
        out[("pair", g, a, b)] = sp(g, a, mk) - sp(g, b, mk)
    for m in J["methods"]:
        for g1, g2 in (("unified", "maxabs"), ("unified", "coef"), ("unified_pairwise", "unified"),
                       ("native_sim", "maxabs"), ("unified", "native_sim_area"),
                       ("unified", "native_sim_area_region")):
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


def controls(rows: pd.DataFrame, prs: pd.DataFrame) -> pd.DataFrame:
    """Numeri pubblicati con la GT maxabs che devono tornare: punti e la differenza appaiata e108 - Chamfer eval."""
    out = []
    sp = pd.read_csv(RUNS / "data_scale_ood" / "arcface_vs_scale_hifi3d" / "spearman.csv")
    spp = pd.read_csv(RUNS / "data_scale_ood" / "arcface_vs_scale_hifi3d" / "spearman_paired.csv")
    for r in sp[sp["group"].isin(["nocrop_cross", "subject_pair_mean"])].itertuples():
        m = "chamfer_eval" if r.method == "chamfer" else r.method
        new = rows[(rows["domain"] == "hifi3d") & (rows["group"] == r.group) & (rows["gt"] == "maxabs")
                   & (rows["method"] == m)]
        if len(new):
            out.append({"domain": "hifi3d", "what": f"{m} {r.group} (punto)", "published": r.point,
                        "recomputed": float(new["point"].iloc[0])})
    for r in spp[(spp["a"] == "scale_e108") & (spp["b"] == "chamfer")
                 & spp["group"].isin(["nocrop_cross", "subject_pair_mean"])].itertuples():
        new = prs[(prs["domain"] == "hifi3d") & (prs["group"] == r.group) & (prs["gt"] == "maxabs")
                  & (prs["a"] == "scale_e108") & (prs["b"] == "chamfer_eval")].iloc[0]
        for c in ("point", "ci_low", "ci_high"):
            out.append({"domain": "hifi3d", "what": f"e108 - Chamfer eval {r.group} ({c})",
                        "published": getattr(r, c), "recomputed": float(new[c])})
    bl = pd.read_csv(RUNS / "data_scale_ood" / "hifi" / "baselines.csv")
    for setting, group in (("nocrop_cross_topology", "nocrop_cross"), ("original_to_original", "original_to_original")):
        for gt in ("maxabs", "coef"):
            for m in FB:
                r = bl[(bl["metric"] == m) & (bl["setting"] == setting) & (bl["gt"] == gt)]
                new = rows[(rows["domain"] == "hifi3d") & (rows["group"] == group) & (rows["gt"] == gt)
                           & (rows["method"] == f"fb_{m}")]
                if len(r) and len(new):
                    out.append({"domain": "hifi3d", "what": f"fb_{m} {group} GT {gt} (punto)",
                                "published": float(r["spearman"].iloc[0]), "recomputed": float(new["point"].iloc[0])})
    cells = pd.read_csv(RUNS / "data_scale_ood" / "hifi" / "table_cells.csv")
    for proto, group in (("mesh_pair_nocrop_cross", "nocrop_cross"), ("subject_pair_mean", "subject_pair_mean")):
        for gt in ("maxabs", "coef"):
            for arm, m in (("joint", "joint"), ("scale_e108", "scale_e108")):
                r = cells[(cells["model"] == arm) & (cells["gt"] == gt) & (cells["protocol"] == proto)
                          & (cells["scenario"] == "clean")]
                new = rows[(rows["domain"] == "hifi3d") & (rows["group"] == group) & (rows["gt"] == gt)
                           & (rows["method"] == m)]
                if len(r) and len(new):
                    r = r.iloc[0]
                    pub = r["latent_point_check"] if np.isfinite(r["latent_point_check"]) else r["latent_spearman"]
                    out.append({"domain": "hifi3d", "what": f"{m} {group} GT {gt} (punto)",
                                "published": float(pub), "recomputed": float(new["point"].iloc[0])})
    sec = pd.read_csv(RUNS / "ws_faceverse_expr" / "secondary.csv")
    for proto, group in (("mesh_pair_nocrop", "mesh_pair_nocrop"), ("subject_pair_mean_nocrop", "subject_pair_mean_nocrop")):
        r = sec[(sec["method"] == "scale_e108@bfm") & (sec["protocol"] == proto) & (sec["col_b"] == "raw_chamfer")]
        if len(r):
            r = r.iloc[0]
            new = prs[(prs["domain"] == "faceverse") & (prs["group"] == group) & (prs["gt"] == "maxabs")
                      & (prs["a"] == "scale_e108") & (prs["b"] == "chamfer_eval")].iloc[0]
            for c, c2 in (("point", "diff"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                out.append({"domain": "faceverse", "what": f"e108 - Chamfer eval {group} ({c})",
                            "published": float(r[c2]), "recomputed": float(new[c])})
        for m, col in (("chamfer_eval", "raw_chamfer"),) + tuple((f"fb_{x}", x) for x in FB):
            rb = sec[(sec["method"] == "baseline") & (sec["protocol"] == proto) & (sec["col_a"] == col)]
            new = rows[(rows["domain"] == "faceverse") & (rows["group"] == group) & (rows["gt"] == "maxabs")
                       & (rows["method"] == m)]
            if len(rb) and len(new):
                out.append({"domain": "faceverse", "what": f"{m} {group} (punto)",
                            "published": float(rb["spearman"].iloc[0]), "recomputed": float(new["point"].iloc[0])})
    df = pd.DataFrame(out)
    df["abs_diff"] = (df["published"] - df["recomputed"]).abs()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--domains", default="hifi3d,faceverse")
    a = p.parse_args()
    workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")), 32)
    R, P, G = [], [], []
    for dom in a.domains.split(","):
        frames, subjects = hifi_frames() if dom == "hifi3d" else fv_frames()
        for group, (df, methods, seed) in frames.items():
            r, pr, gd, n_s = evaluate(dom, group, df, methods, seed, a.n_bootstrap, workers)
            print(f"[e8-methods] {dom} {group}: {len(df)} righe, {n_s} soggetti", flush=True)
            R.append(r)
            P.append(pr)
            G.append(gd)
    R, P, G = pd.concat(R), pd.concat(P), pd.concat(G)
    R.to_csv(C.EVID_DIR / "methods_spearman.csv", index=False)
    P.to_csv(C.EVID_DIR / "methods_paired.csv", index=False)
    G.to_csv(C.EVID_DIR / "methods_gt_diff.csv", index=False)
    ctrl = controls(R, P)
    ctrl.to_csv(C.EVID_DIR / "methods_controls.csv", index=False)
    print(ctrl.to_string(index=False))
    piv = R.pivot_table(index=["domain", "group", "method"], columns="gt", values="point")
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
