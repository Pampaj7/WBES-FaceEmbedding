#!/usr/bin/env python3
"""CLIP e DINOv2 su render di sola geometria: riconoscimento e ranking con le funzioni delle tabelle esistenti.

    aau/run.sh aau/baselines_extra/vfm_summarize.py --domain fv_expr      (riconoscimento)
    aau/run.sh aau/baselines_extra/vfm_summarize.py --domain hifi3d       (riconoscimento + ranking)
    aau/run.sh aau/baselines_extra/vfm_summarize.py --domain bfm_heldout  (ranking in dominio)
    (be_summarize.sbatch)

Protocollo: ``aau/runs/baselines_extra/protocol.md``, dichiarato prima dei numeri e copiato in testa al summary.
Nessun calcolo e' riscritto:

- distanze degli encoder: ``zs_arcface_summarize.arcface_distances`` (media delle viste L2, 1 - coseno) sugli
  npz di ``vfm_embed.py``, che hanno lo stesso formato di ``arcface_views.npz``;
- riconoscimento (FaceVerse con espressioni, HIFI3D): ``zs_expr_summarize`` (``Index``, ``_recog_task``,
  ``bootstrap_counts`` con lo STESSO seme ``stable_seed(1234, "expr_recognition")``), cioe' lo stesso codice
  di ``zs_arcface_summarize.py``; le righe gia' presenti in ``aau/runs/arcface_render_zs/<dominio>/recognition.csv``
  devono coincidere (controllo nel summary);
- ranking (HIFI3D, BFM held-out): Spearman con la GT e CI per soggetto con ``weighted_bootstrap_spearman``
  del paper (``ict_summarize.bootstrap_row``), delta appaiati con ``zs_summarize.paired_bootstrap`` (stesse
  repliche per le due colonne; stesso seme per tutti i confronti di un setting).

Coppie del ranking (come ``aau/baselines/rank_from_matrix.py``): per ogni coppia ordinata di topologie del
setting (``common.setting_topology_pairs``) le 4950 coppie di soggetti i<j, soggetto i in tA e j in tB.

Scrive ``<summary-dir>/<domain>/{recognition,recognition_paired,ranking,ranking_paired}.csv`` e
``<summary-dir>/results_<domain>.md``, poi ricompone ``<summary-dir>/summary.md`` = protocol.md + results_*.md
+ other_baselines.md + giudizio.md (quelli presenti).
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

THIS_DIR = Path(__file__).resolve().parent
AAU_DIR = THIS_DIR.parent
REPO_ROOT = AAU_DIR.parent
sys.path.insert(0, str(AAU_DIR / "zs3dmm"))

import zs_arcface_summarize as zas  # noqa: E402
import zs_expr_summarize as zes  # noqa: E402
from zs_stage import TOPOLOGIES, select_subjects  # noqa: E402

zsum, base, common = zes.zsum, zes.base, zes.common
RUNS = AAU_DIR / "runs"
VIEWS3 = zas.VIEWS["3v"]
REFERENCES = ("clip_l14_crop", "dinov2_b14_crop")
ABLATIONS = (("clip_l14_full", "clip_l14_crop"), ("dinov2_b14_full", "dinov2_b14_crop"),
             ("clip_l14_crop", "dinov2_b14_crop"))
RANK_SETTINGS = ("nocrop_cross_topology", "original_to_original")
LABEL = {
    "clip_l14_crop": "CLIP ViT-L/14, normal map, crop, 3 viste (riferimento)",
    "dinov2_b14_crop": "DINOv2 ViT-B/14, normal map, crop, 3 viste (riferimento)",
    "clip_l14_full": "CLIP ViT-L/14, normal map, render intero, 3 viste",
    "dinov2_b14_full": "DINOv2 ViT-B/14, normal map, render intero, 3 viste",
    "arcface_normals_3v": "ArcFace, normal map, 3 viste (stessi render)",
    "arcface_shaded_3v": "ArcFace, ombreggiato, 3 viste",
    "joint@bfm": "BFM+ICT congiunto, convenzione BFM",
    "ws1_chamfer": "Chamfer faceBench (WS1)", "ws1_lpips": "LPIPS AlexNet, ombreggiato (WS1)",
    "ws1_arcface": "ArcFace, ombreggiato, crop fisso (WS1)",
    "ws1_clip": "CLIP ViT-B/32 laion2b, ombreggiato intero (WS1)",
    "ws1_dinov2": "DINOv2 ViT-S/14, ombreggiato intero (WS1)",
    "neurips": "Modello NeurIPS (BFM, v1)",
}
DOMAINS = {
    "fv_expr": {"label": "FaceVerse v2 con espressioni casuali", "view": "datasets/FACEVERSE_ZS/expr_view/npz",
                "runs": "aau/runs/ws_faceverse_expr/data_736f96956a", "joint": "joint_flip_topology/zs_zeroshot",
                "recognition": True, "gt": None},
    "hifi3d": {"label": "HIFI3D, neutre", "view": "datasets/HIFI3D/eval_view/npz",
               "runs": "aau/runs/ws_hifi3d/data_328f2bfc1a", "joint": "joint_frame-xmymz_flip_ranking/zs_zeroshot",
               "recognition": True, "gt": {"maxabs": "datasets/HIFI3D/eval_view/gt_matrix.npz",
                                           "coef": "datasets/HIFI3D/eval_view/gt_coef_matrix.npz"}},
    "bfm_heldout": {"label": "BFM held-out (in dominio, i 100 soggetti di WS1)", "view": None, "runs": None,
                    "joint": None, "recognition": False, "gt": {"paper": None}},
}
RANK_COMPARATORS = {"hifi3d": ("chamfer", "nicp_p2tri", "arcface_normals_3v", "joint@bfm"),
                    "bfm_heldout": ("ws1_chamfer", "ws1_lpips", "ws1_arcface", "arcface_normals_3v", "neurips")}
RECOG_COMPARATORS = ("arcface_normals_3v", "arcface_shaded_3v", "nicp_p2tri", "chamfer", "joint@bfm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    p.add_argument("--summary-dir", type=Path, default=RUNS / "baselines_extra")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-seed", type=int, default=1234)
    return p.parse_args()


def label(name: str) -> str:
    return LABEL.get(name) or zes.BL_LABEL.get(name, name)


# ------------------------------------------------------------------------ distanze 600 x 600

def encoder_distances(dom: str, args, idx: zes.Index) -> dict:
    D = {}
    for name in ("clip_l14_crop", "dinov2_b14_crop", "clip_l14_full", "dinov2_b14_full"):
        D[name] = zas.arcface_distances(args.summary_dir / dom / f"{name}_views.npz", idx, VIEWS3)
    arc_root = RUNS / "arcface_render_zs" / dom if dom != "bfm_heldout" else args.summary_dir / dom
    for mode in ("normals", "shaded"):
        D[f"arcface_{mode}_3v"] = zas.arcface_distances(arc_root / mode / "arcface_views.npz", idx, VIEWS3)
    return D


# ------------------------------------------------------------------------------ ranking

PAIR_KEYS = ["subject_a", "subject_b", "topology_a", "topology_b"]


def frame_from_D(D: np.ndarray, idx: zes.Index, setting: str, col: str) -> pd.DataFrame:
    i, j = common.subject_pair_indices(len(idx.subjects))
    names = np.asarray(idx.subjects)
    frames = [pd.DataFrame({"subject_a": names[i], "subject_b": names[j], "topology_a": ta, "topology_b": tb,
                            col: D[idx.rows(ta)[i], idx.rows(tb)[j]]})
              for ta, tb in common.setting_topology_pairs(setting)]
    return pd.concat(frames, ignore_index=True)


def frame_from_matrices(root: Path, metric: str, subjects: list[str], setting: str, col: str) -> pd.DataFrame:
    frames = []
    for ta, tb in common.setting_topology_pairs(setting):
        M, subj, _, _, _ = common.load_matrix(common.matrix_path(metric, ta, tb, root))
        if subj != subjects:
            raise SystemExit(f"{root} {metric} {ta}->{tb}: soggetti diversi dal set")
        i, j = common.subject_pair_indices(len(subj))
        names = np.asarray(subj)
        frames.append(pd.DataFrame({"subject_a": names[i], "subject_b": names[j], "topology_a": ta,
                                    "topology_b": tb, col: M[i, j]}))
    return pd.concat(frames, ignore_index=True)


def frame_neurips(subjects: list[str], setting: str) -> pd.DataFrame:
    """``latent_distance`` delle pair table del paper; original->original dalla matrice ``latent_v1`` di WS1."""
    if setting == "original_to_original":
        return frame_from_matrices(common.OUT_ROOT, "latent_v1", subjects, setting, "neurips")
    frames = []
    for ta, tb in common.setting_topology_pairs(setting):
        pm = pd.read_csv(common.PAIR_TABLE_ROOT / f"{ta}__to__{tb}" / "pair_metrics.csv",
                         usecols=PAIR_KEYS + ["latent_distance"])
        pm = pm.groupby(PAIR_KEYS, as_index=False)["latent_distance"].mean()
        if len(pm) != len(subjects) * (len(subjects) - 1) // 2:
            raise SystemExit(f"pair table {ta}->{tb}: {len(pm)} righe")
        frames.append(pm.rename(columns={"latent_distance": "neurips"}))
    return pd.concat(frames, ignore_index=True)


def merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = frames[0]
    for f in frames[1:]:
        m = out.merge(f, on=PAIR_KEYS, how="inner", validate="one_to_one")
        if len(m) != len(out) or len(m) != len(f):
            raise SystemExit(f"righe non allineate fra i metodi: {len(out)}, {len(f)}, comuni {len(m)}")
        out = m
    return out


def ranking(dom: str, args, idx: zes.Index, D: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    subjects = idx.subjects
    if dom == "hifi3d":
        bl_root = REPO_ROOT / DOMAINS[dom]["runs"] / "baselines"
        gts = {tag: zsum.load_gt(REPO_ROOT / p) for tag, p in DOMAINS[dom]["gt"].items()}
    else:
        G = common.load_gt_submatrix(subjects)
        gts = {"paper": (np.asarray(G, np.float64), {s: k for k, s in enumerate(subjects)})}
    methods = list(D)
    wide = {}
    for setting in RANK_SETTINGS:
        frames = [frame_from_D(D[m], idx, setting, m) for m in methods]
        if dom == "hifi3d":
            frames += [frame_from_matrices(bl_root, m, subjects, setting, m) for m in zes.BL_FACEBENCH]
        else:
            for m in ("chamfer", "lpips", "arcface", "clip", "dinov2"):
                frames.append(frame_from_matrices(common.OUT_ROOT, m, subjects, setting, f"ws1_{m}"))
            frames.append(frame_neurips(subjects, setting))
        wide[setting] = merge_frames(frames)
        wide[setting]["gt_distance"] = 0.0
    cols = [c for c in wide[RANK_SETTINGS[0]].columns if c not in PAIR_KEYS + ["gt_distance"]]

    tasks = []
    for setting in RANK_SETTINGS:
        for gt_tag, gt in gts.items():
            df = zsum.with_gt(wide[setting], gt)
            for c in cols:
                tasks.append(((setting, gt_tag, c), df, c, args.n_bootstrap,
                              base.stable_seed(args.seed, "be", dom, setting, gt_tag, c)))
            pseed = base.stable_seed(args.seed, "be_paired", dom, setting, gt_tag)
            for a in REFERENCES:
                for b in RANK_COMPARATORS[dom] + tuple(x for x in REFERENCES if x != a):
                    tasks.append(((setting, gt_tag, a, b), df, (a, b), args.n_bootstrap, pseed))
            for a, b in ABLATIONS[:2]:
                tasks.append(((setting, gt_tag, a, b), df, (a, b), args.n_bootstrap, pseed))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[be-sum] {dom}: {len(tasks)} bootstrap di ranking su {workers} processi", flush=True)
    res = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for key, r in pool.imap_unordered(zsum._boot_task, tasks):
            res[key] = r
    single = pd.DataFrame([{"setting": k[0], "gt": k[1], "method": k[2],
                            "n_nan": int(wide[k[0]][k[2]].isna().sum()), **r} for k, r in res.items() if len(k) == 3])
    # paired_bootstrap chiama "a" e "b" i due Spearman: rinominati, i nomi dei metodi vanno in "a"/"b".
    paired = pd.DataFrame([{"setting": k[0], "gt": k[1],
                            **{(f"spearman_{x}" if x in ("a", "b") else x): v for x, v in r.items()},
                            "a": k[2], "b": k[3]} for k, r in res.items() if len(k) == 4])

    lines = []
    for gt_tag in gts:
        lines += [f"\n### Spearman con la GT `{gt_tag}`, CI 95% bootstrap per soggetto ({args.n_bootstrap} repliche)\n",
                  "| metodo | " + " | ".join(RANK_SETTINGS) + " | distanze NaN (cross) |",
                  "| --- | " + " | ".join("---" for _ in RANK_SETTINGS) + " | --- |"]
        sub = single[single["gt"] == gt_tag].set_index(["setting", "method"])
        for c in cols:
            cells = [zes.fmt(*sub.loc[(s, c), ["spearman", "ci_low", "ci_high"]]) for s in RANK_SETTINGS]
            lines.append(f"| {label(c)} | " + " | ".join(cells) + f" | {sub.loc[(RANK_SETTINGS[0], c), 'n_nan']} |")
        lines += ["\nDelta appaiati A - B dello Spearman (stesse repliche), [CI 95%] (P<=0), righe finite per entrambi:\n",
                  "| A | B | " + " | ".join(RANK_SETTINGS) + " |", "| --- | --- | " + " | ".join("---" for _ in RANK_SETTINGS) + " |"]
        sp = paired[paired["gt"] == gt_tag].set_index(["setting", "a", "b"])
        order = [(a, b) for a in REFERENCES for b in RANK_COMPARATORS[dom] + REFERENCES if a != b] + list(ABLATIONS[:2])
        for a, b in order:
            cells = []
            for s in RANK_SETTINGS:
                r = sp.loc[(s, a, b)]
                cells.append(f"{zes.fmt(r['diff'], r['ci_low'], r['ci_high'], True)} ({r['p_boot_le0']:.3f})")
            lines.append(f"| {label(a)} | {label(b)} | " + " | ".join(cells) + " |")
    n = len(subjects) * (len(subjects) - 1) // 2
    head = [f"Coppie: `nocrop_cross_topology` = 20 coppie ordinate di topologie x {n} = {20 * n} righe; "
            f"`original_to_original` = {n}."]
    return single, paired, head + lines


# ------------------------------------------------------------------------ riconoscimento

def recognition(dom: str, args, idx: zes.Index, D: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cfg = DOMAINS[dom]
    runs = REPO_ROOT / cfg["runs"]
    D = dict(D)
    joint = runs / cfg["joint"]
    D["joint@bfm"] = zes.model_distances(joint, idx)
    staged = json.loads((joint.parent / "subjects.json").read_text())
    if sorted(staged["subjects"]) != idx.subjects:
        raise SystemExit("congiunto: zs_stage ha valutato soggetti diversi")
    for m in zes.BL_FACEBENCH:
        D[m] = zes.facebench_distances(runs / "baselines", m, idx)
    counts = zes.bootstrap_counts(len(idx.subjects), args.n_bootstrap, base.stable_seed(args.seed, "expr_recognition"))
    blocks = {
        "nocrop": ([(a, b) for a in zes.NOCROP for b in zes.NOCROP if a != b],
                   [(a, b) for i, a in enumerate(zes.NOCROP) for b in zes.NOCROP[i + 1:]]),
        "crop": ([(a, b) for a in TOPOLOGIES for b in TOPOLOGIES if a != b and "crop" in (a, b)],
                 [("crop", b) for b in zes.NOCROP]),
    }
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    rec = {}
    with mp.get_context("fork").Pool(min(workers, 16)) as pool:
        tasks = [((blk, name), D[name], idx, pr, pv, counts) for blk, (pr, pv) in blocks.items() for name in D]
        for key, vals, n_nan in pool.imap_unordered(zes._recog_task, tasks):
            rec[key] = (vals, n_nan)
    rows = []
    for (blk, name), (vals, n_nan) in rec.items():
        r = {"block": blk, "method": name, "n_nan_distances": n_nan}
        for m in ("rank1", "map", "auc"):
            r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(vals[m][0]), zes.ci(vals[m])
        rows.append(r)
    table = pd.DataFrame(rows)
    comparisons = [(a, b) for a in REFERENCES for b in RECOG_COMPARATORS] + list(ABLATIONS)
    deltas = []
    for blk in blocks:
        for a_name, b_name in comparisons:
            a, b = rec[(blk, a_name)][0], rec[(blk, b_name)][0]
            r = {"block": blk, "model": a_name, "baseline": b_name}
            for m in ("rank1", "map", "auc"):
                d = a[m] - b[m]
                r[m], (r[f"{m}_ci_low"], r[f"{m}_ci_high"]) = float(d[0]), zes.ci(d)
                r[f"{m}_p_le0"] = float((d[1:] <= 0).mean())
            deltas.append(r)
    delta = pd.DataFrame(deltas)
    order = list(REFERENCES) + [n for n in D if n not in REFERENCES]

    def rec_md(blk: str) -> list[str]:
        out = ["| metodo | rank-1 | mAP | AUC verifica | distanze NaN |", "| --- | --- | --- | --- | --- |"]
        sub = table[table["block"] == blk].set_index("method")
        for name in order:
            r = sub.loc[name]
            out.append(f"| {label(name)} | " + " | ".join(zes.fmt(r[m], r[f"{m}_ci_low"], r[f"{m}_ci_high"])
                                                          for m in ("rank1", "map", "auc")) + f" | {r['n_nan_distances']} |")
        return out

    def delta_md(blk: str, pairs) -> list[str]:
        out = ["| A | B | rank-1: A - B [CI 95%] (P<=0) | mAP (P<=0) | AUC (P<=0) |", "| --- | --- | --- | --- | --- |"]
        sub = delta[delta["block"] == blk]
        for a_name, b_name in pairs:
            r = sub[(sub["model"] == a_name) & (sub["baseline"] == b_name)].iloc[0]
            out.append(f"| {label(a_name)} | {label(b_name)} | "
                       + " | ".join(f"{zes.fmt(r[m], r[m + '_ci_low'], r[m + '_ci_high'], True)} ({r[m + '_p_le0']:.3f})"
                                    for m in ("rank1", "map", "auc")) + " |")
        return out

    # Controllo: le righe gia' calcolate da zs_arcface_summarize.py devono coincidere.
    ref_csv = RUNS / "arcface_render_zs" / dom / "recognition.csv"
    ref = pd.read_csv(ref_csv).set_index(["block", "method"])
    cols = [f"{m}{s}" for m in ("rank1", "map", "auc") for s in ("", "_ci_low", "_ci_high")]
    diffs = {(r["block"], r["method"]): max(abs(r[c] - ref.loc[(r["block"], r["method"]), c]) for c in cols)
             for r in rows if (r["block"], r["method"]) in ref.index}
    check = (f"- riproduzione di `{ref_csv}` (stesse repliche): max |diff| su punto e CI di rank-1, mAP, AUC = "
             f"{max(diffs.values()):.2e} su {len(diffs)} righe ({', '.join(sorted({m for _, m in diffs}))})")
    n_q = len(blocks["nocrop"][0]) * len(idx.subjects)
    primary = [c for c in comparisons if c[0] in REFERENCES and c[1] in RECOG_COMPARATORS]
    lines = ["## Riconoscimento d'identita'\n",
             f"Baseline da `{runs / 'baselines'}`, congiunto da `{joint}`. CI 95% bootstrap per soggetto, "
             f"{args.n_bootstrap} repliche (le stesse per tutte le righe).\n",
             f"### PRIMARIO: 5 topologie senza crop ({n_q} query; 1000 coppie stessa persona, 99000 diverse)\n",
             *rec_md("nocrop"), "\n#### Delta appaiati, righe di riferimento - confronti\n", *delta_md("nocrop", primary),
             "\n#### Ablazioni, delta appaiati\n", *delta_md("nocrop", ABLATIONS),
             "\n### A parte: crop (coppie di topologie con crop da un lato)\n", *rec_md("crop"),
             "\n#### Delta appaiati, crop\n", *delta_md("crop", primary + list(ABLATIONS)),
             "\n### Controllo\n", check]
    return table, delta, lines


def main() -> None:
    args = parse_args()
    dom, cfg = args.domain, DOMAINS[args.domain]
    if dom == "bfm_heldout":
        subjects = common.heldout_subjects()
        staged = json.loads((args.summary_dir / dom / "subjects.json").read_text())
        if staged["subjects"] != subjects:
            raise SystemExit("subjects.json dei render diverso da common.heldout_subjects")
    else:
        subjects = select_subjects(REPO_ROOT / cfg["view"], args.eval_seed)
    idx = zes.Index(subjects)
    print(f"[be-sum] {dom}: {len(subjects)} soggetti (primi {subjects[:3]})", flush=True)
    D = encoder_distances(dom, args, idx)

    out = args.summary_dir / dom
    parts = [f"# Risultati: {cfg['label']}\n",
             f"Soggetti: {len(subjects)}; embedding da `{out}` (vfm_embed.py), ArcFace da "
             f"`{RUNS / 'arcface_render_zs' / dom if dom != 'bfm_heldout' else out}`.\n"]
    if cfg["recognition"]:
        table, delta, lines = recognition(dom, args, idx, D)
        table.to_csv(out / "recognition.csv", index=False)
        delta.to_csv(out / "recognition_paired.csv", index=False)
        parts += lines
    if cfg["gt"] is not None:
        if dom == "hifi3d":
            D["joint@bfm"] = zes.model_distances(REPO_ROOT / cfg["runs"] / cfg["joint"], idx)
        single, paired, lines = ranking(dom, args, idx, D)
        single.to_csv(out / "ranking.csv", index=False)
        paired.to_csv(out / "ranking_paired.csv", index=False)
        parts += ["\n## Ranking: Spearman con la GT\n", *lines]
    (args.summary_dir / f"results_{dom}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    sections = [(args.summary_dir / "protocol.md").read_text().rstrip()]
    for name in ("results_fv_expr.md", "results_hifi3d.md", "results_bfm_heldout.md", "other_baselines.md",
                 "giudizio.md"):
        if (args.summary_dir / name).exists():
            sections += ["\n---\n", (args.summary_dir / name).read_text().rstrip()]
    (args.summary_dir / "summary.md").write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[be-sum] scritto {args.summary_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
