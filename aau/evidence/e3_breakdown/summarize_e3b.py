#!/usr/bin/env python3
"""E3b: e108 con rimesh uniforme al test (e le diagnosi di centro e pooling) contro e108 com'e', stessi protocolli.

    aau/run.sh aau/evidence/e3_breakdown/summarize_e3b.py      (summary.sbatch, CPU)

Stessa macchina di ``aau/evidence/e2_canon/summarize_e2.py`` (importata): Spearman HIFI3D sulle righe delle
pair_metrics di e108 col seme della riga pubblicata per gruppo; riconoscimento con ``bootstrap_counts`` del seme
``expr_recognition``; differenze appaiate variante - e108 sulle stesse repliche. Su FaceVerse anche e108 sul
FaceVerse NEUTRO (stessi soggetti e topologie, convenzione BFM): neutro - espressioni = costo dell'espressione.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "aau" / "evidence" / "e2_canon"))

import summarize_e2 as s2  # noqa: E402

zes, zsum, base = s2.zes, s2.zsum, s2.base
E3 = s2.RUNS / "evidence" / "e3"
LABEL = {"e108": "e108 (mesh come sono)", "e108_remesh": "e108, rimesh uniforme al test",
         "e108_areacenter": "e108, centro per area", "e108_areapool": "e108, pooling medio per area",
         "e108_areaboth": "e108, centro e pooling per area", "e108_neutral": "e108 su FaceVerse NEUTRO"}


def dist(stage: Path, idx) -> np.ndarray | None:
    return zes.model_distances(stage, idx) if (stage / "embeddings.npz").exists() else None


def variant_dir(v: str) -> Path:
    tmp = E3 / "hifi" / "variants_e108" / f"_{v}"
    tmp.mkdir(parents=True, exist_ok=True)
    f = E3 / "hifi" / "variants_e108" / f"embeddings_{v}.npz"
    if f.exists() and not (tmp / "embeddings.npz").exists():
        (tmp / "embeddings.npz").symlink_to(f)
    return tmp


def spearman(idx, D: dict, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm = base.read_pair_metrics(s2.HIFI_RUNS / "scale_e108_topology" / zsum.STAGE)
    keys = zsum.PAIR_KEYS
    df = pm[keys + ["gt_distance", "latent_distance"]].rename(columns={"latent_distance": "e108"})
    df["subject_a"], df["subject_b"] = df["subject_a"].astype(str), df["subject_b"].astype(str)
    ia = np.asarray([idx.pos[k] for k in zip(df["subject_a"], df["topology_a"])])
    ib = np.asarray([idx.pos[k] for k in zip(df["subject_b"], df["topology_b"])])
    cols = ["e108"] + [m for m in D if m != "e108"]
    for m in cols[1:]:
        df[m] = D[m][ia, ib]
    frames = {"nocrop_cross": df[df.topology_a.ne("crop") & df.topology_b.ne("crop")], "all_cross": df,
              "subject_pair_mean": df.groupby(["subject_a", "subject_b"], as_index=False)[["gt_distance"] + cols].mean()}
    tasks = []
    for g, fr in frames.items():
        seed = base.stable_seed(args.seed, "scale_e108", "maxabs", s2.TOK[g], "latent")
        for c in cols:
            tasks.append(((g, c, "ref"), fr, c, args.n_bootstrap, seed))
        for c in cols[1:]:
            tasks.append(((g, c, "e108"), fr, (c, "e108"), args.n_bootstrap, seed))
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = dict(pool.imap_unordered(zsum._boot_task, tasks))
    rows = [{"group": g, "method": a, "point": r["spearman"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
            for (g, a, b), r in res.items() if b == "ref"]
    pairs = [{"group": g, "a": a, "b": b, "diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
              "p_le0": r["p_boot_le0"]} for (g, a, b), r in res.items() if b != "ref"]
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    p.add_argument("--out-md", type=Path, default=E3 / "summary_e3b.md")
    args = p.parse_args()
    out = E3 / "tables"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# E3b: rimesh uniforme al test (e diagnosi di centro e pooling), e108 senza training\n",
          "Protocollo e lato L dichiarati prima in `aau/runs/evidence/e2/protocol.md`; generatore in "
          "`aau/evidence/e3_breakdown/remesh.py` (record per mesh in `aau/runs/evidence/e3/<dominio>/remesh_records.csv`). "
          f"IC 95% bootstrap per soggetto, {args.n_bootstrap} repliche, le stesse dei riferimenti; P(<=0) = frazione di "
          "repliche con differenza <= 0.\n"]
    for dom, view, emb in (("hifi", s2.REPO_ROOT / "datasets/HIFI3D/eval_view/npz", s2.HIFI_RUNS / "scale_e108_embed" / zsum.STAGE),
                           ("fv", s2.REPO_ROOT / "datasets/FACEVERSE_ZS/expr_view/npz",
                            s2.FV_RUNS / "scale_e108_flip_topology" / zsum.STAGE)):
        subjects = s2.select_subjects(view, 1234)
        idx = zes.Index(subjects)
        D = {"e108": zes.model_distances(emb, idx)}
        for name, stage in (("e108_remesh", E3 / dom / "remesh_e108"),) + (
                tuple((f"e108_{v}", variant_dir(v)) for v in ("areacenter", "areapool", "areaboth")) if dom == "hifi"
                else (("e108_neutral", E3 / "fv" / "neutral_e108_flip"),)):
            d = dist(stage, idx)
            if d is not None:
                D[name] = d
        recognition = s2.recognition(D, idx, zes.bootstrap_counts(len(subjects), args.n_bootstrap,
                                                                         base.stable_seed(args.seed, "expr_recognition")),
                                           s2.blocks_rec(), args.workers)
        comps = [(m, "e108") for m in D if m != "e108"]
        rows, d = s2.rec_tables(recognition, comps)
        rows.to_csv(out / f"e3b_{dom}_recognition.csv", index=False)
        d.to_csv(out / f"e3b_{dom}_recognition_paired.csv", index=False)
        title = "HIFI3D" if dom == "hifi" else "FaceVerse con espressioni (convenzione BFM)"
        if dom == "hifi":
            sr, sp = spearman(idx, D, args)
            sr.to_csv(out / "e3b_hifi_spearman.csv", index=False)
            sp.to_csv(out / "e3b_hifi_spearman_paired.csv", index=False)
            groups = ("nocrop_cross", "all_cross", "subject_pair_mean")
            md += [f"## {title}: Spearman con la GT maxabs\n",
                   *s2.table(["metodo"] + list(groups),
                             [[LABEL[m]] + [s2.fmt(*sr[(sr.group == g) & (sr.method == m)][["point", "ci_low", "ci_high"]].iloc[0])
                                            for g in groups] for m in D]),
                   "\nDifferenze appaiate variante - e108:\n",
                   *s2.table(["variante - e108"] + [f"{g} [IC] (P<=0)" for g in groups],
                             [[LABEL[m]] + [f"{s2.fmt(r['diff'], r.ci_low, r.ci_high, True)} ({r.p_le0:.3f})"
                                            for r in [sp[(sp.group == g) & (sp.a == m)].iloc[0] for g in groups]]
                              for m in D if m != "e108"]), ""]
        for blk, sub in (("nocrop", "5 topologie senza crop"), ("crop", "coppie con crop (a parte)")):
            md += [f"## {title}: riconoscimento, {sub}\n",
                   *s2.table(["metodo", "rank-1", "mAP", "AUC"],
                             [[LABEL[m]] + [s2.fmt(r[x], r[f"{x}_ci_low"], r[f"{x}_ci_high"]) for x in ("rank1", "map", "auc")]
                              for m in D for r in [rows[(rows.block == blk) & (rows.method == m)].iloc[0]]]),
                   "\nDifferenze appaiate (stesse repliche):\n",
                   *s2.table(["A - B", "rank-1 [IC] (P<=0)", "mAP", "AUC"],
                             [[f"{LABEL[r.a]} - {LABEL[r.b]}"] + [f"{s2.fmt(getattr(r, x), getattr(r, x + '_ci_low'), getattr(r, x + '_ci_high'), True)} "
                                                                  f"({getattr(r, x + '_p_le0'):.3f})" for x in ("rank1", "map", "auc")]
                              for r in (d[d["block"] == blk].itertuples() if not d.empty else [])]), ""]
        if not (E3 / dom / "remesh_records.csv").exists():
            continue
        rr = pd.read_csv(E3 / dom / "remesh_records.csv")
        md += [f"Rimesh, {title}: vertici in uscita per topologia (mediana) e qualita':\n",
               *s2.table(["topologia", "vertici in ingresso", "vertici in uscita", "lato medio / L", "CV dei lati",
                          "area uscita / ingresso", "spigoli non-manifold (totale)", "s/mesh"],
                         [[t, int(g.n_verts_in.median()), int(g.n_verts_out.median()), f"{g.edge_mean_over_L.median():.3f}",
                           f"{g.edge_cv.median():.3f}", f"{g.area_ratio_out_in.median():.3f}", int(g.nonmanifold_edges.sum()),
                           f"{g.seconds.median():.1f}"] for t, g in rr.groupby("topology")]), ""]
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[e3b-sum] scritto {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
