#!/usr/bin/env python
"""Phase-0 driver: extend the v1 pair-level benchmark tables with
perceptual-2D and varifold/currents metrics, on the exact same pair sets.

Inputs
  - pair tables : paper_artifacts/bootstrap_ci/table1_pairlevel_exact/<topoA>__to__<topoB>/pair_metrics.csv
                  (subject_a, subject_b, topology_a, topology_b, gt_distance, latent_distance, raw_chamfer)
  - meshes      : datasets/REMESH/npz_data_topo_500/<sid>_GTready_<topo>.npz  (V/F)

Stages (each cached, resumable)
  1. render     : one PNG per mesh                      -> v2_work/phase0/cache/renders/
  2. embed      : arcface/clip/dinov2 per mesh          -> v2_work/phase0/cache/embeddings_<name>.npz
  3. measures   : varifold/currents mesh measures       -> in-memory (fast) per run
  4. pairs      : per topo-pair extended pair_metrics   -> v2_work/phase0/extended_pair_metrics/<pair>/pair_metrics.csv
  5. summary    : Spearman-vs-GT per metric per pair    -> v2_work/phase0/gate1_summary.csv (+ .md)
  6. proxy      : arcface distances vs D_GT (original)  -> v2_work/phase0/arcface_vs_gt_proxy.json

Usage:
  .conda_env/bin/python v2_work/phase0/run_phase0.py --stages render,embed
  .conda_env/bin/python v2_work/phase0/run_phase0.py --stages pairs,summary,proxy
  Optional: --extractors arcface,clip,dinov2  --lpips-subjects 20  --geom-max-tris 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

PAIR_TABLE_ROOT = REPO_ROOT / "paper_artifacts" / "bootstrap_ci" / "table1_pairlevel_exact"
MESH_ROOT = REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500"
GT_MATRIX = (
    REPO_ROOT
    / "face_embedding" / "gt_encdec" / "autoencoder" / "latent_analysis"
    / "gt_distance_matrix" / "normalized_matrix_distances.npz"
)
CACHE = THIS_DIR / "cache"
OUT_PAIRS = THIS_DIR / "extended_pair_metrics"

TOPOLOGIES = ["crop", "down8k", "noisy", "original", "remesh", "up60k"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="render,embed,pairs,summary,proxy")
    p.add_argument("--extractors", default="arcface,clip,dinov2")
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--geom-max-tris", type=int, default=2000)
    p.add_argument("--geom-kinds", default="varifold,currents")
    p.add_argument("--lpips-subjects", type=int, default=0, help="0 = skip LPIPS")
    p.add_argument("--limit-pairs", type=int, default=0, help="debug: cap pairs per topo-pair")
    return p.parse_args()


# ---------------------------------------------------------------- discovery

def load_pair_tables() -> dict[str, list[dict]]:
    tables = {}
    for d in sorted(PAIR_TABLE_ROOT.iterdir()):
        f = d / "pair_metrics.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            tables[d.name] = list(csv.DictReader(fh))
    if not tables:
        raise RuntimeError(f"no pair tables under {PAIR_TABLE_ROOT}")
    return tables


def mesh_name(sid: str, topo: str) -> str:
    return f"{sid}_GTready_{topo}"


def all_mesh_names(tables) -> list[str]:
    names = set()
    for rows in tables.values():
        for r in rows:
            names.add(mesh_name(r["subject_a"], r["topology_a"]))
            names.add(mesh_name(r["subject_b"], r["topology_b"]))
    return sorted(names)


# ---------------------------------------------------------------- stages

def stage_render(names, size):
    from render_mesh import render_npz
    from PIL import Image

    out = CACHE / "renders"
    out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out / f"{n}.png").exists()]
    print(f"[render] {len(todo)}/{len(names)} to render")
    t0 = time.time()
    for i, n in enumerate(todo):
        img = render_npz(MESH_ROOT / f"{n}.npz", size=size)
        Image.fromarray(img).save(out / f"{n}.png")
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"[render] {i+1}/{len(todo)} ({rate:.1f}/s)", flush=True)
    print(f"[render] done in {time.time()-t0:.0f}s")


def stage_embed(names, extractor_names):
    from PIL import Image
    from perceptual_embed import EXTRACTORS

    rend = CACHE / "renders"
    for ex_name in extractor_names:
        out_f = CACHE / f"embeddings_{ex_name}.npz"
        done: dict[str, np.ndarray] = {}
        if out_f.exists():
            with np.load(out_f) as z:
                done = {k: z[k] for k in z.files}
        todo = [n for n in names if n not in done]
        if not todo:
            print(f"[embed:{ex_name}] cached ({len(done)})")
            continue
        print(f"[embed:{ex_name}] {len(todo)} to embed")
        ex = EXTRACTORS[ex_name]()
        t0 = time.time()
        for i, n in enumerate(todo):
            img = np.asarray(Image.open(rend / f"{n}.png").convert("RGB"))
            done[n] = ex(img)
            if (i + 1) % 50 == 0:
                print(f"[embed:{ex_name}] {i+1}/{len(todo)} "
                      f"({(i+1)/(time.time()-t0):.1f}/s)", flush=True)
            if (i + 1) % 200 == 0:
                np.savez(out_f, **done)  # periodic checkpoint
        np.savez(out_f, **done)
        extra = f" fallback={ex.n_fallback}" if hasattr(ex, "n_fallback") else ""
        print(f"[embed:{ex_name}] done in {time.time()-t0:.0f}s{extra}")


def load_embeddings(extractor_names, prefix: str = "embeddings") -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for ex_name in extractor_names:
        f = CACHE / f"{prefix}_{ex_name}.npz"
        if f.exists():
            with np.load(f) as z:
                out[ex_name] = {k: z[k] for k in z.files}
    return out


def stage_pairs(tables, names, args):
    from measure_distances import mesh_measure, varifold_distance, currents_distance

    embs = load_embeddings(args.extractors.split(","))
    geom_kinds = [k for k in args.geom_kinds.split(",") if k]

    measures: dict[str, dict] = {}
    if geom_kinds:
        print(f"[pairs] building {len(names)} mesh measures (max_tris={args.geom_max_tris})")
        for i, n in enumerate(names):
            measures[n] = mesh_measure(MESH_ROOT / f"{n}.npz", max_tris=args.geom_max_tris)
            if (i + 1) % 100 == 0:
                print(f"[pairs] measures {i+1}/{len(names)}", flush=True)

    lpips_model = None
    lpips_subjects: set[str] = set()
    if args.lpips_subjects > 0:
        import lpips as lpips_mod
        import torch
        from PIL import Image

        lpips_model = lpips_mod.LPIPS(net="alex")
        subj_all = sorted({r["subject_a"] for rows in tables.values() for r in rows})
        lpips_subjects = set(subj_all[: args.lpips_subjects])

        def lpips_dist(na, nb):
            def load(n):
                im = np.asarray(Image.open(CACHE / "renders" / f"{n}.png").convert("RGB"))
                t = torch.from_numpy(im).float().permute(2, 0, 1) / 127.5 - 1.0
                return t.unsqueeze(0)
            with torch.no_grad():
                return float(lpips_model(load(na), load(nb)).item())
    OUT_PAIRS.mkdir(parents=True, exist_ok=True)

    for pair_label, rows in tables.items():
        out_d = OUT_PAIRS / pair_label
        out_f = out_d / "pair_metrics.csv"
        if out_f.exists():
            print(f"[pairs] {pair_label} cached")
            continue
        out_d.mkdir(exist_ok=True)
        if args.limit_pairs > 0:
            rows = rows[: args.limit_pairs]
        t0 = time.time()
        out_rows = []
        for r in rows:
            na = mesh_name(r["subject_a"], r["topology_a"])
            nb = mesh_name(r["subject_b"], r["topology_b"])
            row = dict(r)
            for ex_name, table in embs.items():
                if na in table and nb in table:
                    row[f"{ex_name}_dist"] = float(1.0 - np.dot(table[na], table[nb]))
            if geom_kinds and na in measures and nb in measures:
                if "varifold" in geom_kinds:
                    row["varifold_dist"] = varifold_distance(measures[na], measures[nb])
                if "currents" in geom_kinds:
                    row["currents_dist"] = currents_distance(measures[na], measures[nb])
            if lpips_model is not None and r["subject_a"] in lpips_subjects and r["subject_b"] in lpips_subjects:
                row["lpips_dist"] = lpips_dist(na, nb)
            out_rows.append(row)
        keys = list(out_rows[0].keys())
        with open(out_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(out_rows)
        print(f"[pairs] {pair_label}: {len(out_rows)} rows in {time.time()-t0:.0f}s", flush=True)


def _spearman(x, y):
    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(spearmanr(x[m], y[m]).statistic)


def stage_summary(out_pairs: Path = OUT_PAIRS, out_csv: Path | None = None):
    out_csv = out_csv or THIS_DIR / "gate1_summary.csv"
    metric_cols = None
    rows_out = []
    all_rows = []
    for d in sorted(out_pairs.iterdir()):
        f = d / "pair_metrics.csv"
        if not f.exists():
            continue
        with open(f, newline="") as fh:
            rows = list(csv.DictReader(fh))
        all_rows.extend(rows)
        if metric_cols is None:
            skip = {"subject_a", "subject_b", "topology_a", "topology_b",
                    "gt_distance", "n_mesh_pairs", "mesh_pair_index"}
            metric_cols = [k for k in rows[0].keys() if k not in skip]
        gt = [r["gt_distance"] for r in rows]
        out = {"pair_label": d.name, "n_pairs": len(rows)}
        for c in metric_cols:
            out[f"spearman_{c}"] = _spearman(gt, [r.get(c, "nan") or "nan" for r in rows])
        rows_out.append(out)

    gt = [r["gt_distance"] for r in all_rows]
    overall = {"pair_label": "OVERALL", "n_pairs": len(all_rows)}
    for c in metric_cols:
        overall[f"spearman_{c}"] = _spearman(gt, [r.get(c, "nan") or "nan" for r in all_rows])
    rows_out.insert(0, overall)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[summary] wrote {out_csv}")
    print(json.dumps(rows_out[0], indent=2))
    return rows_out


def stage_proxy(args, prefix: str = "embeddings", out_name: str = "arcface_vs_gt_proxy.json"):
    """ArcFace-on-renders vs D_GT: external sanity of the GT matrix (D2.2)."""
    embs = load_embeddings(["arcface", "clip", "dinov2"], prefix=prefix)
    with np.load(GT_MATRIX, allow_pickle=True) as z:
        D = z["D_orig"]
        gt_names = [str(n) for n in z["names"]]
    name_to_idx = {n.split("_GTready")[0]: i for i, n in enumerate(gt_names)}

    out = {}
    for ex_name, table in embs.items():
        subs = sorted({n.split("_GTready_")[0] for n in table
                       if n.endswith("_original")})
        subs = [s for s in subs if s in name_to_idx]
        gt_vals, emb_vals = [], []
        for i, sa in enumerate(subs):
            for sb in subs[i + 1:]:
                gt_vals.append(D[name_to_idx[sa], name_to_idx[sb]])
                ea = table[mesh_name(sa, "original")]
                eb = table[mesh_name(sb, "original")]
                emb_vals.append(1.0 - float(np.dot(ea, eb)))
        out[ex_name] = {
            "n_subjects": len(subs),
            "n_pairs": len(gt_vals),
            "spearman_vs_gt": _spearman(gt_vals, emb_vals),
        }
    with open(THIS_DIR / out_name, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[proxy] {json.dumps(out, indent=2)}")
    return out


def main():
    args = parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    tables = load_pair_tables()
    names = all_mesh_names(tables)
    print(f"pair tables: {len(tables)} | unique meshes: {len(names)}")

    if "render" in stages:
        stage_render(names, args.render_size)
    if "embed" in stages:
        stage_embed(names, args.extractors.split(","))
    if "pairs" in stages:
        stage_pairs(tables, names, args)
    if "summary" in stages:
        stage_summary()
    if "proxy" in stages:
        stage_proxy(args)


if __name__ == "__main__":
    main()
