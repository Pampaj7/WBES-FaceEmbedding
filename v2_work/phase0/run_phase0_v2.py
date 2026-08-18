#!/usr/bin/env python
"""Phase-0 v2: bias-fixed, multi-view perceptual metrics on the v1 pair sets.

Two changes over run_phase0.py (v1):
  1. shared frame  -- every topology of a subject is rendered with the centre and
     scale of that subject's `_GTready_original` mesh, so `crop` is no longer
     blown up by its own tighter bbox (v1: fg 54% vs 48%).
  2. 3 views       -- yaw in {-30, 0, +30} deg; the per-mesh embedding is the
     L2-normalised mean of the three per-view embeddings.

Everything else (extractors, pair sets, Spearman-vs-GT summary, GT proxy) is v1's,
imported from run_phase0 rather than re-implemented.

Outputs
  cache/renders_v2/<mesh>_y{-30,0,30}.png      cache/embeddings_v2_<extractor>.npz
  extended_pair_metrics_v2/<pair>/pair_metrics.csv
  gate1_summary_v2.csv                         arcface_vs_gt_proxy_v2.json

Usage:
  .conda_env/bin/python v2_work/phase0/run_phase0_v2.py --stages render,embed
  .conda_env/bin/python v2_work/phase0/run_phase0_v2.py --stages pairs,summary,proxy
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import run_phase0 as v1  # noqa: E402  (needs THIS_DIR on the path first)
from render_mesh import mesh_frame, render_npz  # noqa: E402

YAWS = (-30, 0, 30)
RENDERS = v1.CACHE / "renders_v2"
OUT_PAIRS = THIS_DIR / "extended_pair_metrics_v2"
SUMMARY_CSV = THIS_DIR / "gate1_summary_v2.csv"
V1_SUMMARY_CSV = THIS_DIR / "gate1_summary.csv"


def view_names(mesh: str) -> list[str]:
    return [f"{mesh}_y{y}" for y in YAWS]


# ---------------------------------------------------------------- stages

def stage_render(names, size):
    from PIL import Image

    RENDERS.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not all((RENDERS / f"{v}.png").exists() for v in view_names(n))]
    print(f"[render] {len(todo)}/{len(names)} meshes to render ({len(YAWS)} views each)")
    frames: dict[str, tuple[np.ndarray, float]] = {}
    t0 = time.time()
    for i, n in enumerate(todo):
        sid = n.split("_GTready_")[0]
        if sid not in frames:  # shared frame = this subject's `original` mesh
            with np.load(v1.MESH_ROOT / f"{v1.mesh_name(sid, 'original')}.npz") as d:
                frames[sid] = mesh_frame(d["V"] if "V" in d else d["verts"])
        center, scale = frames[sid]
        for yaw, out_name in zip(YAWS, view_names(n)):
            f = RENDERS / f"{out_name}.png"
            if f.exists():
                continue
            img = render_npz(v1.MESH_ROOT / f"{n}.npz", size=size,
                             scale=scale, center=center, yaw=yaw)
            Image.fromarray(img).save(f)
        if (i + 1) % 50 == 0:
            print(f"[render] {i+1}/{len(todo)} ({(i+1)/(time.time()-t0):.1f} mesh/s)", flush=True)
    print(f"[render] done in {time.time()-t0:.0f}s")


def stage_embed(names, extractor_names):
    from PIL import Image
    from perceptual_embed import EXTRACTORS

    for ex_name in extractor_names:
        out_f = v1.CACHE / f"embeddings_v2_{ex_name}.npz"
        done: dict[str, np.ndarray] = {}
        if out_f.exists():
            with np.load(out_f) as z:
                done = {k: z[k] for k in z.files}
        todo = [n for n in names if n not in done]
        if not todo:
            print(f"[embed:{ex_name}] cached ({len(done)})")
            continue
        print(f"[embed:{ex_name}] {len(todo)} meshes x {len(YAWS)} views to embed")
        ex = EXTRACTORS[ex_name]()
        t0 = time.time()
        for i, n in enumerate(todo):
            views = [ex(np.asarray(Image.open(RENDERS / f"{v}.png").convert("RGB")))
                     for v in view_names(n)]
            e = np.mean(views, axis=0)
            done[n] = (e / max(float(np.linalg.norm(e)), 1e-9)).astype(np.float32)
            if (i + 1) % 50 == 0:
                print(f"[embed:{ex_name}] {i+1}/{len(todo)} "
                      f"({(i+1)/(time.time()-t0):.2f} mesh/s)", flush=True)
            if (i + 1) % 100 == 0:
                np.savez(out_f, **done)  # checkpoint
        np.savez(out_f, **done)
        extra = f" fallback={ex.n_fallback}" if hasattr(ex, "n_fallback") else ""
        print(f"[embed:{ex_name}] done in {time.time()-t0:.0f}s{extra}")


def stage_pairs(tables, extractor_names):
    embs = v1.load_embeddings(extractor_names, prefix="embeddings_v2")
    OUT_PAIRS.mkdir(parents=True, exist_ok=True)
    for pair_label, rows in tables.items():
        out_f = OUT_PAIRS / pair_label / "pair_metrics.csv"
        if out_f.exists():
            print(f"[pairs] {pair_label} cached")
            continue
        out_f.parent.mkdir(exist_ok=True)
        out_rows = []
        for r in rows:
            na = v1.mesh_name(r["subject_a"], r["topology_a"])
            nb = v1.mesh_name(r["subject_b"], r["topology_b"])
            row = dict(r)
            for ex_name, table in embs.items():
                if na in table and nb in table:
                    row[f"{ex_name}_v2_dist"] = float(1.0 - np.dot(table[na], table[nb]))
            out_rows.append(row)
        with open(out_f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"[pairs] {pair_label}: {len(out_rows)} rows", flush=True)


def stage_summary(extractor_names):
    rows = v1.stage_summary(OUT_PAIRS, SUMMARY_CSV)
    v2_overall = rows[0]
    with open(V1_SUMMARY_CSV, newline="") as fh:
        v1_overall = next(r for r in csv.DictReader(fh) if r["pair_label"] == "OVERALL")
    print("\nOVERALL Spearman vs GT, v1 (single view, per-mesh bbox) -> v2 (shared frame, 3 views)")
    for ex in extractor_names:
        a = float(v1_overall[f"spearman_{ex}_dist"])
        b = float(v2_overall[f"spearman_{ex}_v2_dist"])
        print(f"  {ex:8s} {a:+.3f} -> {b:+.3f}  ({b-a:+.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stages", default="render,embed,pairs,summary,proxy")
    p.add_argument("--extractors", default="arcface,clip,dinov2")
    p.add_argument("--render-size", type=int, default=512)
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    extractors = args.extractors.split(",")
    tables = v1.load_pair_tables()
    names = v1.all_mesh_names(tables)
    print(f"pair tables: {len(tables)} | unique meshes: {len(names)} | yaws: {YAWS}")

    if "render" in stages:
        stage_render(names, args.render_size)
    if "embed" in stages:
        stage_embed(names, extractors)
    if "pairs" in stages:
        stage_pairs(tables, extractors)
    if "summary" in stages:
        stage_summary(extractors)
    if "proxy" in stages:
        v1.stage_proxy(None, prefix="embeddings_v2", out_name="arcface_vs_gt_proxy_v2.json")


if __name__ == "__main__":
    main()
