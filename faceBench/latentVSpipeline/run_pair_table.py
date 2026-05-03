#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from fg_metrics import compute_pair_metrics, infer_mesh_path, load_mm_json, load_vertices


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute FG/FaceBench-style geometry metrics for a latent pair table.")
    p.add_argument("--pairs_csv", required=True, help="Input CSV with latent_distance and mesh path/sample columns")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--npz_root", default="", help="Root used to resolve sample_name_a/b to .npz paths")
    p.add_argument("--mm_json", default="", help="Optional morphable-model JSON for topology correction")
    p.add_argument("--stages", default="raw,rigid,fg", help="Comma-separated stages: raw,rigid,fg")
    p.add_argument("--max_pairs", type=int, default=0, help="0 means all")
    p.add_argument("--icp_points", type=int, default=4096)
    p.add_argument("--icp_iter", type=int, default=30)
    p.add_argument("--p2tri_points", type=int, default=4096, help="0 means all aligned source points")
    p.add_argument("--vertex_scale", type=float, default=1.0, help="Multiply loaded vertices by this value")
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def read_rows(path: Path, max_pairs: int) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.pairs_csv), max_pairs=int(args.max_pairs))
    mm = load_mm_json(args.mm_json)
    stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    out_rows: List[Dict[str, object]] = []

    for idx, row in enumerate(rows):
        out = dict(row)
        try:
            mesh_a = infer_mesh_path(row, "a", args.npz_root)
            mesh_b = infer_mesh_path(row, "b", args.npz_root)
            out["mesh_a"] = mesh_a
            out["mesh_b"] = mesh_b
            X = load_vertices(mesh_a) * float(args.vertex_scale)
            Y = load_vertices(mesh_b) * float(args.vertex_scale)
            metrics = compute_pair_metrics(
                X,
                Y,
                stages=stages,
                mm=mm,
                icp_points=int(args.icp_points),
                icp_iter=int(args.icp_iter),
                p2tri_points=int(args.p2tri_points),
                seed=int(args.seed) + idx,
            )
            out.update(metrics.__dict__)
        except Exception as exc:
            out["status"] = "failed"
            out["error"] = f"{type(exc).__name__}: {exc}"
        out_rows.append(out)
        if idx == 0 or (idx + 1) == len(rows) or (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(rows)} pairs", flush=True)

    out_csv = out_dir / "pair_metrics_with_fg.csv"
    write_rows(out_csv, out_rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
