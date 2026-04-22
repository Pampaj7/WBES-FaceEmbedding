#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a REMESH GT surrogate in the same per-mesh normalized space used by the benchmark.")
    p.add_argument(
        "--data_dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"),
    )
    p.add_argument(
        "--topology",
        type=str,
        default="original",
        help="Topology suffix used to pick one representative mesh per subject.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "checking_assumptions" / "outputs" / "gt_surrogate_normalized_original"),
    )
    return p.parse_args()


def _normalize_vertices(V: np.ndarray) -> np.ndarray:
    Vn = V.astype(np.float64, copy=True)
    Vn -= Vn.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(Vn)))
    if scale > 1e-6:
        Vn /= scale
    else:
        Vn *= 0.0
    return Vn


def _mean_vertex_l2(Va: np.ndarray, Vb: np.ndarray) -> float:
    diff = Va - Vb
    return float(np.sqrt((diff * diff).sum(axis=1)).mean())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(data_dir.glob(f"*_GTready_{args.topology}.npz"))
    if not files:
        raise RuntimeError(f"No files found for topology={args.topology} in {data_dir}")

    subjects: List[str] = []
    verts_norm: List[np.ndarray] = []
    for path in files:
        sid = path.name.split("_")[0].lower()
        subjects.append(sid)
        data = np.load(path, allow_pickle=False)
        V = data["verts"].astype(np.float64)
        data.close()
        verts_norm.append(_normalize_vertices(V))

    n = len(subjects)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _mean_vertex_l2(verts_norm[i], verts_norm[j])
            D[i, j] = D[j, i] = d

    mask = D > 0
    if not mask.any():
        raise RuntimeError("Surrogate distance matrix has no positive entries")
    D_norm = D / float(D[mask].max())

    npz_path = out_dir / "normalized_matrix_distances_surrogate.npz"
    np.savez(npz_path, D_orig=D_norm.astype(np.float32), names=np.asarray([f"{sid}_GTready" for sid in subjects], dtype=object))

    summary = {
        "data_dir": str(data_dir),
        "topology": str(args.topology),
        "subject_count": int(n),
        "matrix_shape": [int(n), int(n)],
        "positive_min": float(D_norm[mask].min()),
        "positive_max": float(D_norm[mask].max()),
        "npz_path": str(npz_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Normalized GT Surrogate",
                "",
                "This surrogate GT matrix is built directly from REMESH `original` meshes after applying the same per-mesh normalization used by `GTReadyDatasetNPZ`.",
                "",
                f"- topology source: `{args.topology}`",
                f"- subject count: `{n}`",
                f"- output: `{npz_path}`",
                "",
                "Distance definition:",
                "- subtract per-mesh centroid",
                "- divide each mesh by its own max absolute coordinate",
                "- compute mean per-vertex L2 distance",
                "- normalize the final matrix by its positive max",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] wrote surrogate GT to {npz_path}")


if __name__ == "__main__":
    main()
