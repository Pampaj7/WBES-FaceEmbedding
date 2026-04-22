#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
AUTOENCODER_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "autoencoder"
if str(AUTOENCODER_DIR) not in sys.path:
    sys.path.append(str(AUTOENCODER_DIR))

from face_embedding.gt_encdec.remeshing.intrinsic.intrinsic_utils import (  # noqa: E402
    SUBJECT_RE_ANY,
    build_subject_map,
    load_gt_distance_matrix,
)
from face_embedding.gt_encdec.autoencoder.dataset_gtready import GTReadyDatasetNPZ  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check the REMESH GT distance matrix.")
    p.add_argument(
        "--dist_npz",
        type=str,
        default=str(
            REPO_ROOT
            / "face_embedding"
            / "gt_encdec"
            / "autoencoder"
            / "latent_analysis"
            / "gt_distance_matrix"
            / "normalized_matrix_distances.npz"
        ),
    )
    p.add_argument(
        "--remesh_dir",
        type=str,
        default=str(REPO_ROOT / "datasets" / "REMESH" / "npz_data_topo_500_withops"),
    )
    p.add_argument("--sample_subjects", type=int, default=80)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(REPO_ROOT / "checking_assumptions" / "outputs" / "gt_matrix_sanity"),
    )
    return p.parse_args()


def _decode_name(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


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


def _normalize_distance_matrix(D: np.ndarray) -> np.ndarray:
    out = D.astype(np.float64, copy=True)
    mask = out > 0
    if mask.any():
        out /= float(out[mask].max())
    return out


def _corr_summary(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
        "mae": float(np.mean(np.abs(x - y))),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = np.load(args.dist_npz, allow_pickle=True)
    D_raw = pack["D_orig"].astype(np.float64)
    names_raw = [_decode_name(x) for x in pack["names"]]

    parsed_ids: List[str | None] = []
    for n in names_raw:
        m = SUBJECT_RE_ANY.search(n)
        parsed_ids.append(m.group(1).lower() if m else None)

    parsed_ok = [x for x in parsed_ids if x is not None]
    duplicate_ids = sorted((k, v) for k, v in Counter(parsed_ok).items() if v > 1)

    structure_rows = [
        {
            "metric": "shape_n",
            "value": int(D_raw.shape[0]),
        },
        {
            "metric": "diag_abs_max",
            "value": float(np.abs(np.diag(D_raw)).max()),
        },
        {
            "metric": "sym_abs_max",
            "value": float(np.abs(D_raw - D_raw.T).max()),
        },
        {
            "metric": "positive_min",
            "value": float(D_raw[D_raw > 0].min()),
        },
        {
            "metric": "positive_max",
            "value": float(D_raw[D_raw > 0].max()),
        },
        {
            "metric": "negative_count",
            "value": int((D_raw < 0).sum()),
        },
        {
            "metric": "zero_count",
            "value": int((D_raw == 0).sum()),
        },
        {
            "metric": "parsed_name_count",
            "value": int(len(parsed_ok)),
        },
        {
            "metric": "unparsed_name_count",
            "value": int(len(parsed_ids) - len(parsed_ok)),
        },
        {
            "metric": "duplicate_subject_id_count",
            "value": int(len(duplicate_ids)),
        },
    ]
    _write_csv(out_dir / "matrix_structure.csv", structure_rows)

    dataset = GTReadyDatasetNPZ(args.remesh_dir, verbose=False)
    subject_map = build_subject_map(dataset.files, subject_re=SUBJECT_RE_ANY)
    _, name_to_idx = load_gt_distance_matrix(args.dist_npz, subject_re=SUBJECT_RE_ANY, dtype=np.float64)
    dataset_subjects = sorted(subject_map.keys())
    gt_subjects = sorted(name_to_idx.keys())
    overlap = sorted(set(dataset_subjects) & set(gt_subjects))

    overlap_rows = [
        {
            "dataset_subjects": len(dataset_subjects),
            "gt_subjects": len(gt_subjects),
            "overlap_subjects": len(overlap),
            "dataset_not_in_gt": len(set(dataset_subjects) - set(gt_subjects)),
            "gt_not_in_dataset": len(set(gt_subjects) - set(dataset_subjects)),
            "min_meshes_per_subject": min(len(v) for v in subject_map.values()),
            "max_meshes_per_subject": max(len(v) for v in subject_map.values()),
        }
    ]
    _write_csv(out_dir / "dataset_overlap.csv", overlap_rows)

    rng = np.random.default_rng(int(args.seed))
    original_files = sorted(Path(args.remesh_dir).glob("*_original.npz"))
    if len(original_files) < 2:
        raise RuntimeError(f"Need at least 2 original-topology files in {args.remesh_dir}")

    subject_by_file = [f.name.split("_")[0].lower() for f in original_files]
    eligible_idx = [i for i, sid in enumerate(subject_by_file) if sid in name_to_idx]
    if len(eligible_idx) < 2:
        raise RuntimeError("No original-topology REMESH subjects overlap with GT matrix")

    n_pick = min(int(args.sample_subjects), len(eligible_idx))
    picked = np.sort(rng.choice(np.asarray(eligible_idx, dtype=np.int64), size=n_pick, replace=False))
    picked_files = [original_files[int(i)] for i in picked]
    picked_subjects = [subject_by_file[int(i)] for i in picked]

    verts_raw: List[np.ndarray] = []
    verts_norm: List[np.ndarray] = []
    for path in picked_files:
        data = np.load(path, allow_pickle=False)
        V = data["verts"].astype(np.float64)
        data.close()
        verts_raw.append(V)
        verts_norm.append(_normalize_vertices(V))

    n = len(picked_files)
    D_subset_gt = D_raw[np.ix_([name_to_idx[s] for s in picked_subjects], [name_to_idx[s] for s in picked_subjects])]
    D_subset_raw = np.zeros((n, n), dtype=np.float64)
    D_subset_norm = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d_raw = _mean_vertex_l2(verts_raw[i], verts_raw[j])
            d_norm = _mean_vertex_l2(verts_norm[i], verts_norm[j])
            D_subset_raw[i, j] = D_subset_raw[j, i] = d_raw
            D_subset_norm[i, j] = D_subset_norm[j, i] = d_norm

    D_subset_raw_normed = _normalize_distance_matrix(D_subset_raw)
    D_subset_norm_normed = _normalize_distance_matrix(D_subset_norm)
    iu = np.triu_indices(n, k=1)

    comparison_rows = []
    for label, A, B in [
        ("raw_original_vs_gt", D_subset_raw_normed[iu], D_subset_gt[iu]),
        ("per_mesh_normalized_original_vs_gt", D_subset_norm_normed[iu], D_subset_gt[iu]),
        ("raw_original_vs_per_mesh_normalized_original", D_subset_raw_normed[iu], D_subset_norm_normed[iu]),
    ]:
        stats = _corr_summary(A, B)
        comparison_rows.append(
            {
                "comparison": label,
                "n_subjects": n,
                "n_pairs": int(len(A)),
                **stats,
            }
        )
    _write_csv(out_dir / "original_topology_subset_comparison.csv", comparison_rows)

    summary = {
        "dist_npz": str(args.dist_npz),
        "remesh_dir": str(args.remesh_dir),
        "structure": {row["metric"]: row["value"] for row in structure_rows},
        "dataset_overlap": overlap_rows[0],
        "sample_subjects": picked_subjects,
        "original_topology_subset_comparison": comparison_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = "\n".join(
        [
            "# GT Matrix Sanity",
            "",
            "## Structural checks",
            f"- matrix shape: {D_raw.shape[0]} x {D_raw.shape[1]}",
            f"- diagonal abs max: {float(np.abs(np.diag(D_raw)).max()):.6g}",
            f"- symmetry abs max: {float(np.abs(D_raw - D_raw.T).max()):.6g}",
            f"- parsed subject ids: {len(parsed_ok)} / {len(parsed_ids)}",
            f"- duplicate parsed ids: {len(duplicate_ids)}",
            "",
            "## Dataset overlap",
            f"- REMESH subjects: {len(dataset_subjects)}",
            f"- GT subjects: {len(gt_subjects)}",
            f"- overlap: {len(overlap)}",
            f"- meshes per REMESH subject: {overlap_rows[0]['min_meshes_per_subject']} to {overlap_rows[0]['max_meshes_per_subject']}",
            "",
            "## Original-topology subset comparison",
            f"- subset size: {n} subjects",
            f"- raw original vs GT: pearson {comparison_rows[0]['pearson']:.6f}, spearman {comparison_rows[0]['spearman']:.6f}",
            f"- per-mesh normalized original vs GT: pearson {comparison_rows[1]['pearson']:.6f}, spearman {comparison_rows[1]['spearman']:.6f}",
            f"- raw original vs per-mesh normalized original: pearson {comparison_rows[2]['pearson']:.6f}, spearman {comparison_rows[2]['spearman']:.6f}",
            "",
            "Interpretation:",
            "- If `raw original vs GT` is ~1.0 while `per-mesh normalized original vs GT` is noticeably lower, then the GT matrix is not arbitrary or randomly wrong.",
            "- Instead, it is behaving like a matrix built from raw original-space vertex distances, while the benchmark metrics are evaluated on per-mesh normalized geometry.",
        ]
    )
    (out_dir / "README.md").write_text(readme + "\n", encoding="utf-8")

    print(readme)


if __name__ == "__main__":
    main()
