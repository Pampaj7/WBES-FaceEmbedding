from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PERTURBATED_DIR = REPO_ROOT / "face_embedding" / "gt_encdec" / "remeshing" / "intrinsic" / "perturbated"
if str(PERTURBATED_DIR) not in sys.path:
    sys.path.insert(0, str(PERTURBATED_DIR))

import compare_model_vs_chamfer_rankings as base  # noqa: E402
import registration_utils as reg_utils  # noqa: E402


DEFAULT_DATA_DIR = (
    REPO_ROOT
    / "datasets"
    / "REMESH"
    / "npz_data_topo_500_withops"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sanity-check the registration pipeline on synthetic rigid nuisance. "
            "If rigid ICP cannot recover obvious self-to-self transforms, the large ranking result is suspicious."
        )
    )
    p.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out_dir", type=str, default=str(REPO_ROOT / "checking_assumptions" / "outputs" / "synthetic_icp_sanity"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n_subjects", type=int, default=2)
    p.add_argument("--topology_labels", type=str, default="crop,original,remesh")
    p.add_argument("--icp_points", type=int, default=128)
    p.add_argument("--cpd_points", type=int, default=64)
    p.add_argument("--registration_workers", type=int, default=4)
    p.add_argument("--icp_max_correspondence_distance", type=float, default=0.05)
    p.add_argument("--icp_max_iteration", type=int, default=20)
    p.add_argument("--cpd_alpha", type=float, default=2.0)
    p.add_argument("--cpd_beta", type=float, default=0.2)
    p.add_argument("--cpd_max_iteration", type=int, default=15)
    p.add_argument("--cpd_tolerance", type=float, default=1.0e-4)
    p.add_argument("--cpd_outlier_weight", type=float, default=0.05)
    p.add_argument("--warp_chunk_size", type=int, default=4096)
    return p.parse_args()


def _rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _apply_known_rigid(V: torch.Tensor, *, rx_deg: float, ry_deg: float, rz_deg: float, translation_xyz: tuple[float, float, float]) -> torch.Tensor:
    R = torch.as_tensor(_rotation_matrix_xyz(rx_deg, ry_deg, rz_deg), dtype=V.dtype, device=V.device)
    t = torch.as_tensor(np.asarray(translation_xyz, dtype=np.float32), dtype=V.dtype, device=V.device)
    return V @ R.T + t


def _pick_records(dataset: base.GTReadyDataset, *, n_subjects: int, topology_labels: List[str]) -> List[dict]:
    subject_map = base.build_subject_map(dataset.files, subject_re=base.SUBJECT_RE_ANY)
    chosen_records: List[dict] = []
    for sid in sorted(subject_map.keys()):
        subject_files = subject_map[sid]
        selected_for_subject = []
        for idx in subject_files:
            sample_name = Path(dataset.files[int(idx)]).stem.lower()
            for topology in topology_labels:
                if topology in sample_name:
                    selected_for_subject.append({"subject_id": sid, "dataset_idx": int(idx), "sample_name": dataset.files[int(idx)], "topology_label": topology})
                    break
        if selected_for_subject:
            seen = set()
            for rec in selected_for_subject:
                key = rec["topology_label"]
                if key in seen:
                    continue
                chosen_records.append(rec)
                seen.add(key)
        if len({rec["subject_id"] for rec in chosen_records}) >= int(n_subjects):
            break
    return chosen_records


def _metric_values_for_pair(
    V_src: torch.Tensor,
    V_tgt: torch.Tensor,
    *,
    icp_points: int,
    cpd_points: int,
    registration_workers: int,
    icp_max_correspondence_distance: float,
    icp_max_iteration: int,
    cpd_alpha: float,
    cpd_beta: float,
    cpd_max_iteration: int,
    cpd_tolerance: float,
    cpd_outlier_weight: float,
    warp_chunk_size: int,
) -> Dict[str, float]:
    vertex_sets = [V_src.contiguous(), V_tgt.contiguous()]
    src_np = V_src.detach().cpu().numpy()
    tgt_np = V_tgt.detach().cpu().numpy()

    icp_idx_src = reg_utils.build_sample_vertex_indices(vertex_count=src_np.shape[0], n_points=icp_points, seed=11)
    icp_idx_tgt = reg_utils.build_sample_vertex_indices(vertex_count=tgt_np.shape[0], n_points=icp_points, seed=17)
    cpd_idx_src = reg_utils.build_sample_vertex_indices(vertex_count=src_np.shape[0], n_points=cpd_points, seed=23)
    cpd_idx_tgt = reg_utils.build_sample_vertex_indices(vertex_count=tgt_np.shape[0], n_points=cpd_points, seed=29)

    raw = base.compute_pairwise_chamfer_values(
        vertex_sets=vertex_sets,
        pair_i=np.asarray([0], dtype=np.int64),
        pair_j=np.asarray([1], dtype=np.int64),
        batch_pairs=1,
        progress_desc="",
        show_progress=False,
    )[0]

    rigid_vals, rigid_timings = reg_utils.compute_pairwise_registered_chamfer_values(
        vertex_sets=vertex_sets,
        icp_point_sets=[
            reg_utils.extract_point_subset(V_src, icp_idx_src),
            reg_utils.extract_point_subset(V_tgt, icp_idx_tgt),
        ],
        cpd_point_sets=[
            reg_utils.extract_point_subset(V_src, cpd_idx_src),
            reg_utils.extract_point_subset(V_tgt, cpd_idx_tgt),
        ],
        pair_i=np.asarray([0], dtype=np.int64),
        pair_j=np.asarray([1], dtype=np.int64),
        batch_pairs=1,
        registration_workers=registration_workers,
        icp_max_correspondence_distance=icp_max_correspondence_distance,
        icp_max_iteration=icp_max_iteration,
        use_nonrigid_cpd=False,
        cpd_alpha=cpd_alpha,
        cpd_beta=cpd_beta,
        cpd_max_iteration=cpd_max_iteration,
        cpd_tolerance=cpd_tolerance,
        cpd_outlier_weight=cpd_outlier_weight,
        warp_chunk_size=warp_chunk_size,
        progress_desc="",
        show_progress=False,
    )

    cpd_vals, cpd_timings = reg_utils.compute_pairwise_registered_chamfer_values(
        vertex_sets=vertex_sets,
        icp_point_sets=[
            reg_utils.extract_point_subset(V_src, icp_idx_src),
            reg_utils.extract_point_subset(V_tgt, icp_idx_tgt),
        ],
        cpd_point_sets=[
            reg_utils.extract_point_subset(V_src, cpd_idx_src),
            reg_utils.extract_point_subset(V_tgt, cpd_idx_tgt),
        ],
        pair_i=np.asarray([0], dtype=np.int64),
        pair_j=np.asarray([1], dtype=np.int64),
        batch_pairs=1,
        registration_workers=registration_workers,
        icp_max_correspondence_distance=icp_max_correspondence_distance,
        icp_max_iteration=icp_max_iteration,
        use_nonrigid_cpd=True,
        cpd_alpha=cpd_alpha,
        cpd_beta=cpd_beta,
        cpd_max_iteration=cpd_max_iteration,
        cpd_tolerance=cpd_tolerance,
        cpd_outlier_weight=cpd_outlier_weight,
        warp_chunk_size=warp_chunk_size,
        progress_desc="",
        show_progress=False,
    )

    return {
        "raw_chamfer": float(raw),
        "rigid_registered_chamfer": float(rigid_vals[0]),
        "cpd_registered_chamfer": float(cpd_vals[0]),
        "rigid_icp_seconds": float(rigid_timings["rigid_icp_seconds"][0]),
        "rigid_total_seconds": float(rigid_timings["total_pair_seconds"][0]),
        "cpd_total_seconds": float(cpd_timings["total_pair_seconds"][0]),
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))

    dataset = base.GTReadyDataset(str(Path(args.data_dir).expanduser().resolve()))
    topology_labels = [tok.strip().lower() for tok in str(args.topology_labels).split(",") if tok.strip()]
    picked = _pick_records(dataset, n_subjects=int(args.n_subjects), topology_labels=topology_labels)
    if not picked:
        raise RuntimeError("Could not find any records matching the requested topology labels")

    transforms = [
        {"transform_name": "identity", "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0, "translation_xyz": (0.0, 0.0, 0.0)},
        {"transform_name": "translation_only", "rx_deg": 0.0, "ry_deg": 0.0, "rz_deg": 0.0, "translation_xyz": (0.03, -0.015, 0.02)},
        {"transform_name": "rotation_only", "rx_deg": 0.0, "ry_deg": 8.0, "rz_deg": -5.0, "translation_xyz": (0.0, 0.0, 0.0)},
        {"transform_name": "rigid_combo", "rx_deg": 4.0, "ry_deg": 7.0, "rz_deg": -6.0, "translation_xyz": (0.02, 0.01, -0.015)},
    ]

    rows: List[dict] = []
    for rec in picked:
        sample = dataset[int(rec["dataset_idx"])]
        V_src = sample["verts"].contiguous()
        for transform in transforms:
            V_tgt = _apply_known_rigid(
                V_src,
                rx_deg=float(transform["rx_deg"]),
                ry_deg=float(transform["ry_deg"]),
                rz_deg=float(transform["rz_deg"]),
                translation_xyz=tuple(transform["translation_xyz"]),
            )
            metric_values = _metric_values_for_pair(
                V_src=V_src,
                V_tgt=V_tgt,
                icp_points=int(args.icp_points),
                cpd_points=int(args.cpd_points),
                registration_workers=int(args.registration_workers),
                icp_max_correspondence_distance=float(args.icp_max_correspondence_distance),
                icp_max_iteration=int(args.icp_max_iteration),
                cpd_alpha=float(args.cpd_alpha),
                cpd_beta=float(args.cpd_beta),
                cpd_max_iteration=int(args.cpd_max_iteration),
                cpd_tolerance=float(args.cpd_tolerance),
                cpd_outlier_weight=float(args.cpd_outlier_weight),
                warp_chunk_size=int(args.warp_chunk_size),
            )
            rows.append(
                {
                    "subject_id": str(rec["subject_id"]),
                    "topology_label": str(rec["topology_label"]),
                    "sample_name": str(rec["sample_name"]),
                    "transform_name": str(transform["transform_name"]),
                    "rx_deg": float(transform["rx_deg"]),
                    "ry_deg": float(transform["ry_deg"]),
                    "rz_deg": float(transform["rz_deg"]),
                    "tx": float(transform["translation_xyz"][0]),
                    "ty": float(transform["translation_xyz"][1]),
                    "tz": float(transform["translation_xyz"][2]),
                    **metric_values,
                }
            )

    summary_rows: List[dict] = []
    by_transform: Dict[str, List[dict]] = {}
    for row in rows:
        by_transform.setdefault(str(row["transform_name"]), []).append(row)
    for transform_name, group in sorted(by_transform.items()):
        summary_rows.append(
            {
                "transform_name": transform_name,
                "n_cases": int(len(group)),
                "raw_mean": float(np.mean([float(r["raw_chamfer"]) for r in group])),
                "rigid_mean": float(np.mean([float(r["rigid_registered_chamfer"]) for r in group])),
                "cpd_mean": float(np.mean([float(r["cpd_registered_chamfer"]) for r in group])),
                "rigid_beats_raw_frac": float(np.mean([float(r["rigid_registered_chamfer"] < r["raw_chamfer"]) for r in group])),
                "cpd_beats_rigid_frac": float(np.mean([float(r["cpd_registered_chamfer"] < r["rigid_registered_chamfer"]) for r in group])),
            }
        )

    payload = {
        "seed": int(args.seed),
        "n_cases": int(len(rows)),
        "picked_records": picked,
        "summary_by_transform": summary_rows,
    }

    _write_csv(out_dir / "pair_metrics.csv", rows)
    _write_csv(out_dir / "summary_by_transform.csv", summary_rows)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    md_lines = [
        "# Synthetic ICP Sanity",
        "",
        "Question: can the current registration stack recover obvious rigid nuisance on identical meshes?",
        "",
        f"- Cases: `{len(rows)}`",
        f"- Subjects covered: `{len(sorted({row['subject_id'] for row in rows}))}`",
        "",
        "## Summary By Transform",
        "",
        "| transform | n | raw mean | rigid mean | cpd mean | rigid<raw frac | cpd<rigid frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        md_lines.append(
            "| {transform_name} | {n_cases} | {raw_mean:.6f} | {rigid_mean:.6f} | {cpd_mean:.6f} | {rigid_beats_raw_frac:.3f} | {cpd_beats_rigid_frac:.3f} |".format(
                **row
            )
        )
    md_lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- If `translation_only` and `rotation_only` do not improve strongly after rigid ICP, something is wrong in the registration pipeline or parameters.",
            "- If rigid ICP succeeds here but still hurts the benchmark, that supports the idea that the benchmark failure is not a basic implementation bug.",
        ]
    )
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[done] wrote synthetic sanity outputs to {out_dir}")


if __name__ == "__main__":
    main()
