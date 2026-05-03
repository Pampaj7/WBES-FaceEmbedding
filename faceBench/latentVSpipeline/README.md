# Latent vs FG Pipeline

This folder contains the comparison experiment for the paper:

> Compare our latent ranking against a transparent geometry pipeline instead of
> relying on an opaque external black box.

The intended claim is not that FG is a new learned baseline. FG is a controlled,
inspectable geometry pipeline. We use it to ask whether the pair ranking induced
by geometry-only processing agrees with the identity structure kept by the

## Scripts

### `run_facebench_remesh.py` — **USE THIS** (full facebench pipeline, at scale)

Mirrors `facebench/examples/large_conf_test.py` but for REMESH cross-topology pairs.
Uses the real `facebench` library (not the DIY `fg_metrics.py`).

Pipeline stages:
- `raw`   — symmetric Chamfer (no alignment)
- `rigid` — ICP (bbox prealign) → P2P
- `nicp`  — rigid ICP → non-rigid ICP → Chamfer correspondence → P2P + P2Tri

```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/run_facebench_remesh.py \
  --npz_root datasets/REMESH/npz_data_topo_500 \
  --withops_root datasets/REMESH/npz_data_topo_500_withops \
  --checkpoint face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth \
  --model_config face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json \
  --gt_matrix face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
  --out_dir faceBench/latentVSpipeline/outputs/facebench_remesh_full \
  --max_subjects 100 \
  --stages raw,rigid,nicp \
  --max_sample_points 4096 \
  --workers 8
```

Quick smoke test (5 subjects, raw+rigid only, no NICP):
```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/run_facebench_remesh.py \
  --npz_root datasets/REMESH/npz_data_topo_500 \
  --withops_root datasets/REMESH/npz_data_topo_500_withops \
  --checkpoint face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth \
  --model_config face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json \
  --gt_matrix face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
  --out_dir faceBench/latentVSpipeline/outputs/smoke_facebench \
  --max_subjects 5 \
  --stages raw,rigid \
  --topo_pairs "original,remesh;original,down8k" \
  --workers 4
```

### Legacy scripts (small scale, DIY geometry, not recommended)

- `run_pair_table.py` — DIY fg_metrics.py, only 3–200 pairs from same_vs_diff_gap_probe
- `analyze_rankings.py` — ranking agreement analysis (compatible with new script output too)
- `summarize_existing_rankings.py` / `compare_existing_methods.py` — aggregate existing summaries

---

## Original description
latent space.

## Files

- `baseline_config.json`
  Pins the baseline run used for this experiment:
  `face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1`.
- `run_pair_table.py`
  Computes geometry-pipeline distances for a CSV of mesh pairs.
- `analyze_rankings.py`
  Computes ranking agreement, top-k overlap, and same-vs-different summaries.
- `summarize_existing_rankings.py`
  Collects already-produced `ranking_summary.csv` files into one table.
- `compare_existing_methods.py`
  Aggregates the collected summaries by method/scenario/topology pair.

## Pair Table Format

Minimum columns:

```text
latent_distance,mesh_a,mesh_b
```

Recommended columns:

```text
scenario,pair_type,subject_a,topology_a,sample_name_a,subject_b,topology_b,sample_name_b,gt_distance,latent_distance,mesh_a,mesh_b
```

`mesh_a` and `mesh_b` may point to `.npz` files with key `V`, or plain text
files with `N x 3` vertices. If the table only has `sample_name_a` and
`sample_name_b`, pass `--npz_root`; paths are inferred as
`<npz_root>/<sample_name>.npz`.

## Quick Commands

The baseline for the paper comparison is:

```text
/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1
```

with checkpoint:

```text
mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth
```

Compute FG metrics on a pair table:

```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/run_pair_table.py \
  --pairs_csv checking_assumptions/outputs/same_vs_diff_gap_probe_all12/clean/pair_metrics.csv \
  --npz_root datasets/REMESH/npz_data_topo_500 \
  --out_dir faceBench/latentVSpipeline/outputs/smoke_clean \
  --stages raw,rigid,fg \
  --vertex_scale 1e-6 \
  --max_pairs 200
```

Analyze ranking agreement:

```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/analyze_rankings.py \
  --pair_metrics faceBench/latentVSpipeline/outputs/smoke_clean/pair_metrics_with_fg.csv \
  --out_dir faceBench/latentVSpipeline/outputs/smoke_clean/analysis
```

Collect already-existing ranking summaries:

```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/summarize_existing_rankings.py \
  --roots \
    /deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/perturbation_ranking_vs_chamfer_topology_breakdown \
    /deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/perturbation_ranking_vs_registered_chamfer_topology_breakdown \
    /deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/perturbation_ranking_vs_nicp_correspondence_topology_breakdown \
  --out_dir faceBench/latentVSpipeline/outputs/baseline_dn_mixed_topology_v1
```

Aggregate that collected table:

```bash
.venv_twotower_robust_312/bin/python faceBench/latentVSpipeline/compare_existing_methods.py \
  --summary_csv faceBench/latentVSpipeline/outputs/baseline_dn_mixed_topology_v1/existing_ranking_summaries.csv \
  --out_dir faceBench/latentVSpipeline/outputs/baseline_dn_mixed_topology_v1
```

## Stage Definitions

- `raw`: symmetric Chamfer on the two input point clouds.
- `rigid`: sample-based rigid ICP, then symmetric Chamfer.
- `fg`: rigid ICP, correspondence by nearest neighbor, then point-to-point and
  point-to-triangle distances. When `--mm_json` is provided and compatible with
  the target vertex count, it also applies FaceBench topology-consistency
  correction before distance computation.

For REMESH `.npz` assets in this repository, use `--vertex_scale 1e-6` when you
want distances in the same meter-like scale as the existing FaceBench examples.

This intentionally exposes multiple levels instead of reporting only one number:
raw geometry, registered geometry, and the full FG-style distance can fail in
different ways.
