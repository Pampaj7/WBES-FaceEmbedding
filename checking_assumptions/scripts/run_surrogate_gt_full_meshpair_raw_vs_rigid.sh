#!/usr/bin/env bash
set -euo pipefail

REPO=/deck/datasets/WBES-FaceEmbedding
CKPT=$REPO/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth
DATA_DIR=$REPO/datasets/REMESH/npz_data_topo_500_withops
DIST_NPZ=$REPO/checking_assumptions/outputs/gt_surrogate_normalized_original/normalized_matrix_distances_surrogate.npz
VENV=$REPO/.venv_twotower_robust_312
PYTHON=$REPO/.venv_twotower_robust_312/bin/python
RAW_SCRIPT=$REPO/face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_topology_breakdown_scenarios.py
RIGID_SCRIPT=$REPO/face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_registered_chamfer_topology_breakdown.py

ROOT_OUT=$REPO/checking_assumptions/outputs/full_meshpair_surrogate_gt_raw_vs_rigid
RAW_OUT=$ROOT_OUT/raw_noicp_clean_translation
RIGID_OUT=$ROOT_OUT/rigid_only_clean_translation
QUEUE_LOG=$ROOT_OUT/queue.log

mkdir -p "$RAW_OUT" "$RIGID_OUT"
exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Activating environment"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
export WBES_DIFFUSION_NET_SRC=$REPO/diffusion-net/src
export PYTHONUNBUFFERED=1

echo "[$(timestamp)] Starting surrogate-GT raw meshpair benchmark"
"$PYTHON" "$RAW_SCRIPT" \
  --model_path "$CKPT" \
  --data_dir "$DATA_DIR" \
  --dist_npz "$DIST_NPZ" \
  --subject_split eval \
  --eval_fraction 0.2 \
  --seed 1234 \
  --max_subjects 500 \
  --max_meshes_per_subject_eval 10 \
  --preload_workers 8 \
  --pair_mode cross_topology \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --mesh_pair_level \
  --scenarios clean,translation \
  --translation_sigma 0.05 \
  --rigid_rot_deg 20 \
  --rigid_trans_scale 0.05 \
  --rigid_rot_deg_min 1 \
  --rigid_trans_scale_min 0.002 \
  --chamfer_batch_pairs 256 \
  --chamfer_cache_verts force \
  --chamfer_cache_verts_max_mb 4096 \
  --save_pair_timings \
  --write_per_pair_outputs \
  --out_dir "$RAW_OUT" \
  2>&1 | tee "$RAW_OUT/run.log"

echo "[$(timestamp)] Starting surrogate-GT rigid-only meshpair benchmark"
"$PYTHON" "$RIGID_SCRIPT" \
  --model_path "$CKPT" \
  --data_dir "$DATA_DIR" \
  --dist_npz "$DIST_NPZ" \
  --subject_split eval \
  --eval_fraction 0.2 \
  --seed 1234 \
  --max_subjects 500 \
  --max_meshes_per_subject_eval 10 \
  --pair_mode cross_topology \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  --mesh_pair_level \
  --scenarios clean,translation \
  --translation_sigma 0.05 \
  --registration_workers 8 \
  --icp_points 128 \
  --icp_max_correspondence_distance 0.05 \
  --icp_max_iteration 20 \
  --no-use_nonrigid_cpd \
  --warp_chunk_size 4096 \
  --chamfer_batch_pairs 8 \
  --save_pair_timings \
  --write_per_pair_outputs \
  --out_dir "$RIGID_OUT" \
  2>&1 | tee "$RIGID_OUT/run.log"

echo "[$(timestamp)] Surrogate-GT full meshpair raw-vs-rigid benchmark finished"
