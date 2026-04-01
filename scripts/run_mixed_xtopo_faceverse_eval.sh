#!/usr/bin/env bash
set -euo pipefail

ROOT=/deck/datasets/WBES-FaceEmbedding/datasets/FaceVerse
CKPT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth
VENV=/deck/datasets/WBES-FaceEmbedding/.venv_twotower_robust_312
PYTHON=/deck/datasets/WBES-FaceEmbedding/.venv_twotower_robust_312/bin/python
RANK_SCRIPT=/deck/datasets/WBES-FaceEmbedding/datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse.py
SIGMA_SCRIPT=/deck/datasets/WBES-FaceEmbedding/datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse_sigma_sweep.py

RANK_OUT=$ROOT/faceverse_ranking_vs_gt_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_noicp
SIGMA_OUT=$ROOT/faceverse_ranking_vs_gt_sigma_sweep_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_noicp
QUEUE_LOG=$ROOT/mixed_xtopo_faceverse_queue.log

mkdir -p "$RANK_OUT" "$SIGMA_OUT"

exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

common_args=(
  --model_path "$CKPT"
  --pose_ids 01
  --pattern '*_01.npz'
  --gt_pattern '*_01.ply'
  --subject_split all
  --max_subjects 0
  --max_meshes_per_subject_eval 1
  --preload_workers 8
  --pair_mode within_topology
  --aggregation_level subject_pair_mean
  --rigid_rot_deg 20
  --rigid_trans_scale 0.05
  --rigid_rot_deg_min 1
  --rigid_trans_scale_min 0.002
  --chamfer_batch_pairs 256
  --chamfer_cache_verts force
  --chamfer_cache_verts_max_mb 4096
  --no-chamfer_use_icp
)

echo "[$(timestamp)] Activating environment"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
export WBES_DIFFUSION_NET_SRC=/deck/datasets/WBES-FaceEmbedding/diffusion-net/src
export PYTHONUNBUFFERED=1
"$PYTHON" -c "import torch; print(f'Activated {\"$VENV\"}'); print(f'WBES_DIFFUSION_NET_SRC={\"$WBES_DIFFUSION_NET_SRC\"}'); print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

echo "[$(timestamp)] Starting FaceVerse ranking benchmark"
"$PYTHON" "$RANK_SCRIPT" \
  "${common_args[@]}" \
  --out_dir "$RANK_OUT" \
  --scenarios clean,jitter,translation,rotation,mixed \
  --jitter_sigma 0.05 \
  --translation_sigma 0.05 \
  --rotation_sigma 0.05 \
  --mixed_jitter_sigma 0.05 \
  --mixed_translation_sigma 0.05 \
  --mixed_rotation_sigma 0.05 \
  2>&1 | tee "$RANK_OUT/run.log"

echo "[$(timestamp)] Starting FaceVerse sigma sweep"
"$PYTHON" "$SIGMA_SCRIPT" \
  "${common_args[@]}" \
  --out_dir "$SIGMA_OUT" \
  --sweep_scenarios jitter,translation,rotation,mixed \
  --sigma_values 0.00,0.02,0.05,0.10,0.15,0.20 \
  --include_clean_once \
  --progressive_output_layout \
  2>&1 | tee "$SIGMA_OUT/run.log"

echo "[$(timestamp)] FaceVerse pipeline finished"
