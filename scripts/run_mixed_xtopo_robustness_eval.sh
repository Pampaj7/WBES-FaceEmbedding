#!/usr/bin/env bash
set -euo pipefail

ROOT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1
CKPT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth
DATA_DIR=/deck/datasets/WBES-FaceEmbedding/datasets/REMESH/npz_data_topo_500_withops
DIST_NPZ=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz
VENV=/deck/datasets/WBES-FaceEmbedding/.venv_twotower_robust_312
PYTHON=/deck/datasets/WBES-FaceEmbedding/.venv_twotower_robust_312/bin/python
RANK_SCRIPT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_rankings.py
SIGMA_SCRIPT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_rankings_sigma_sweep.py
TOPO_SCRIPT=/deck/datasets/WBES-FaceEmbedding/face_embedding/gt_encdec/remeshing/intrinsic/perturbated/compare_model_vs_chamfer_topology_breakdown.py

RANK_OUT=$ROOT/perturbation_ranking_vs_chamfer/best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_agglvl-subject_pair_mean_subjects-100_meshes-10_scenarios-clean-jitter-translation-rotation-mixed
TOPO_OUT=$ROOT/perturbation_ranking_vs_chamfer_topology_breakdown/best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_topopairs-cross_only_ordered-1_subjects-100_meshes-10_level-subject_pair_mean
SIGMA_OUT=$ROOT/perturbation_ranking_vs_chamfer_sigma_sweep/best_by_xtopo_mesh_clean_split-eval_pairs-cross_topology_agglvl-subject_pair_mean_subjects-100_meshes-10_sweep-jitter-translation-rotation-mixed_sigmas-0p00-0p02-0p05-0p10-0p15-0p20
QUEUE_LOG=$ROOT/robustness_queue.log

mkdir -p "$RANK_OUT" "$TOPO_OUT" "$SIGMA_OUT"

exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

common_args=(
  --model_path "$CKPT"
  --checkpoint_selector best_by_clean
  --data_dir "$DATA_DIR"
  --dist_npz "$DIST_NPZ"
  --subject_split eval
  --eval_fraction 0.2
  --seed 1234
  --max_subjects 500
  --max_meshes_per_subject_eval 10
  --preload_workers 8
  --pair_mode cross_topology
  --rigid_rot_deg 20
  --rigid_trans_scale 0.05
  --rigid_rot_deg_min 1
  --rigid_trans_scale_min 0.002
  --chamfer_batch_pairs 256
  --chamfer_cache_verts force
  --chamfer_cache_verts_max_mb 4096
)

echo "[$(timestamp)] Activating environment"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME
export WBES_DIFFUSION_NET_SRC=/deck/datasets/WBES-FaceEmbedding/diffusion-net/src
export PYTHONUNBUFFERED=1
"$PYTHON" -c "import torch; print(f'Activated {\"$VENV\"}'); print(f'WBES_DIFFUSION_NET_SRC={\"$WBES_DIFFUSION_NET_SRC\"}'); print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')"

echo "[$(timestamp)] Starting ranking benchmark"
"$PYTHON" "$RANK_SCRIPT" \
  "${common_args[@]}" \
  --out_dir "$RANK_OUT" \
  --aggregation_level subject_pair_mean \
  --scenarios clean,jitter,translation,rotation,mixed \
  --jitter_sigma 0.05 \
  --translation_sigma 0.05 \
  --rotation_sigma 0.05 \
  --mixed_jitter_sigma 0.05 \
  --mixed_translation_sigma 0.05 \
  --mixed_rotation_sigma 0.05 \
  2>&1 | tee "$RANK_OUT/run.log"

echo "[$(timestamp)] Starting topology breakdown"
"$PYTHON" "$TOPO_SCRIPT" \
  "${common_args[@]}" \
  --out_dir "$TOPO_OUT" \
  --topology_pair_mode cross_only \
  --ordered_topology_pairs \
  2>&1 | tee "$TOPO_OUT/run.log"

echo "[$(timestamp)] Starting sigma sweep"
"$PYTHON" "$SIGMA_SCRIPT" \
  "${common_args[@]}" \
  --out_dir "$SIGMA_OUT" \
  --aggregation_level subject_pair_mean \
  --sweep_scenarios jitter,translation,rotation,mixed \
  --sigma_values 0.00,0.02,0.05,0.10,0.15,0.20 \
  --include_clean_once \
  --progressive_output_layout \
  2>&1 | tee "$SIGMA_OUT/run.log"

echo "[$(timestamp)] Robustness pipeline finished"
