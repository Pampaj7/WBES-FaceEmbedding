#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv_twotower_robust_312/bin/python}"

SHOT="${SHOT:-10}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_by_xtopo_mesh_clean}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
AGGREGATION_LEVEL="${AGGREGATION_LEVEL:-subject_pair_mean}"

FINE_ROOT="${REPO_ROOT}/datasets/FaceVerse/FINE_tuning"
RUN_DIR="${RUN_DIR:-${FINE_ROOT}/runs/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_warmstart_seed1234__b19558f8}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/checkpoints/${CHECKPOINT_KIND}.pth}"
SUBJECT_IDS_TXT="${SUBJECT_IDS_TXT:-${FINE_ROOT}/shot$(printf "%02d" "${SHOT}")_cross_topology_with_ops_heldout_subject_ids.txt}"
OUT_DIR="${OUT_DIR:-${FINE_ROOT}/evals/shot$(printf "%02d" "${SHOT}")_heldout_xtopo10k_${CHECKPOINT_KIND}_postperturb_icp}"
LOG_DIR="${LOG_DIR:-${FINE_ROOT}/logs}"
LOG_PATH="${LOG_DIR}/faceverse_shot$(printf "%02d" "${SHOT}")_heldout_xtopo_sweep_${CHECKPOINT_KIND}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model checkpoint not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${SUBJECT_IDS_TXT}" ]]; then
  echo "Held-out subject ids file not found: ${SUBJECT_IDS_TXT}" >&2
  exit 1
fi

mapfile -t SUBJECT_IDS < "${SUBJECT_IDS_TXT}"
if [[ "${#SUBJECT_IDS[@]}" -lt 3 ]]; then
  echo "Need at least 3 held-out subjects, got ${#SUBJECT_IDS[@]}" >&2
  exit 1
fi

echo "Model: ${MODEL_PATH}"
echo "Held-out subjects: ${#SUBJECT_IDS[@]} (${SUBJECT_IDS_TXT})"
echo "Output: ${OUT_DIR}"
echo "Logging to ${LOG_PATH}"

cd "${REPO_ROOT}"
"${PYTHON}" datasets/FaceVerse/compare_model_vs_chamfer_rankings_faceverse_sigma_sweep.py \
  --model_path "${MODEL_PATH}" \
  --config_json "${RUN_DIR}/config.json" \
  --out_dir "${OUT_DIR}" \
  --device cuda \
  --data_dir "${REPO_ROOT}/datasets/FaceVerse/cross_topology_10k_with_ops" \
  --pattern "*_01*.npz" \
  --gt_mesh_dir "${REPO_ROOT}/datasets/FaceVerse/extracted/detail" \
  --gt_pattern "*_01.ply" \
  --dist_npz "${REPO_ROOT}/datasets/FaceVerse/gt_distance_matrix/faceverse_detail_pose01_vertex_mean_l2_normalized.npz" \
  --pose_ids 01 \
  --subject_ids "${SUBJECT_IDS[@]}" \
  --subject_split all \
  --max_subjects 0 \
  --max_meshes_per_subject_eval 2 \
  --preload_eval_samples \
  --preload_workers 4 \
  --pair_mode cross_topology \
  --aggregation_level "${AGGREGATION_LEVEL}" \
  --sweep_scenarios jitter,translation,rotation,mixed \
  --sigma_values 0.00,0.02,0.05,0.10,0.15,0.20 \
  --include_clean_once \
  --chamfer_use_icp \
  --icp_alignment_stage perturbed_pairs \
  --icp_points 2048 \
  --icp_max_correspondence_distance 0.05 \
  --icp_max_iteration 20 \
  --icp_workers 8 \
  --chamfer_batch_pairs 256 \
  --chamfer_pair_progress \
  --chamfer_cache_verts force 2>&1 | tee "${LOG_PATH}"
