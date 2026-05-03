#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv_twotower_robust_312/bin/python}"

SHOT="${SHOT:-10}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-5}"
P_NOISE="${P_NOISE:-0.6}"
LAMBDA_ID="${LAMBDA_ID:-0.25}"
LAMBDA_RANK="${LAMBDA_RANK:-0.5}"
SIGMA_MIN="${SIGMA_MIN:-5e-4}"
SIGMA_MAX="${SIGMA_MAX:-2e-2}"
NOISE_MODES="${NOISE_MODES:-translation,rotation,jitter}"
NOISE_MODE_WEIGHTS="${NOISE_MODE_WEIGHTS:-translation=4,rotation=2,jitter=1}"
RIGID_ROT_DEG="${RIGID_ROT_DEG:-12.0}"
RIGID_ROT_DEG_MIN="${RIGID_ROT_DEG_MIN:-0.5}"
RIGID_TRANS_SCALE="${RIGID_TRANS_SCALE:-0.03}"
RIGID_TRANS_SCALE_MIN="${RIGID_TRANS_SCALE_MIN:-0.001}"
OUTLIER_FRAC="${OUTLIER_FRAC:-0.02}"
OUTLIER_SCALE="${OUTLIER_SCALE:-6.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

FINE_ROOT="${REPO_ROOT}/datasets/FaceVerse/FINE_tuning"
DATA_DIR="${DATA_DIR:-${FINE_ROOT}/shot$(printf "%02d" "${SHOT}")_cross_topology_with_ops}"
DIST_NPZ="${DIST_NPZ:-${FINE_ROOT}/faceverse_id_distance_matrix.npz}"
RUNS_ROOT="${RUNS_ROOT:-${FINE_ROOT}/runs}"
LOG_DIR="${LOG_DIR:-${FINE_ROOT}/logs}"
INIT_CKPT="${INIT_CKPT:-${REPO_ROOT}/face_embedding/gt_encdec/remeshing/intrinsic/newdata/dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/checkpoints/best_by_xtopo_mesh_clean.pth}"

mkdir -p "${RUNS_ROOT}" "${LOG_DIR}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Fine-tuning data dir not found: ${DATA_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DIST_NPZ}" ]]; then
  echo "Distance matrix not found: ${DIST_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Init checkpoint not found: ${INIT_CKPT}" >&2
  exit 1
fi

LOG_PATH="${LOG_DIR}/faceverse_shot$(printf "%02d" "${SHOT}")_finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_PATH}"

cd "${REPO_ROOT}"
"${PYTHON}" face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py \
  --data_dir "${DATA_DIR}" \
  --dist_npz "${DIST_NPZ}" \
  --runs_root "${RUNS_ROOT}" \
  --init_checkpoint "${INIT_CKPT}" \
  --device cuda \
  --model xyz_dn \
  --epochs "${EPOCHS}" \
  --batch_subjects 5 \
  --latent_dim 256 \
  --width 128 \
  --n_blocks 4 \
  --dropout 0.1 \
  --pool_mode meanmax \
  --k_spec 0 \
  --lr "${LR}" \
  --weight_decay 1e-6 \
  --max_meshes_per_subject_train 2 \
  --max_meshes_per_subject_eval 2 \
  --max_subjects_eval_train 0 \
  --preload_eval_samples_train \
  --preload_eval_workers_train 2 \
  --train_level mixed \
  --train_pair_mode cross_topology \
  --lambda_subject 1.0 \
  --lambda_mesh 1.0 \
  --p_noise "${P_NOISE}" \
  --sigma_min "${SIGMA_MIN}" \
  --sigma_max "${SIGMA_MAX}" \
  --noise_modes "${NOISE_MODES}" \
  --noise_mode_weights "${NOISE_MODE_WEIGHTS}" \
  --outlier_frac "${OUTLIER_FRAC}" \
  --outlier_scale "${OUTLIER_SCALE}" \
  --rigid_rot_deg "${RIGID_ROT_DEG}" \
  --rigid_rot_deg_min "${RIGID_ROT_DEG_MIN}" \
  --rigid_trans_scale "${RIGID_TRANS_SCALE}" \
  --rigid_trans_scale_min "${RIGID_TRANS_SCALE_MIN}" \
  --use_id_loss \
  --lambda_id "${LAMBDA_ID}" \
  --lambda_rank "${LAMBDA_RANK}" \
  --rank_pairs 1024 \
  --rank_margin 0.05 \
  --rank_tau 0.02 \
  --rank_hard_frac 0.7 \
  --sigma_min_eval 1e-3 \
  --sigma_max_eval 1e-1 \
  --n_sigma_eval 6 \
  --eval_mode average \
  --save_every 5 \
  --eval_every 2 2>&1 | tee "${LOG_PATH}"
