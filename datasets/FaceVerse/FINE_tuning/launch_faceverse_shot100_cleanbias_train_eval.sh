#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FINE_ROOT="${REPO_ROOT}/datasets/FaceVerse/FINE_tuning"

SHOT="${SHOT:-100}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-5}"
P_NOISE="${P_NOISE:-0.3}"
LAMBDA_ID="${LAMBDA_ID:-0.5}"
LAMBDA_RANK="${LAMBDA_RANK:-0.5}"
RUNS_ROOT="${RUNS_ROOT:-${FINE_ROOT}/runs_cleanbias/shot$(printf "%02d" "${SHOT}")_pnoise${P_NOISE}_id${LAMBDA_ID}}"

SHOT="${SHOT}" EPOCHS="${EPOCHS}" LR="${LR}" P_NOISE="${P_NOISE}" \
  LAMBDA_ID="${LAMBDA_ID}" LAMBDA_RANK="${LAMBDA_RANK}" RUNS_ROOT="${RUNS_ROOT}" \
  "${FINE_ROOT}/launch_faceverse_finetune.sh"

mapfile -t run_dirs < <(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
if [[ "${#run_dirs[@]}" -ne 1 ]]; then
  echo "Expected exactly one run dir in ${RUNS_ROOT}, found ${#run_dirs[@]}" >&2
  printf '%s\n' "${run_dirs[@]:-}" >&2
  exit 1
fi

run_dir="${run_dirs[0]}"
for checkpoint_kind in best_by_clean best_by_xtopo_mesh_clean; do
  out_dir="${FINE_ROOT}/evals_cleanbias/shot$(printf "%02d" "${SHOT}")_heldout_xtopo10k_pnoise${P_NOISE}_id${LAMBDA_ID}_${checkpoint_kind}_postperturb_icp"
  echo
  echo "=== Clean-bias held-out sweep: shot=${SHOT}, ${checkpoint_kind} ==="
  SHOT="${SHOT}" CHECKPOINT_KIND="${checkpoint_kind}" RUN_DIR="${run_dir}" \
    MODEL_PATH="${run_dir}/checkpoints/${checkpoint_kind}.pth" OUT_DIR="${out_dir}" \
    "${FINE_ROOT}/launch_faceverse_heldout_xtopo_sweep.sh"
done
