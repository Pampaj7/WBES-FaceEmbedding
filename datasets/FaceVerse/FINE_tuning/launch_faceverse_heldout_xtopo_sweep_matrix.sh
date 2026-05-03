#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FINE_ROOT="${REPO_ROOT}/datasets/FaceVerse/FINE_tuning"

SHOTS="${SHOTS:-6 10 20 50 100}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-best_by_xtopo_mesh_clean}"

for shot in ${SHOTS}; do
  shot_label="$(printf "shot%02d" "${shot}")"
  mapfile -t run_dirs < <(
    find "${FINE_ROOT}/runs_matrix/${shot_label}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort
  )
  if [[ "${#run_dirs[@]}" -ne 1 ]]; then
    echo "Expected exactly one run dir for ${shot_label}, found ${#run_dirs[@]}" >&2
    printf '%s\n' "${run_dirs[@]:-}" >&2
    exit 1
  fi

  run_dir="${run_dirs[0]}"
  model_path="${run_dir}/checkpoints/${CHECKPOINT_KIND}.pth"
  out_dir="${FINE_ROOT}/evals_matrix/${shot_label}_heldout_xtopo10k_${CHECKPOINT_KIND}_postperturb_icp"

  echo
  echo "=== FaceVerse held-out sweep matrix: ${shot_label}, ${CHECKPOINT_KIND} ==="
  SHOT="${shot}" RUN_DIR="${run_dir}" MODEL_PATH="${model_path}" OUT_DIR="${out_dir}" \
    "${FINE_ROOT}/launch_faceverse_heldout_xtopo_sweep.sh"
done

echo
echo "Held-out sweep matrix complete."
