#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FINE_ROOT="${REPO_ROOT}/datasets/FaceVerse/FINE_tuning"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION-faceverse_ft_shot10_eval}"
SHOTS="${SHOTS:-6 10 20 50 100}"
LR="${LR:-1e-5}"

# With two FaceVerse topologies per subject and batch_subjects=5, steps/epoch are
# roughly shot/5. These epoch counts keep total optimization steps in the same
# ballpark while still letting the larger-shot runs see more subject diversity.
epochs_for_shot() {
  case "$1" in
    6) echo "${EPOCHS_SHOT06:-200}" ;;
    10) echo "${EPOCHS_SHOT10:-200}" ;;
    20) echo "${EPOCHS_SHOT20:-150}" ;;
    50) echo "${EPOCHS_SHOT50:-100}" ;;
    100) echo "${EPOCHS_SHOT100:-100}" ;;
    *) echo "${EPOCHS_DEFAULT:-100}" ;;
  esac
}

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "Waiting for tmux session '${WAIT_FOR_SESSION}' to finish before training matrix..."
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

for shot in ${SHOTS}; do
  epochs="$(epochs_for_shot "${shot}")"
  shot_label="$(printf "shot%02d" "${shot}")"
  echo
  echo "=== FaceVerse fine-tune matrix: shot=${shot}, epochs=${epochs}, lr=${LR} ==="
  SHOT="${shot}" EPOCHS="${epochs}" LR="${LR}" RUNS_ROOT="${FINE_ROOT}/runs_matrix/${shot_label}" \
    "${FINE_ROOT}/launch_faceverse_finetune.sh"
done

echo
echo "Fine-tune matrix complete."
