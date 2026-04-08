#!/usr/bin/env bash
set -euo pipefail

ROOT=/deck/datasets/WBES-FaceEmbedding
FACEVERSE_DIR="$ROOT/datasets/FaceVerse"
LOG_PATH="$FACEVERSE_DIR/mixed_xtopo_faceverse_postperturb_icp_comparison_queue.log"
PID_PATH="$FACEVERSE_DIR/mixed_xtopo_faceverse_postperturb_icp_comparison_queue.pid"
WAIT_SESSION="mixed_faceverse_postperturb_icp_queue"
PYTHON="$ROOT/.venv_twotower_robust_312/bin/python"
COMPARE_SCRIPT="$FACEVERSE_DIR/compare_faceverse_result_dirs.py"

OLD_RANKING="$FACEVERSE_DIR/faceverse_ranking_vs_gt_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_noicp/ranking_summary.csv"
NEW_RANKING="$FACEVERSE_DIR/faceverse_ranking_vs_gt_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_postperturb_icp/ranking_summary.csv"
OLD_SIGMA="$FACEVERSE_DIR/faceverse_ranking_vs_gt_sigma_sweep_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_noicp/sigma_sweep_summary.csv"
NEW_SIGMA="$FACEVERSE_DIR/faceverse_ranking_vs_gt_sigma_sweep_neutral_full_mixed_xtopo_9a81466d_best_by_xtopo_mesh_clean_postperturb_icp/sigma_sweep_summary.csv"
OUT_DIR="$FACEVERSE_DIR/faceverse_comparison_mixed_xtopo_9a81466d_noicp_vs_postperturb_icp"

mkdir -p "$OUT_DIR"
echo "$$" > "$PID_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] Waiting for FaceVerse post-perturb ICP benchmark queue to finish"
while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  echo "[$(timestamp)] Session '$WAIT_SESSION' still running; sleeping 120s"
  sleep 120
done

echo "[$(timestamp)] FaceVerse post-perturb ICP queue finished; checking outputs"
for path in "$OLD_RANKING" "$NEW_RANKING" "$OLD_SIGMA" "$NEW_SIGMA"; do
  if [[ ! -f "$path" ]]; then
    echo "[$(timestamp)] Missing required file: $path"
    exit 1
  fi
done

echo "[$(timestamp)] Running noicp-vs-postperturb-icp FaceVerse comparison"
"$PYTHON" "$COMPARE_SCRIPT" \
  --old_ranking_csv "$OLD_RANKING" \
  --new_ranking_csv "$NEW_RANKING" \
  --old_sigma_csv "$OLD_SIGMA" \
  --new_sigma_csv "$NEW_SIGMA" \
  --out_dir "$OUT_DIR" \
  --old_label "noicp" \
  --new_label "postperturb_icp"

echo "[$(timestamp)] FaceVerse noicp-vs-postperturb-icp comparison finished"
