#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_SESSION=${1:-mixed_robustness_queue}
QUEUE_LOG=/deck/datasets/WBES-FaceEmbedding/datasets/FaceVerse/mixed_xtopo_faceverse_queue.log
LAUNCHER=/deck/datasets/WBES-FaceEmbedding/scripts/run_mixed_xtopo_faceverse_eval.sh

exec > >(tee -a "$QUEUE_LOG") 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] FaceVerse queue armed; waiting for session '$WAIT_FOR_SESSION'"
while tmux has-session -t "$WAIT_FOR_SESSION" 2>/dev/null; do
  echo "[$(timestamp)] Session '$WAIT_FOR_SESSION' still running; sleeping 120s"
  sleep 120
done

echo "[$(timestamp)] Wait condition satisfied; launching FaceVerse eval"
exec /bin/bash "$LAUNCHER"
