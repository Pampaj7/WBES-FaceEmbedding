#!/usr/bin/env bash
# Giro di ablazioni v3 sulla cella E1 C3F (protocollo aau/runs/evidence/trainer_v3/ablation_protocol.md, sha256 in
# ablation_protocol.sha256): store degli operatori, poi i bracci in ordine (ctrl, area, arearobust, bal, ugtmix),
# ognuno con la sua eval. Tutto su nv-ai-04 (A100, prelazionabile, --requeue). Job in c3f_jobs.txt.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/aau/env.sh"
D="$WBES_ROOT/v3_work/trainer/ablations/c3f"
LOG="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_jobs.txt"
S=$("$AAU_DIR/submit.sh" "$D/build_store.sbatch" --parsable)
echo "$(date +%F_%T) store=$S" | tee -a "$LOG"
for arm in ${WBES_V3_ARMS:-ctrl area arearobust bal ugtmix}; do
  T=$(WBES_V3_ARM="$arm" "$AAU_DIR/submit.sh" "$D/train.sbatch" --dependency="afterok:$S" --kill-on-invalid-dep=yes \
      --job-name="wbes-v3-c3f-$arm" --parsable)
  E=$(WBES_V3_ARM="$arm" "$AAU_DIR/submit.sh" "$D/eval.sbatch" --dependency="afterok:$T" --kill-on-invalid-dep=yes \
      --job-name="wbes-v3-c3f-eval-$arm" --parsable)
  echo "$(date +%F_%T) $arm training=$T eval=$E" | tee -a "$LOG"
done
