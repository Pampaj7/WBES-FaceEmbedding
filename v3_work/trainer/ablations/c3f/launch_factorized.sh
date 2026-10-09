#!/usr/bin/env bash
# Bracci C3F della modalita' fattorizzata (aau/runs/evidence/trainer_v3/factorized.md), PREPARATI E NON LANCIATI:
# serve prima un emendamento del protocollo (confronto factorized contro robal, regola) con lo sha256 registrato.
# Lo store di /tmp/wbes_v3_store_c3f (job 1062011) non esiste piu' (l'epilogo di nv-ai-04 ha pulito /tmp quando sono
# finiti i custodi 1062017 e 1062076): si ricostruisce con la stessa ricetta (build_store.sbatch, ~2.6 h su 224 CPU);
# entrambi i bracci leggono lo STESSO store nuovo, quindi factorized e robal hanno gli stessi operatori.
#   1. dati del braccio (build_factorized_data.sbatch, CPU) e store, in parallelo;
#   2. custode dello store (keep_store.sbatch) sui due bracci;
#   3. bracci (WBES_V3_FACT_ARMS, default "robal factorized") con la loro eval (eval_body.sh: ramo factorized).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/aau/env.sh"
D="$WBES_ROOT/v3_work/trainer/ablations/c3f"
LOG="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_jobs.txt"
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
DEP=""
if [[ ! -f "$C/scale_table.npz" || ! -f "$C/factorized/gt_shape.npz" ]]; then
  F=$("$AAU_DIR/submit.sh" "$D/build_factorized_data.sbatch" --parsable)
  DEP=":$F"
  echo "$(date +%F_%T) dati factorized=$F" | tee -a "$LOG"
fi
S=$("$AAU_DIR/submit.sh" "$D/build_store.sbatch" --parsable)
echo "$(date +%F_%T) store=$S (ricostruito, stessa ricetta di 1062011)" | tee -a "$LOG"
TRAIN=()
for arm in ${WBES_V3_FACT_ARMS:-robal factorized}; do
  T=$(WBES_V3_ARM="$arm" "$AAU_DIR/submit.sh" "$D/train.sbatch" --dependency="afterok:$S$DEP" --kill-on-invalid-dep=yes \
      --job-name="wbes-v3-c3f-$arm" --parsable)
  E=$(WBES_V3_ARM="$arm" "$AAU_DIR/submit.sh" "$D/eval.sbatch" --dependency="afterok:$T" --kill-on-invalid-dep=yes \
      --job-name="wbes-v3-c3f-eval-$arm" --parsable)
  TRAIN+=("$T")
  echo "$(date +%F_%T) $arm training=$T eval=$E" | tee -a "$LOG"
done
K=$(WBES_V3_KEEP_JOBS="$(IFS=,; echo "$S,${TRAIN[*]}")" "$AAU_DIR/submit.sh" "$D/keep_store.sbatch" --parsable)
echo "$(date +%F_%T) custode dello store=$K (attende $S ${TRAIN[*]})" | tee -a "$LOG"
