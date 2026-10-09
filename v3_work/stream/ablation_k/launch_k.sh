#!/usr/bin/env bash
# Ablazione k_eig 64 contro 128 (C3F, testa ctrl con la ricetta adottata robal), PREPARATA E NON LANCIATA.
# Da lanciare SOLO quando i 6 bracci C3F in corso sulle A100 sono finiti (controllare con gpufree), o con la
# dipendenza qui sotto (WBES_K_AFTER, i loro job id). Due training + due eval per seme (WBES_K_SEEDS, default 1234).
#   v3_work/stream/ablation_k/launch_k.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/aau/env.sh"
D="$WBES_ROOT/v3_work/stream/ablation_k"
AFTER="${WBES_K_AFTER:-1062768:1062770:1062772:1062774:1062987:1062989}"
LOG="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_jobs.txt"
[[ -e "$AAU_RUNS/evidence/trainer_v3/factorized/HOLD_STORE" ]] \
  || echo "AVVISO: HOLD_STORE assente: lo store C3F resta solo finche' ci sono job wbes-v3-c3f-fact* in coda"
for seed in ${WBES_K_SEEDS:-1234}; do
  for k in 128 64; do
    sfx=""; [[ "$seed" != 1234 ]] && sfx="_s$seed"
    T=$(WBES_K=$k WBES_V3_SEED=$seed "$AAU_DIR/submit.sh" "$D/train_k.sbatch" --dependency="afterany:$AFTER" \
        --job-name="wbes-v3-c3f-fact-k$k$sfx" --parsable)
    E=$(WBES_K=$k WBES_V3_SEED=$seed "$AAU_DIR/submit.sh" "$D/eval_k.sbatch" --dependency="afterok:$T" \
        --kill-on-invalid-dep=yes --job-name="wbes-v3-c3f-eval-k$k$sfx" --parsable)
    echo "$(date +%F_%T) ablazione k: k=$k seme=$seed training=$T eval=$E (dopo $AFTER)" | tee -a "$LOG"
  done
done
