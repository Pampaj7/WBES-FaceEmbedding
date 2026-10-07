#!/usr/bin/env bash
# Lancia il run grande e la sua catena di eval (aau/data_scale/PLAN.md). Dal frontend:
#
#   aau/data_scale/launch_scale.sh                                  # frame spento (default)
#   WBES_DS_CANON='{"bfm": {"R": [[1,0,0],[0,-1,0],[0,0,-1]], "flip_faces": true}}' aau/data_scale/launch_scale.sh
#       (Rx 180 gradi + inversione delle facce: BFM nel frame ICT, misurato in check_canon.py)
#   WBES_DS_AUG='{"rot_deg": 180, "reflect_p": 0.5}' aau/data_scale/launch_scale.sh
#
# Le variabili WBES_DS_* (vedi train_scale.sbatch) passano ai job con l'ambiente.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
export WBES_DS_CANON="${WBES_DS_CANON:-null}" WBES_DS_AUG="${WBES_DS_AUG:-null}"
TAG="$( [[ "$WBES_DS_CANON" == null ]] && echo nocanon || echo canon )_$( [[ "$WBES_DS_AUG" == null ]] && echo noaug || echo aug )"
export WBES_RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_RUNS/data_scale_runs/scale_bfm_ict_gnm_s${WBES_DS_SEED:-1234}_${TAG}_$(date +%Y%m%d_%H%M)}"
mkdir -p "$WBES_RUNS_ROOT"
TRAIN=$("$AAU_DIR/submit.sh" data_scale/train_scale.sbatch --parsable)
# la catena legge la run dir VERA scritta dal training (puo' avere un suffisso _rerunN)
CHAIN=$(WBES_DS_TRAIN_JOB="$TRAIN" "$AAU_DIR/submit.sh" data_scale/eval_chain_scale.sbatch --parsable \
        --dependency=afterok:"$TRAIN")
echo "training=$TRAIN catena_eval=$CHAIN runs_root=$WBES_RUNS_ROOT"
