#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-eval-k
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=10:00:00
#
# Eval dei bracci robal_k<K> (train_k.sbatch) con v3_work/trainer/ablations/c3f/eval_body.sh copiato all'avvio e
# ogni lancio di eval_v3.py fatto dentro ktrunc.py (WBES_K_TRUNC = K: gli operatori delle viste di eval troncati
# come al training). Stesse misure dei bracci C3F (hifi, devfs, fv, now).
#   WBES_K=64|128 [WBES_V3_SEED=2345] sbatch --dependency=afterok:<training> v3_work/stream/ablation_k/eval_k.sbatch
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
K="${WBES_K:?WBES_K=64|128}"
SEED="${WBES_V3_SEED:-1234}"
SFX=""; [[ "$SEED" != 1234 ]] && SFX="_s$SEED"
B="/tmp/v3c3f_evalK_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed 's|v3_work/trainer/eval_v3.py --|v3_work/stream/ktrunc.py v3_work/trainer/eval_v3.py --|g' \
    "$WBES_ROOT/v3_work/trainer/ablations/c3f/eval_body.sh" > "$B"
[[ $(grep -c "ktrunc.py v3_work/trainer/eval_v3.py --" "$B") -ge 3 ]] || { echo "ERRORE: eval_body.sh non adattato" >&2; exit 1; }
[[ "$K" == 128 ]] && export WBES_K_TRUNC=0 || export WBES_K_TRUNC="$K"
export WBES_V3_ARM="robal_k$K"
export WBES_V3_EVAL_RR="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/robal_k$K$SFX"
exec bash "$B"
