#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-c3f-fact-k
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --nodelist=nv-ai-04
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=130G
#SBATCH --time=12:00:00
#
# Ablazione k_eig 64 contro 128 sul sottoinsieme C3F (testa ctrl, ricetta adottata = robal: arearobust + bal), 21.096
# passi, lo store C3F di nv-ai-04 (k 128) e, per k 64, i primi 64 autovettori (v3_work/stream/ktrunc.py, verificato
# in aau/runs/evidence/stream/ktrunc_check.json). Il corpo e' il braccio robal di v3_work/trainer/ablations/c3f/
# train_body.sh, copiato all'avvio e cambiato in due punti: la run dir (c3f_runs/robal_k<K>) e il lancio dentro
# ktrunc.py (con WBES_K_TRUNC = K; per K 128 nessuna troncatura, stesso percorso).
#   WBES_K=64|128 [WBES_V3_SEED=2345] sbatch v3_work/stream/ablation_k/train_k.sbatch     (launch_k.sh)
# Il nome comincia con wbes-v3-c3f-fact: il custode dello store (keep_store_fact.sbatch) resta vivo finche' e' in coda.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
K="${WBES_K:?WBES_K=64|128}"
B="/tmp/v3c3f_trainK_body_${SLURM_JOB_ID:-manual}_${SLURM_RESTART_COUNT:-0}.sh"
sed -e 's|^RUNS_ROOT="\$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/\$ARM"$|RUNS_ROOT="$AAU_RUNS/evidence/trainer_v3/ablations/c3f_runs/${ARM}_k'"$K"'"|' \
    -e 's|"\$AAU_DIR/run.sh" "\${CMD\[@\]}"|"$AAU_DIR/run.sh" v3_work/stream/ktrunc.py "${CMD[@]}"|' \
    -e 's|^\[\[ "\$0" == /tmp/v3c3f_train_body_\* \]\] && rm -f "\$0"$|rm -f "$0"|' \
    "$WBES_ROOT/v3_work/trainer/ablations/c3f/train_body.sh" > "$B"
grep -q "c3f_runs/\${ARM}_k$K\"" "$B" && grep -q 'run.sh" v3_work/stream/ktrunc.py "${CMD' "$B" \
  || { echo "ERRORE: train_body.sh non adattato (cambiato?)" >&2; exit 1; }
[[ "$K" == 128 ]] && export WBES_K_TRUNC=0 || export WBES_K_TRUNC="$K"
export WBES_V3_ARM=robal
exec bash "$B"
