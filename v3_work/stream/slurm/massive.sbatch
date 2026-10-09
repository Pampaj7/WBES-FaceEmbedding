#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-massive
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=prioritized
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem-per-gpu=80G
#SBATCH --time=02:00:00
#SBATCH --requeue
#
# Run a streaming (P1) con la GT di E12, su uno o piu' nodi: per OGNI nodo produttori + anello in RAM + i rank DDP
# del nodo (torchrun con rendezvous c10d fra i nodi), corpo in massive_node.sh (uno srun per nodo).
#   prova, 1 GPU:     sbatch v3_work/stream/slurm/massive.sbatch                    (STREAM_STEPS=2000 ...)
#   run principale, 12 L40S = 6 + 6 (STESSO numero di GPU per nodo):
#     sbatch --nodes=2 --gpus-per-node=l40s:6 --time=2-00:00:00 v3_work/stream/slurm/massive.sbatch
#   NIENTE 8 + 4: con un numero di GPU diverso fra i nodi la prima all_reduce di NCCL non finisce mai (misurato il 9
#   ottobre con 2 + 1 GPU: RoCE, socket, Ring, Tree, PXN/P2P/SHM spenti, NCCL 2.22 e quello di pytorch_25.09; 1 + 1,
#   1 + 1 + 1 e 2 + 2 funzionano). Un job eterogeneo (componenti separati da ':') parte solo con STREAM_ALLOW_HETERO=1
#   e GPU uguali per componente (es. 2 A100 di nv-ai-04 + 2 L40S, provato).
#   A100 (secondari): sbatch -p aicentre-a100 --qos=unprivileged --gpus-per-node=a100:8 --cpus-per-gpu=28 \
#            --mem-per-gpu=110G --time=2-00:00:00 v3_work/stream/slurm/massive.sbatch
# Ripresa: --requeue + --resume auto del trainer (la run dir non dipende dal job ne' dai riavvii); l'anello si
# riempie da capo, i produttori ripartono con semi nuovi (SEME + 1000 x nodo + 100000 x riavvio, scritti in ogni gruppo).
# Variabili (default = la prova breve):
#   STREAM_ARM      factorized | factorized2 | ctrlfr (testa standard su GT-FR) | unified (la GT unificata di prima)
#   STREAM_SOURCES  massive (bfm2019, ict, gnm, flame2023, famos) | open_core (gnm, ict, flame2023) | lista
#   STREAM_STEPS, STREAM_SPE (passi per epoca), STREAM_K (64|128), STREAM_REUSE, STREAM_EXPR_FRAC (0.5)
#   STREAM_MM_AUG   probabilita' dei moltiplicatori (default hybrid=0.3,expr_transfer=0.15,rbf=0.15; vuoto = puri)
#   STREAM_RING_GB_PER_GPU (10), STREAM_TRAIN_CPUS_PER_RANK (4), STREAM_EXTRA (anelli condivisi, es. CephFS)
#   STREAM_OUT (aau/runs/evidence/stream/massive_<job>), STREAM_SEED (1234), STREAM_TEST_CRASH (passo, solo test)
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export STREAM_OUT="${STREAM_OUT:-$AAU_RUNS/evidence/stream/massive_${SLURM_JOB_ID:-manual}}"
mkdir -p "$STREAM_OUT"
# nodi: quelli del job o, in un job eterogeneo, dei suoi componenti in ordine (massive_node.sh ne ricava indice e numero)
if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
  NL=""; for i in $(seq 0 $(( SLURM_HET_SIZE - 1 ))); do v="SLURM_JOB_NODELIST_HET_GROUP_$i"; NL+="${NL:+,}${!v}"; done
  SRUN=(--het-group="$(seq -s, 0 $(( SLURM_HET_SIZE - 1 )))")
else
  NL="$SLURM_JOB_NODELIST"; SRUN=(--ntasks-per-node=1)
fi
export STREAM_NODES=$(scontrol show hostnames "$NL" | awk '!s[$0]++' | xargs)
export STREAM_MASTER=${STREAM_NODES%% *}
if [[ -n "${SLURM_HET_SIZE:-}" && $(wc -w <<< "$STREAM_NODES") -ne $SLURM_HET_SIZE ]]; then
  echo "[massive] ERRORE: due componenti sullo stesso nodo ($NL): usare --nodelist o --exclude per componente" >&2
  exit 2
fi
if [[ -n "${SLURM_HET_SIZE:-}" && "${STREAM_ALLOW_HETERO:-0}" != 1 ]]; then
  echo "[massive] ERRORE: job eterogeneo: NCCL si blocca se le GPU per nodo sono diverse (vedi sopra);" \
       "STREAM_ALLOW_HETERO=1 solo con lo stesso numero di GPU in ogni componente" >&2
  exit 2
fi
export STREAM_PORT=$(( 29400 + ${SLURM_JOB_ID:-0} % 500 ))
echo "[massive] $(date +%F_%T) job=${SLURM_JOB_ID:-} riavvio=${SLURM_RESTART_COUNT:-0} nodi=$STREAM_NODES" \
     "rendezvous=$STREAM_MASTER:$STREAM_PORT arm=${STREAM_ARM:-factorized} fonti=${STREAM_SOURCES:-massive}" \
  | tee -a "$STREAM_OUT/massive.log"
# un task per nodo, con tutte le CPU e le GPU del nodo nel job
srun "${SRUN[@]}" --kill-on-bad-exit=1 bash v3_work/stream/slurm/massive_node.sh
echo "[massive] $(date +%F_%T) fine" | tee -a "$STREAM_OUT/massive.log"
