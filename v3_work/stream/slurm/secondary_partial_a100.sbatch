#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-partial
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:8
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-gpu=110G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR2@600
#
# Run SECONDARIO (ablazione): la ricetta c3m del run principale (preset validated, massive_node.sh) piu' la parzialita'
# variabile per vista (partial_aug.py, producer.py --partial-p), su 8 A100 di nv-ai-04 con --qos=unprivileged:
# prelazionabile, quindi ripartibile.
#   STREAM_STEPS=<T del run principale> sbatch v3_work/stream/slurm/secondary_partial_a100.sbatch
# Ripresa:
#   * prelazione (PreemptMode REQUEUE, GraceTime 0) e scontrol requeue: SIGTERM SOLO al capo di ogni task (questo
#     script e massive_node.sh su ogni nodo; proctrack/cgroup, misurato: i figli no), SIGKILL a tutti dopo KillWait
#     (30 s). massive_node.sh inoltra il SIGTERM al trainer (STREAM_PREEMPTIBLE=1 -> WBES_PREEMPT_SAVE=1,
#     train_stream.py), che salva last.pth al passo in corso ed esce; al riavvio --resume auto riparte da li' (stessa run
#     dir: anello /tmp/<job>_stream/r0, il job id resta col requeue; codice congelato in STREAM_OUT/code,
#     massive.sbatch), anello nuovo con semi nuovi;
#   * --signal NON arriva sulla prelazione, solo 600 s prima del limite di tempo: qui (B:, solo questo script) USR2
#     fa ``scontrol requeue`` del job, cioe' lo stesso percorso della prelazione (SIGTERM, salvataggio, coda) invece
#     di un SIGKILL al limite; il run continua finche' non fa STREAM_STEPS passi;
#   * checkpoint di ripresa anche ogni STREAM_CKPT_MIN minuti (default qui 5: perdita massima se il salvataggio su
#     segnale non fa in tempo, es. durante l'eval di fine epoca).
# Differenze dal run principale (12 L40S = 6 + 6): 8 rank su un nodo, quindi batch globale 8 x 5 identita' invece di
# 12 x 5 (STREAM_IDS_PER_RANK per cambiarlo); tutto il resto e' la ricetta c3m, sovrascrivibile con le variabili di
# massive.sbatch. Parzialita': STREAM_PARTIAL (0.5), STREAM_PARTIAL_AREA (vuoto = 0.03,0.40 di partial_aug).
set -euo pipefail
if [[ -z "${STREAM_STEPS:-}" ]]; then
  echo "[secondary] ERRORE: STREAM_STEPS obbligatorio (lo stesso T del run principale), nessun default" >&2
  exit 2
fi
export STREAM_SOURCES="${STREAM_SOURCES:-validated}"
export STREAM_RECIPE="${STREAM_RECIPE:-c3m}"
export STREAM_PARTIAL="${STREAM_PARTIAL:-0.5}"
export STREAM_PREEMPTIBLE=1
export STREAM_CKPT_MIN="${STREAM_CKPT_MIN:-5}"
export STREAM_OUT="${STREAM_OUT:-/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/runs/evidence/stream/secondary_partial_${SLURM_JOB_ID:-manual}}"
mkdir -p "$STREAM_OUT"
echo "[secondary] $(date +%F_%T) job=${SLURM_JOB_ID:-} riavvio=${SLURM_RESTART_COUNT:-0} T=$STREAM_STEPS" \
     "parzialita' p=$STREAM_PARTIAL area=${STREAM_PARTIAL_AREA:-default} ckpt ogni ${STREAM_CKPT_MIN} min" \
  | tee -a "$STREAM_OUT/secondary.log"
on_usr2() {
  echo "[secondary] $(date +%F_%T) USR2 (limite di tempo fra 600 s): scontrol requeue" | tee -a "$STREAM_OUT/secondary.log"
  scontrol requeue "$SLURM_JOB_ID"
}
trap on_usr2 USR2
# in background: wait si interrompe sul segnale (un comando in primo piano lo rimanderebbe alla sua fine)
bash v3_work/stream/slurm/massive.sbatch &
PID=$!
rc=0
while :; do
  set +e
  wait "$PID"
  rc=$?
  set -e
  kill -0 "$PID" 2>/dev/null || break    # ancora vivo: wait interrotto da USR2
done
echo "[secondary] $(date +%F_%T) fine rc=$rc" | tee -a "$STREAM_OUT/secondary.log"
exit "$rc"
