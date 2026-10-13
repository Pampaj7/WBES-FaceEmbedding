#!/usr/bin/env bash
# Lancio del modello da rilasciare (PLAN_MASSIVE sez. 23) con massive.sbatch: UNA riga di lancio per smoke e run.
#   aau/massive/launch.sh smoke <gpu> <n> [passi]      prova breve su 1 nodo (es. smoke l40s 2 600; a10 per il codice)
#   aau/massive/launch.sh main  <nodi> <gpu per nodo> <passi> <time>
#   continuazione dopo un TIMEOUT: STREAM_OUT=aau/runs/massive_release/main aau/massive/launch.sh main <stessi argomenti>
# Ricetta (dichiarata in PLAN_MASSIVE sez. 23): testa factorized, GT-SR + log S (E12), k_eig 128, lr 1e-4 costante con
# decadimento a 1e-5 al 90% dei passi, EMA 0.999, 5 identita' x 6 viste per rank, batch a dominio singolo con passi
# uguali fra gli 8 domini, quota d'espressioni 0.5, 6 discretizzazioni senza reinserimento (crop compreso), nessuna
# rotazione a ogni uso, niente mm_aug. Fonti: preset release (sources.SOURCE_PRESETS), FaMoS senza le persone held-out.
# MAI --qos=unprivileged: QoS normal, L40S, al massimo 8 GPU per questo run.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:?smoke | main}"
export STREAM_SOURCES=release STREAM_RELEASE=1 STREAM_ARM=factorized STREAM_RECIPE=c3m STREAM_EXPR_FRAC=0.5
export STREAM_ROT=0 STREAM_LABEL_DRAW=perm STREAM_K=128 STREAM_SEED=1234
export STREAM_HELDOUT="${STREAM_HELDOUT:-$PWD/datasets/MASSIVE_HELDOUT}"
[[ -f "$STREAM_HELDOUT/heldout.json" ]] || { echo "manca $STREAM_HELDOUT/heldout.json (aau/massive/heldout.sbatch)" >&2; exit 2; }
COMMON=(-p prioritized --ntasks-per-node=1 --cpus-per-gpu=16 --requeue)
case "$MODE" in
  smoke)
    GPU="${2:?gpu}" N="${3:?n}" T="${4:-600}"
    export STREAM_STEPS="$T" STREAM_SPE=100 STREAM_SAVE_EVERY=1 STREAM_EVAL_EVERY=3 STREAM_CKPT_MIN=5
    export STREAM_LR_STEPS="$(( T * 9 / 10 )):1e-5"
    export STREAM_OUT="${STREAM_OUT:-$PWD/aau/runs/massive_release/smoke_$(date +%m%d_%H%M)}"
    [[ "$GPU" == l40s ]] || export STREAM_TRAIN_CPUS_PER_RANK=4      # prova del codice su GPU piccole
    sbatch -J wbes-massive-rel-smoke "${COMMON[@]}" --nodes=1 --gpus-per-node="$GPU:$N" --mem-per-gpu=40G \
      --time=02:00:00 v3_work/stream/slurm/massive.sbatch ;;
  main)
    NN="${2:?nodi}" G="${3:?gpu per nodo}" T="${4:?passi}" TIME="${5:?time}"
    (( NN * G <= 8 )) || { echo "al massimo 8 GPU per il massivo" >&2; exit 2; }
    export STREAM_STEPS="$T" STREAM_SPE=1000 STREAM_SAVE_EVERY=8 STREAM_CKPT_MIN=20
    export STREAM_LR_STEPS="$(( T * 9 / 10 )):1e-5"
    export STREAM_OUT="${STREAM_OUT:-$PWD/aau/runs/massive_release/main}"
    sbatch -J wbes-massive-release "${COMMON[@]}" --nodes="$NN" --gpus-per-node="l40s:$G" --mem-per-gpu=40G \
      --time="$TIME" v3_work/stream/slurm/massive.sbatch ;;
  *) echo "modo $MODE (smoke | main)" >&2; exit 2 ;;
esac
echo "STREAM_OUT=$STREAM_OUT"
