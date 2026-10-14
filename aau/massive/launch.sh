#!/usr/bin/env bash
# Lancio del modello da rilasciare (PLAN_MASSIVE sez. 23) con massive.sbatch: UNA riga di lancio per smoke e run.
#   aau/massive/launch.sh smoke <gpu> <n> [passi]      prova breve su 1 nodo (es. smoke l40s 2 600; a10 per il codice)
#   aau/massive/launch.sh main  <nodi> <gpu per nodo> <passi> <time>
#   aau/massive/launch.sh a100  <gpu> <passi> <time>    UN nodo nv-ai-04, -p aicentre-a100 --qos=unprivileged
#       (prelazionabile: --requeue + ripresa da last.pth; autorizzato dall'utente il 9 ottobre per run lunghi secondari
#       ripartibili; il modello da rilasciare non e' il run del paper) -> aau/runs/massive_release/main_a100
#   aau/massive/launch.sh a10   <gpu> <passi> <time>    UN nodo A10, prioritized, QoS normal
#   continuazione dopo un TIMEOUT: STREAM_OUT=<la stessa directory> aau/massive/launch.sh <stesso modo e argomenti>
# Ricetta (dichiarata in PLAN_MASSIVE sez. 23): testa factorized, GT-SR + log S (E12), k_eig 128, lr 1e-4 costante con
# decadimento a 1e-5 al 90% dei passi, EMA 0.999, 5 identita' x 6 viste per rank, batch a dominio singolo con passi
# uguali fra gli 8 domini, quota d'espressioni 0.5, 6 discretizzazioni senza reinserimento (crop compreso), nessuna
# rotazione a ogni uso, niente mm_aug. Fonti: preset release (sources.SOURCE_PRESETS), FaMoS senza le persone held-out.
# L40S: QoS normal. A100 (modo a100): unprivileged solo su decisione del PI del 13 ottobre. Al massimo 8 GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE="${1:?smoke | main | a100}"
# RELEASE_SET=open: il modello a nucleo aperto (PLAN_MASSIVE sez. 24): solo GNM, ICT, FLAME 2023 Open (sources open_core),
# senza --release (nessun 3DMM di test, niente FaMoS), held-out ristretto a quei domini, run dir release_open
SET="${RELEASE_SET:-full}"
case "$SET" in
  full) export STREAM_SOURCES=release STREAM_RELEASE=1; NAME=wbes-massive-release; SUB=main ;;
  open) export STREAM_SOURCES=open_core STREAM_RELEASE=0 STREAM_STEP_JSONL=1 STREAM_HELDOUT_DOMAINS=gnm,ict,flame2023
        NAME=wbes-massive-release-open; SUB=release_open ;;
  *) echo "RELEASE_SET=$SET (full | open)" >&2; exit 2 ;;
esac
export STREAM_ARM=factorized STREAM_RECIPE=c3m STREAM_EXPR_FRAC=0.5
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
  a100)
    G="${2:?gpu}" T="${3:?passi}" TIME="${4:?time}"
    (( G <= 8 )) || { echo "al massimo 8 GPU per il massivo" >&2; exit 2; }
    export STREAM_STEPS="$T" STREAM_SPE=1000 STREAM_SAVE_EVERY=8 STREAM_CKPT_MIN=20
    export STREAM_LR_STEPS="$(( T * 9 / 10 )):1e-5"
    export STREAM_OUT="${STREAM_OUT:-$PWD/aau/runs/massive_release/${SUB}_a100}"
    [[ "$SET" == open ]] && STREAM_OUT="${STREAM_OUT%_a100}"
    # produttori ~5 volte piu' lenti per core sulle EPYC 7742 (PLAN_MASSIVE 22.12): 32 CPU per GPU (oltre, Slurm rifiuta la richiesta)
    sbatch -J "$NAME" -p aicentre-a100 --qos=unprivileged --ntasks-per-node=1 --cpus-per-gpu=32 --requeue \
      --nodes=1 --gpus-per-node="a100:$G" --mem-per-gpu=80G --time="$TIME" v3_work/stream/slurm/massive.sbatch ;;
  a10)
    # UN nodo A10, prioritized, QoS normal (non prelazionabile). Stesse variabili del modo a100: una continuazione
    # (STREAM_OUT=<run>) ritrova la stessa run dir; con lo stesso numero di GPU il batch globale non cambia
    G="${2:?gpu}" T="${3:?passi}" TIME="${4:?time}"
    (( G <= 4 )) || { echo "i nodi A10 hanno 4 GPU" >&2; exit 2; }
    export STREAM_STEPS="$T" STREAM_SPE=1000 STREAM_SAVE_EVERY=8 STREAM_CKPT_MIN=20
    export STREAM_LR_STEPS="$(( T * 9 / 10 )):1e-5"
    export STREAM_OUT="${STREAM_OUT:-$PWD/aau/runs/massive_release/${SUB}_a10}"
    [[ "$SET" == open ]] && STREAM_OUT="${STREAM_OUT%_a10}"
    # Slurm accetta al piu' 32 CPU per job su questi nodi (64 in tutto): 4 per rank (fuori dall'hash della run dir,
    # train_v3.make_run_dir), il resto ai produttori
    export STREAM_TRAIN_CPUS_PER_RANK="${STREAM_TRAIN_CPUS_PER_RANK:-4}"
    sbatch -J "$NAME" -p prioritized --ntasks-per-node=1 --cpus-per-gpu=$(( 32 / G )) --requeue \
      --nodes=1 --gpus-per-node="a10:$G" --mem-per-gpu=40G --time="$TIME" v3_work/stream/slurm/massive.sbatch ;;
  *) echo "modo $MODE (smoke | main | a100 | a10)" >&2; exit 2 ;;
esac
echo "STREAM_OUT=$STREAM_OUT"
