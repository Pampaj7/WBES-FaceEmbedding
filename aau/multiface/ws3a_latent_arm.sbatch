#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3a-latent-arm
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# WS3a duro (Multiface) per i bracci con frame rms e token di taglia: ws3a_latent.sbatch, ma
# lanciato da aau/multiface/ws3a_latent_arm.py con gli hook delle ablazioni. Tempi e risorse come
# ws3a_latent.sbatch (I/O bound, T4).
#
#   WBES_CKPT=<pth> WBES_MF_FRAME=rms WBES_MF_TOKEN_DIR=<dir per topologia> \
#   WBES_WS3A_PAIRS=hard WBES_WS3A_OUT=aau/runs/multiface_ws3a_hard \
#     sbatch ... aau/multiface/ws3a_latent_arm.sbatch --metric latent_abl_B_tokA
#
# Piu' metriche con lo stesso checkpoint in un job solo (stessi operatori, token diversi):
#   WBES_MF_RUNS="latent_abl_E_tokA=<dir a> latent_abl_E_tok0=<dir b>"  (niente --metric dopo lo script)
#
# Operatori robusti ad area 1 (braccio E): WBES_MF_RA1_STAGE=1 li calcola DENTRO il job su /tmp del
# nodo (aau/models/robust_area1_operators.py, un processo per core, ~23 CPU-ore per le 6 topologie,
# ~60 GB compressi) e punta WBES_MF_PREP_DIR a una prep in /tmp con il manifest originale. Non si
# scrivono su CephFS perche' la quota da 1 TB e' piena: il primo tentativo (job 1055540-1055545) l'ha
# saturata. /tmp e' RAM e conta contro --mem: per lo staging chiedere --cpus-per-task=48 --mem=200G.
#
# Gli argomenti dopo lo script vanno a ws3a_latent.py: si sottomette con sbatch e non con
# aau/submit.sh, che li metterebbe fra le opzioni di sbatch (job 1055168: --wrap con array bash
# eseguito da /bin/sh, morto in un secondo).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
# WBES_CKPT puo' essere anche il runs_root di un training (si sottomette prima che esista la run dir).
if [[ -d "$WBES_CKPT" ]]; then
    _m=("$WBES_CKPT"/*/checkpoints/best_by_xtopo_mesh_clean.pth)
    [[ ${#_m[@]} -eq 1 && -f "${_m[0]}" ]] || { echo "ERRORE: checkpoint in $WBES_CKPT: ${_m[*]}" >&2; exit 2; }
    export WBES_CKPT="${_m[0]}"
fi
aau_require_checkpoint

FRAME="${WBES_MF_FRAME:?WBES_MF_FRAME=current|rms}"
TOPOS=(tracked remesh crop noisy down up)

echo "[ws3a-arm] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[ws3a-arm] ckpt=$WBES_CKPT frame=$FRAME coppie=${WBES_WS3A_PAIRS:-clean} out=${WBES_WS3A_OUT:-default}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ "${WBES_MF_RA1_STAGE:-0}" == "1" ]]; then
    SRC_PREP="${WBES_MF_PREP_DIR:-$WBES_ROOT/datasets/Multiface/prep}"
    STAGE="/tmp/${SLURM_JOB_ID:-manual}/prep"
    trap 'rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
    mkdir -p "$STAGE"
    ln -s "$SRC_PREP/manifest.csv" "$STAGE/manifest.csv"
    PROCS="${SLURM_CPUS_PER_TASK:-8}"
    t0=$SECONDS
    for t in "${TOPOS[@]}"; do
        mkdir -p "$STAGE/${t}_withops_robust_area1"
        # up (~60k vertici) costa ~4 GB a processo: meta' dei processi, per stare in --mem.
        np_t=$PROCS
        [[ "$t" == up ]] && np_t=$(( (PROCS + 1) / 2 ))
        pids=()
        for (( p = 0; p < np_t; p++ )); do
            OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AAU_NV="" \
            "$AAU_DIR/run.sh" aau/models/robust_area1_operators.py \
                --input-dir "$SRC_PREP/$t" --output-dir "$STAGE/${t}_withops_robust_area1" \
                --k-eig 128 --shard "$p/$np_t" > "$STAGE/ra1_${t}_$p.log" 2>&1 &
            pids+=($!)
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || { echo "ERRORE: operatori robusti $t shard $i" >&2; tail -5 "$STAGE/ra1_${t}_$i.log" >&2; exit 1; }
        done
        n=$(find "$STAGE/${t}_withops_robust_area1" -maxdepth 1 -name '*.npz' ! -name '.*' | wc -l)
        n_in=$(find "$SRC_PREP/$t" -maxdepth 1 -name '*.npz' | wc -l)
        [[ "$n" == "$n_in" ]] || { echo "ERRORE: $t: $n operatori su $n_in mesh" >&2; exit 1; }
        echo "[ws3a-arm] operatori robusti $t: $n npz, $(du -sh "$STAGE/${t}_withops_robust_area1" | cut -f1), $(( SECONDS - t0 )) s"
    done
    export WBES_MF_PREP_DIR="$STAGE"
fi
echo "[ws3a-arm] prep=${WBES_MF_PREP_DIR:-default} ops=${WBES_MF_OPS_SUFFIX:-_withops}"

# singularity muore con "unknown userid" se il lookup dell'utente fallisce sul nodo (job 1055552, dopo
# 31 min di operatori calcolati; anche 1054618): prima di ogni lancio si aspetta che getent risponda.
wait_user() {
    local i
    for (( i = 0; i < 60; i++ )); do
        getent passwd "$(id -u)" > /dev/null && return 0
        echo "[ws3a-arm] getent passwd $(id -u) fallito, riprovo fra 10 s" >&2
        sleep 10
    done
    echo "ERRORE: utente $(id -u) non risolvibile sul nodo" >&2
    return 1
}

if [[ -n "${WBES_MF_RUNS:-}" ]]; then
    for spec in $WBES_MF_RUNS; do
        metric="${spec%%=*}"
        tok="${spec#*=}"
        echo "[ws3a-arm] metrica $metric, token da $tok"
        wait_user
        "$AAU_DIR/run.sh" aau/multiface/ws3a_latent_arm.py --frame "$FRAME" --token-dir "$tok" -- \
            --device cuda --metric "$metric" "$@"
    done
else
    ARM_ARGS=(--frame "$FRAME")
    [[ -n "${WBES_MF_TOKEN_DIR:-}" ]] && ARM_ARGS+=(--token-dir "$WBES_MF_TOKEN_DIR")
    echo "[ws3a-arm] token_dir=${WBES_MF_TOKEN_DIR:-no}"
    wait_user
    "$AAU_DIR/run.sh" aau/multiface/ws3a_latent_arm.py "${ARM_ARGS[@]}" -- --device cuda "$@"
fi

echo "[ws3a-arm] OK"
