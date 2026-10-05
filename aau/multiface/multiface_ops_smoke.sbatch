#!/usr/bin/env bash
#SBATCH --job-name=wbes-mf-ops-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#
# WS3a, passo 2a: quanto costa un operatore su una mesh Multiface? 20 mesh per topologia e
# per convenzione, un processo solo, cronometrate. Serve a dimensionare --time e --cpus del
# job vero (`multiface_ops.sbatch` / `multiface_ops_areanorm.sbatch`).
#
#   aau/submit.sh multiface/multiface_ops_smoke.sbatch
#
# Le 20 mesh si ritagliano senza toccare gli script di calcolo: `--pattern` di
# `precompute_operators_npz.py` prende un segmento intero (20 frame), e `--shard 0/N` di
# `areanorm_operators.py` con N = ceil(totale/20) prende 20 file sparsi. L'output finisce
# in una dir usa-e-getta sotto aau/runs, non nel dataset.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

PREP_DIR="${WBES_MF_PREP_DIR:-$WBES_ROOT/datasets/Multiface/prep}"
OUT_ROOT="${WBES_MF_SMOKE_OUT:-$AAU_RUNS/multiface_ops_smoke}"
K_EIG="${WBES_K_EIG:-128}"
N_SAMPLE="${WBES_MF_SMOKE_N:-20}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[mf-smoke] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[mf-smoke] prep=$PREP_DIR out=$OUT_ROOT k_eig=$K_EIG n=$N_SAMPLE"

rm -rf "$OUT_ROOT"
for topology in tracked remesh down; do
    IN_DIR="$PREP_DIR/$topology"
    if [[ ! -d "$IN_DIR" ]]; then
        echo "ERRORE: mesh non trovate in $IN_DIR (gira prima multiface_prep.sbatch)" >&2
        exit 1
    fi
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)

    # Un segmento intero = 20 frame: il pattern e' il nome del primo npz senza il frame.
    # `sed -n 1p` e non `head -1`: head chiude la pipe dopo una riga, sort muore di SIGPIPE
    # e con `set -o pipefail` il job esce subito (misurato: job 1019341, 0 secondi).
    FIRST=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f -printf '%f\n' | sort | sed -n '1p')
    PATTERN="${FIRST%__*}__*.npz"
    # ceil(N_IN / N_SAMPLE): lo stride che lascia N_SAMPLE file nello shard 0.
    N_SHARDS=$(( (N_IN + N_SAMPLE - 1) / N_SAMPLE ))

    t0=$(date +%s)
    "$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_ROOT/${topology}_withops" \
        --input-kind npz --pattern "$PATTERN" --k-eig "$K_EIG" --n-cores 1
    t1=$(date +%s)
    n_std=$(find "$OUT_ROOT/${topology}_withops" -maxdepth 1 -name '*.npz' -type f | wc -l)

    "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_ROOT/${topology}_withops_areanorm" \
        --k-eig "$K_EIG" --shard "0/$N_SHARDS"
    t2=$(date +%s)
    n_area=$(find "$OUT_ROOT/${topology}_withops_areanorm" -maxdepth 1 -name '*.npz' -type f | wc -l)

    echo "[mf-smoke] $topology: $N_IN mesh in ingresso"
    echo "[mf-smoke]   standard: $n_std mesh in $((t1 - t0)) s -> $(echo "$((t1 - t0)) $n_std" | awk '{printf "%.2f", $1/$2}') s/mesh"
    echo "[mf-smoke]   areanorm: $n_area mesh in $((t2 - t1)) s -> $(echo "$((t2 - t1)) $n_area" | awk '{printf "%.2f", $1/$2}') s/mesh"
    echo "[mf-smoke]   stima 2 x $N_IN mesh a processo singolo: $(echo "$((t2 - t0)) $((n_std + n_area)) $N_IN" | awk '{printf "%.1f", $1/$2*$3*2/3600}') CPU-ore"
    du -sh "$OUT_ROOT/${topology}_withops" "$OUT_ROOT/${topology}_withops_areanorm"
done

echo "[mf-smoke] totale dir di prova: $(du -sh "$OUT_ROOT" | cut -f1)"
