#!/usr/bin/env bash
#SBATCH --job-name=wbes-ws3b-gtclip
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# WS3b, latent a patch uguale, passo 1: ricostruzioni ritagliate alla patch GT di
# chamfer_gtclip_mm e loro operatori DiffusionNet (k_eig 128, convenzione standard).
#
#   aau/submit.sh recon/ws3b_gtclip_prep.sbatch
#
# Poi il latent (GPU) e l'analisi (cpu).  Il latent vuole argomenti per lo script, che
# submit.sh non passa: sbatch diretto, le direttive del file valgono per questo checkout.
#   sbatch --gres=gpu:a10:1 --dependency=afterok:<questo> aau/recon/ws3b_latent.sbatch \
#       --mode gt --recon-variant gtclip
#   aau/submit.sh recon/ws3b_analysis.sbatch --dependency=afterok:<latent>
#
# Il ritaglio rifa' l'ICP di ws3b_geometric (~80 s per metodo a 16 worker, misurato nella
# corsa 1019739) e si ferma se la chamfer_gtclip_mm ricalcolata non coincide col csv.
# Gli operatori: 3 min per metodo nella prep originale (job 1019523), stesso script usato
# verbatim, che salta i file gia' presenti, quindi il job e' ripartibile.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net

OUT="${WBES_WS3B_OUT:-$AAU_RUNS/multiface_ws3b}"
export WBES_WS3B_OUT="$OUT"
K_EIG="${WBES_K_EIG:-128}"
N_CORES="${WBES_WS3B_CORES:-${SLURM_CPUS_PER_TASK:-8}}"
METHODS="${WBES_WS3B_METHODS:-3ddfa_v2 synergynet prnet}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "[ws3b-gtclip] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT k_eig=$K_EIG cores=$N_CORES"

echo "[ws3b-gtclip] --- ritaglio alla patch GT ---"
"$AAU_DIR/run.sh" aau/recon/ws3b_gtclip_meshes.py --workers "$N_CORES" \
    --methods "$(echo "$METHODS" | tr ' ' ',')" ${WBES_WS3B_GTCLIP_ARGS:-}

echo "[ws3b-gtclip] --- operatori DiffusionNet ---"
for m in $METHODS; do
    IN_DIR="$OUT/recon_face_gtclip/$m"
    OUT_DIR="$OUT/recon_face_gtclip_withops/$m"
    N_IN=$(find "$IN_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[ws3b-gtclip] $m: $N_IN mesh -> $OUT_DIR"
    t0=$(date +%s)
    "$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
        --input-dir "$IN_DIR" --output-dir "$OUT_DIR" \
        --input-kind npz --k-eig "$K_EIG" --n-cores "$N_CORES"
    N_OUT=$(find "$OUT_DIR" -maxdepth 1 -name '*.npz' -type f | wc -l)
    echo "[ws3b-gtclip] $m: $N_OUT / $N_IN npz in $(( ($(date +%s) - t0) / 60 )) min, $(du -sh "$OUT_DIR" | cut -f1)"
    if (( N_OUT != N_IN )); then
        echo "ERRORE: output incompleto per $m ($N_OUT/$N_IN). Risottometti: riprende." >&2
        exit 1
    fi
done

echo "[ws3b-gtclip] OK"
