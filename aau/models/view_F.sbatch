#!/usr/bin/env bash
#SBATCH --job-name=wbes-viewF
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#
# Dati della variante F (BOARD_DIARY, ipotesi H6), tre stadi su CPU, WBES_F_STAGE:
#   gen    5 crop casuali dell'original per ognuno dei 400 soggetti di training (seed 1234)
#          -> datasets/REMESH/npz_data_topo_500_cropF (mesh-only V/F), statistiche in aau/runs/ablation_F
#   ops    operatori standard con precompute_operators_npz.py dell'autore (k_eig 128), array 1-5:
#          il task K fa i 400 crop_rK -> datasets/REMESH/npz_data_topo_500_cropF_withops
#   check  operatori finiti e completi, vista datasets/REMESH/view_F (symlink alle 3000 standard + 2000
#          crop) e verifica delle etichette come le ricava il trainer
#
#   WBES_F_STAGE=gen   aau/submit.sh aau/models/view_F.sbatch --cpus-per-task=16
#   WBES_F_STAGE=ops   aau/submit.sh aau/models/view_F.sbatch --array=1-5 --dependency=afterok:<gen>
#   WBES_F_STAGE=check aau/submit.sh aau/models/view_F.sbatch --dependency=afterok:<ops>
#
# Ripartibile: gen non riscrive i crop esistenti (stesso seme, stesso risultato), ops salta i file
# gia' calcolati, check rifiuta symlink che puntano altrove.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net
export AAU_NV=""

RAW="$WBES_ROOT/datasets/REMESH/npz_data_topo_500_cropF"
OPS="$WBES_ROOT/datasets/REMESH/npz_data_topo_500_cropF_withops"
VIEW="$WBES_ROOT/datasets/REMESH/view_F"
STAGE="${WBES_F_STAGE:?WBES_F_STAGE=gen|ops|check}"
echo "[viewF] stage=$STAGE host=$(hostname) job=${SLURM_JOB_ID:-none} task=${SLURM_ARRAY_TASK_ID:-}"

case "$STAGE" in
    gen)
        "$AAU_DIR/run.sh" aau/models/random_crops_F.py gen --out-dir "$RAW" --dist-npz "$WBES_DIST_NPZ" \
            --seed 1234 --control-runs-root "$AAU_RUNS/remesh_v1recipe_current_s1234_1055026"
        echo "[viewF] crop scritti: $(ls -1 "$RAW" | grep -c '_crop_r[0-9]\.npz$')"
        ;;
    ops)
        K="${SLURM_ARRAY_TASK_ID:?lo stadio ops va lanciato come array 1-5}"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
        "$AAU_DIR/run.sh" face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
            --input-dir "$RAW" --output-dir "$OPS" --input-kind npz --pattern "*_crop_r${K}.npz" \
            --k-eig 128 --n-cores "${SLURM_CPUS_PER_TASK:-6}"
        echo "[viewF] crop_r$K con operatori: $(ls -1 "$OPS" | grep -c "_crop_r${K}\.npz$")"
        ;;
    check)
        "$AAU_DIR/run.sh" aau/models/random_crops_F.py check --raw-crop-dir "$RAW" --ops-dir "$OPS" \
            --std-dir "$WBES_DATA_DIR" --view-dir "$VIEW" --dist-npz "$WBES_DIST_NPZ" --seed 1234
        echo "[viewF] vista: $(ls -1 "$VIEW" | grep -c '\.npz$') npz, $(du -shL "$OPS" | cut -f1) di crop con operatori"
        ;;
    *) echo "ERRORE: WBES_F_STAGE=$STAGE, attesi gen|ops|check" >&2; exit 2 ;;
esac
