#!/usr/bin/env bash
#SBATCH --job-name=wbes-recon-smoke
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# I tre metodi sulle 3 immagini di prova, piu' i controlli di validita' delle mesh (WS3b).
#   aau/submit.sh recon/smoke.sbatch
#
# T4 di proposito: le L40S sono occupate dal training, e qui la GPU serve solo a far
# girare due reti da 120x120 pixel.  PRNet gira comunque su CPU (tensorflow-cpu).
# Per avere tempi confrontabili fra i tre, 3DDFA_V2 e SynergyNet girano due volte, su
# GPU e su CPU: l'unico numero paragonabile a quello di PRNet e' quello CPU.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OUT="${WBES_RECON_OUT:-$AAU_RUNS/recon}"
export WBES_RECON_OUT="$OUT"
IMAGES="$OUT/test_images"

echo "[recon-smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "[recon-smoke] --- immagini di prova ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/make_test_images.py --mode samples --out "$IMAGES"

echo "[recon-smoke] --- 3DDFA_V2 (gpu) ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/tddfa_v2_run.py \
    --images "$IMAGES" --out "$OUT/3ddfa_v2" --device cuda
echo "[recon-smoke] --- 3DDFA_V2 (cpu, solo per il tempo) ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/tddfa_v2_run.py \
    --images "$IMAGES" --out "$OUT/3ddfa_v2_cpu" --device cpu

echo "[recon-smoke] --- SynergyNet (gpu) ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/synergynet_run.py \
    --images "$IMAGES" --out "$OUT/synergynet" --device cuda
echo "[recon-smoke] --- SynergyNet (cpu, solo per il tempo) ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/synergynet_run.py \
    --images "$IMAGES" --out "$OUT/synergynet_cpu" --device cpu

echo "[recon-smoke] --- PRNet (cpu) ---"
"$AAU_DIR/recon/run_recon.sh" prnet aau/recon/prnet_run.py \
    --images "$IMAGES" --out "$OUT/prnet"

echo "[recon-smoke] --- controlli ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/check_meshes.py --expect 3 \
    "$OUT/3ddfa_v2" "$OUT/3ddfa_v2_cpu" "$OUT/synergynet" "$OUT/synergynet_cpu" "$OUT/prnet"

# Controllo incrociato dei 68 landmark contro 3DDFA_V2: serve soprattutto a SynergyNet,
# che gira su una tabella 3dmm_data ricostruita da un altro repo.
echo "[recon-smoke] --- landmark vs 3DDFA_V2 ---"
"$AAU_DIR/recon/run_recon.sh" ddfa aau/recon/check_meshes.py --expect 3 \
    --landmarks-ref "$OUT/3ddfa_v2" "$OUT/synergynet" "$OUT/prnet"

echo "[recon-smoke] OK"
