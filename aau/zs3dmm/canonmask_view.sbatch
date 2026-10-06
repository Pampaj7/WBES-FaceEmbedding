#!/usr/bin/env bash
#SBATCH --job-name=wbes-canon-view
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=NONE
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#
# Viste a maschera canonica di un dominio (WBES_ZS_DOMAIN=bfm|hifi|ict), canonmask.py build:
#   WBES_CANON_MODE=canonical  -> $ZS_ROOT/canonmask_view
#   WBES_CANON_MODE=offcenter  -> $ZS_ROOT/canonmask_offcenter_view   (controllo negativo)
#   WBES_CANON_MODE=identity   -> $ZS_ROOT/canonmask_identity_view    (controllo d'identita')
# Costanti di dominio (S_dom, rho) fissate in aau/runs/canonmask/PROTOCOL.md PRIMA dei lanci, dalla
# calibrazione aau/runs/canonmask/calibration_<dominio>.csv. Non cambiarle senza riscrivere il protocollo.
#
#   WBES_ZS_DOMAIN=hifi WBES_CANON_MODE=canonical aau/submit.sh zs3dmm/canonmask_view.sbatch
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/zs3dmm/zs_env.sh"
cd "$WBES_ROOT"
aau_require_venv
zs_require_view

case "$WBES_ZS_DOMAIN" in
    hifi) S_DOM=14.9825; RHO=0.40 ;;
    ict)  S_DOM=19.0128; RHO=0.51 ;;
    bfm)  S_DOM=132843;  RHO=0.20 ;;
    *) echo "ERRORE: dominio $WBES_ZS_DOMAIN senza costanti (bfm|hifi|ict)" >&2; exit 2 ;;
esac
MODE="${WBES_CANON_MODE:?WBES_CANON_MODE=canonical|offcenter|identity}"
case "$MODE" in
    canonical) NAME=canonmask ;;
    offcenter|identity) NAME="canonmask_$MODE" ;;
    *) echo "ERRORE: WBES_CANON_MODE=$MODE" >&2; exit 2 ;;
esac
OUT="$ZS_ROOT/${NAME}_view"
echo "[canon-view] dominio=$WBES_ZS_DOMAIN modo=$MODE S_dom=$S_DOM rho=$RHO host=$(hostname) job=${SLURM_JOB_ID:-none} -> $OUT"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$AAU_DIR/run.sh" aau/zs3dmm/canonmask.py build --view-dir "$ZS_DATA_DIR" --out-dir "$OUT" \
    --mode "$MODE" --s-dom "$S_DOM" --rho "$RHO" --n-cores "${SLURM_CPUS_PER_TASK:-8}" ${WBES_CANON_ARGS:-}
du -sh "$OUT"
