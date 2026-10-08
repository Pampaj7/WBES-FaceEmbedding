#!/usr/bin/env bash
#SBATCH --job-name=wbes-v3-store-c3f
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=aicentre-a100
#SBATCH --qos=unprivileged
#SBATCH --requeue
#SBATCH --nodelist=nv-ai-04
#SBATCH --cpus-per-task=224
#SBATCH --mem=220G
#SBATCH --time=10:00:00
#
# Store in mmap degli operatori della cella E1 C3F (5.793 identita' di training + eval online), UNA volta, sul disco
# locale di nv-ai-04 (/tmp e' il disco md0, non RAM): tutti i bracci lo leggono, una copia nella page cache.
# Pre-pass con il build_grad vettorizzato di E9 (prepass_v3.py). Ripartibile. Decisione del PI, 8 ottobre.
# Stime misurate l'8 ottobre: operatori ~0.52 TB, store ~0.5 TB (md0 ha 1.5 TB liberi); evecs nel layout del loader.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV=
C="$AAU_RUNS/evidence/trainer_v3/ablations/c3f"
STORE="${WBES_V3_STORE:-/tmp/wbes_v3_store_c3f}"
echo "[store-c3f] $(date +%T) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} -> $STORE"
"$AAU_DIR/run.sh" v3_work/trainer/tools/build_store.py --spec "$C/spec.json" --split "$C/split.json" --out "$STORE" \
  --prepass-proc 216 --workers 64 --shards 64 --verify 200 --tolerate 50 2>&1 | grep --line-buffered -v "Warning\|sparse.FloatTensor"
du -sh "$STORE" && df -h /tmp
