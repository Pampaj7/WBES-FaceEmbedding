#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-memtest
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#
# Cambio di blocco senza GPU (switch_memtest.py): quattro varianti, ognuna in un processo nuovo.
set -euo pipefail
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
export AAU_NV= OMP_NUM_THREADS=4
W="/tmp/wbes-ds-memtest-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$W"' EXIT
O="$AAU_DIR/data_scale/memtest"
N="${WBES_DS_MEMTEST_N:-1250}"
for mode in pop gc trim; do
  "$AAU_DIR/run.sh" aau/data_scale/switch_memtest.py --mode $mode --n-subj "$N" --work "$W/$mode" \
      --out-json "$O/$mode.json" | grep -v "valid test" || true
  rm -rf "$W/$mode"
done
# soglia mmap fissa: i tensori grandi (> 1 MiB) restano mmap e tornano al sistema appena liberati
MALLOC_MMAP_THRESHOLD_=1048576 "$AAU_DIR/run.sh" aau/data_scale/switch_memtest.py --mode gc --n-subj "$N" \
    --work "$W/mmap" --out-json "$O/gc_mmap1M.json" | grep -v "valid test" || true
