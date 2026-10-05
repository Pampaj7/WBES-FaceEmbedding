#!/usr/bin/env bash
#SBATCH --job-name=wbes-oB-tables
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Le tre tabelle di OUTLINE_B (§4 allineamento e compressione, §5 frame) dalle matrici gia'
# su disco in aau/runs/outlineB/. Solo bootstrap: nessuna distanza ricalcolata.
#   aau/submit.sh outlineB/tables.sbatch
# Una sola tabella: OB_TABLES="frame" (valori: alignment, compression, frame).
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
OUT="$AAU_RUNS/outlineB"
OB_TABLES="${OB_TABLES:-alignment compression frame}"
echo "[oB-tables] host=$(hostname) job=${SLURM_JOB_ID:-none} tabelle=$OB_TABLES"

for table in $OB_TABLES; do
    case "$table" in
        alignment)   "$AAU_DIR/run.sh" aau/outlineB/alignment_table.py --runs "$OUT" ;;
        compression) "$AAU_DIR/run.sh" aau/outlineB/compression.py --runs "$OUT" ;;
        frame)       "$AAU_DIR/run.sh" aau/outlineB/frame_table.py --out-root "$OUT/bfm_heldout" ;;
        *) echo "tabella sconosciuta: $table" >&2; exit 2 ;;
    esac
done
echo "[oB-tables] OK"
