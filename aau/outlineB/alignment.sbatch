#!/usr/bin/env bash
#SBATCH --job-name=wbes-oB-align
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=30
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#
# Pipeline faceBench (Chamfer, rigid ICP, NICP P2P/P2Tri) su un set di soggetti, uno shard
# delle 31 coppie di topologie (original->original + 30 cross). Solo CPU.
#
#   OB_SET=heldout       OB_SHARD=0/4 aau/submit.sh outlineB/alignment.sbatch
#   OB_SET=ict_heldout   OB_SHARD=1/4 aau/submit.sh outlineB/alignment.sbatch
#   OB_SET=facebench_first100 OB_SHARD=2/4 aau/submit.sh outlineB/alignment.sbatch \
#       --partition=prioritized --gres=NONE --cpus-per-task=48
#
# OB_SAMPLES=2048 scrive in <out>_s2048: e' il campionamento con cui sono state prodotte la
# colonna cross no-crop di tab:alignment_effect e la tabella di compressione del paper
# (l'artefatto facebench_remesh_100subj_norm si riproduce esattamente solo con 2048 punti:
# job 1054872, aau/scratch/outlineB/sample_probe.py); original->original invece e' a 4096,
# il default di run_facebench_remesh.
#
# TEMPI misurati (job 1054623, un core): 2.5-2.9 s per coppia, quasi tutto NICP; 4950
# coppie per coppia di topologie = ~3.8 ore-core, cioe' ~8 min a 30 worker.
set -euo pipefail

export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

OB_SET="${OB_SET:?OB_SET=heldout|facebench_first100|ict_heldout}"
OB_SHARD="${OB_SHARD:-0/1}"
OB_SAMPLES="${OB_SAMPLES:-4096}"
case "$OB_SET" in
    heldout)            OUT="$AAU_RUNS/outlineB/bfm_heldout" ;;
    facebench_first100) OUT="$AAU_RUNS/outlineB/bfm_fb100" ;;
    ict_heldout)        OUT="$AAU_RUNS/outlineB/ict_heldout"
                        export WBES_REMESH_NOOPS_DIR="$WBES_ROOT/datasets/ICT/eval_view_heldout"
                        export WBES_DIST_NPZ="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
                        export WBES_EVAL_SEED="${WBES_EVAL_SEED:-1234}" ;;
    *) echo "OB_SET sconosciuto: $OB_SET" >&2; exit 2 ;;
esac
if [[ "$OB_SAMPLES" != "4096" ]]; then
    OUT="${OUT}_s${OB_SAMPLES}"
fi
echo "[oB-align] host=$(hostname) job=${SLURM_JOB_ID:-none} set=$OB_SET shard=$OB_SHARD out=$OUT"

"$AAU_DIR/outlineB/run_o3d.sh" aau/outlineB/alignment_matrix.py \
    --subject-set "$OB_SET" --out-root "$OUT" --shard "$OB_SHARD" \
    --max-sample-points "$OB_SAMPLES" \
    --workers "${SLURM_CPUS_PER_TASK:-8}" ${OB_ARGS:-}
echo "[oB-align] OK"
