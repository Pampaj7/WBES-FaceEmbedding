#!/usr/bin/env bash
#SBATCH --job-name=wbes-bl-t1diag
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# WS0: le 6 celle diagonali (same-topology) della Tabella 1 del paper, che lo stage
# eval_topology non produce.  Tre passi in un solo job, sui 100 soggetti
# facebench_first100 (lo stesso insieme della Tabella 2, quello con cui la diagonale
# pubblicata e' stata calcolata):
#
#   1. chamfer_matrix.py --variant facebench  -> matrices/chamfer/<topo>__to__<topo>.npz
#      (di same-topology su fb100 c'era solo original->original; crop mancava del tutto)
#   2. human_study/latent_matrix.py           -> matrices/latent_v1/<topo>__to__<topo>.npz
#   3. table1_diagonal.py                     -> ranking/table1_diagonal_*.{csv,tex,json}
#
#   aau/submit.sh baselines/table1_diag.sbatch
#
# T4 e non L40S: il latent e' 600 forward in tutto, il tempo se ne va nel leggere gli npz
# con operatori da CephFS; il Chamfer gira su CPU (cKDTree) nella stessa allocazione.
# Tutti e tre i passi sono riprendibili: saltano cio' che e' gia' su disco.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset
aau_require_checkpoint

OUT_ROOT="${WBES_BASELINES_OUT:-$AAU_RUNS/baselines_fb100}"
SUBJECT_SET="${WBES_SUBJECT_SET:-facebench_first100}"
TOPOLOGIES="crop,down8k,noisy,original,remesh,up60k"
SETTINGS="same_crop,same_down8k,same_noisy,same_original,same_remesh,same_up60k"

echo "[t1diag] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT_ROOT set=$SUBJECT_SET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$AAU_DIR/run.sh" aau/baselines/chamfer_matrix.py \
    --variant facebench --settings "$SETTINGS" --subject-set "$SUBJECT_SET" \
    --out-root "$OUT_ROOT" --workers "${SLURM_CPUS_PER_TASK:-8}"

"$AAU_DIR/run.sh" aau/human_study/latent_matrix.py \
    --device cuda --topology "$TOPOLOGIES" --subject-set "$SUBJECT_SET" \
    --out-root "$OUT_ROOT"

"$AAU_DIR/run.sh" aau/baselines/table1_diagonal.py \
    --subject-set "$SUBJECT_SET" --out-root "$OUT_ROOT" --strict ${WBES_BL_ARGS:-}

echo "[t1diag] OK"
