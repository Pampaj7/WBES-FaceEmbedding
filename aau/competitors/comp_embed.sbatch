#!/usr/bin/env bash
#SBATCH --job-name=wbes-comp-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=7G
#SBATCH --time=01:00:00
#
# Embedding di Uni3D-g e OpenShape (PointBERT, vitg14-rgb) delle 600 mesh HIFI3D, frame nativo
# (primario) e Rx(+90) (ablazione): comp_embed.py, nel venv dei competitori (comp_setup.sbatch).
# --mem=7G: i pesi si mappano dal file e i parametri nascono sulla GPU (comp_embed.py); con 48G il job
# restava in coda oltre un giorno, l'unico nodo L40S con GPU libere aveva 8 GB di RAM non allocati.
#
#   aau/submit.sh competitors/comp_embed.sbatch
#   WBES_COMP_LIMIT=16 WBES_COMP_MODELS=uni3d aau/submit.sh competitors/comp_embed.sbatch --time=00:15:00  (prova)
#   WBES_COMP_DEVICE=cpu aau/submit.sh competitors/comp_embed.sbatch --gres=NONE --cpus-per-task=32 --mem=32G
#     (stesso calcolo su CPU, fp32: quando nessuna L40S e' schedulabile a breve)
set -euo pipefail

device="${WBES_COMP_DEVICE:-cuda}"
[[ "$device" == cpu ]] && export AAU_NV=""
source "${WBES_ROOT:-$PWD}/aau/env.sh"
source "$WBES_ROOT/aau/competitors/comp_env.sh"
cd "$WBES_ROOT"
for f in "$COMP_VENV/bin/python3" "$COMP_UNI3D_CKPT" "$COMP_OPENSHAPE_CKPT"; do
    [[ -e "$f" ]] || { echo "ERRORE: manca $f (comp_setup.sbatch, pesi in $COMP_EXT)" >&2; exit 1; }
done

limit="${WBES_COMP_LIMIT:-0}"
tag=""
[[ "$limit" != 0 ]] && tag="_limit${limit}"
echo "[comp-emb] host=$(hostname) job=${SLURM_JOB_ID:-none} limit=$limit device=$device"
for model in ${WBES_COMP_MODELS:-uni3d openshape}; do
    for rot in ${WBES_COMP_ROTATIONS:-none rx90}; do
        suffix=""
        [[ "$rot" != none ]] && suffix="_$rot"
        singularity exec ${AAU_NV} "$CONTAINER" bash -c '
            source "$COMP_VENV/bin/activate"
            exec python3 aau/competitors/comp_embed.py "$@"
        ' _ --model "$model" --rotation "$rot" --view-dir "$COMP_VIEW_DIR" \
            --out "$COMP_OUT/emb_${model}${suffix}${tag}.npz" --eval-seed "${WBES_EVAL_SEED:-1234}" --limit "$limit" \
            --device "$device"
    done
done
