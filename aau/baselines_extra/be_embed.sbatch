#!/usr/bin/env bash
#SBATCH --job-name=wbes-be-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#
# CLIP ViT-L/14 e DINOv2 ViT-B/14 sui render normal map della pipeline ArcFace (vfm_embed.py), per i
# domini di WBES_BE_DOMAINS (default tutti e tre). Embedding in aau/runs/baselines_extra/<dominio>/.
#
#   WBES_BE_DOMAINS="fv_expr hifi3d" aau/submit.sh baselines_extra/be_embed.sbatch
#   WBES_BE_DOMAINS=bfm_heldout      aau/submit.sh baselines_extra/be_embed.sbatch --dependency=afterok:<render>
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

echo "[be-embed] host=$(hostname) job=${SLURM_JOB_ID:-none} gpu=${CUDA_VISIBLE_DEVICES:-none}"
for dom in ${WBES_BE_DOMAINS:-fv_expr hifi3d bfm_heldout}; do
    case "$dom" in
        fv_expr|hifi3d) ROOT="$AAU_RUNS/arcface_render_zs/$dom/normals" ;;
        bfm_heldout)    ROOT="$AAU_RUNS/baselines_extra/bfm_heldout/normals" ;;
        *) echo "ERRORE: dominio '$dom'" >&2; exit 2 ;;
    esac
    "$AAU_DIR/baselines/run_bl.sh" aau/baselines_extra/vfm_embed.py --render-root "$ROOT" \
        --out-dir "$AAU_RUNS/baselines_extra/$dom"
done
echo "[be-embed] OK"
