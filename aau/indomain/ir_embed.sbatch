#!/usr/bin/env bash
#SBATCH --job-name=wbes-ir-embed
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Riconoscimento in dominio: embedding di ogni mesh degli insiemi di aau/indomain/ir_views.py con
# aau/zs3dmm/zs_embed.py (stessa catena dello script di breakdown), quattro combinazioni in fila:
# congiunto su bfm, ict, rexpr; BFM-only su bfm (il summarizer ne tiene solo i 19 soggetti bfm19).
# Argomenti da eval_common.sh, piu' `--subject_split all --max_subjects 0` come ws2_eval.sbatch:
# la vista contiene gia' e solo i soggetti valutati. GT: quella del dominio, come le eval WS2
# (serve solo a filtrare i soggetti). Frame nativo dei dati: niente trasformazioni.
#
#   aau/submit.sh indomain/ir_embed.sbatch
#   WBES_IR_EMBED_COMBOS=joint:ict992 aau/submit.sh indomain/ir_embed.sbatch --time=02:00:00
#     (galleria grande, revisione 1 del protocollo: 5952 mesh lette da CephFS)
#
# Output: aau/runs/indomain_recog/embeddings/<modello>__<insieme>.npz
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

RUN_DIR="mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__9a81466d"
declare -A CKPT=(
    [joint]="$AAU_RUNS/x3dmm_joint_bfm_ict_s1234_1019532/$RUN_DIR/checkpoints/best_by_xtopo_mesh_clean.pth"
    [bfm_only]="$AAU_RUNS/remesh_v1recipe_areanorm_s1234_1019310/$RUN_DIR/checkpoints/best_by_xtopo_mesh_clean.pth"
)
declare -A VIEW=(
    [bfm]="$WBES_ROOT/datasets/WS2_CROSS3DMM/joint__bfm"
    [ict]="$WBES_ROOT/datasets/WS2_CROSS3DMM/joint__ict"
    [rexpr]="$WBES_ROOT/datasets/INDOMAIN_RECOG/joint__rexpr6"
    [ict992]="$WBES_ROOT/datasets/INDOMAIN_RECOG/joint__ict992"
)
declare -A GT=(
    [bfm]="$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
    [ict]="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
    [rexpr]="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
    [ict992]="$WBES_ROOT/datasets/ICT/train_ready/gt_matrix.npz"
)
OUT="$AAU_RUNS/indomain_recog/embeddings"
mkdir -p "$OUT"
echo "[ir-embed] host=$(hostname) job=${SLURM_JOB_ID:-none} out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for combo in ${WBES_IR_EMBED_COMBOS:-joint:bfm joint:ict joint:rexpr bfm_only:bfm}; do
    model="${combo%%:*}" set="${combo##*:}"
    if [[ -f "$OUT/${model}__${set}.npz" ]]; then
        echo "[ir-embed] $model su $set: gia' presente, salto"
        continue
    fi
    echo "[ir-embed] $model su $set (${VIEW[$set]})"
    # Sottoshell: eval_common.sh fissa le sue variabili (common_args, OUT_ROOT) per UN checkpoint.
    (
        export WBES_CKPT="${CKPT[$model]}" WBES_DATA_DIR="${VIEW[$set]}" WBES_DIST_NPZ="${GT[$set]}"
        source "$WBES_ROOT/aau/eval_common.sh"
        common_args+=(--subject_split all --max_subjects 0)
        "$AAU_DIR/run.sh" aau/zs3dmm/zs_embed.py "${common_args[@]}" --embeddings_out "$OUT/${model}__${set}.npz"
    ) 2>&1 | tee "$OUT/${model}__${set}.log"
done
echo "[ir-embed] OK"
