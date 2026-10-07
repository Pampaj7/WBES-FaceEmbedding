#!/usr/bin/env bash
#SBATCH --job-name=wbes-distill-eval
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#
# Embedding dello studente distillato sulle 600 mesh valutate di un dominio di test, nelle due
# convenzioni d'ingresso (protocollo: aau/runs/distill_pilot/protocol.md). Per convenzione: zs_stage.py
# (stessi 100 soggetti di select_subjects, seed 1234) e operatori k_eig 128 ad area unitaria su /tmp,
# come zs_zeroshot.sbatch, poi embed_student.py. Output:
#   aau/runs/distill_pilot/eval/<dominio>/<convenzione>/{embeddings.npz,subjects.json}
#
#   WBES_DISTILL_DOMAIN=fv_expr WBES_DISTILL_CKPT=<run>/checkpoints/last.pth aau/submit.sh distill/eval_distill.sbatch
#   WBES_DISTILL_DOMAIN=hifi3d  ...
#
# Convenzioni (le stesse varianti di zs_stage del congiunto, aau/runs/ws_frame):
#   bfm  convenzione BFM completa, quella della riga del congiunto nel gate:
#        FaceVerse --flip-faces (ha gia' la rotazione BFM); HIFI3D --transform x,-y,-z --flip-faces
#   ict  convenzione ICT: FaceVerse --transform x,-y,-z; HIFI3D nativo (e' gia' in convenzione ICT)
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

CKPT="${WBES_DISTILL_CKPT:?WBES_DISTILL_CKPT (checkpoint dello studente)}"
case "${WBES_DISTILL_DOMAIN:-}" in
    fv_expr) VIEW=datasets/FACEVERSE_ZS/expr_view/npz
             declare -A STAGE_ARGS=([bfm]="--flip-faces" [ict]="--transform=x,-y,-z") ;;
    hifi3d)  VIEW=datasets/HIFI3D/eval_view/npz
             declare -A STAGE_ARGS=([bfm]="--transform=x,-y,-z --flip-faces" [ict]="") ;;
    *) echo "ERRORE: WBES_DISTILL_DOMAIN='${WBES_DISTILL_DOMAIN:-}' (fv_expr|hifi3d)" >&2; exit 2 ;;
esac
OUT="${WBES_DISTILL_EVAL_ROOT:-$AAU_RUNS/distill_pilot/eval${WBES_DISTILL_EVAL_SUFFIX:-}}/$WBES_DISTILL_DOMAIN"
TMP="/tmp/wbes_distill_eval_${SLURM_JOB_ID:-manual}"
rm -rf "$TMP"
trap 'rm -rf "$TMP"' EXIT
N_SHARDS="${SLURM_CPUS_PER_TASK:-8}"
(( N_SHARDS % 6 == 0 )) && N_SHARDS=$(( N_SHARDS - 1 ))   # coprimo con 6 (zs_zeroshot.sbatch)
SHARD_LOGS="$AAU_LOGS/${SLURM_JOB_NAME:-wbes-distill-eval}-${SLURM_JOB_ID:-manual}.shards"
mkdir -p "$SHARD_LOGS"

echo "[distill-eval] dominio=$WBES_DISTILL_DOMAIN ckpt=$CKPT host=$(hostname) job=${SLURM_JOB_ID:-none}"
for conv in ${WBES_DISTILL_CONVENTIONS:-bfm ict}; do
    S="$TMP/$conv"
    mkdir -p "$S/ops" "$OUT/$conv"
    t0=$(date +%s)
    # shellcheck disable=SC2086  (gli argomenti della convenzione vanno spezzati)
    "$AAU_DIR/run.sh" aau/zs3dmm/zs_stage.py --view-dir "$VIEW" --out-dir "$S/in" --seed 1234 ${STAGE_ARGS[$conv]}
    pids=()
    for (( i = 0; i < N_SHARDS; i++ )); do
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$AAU_DIR/run.sh" v2_work/potential/areanorm_operators.py --input-dir "$S/in" --output-dir "$S/ops" \
            --k-eig 128 --shard "$i/$N_SHARDS" > "$SHARD_LOGS/$conv.shard$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "ERRORE: shard $i ($conv), vedi $SHARD_LOGS" >&2; exit 1; }
    done
    N_IN=$(find "$S/in" -name "*.npz" | wc -l); N_OPS=$(find "$S/ops" -name "*.npz" | wc -l)
    echo "[distill-eval] $conv: operatori $N_OPS / $N_IN, $(du -sh "$S/ops" | cut -f1) su /tmp, $(( $(date +%s) - t0 ))s"
    (( N_OPS == N_IN )) || { echo "ERRORE: operatori incompleti" >&2; exit 1; }
    cp "$S/subjects.json" "$OUT/$conv/subjects.json"
    "$AAU_DIR/run.sh" aau/distill/embed_student.py --ckpt "$CKPT" --data-dir "$S/ops" --out "$OUT/$conv/embeddings.npz"
    rm -rf "$S"
done
echo "[distill-eval] OK"
