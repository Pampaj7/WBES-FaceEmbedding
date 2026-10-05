#!/usr/bin/env bash
#SBATCH --job-name=wbes-train-abl
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=40:00:00
#
# Ablazioni v3 (BOARD_DIARY, "Progetto della versione nuova e ablazioni"), ricetta v1 IDENTICA a
# aau/train_remesh_frame.sbatch (stesso wrapper train_fast.py, cache in RAM), appaiate al
# controllo aau/runs/remesh_v1recipe_current_s1234_1055026. Cambiano solo frame, token e
# operatori, attraverso aau/models/train_ablation.py (hook in aau/models/ablation_hooks.py):
#
#   WBES_ABL_ARM=B  --frame rms     operatori standard           + token di taglia
#   WBES_ABL_ARM=C  --frame current operatori robusti ad area 1  (nessun hook)
#   WBES_ABL_ARM=E  --frame rms     operatori robusti ad area 1  + token di taglia
#
#   WBES_ABL_ARM=B aau/submit.sh aau/models/train_ablation.sbatch --job-name=wbes-train-abl-B
#
# --mem=180G: con 100G la cache in RAM ha mandato in thrashing due training (BOARD_DIARY,
# incidente del 5 ottobre).
#
# Braccio, frame, operatori e tabella dei token finiscono in <runs_root>/ablation_arm.txt: e' da
# li' che aau/models/eval_ablation_topology.sbatch e ict_ablation_rank.sbatch capiscono come
# caricare e valutare il modello (il frame non entra nel fingerprint ne' in config.json).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset

ARM="${WBES_ABL_ARM:?WBES_ABL_ARM=B|C|E}"
SEED="${WBES_SEED:-1234}"
RA1_DIR="${WBES_RA1_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_robust_area1}"
TOKEN_JSON="${WBES_SIZE_TOKEN_JSON:-$AAU_DIR/runs/ablations_v3/size_token_bfm_s${SEED}.json}"
case "$ARM" in
    B) FRAME=rms;     DATA_DIR="$WBES_DATA_DIR"; TOKEN=1 ;;
    C) FRAME=current; DATA_DIR="$RA1_DIR";       TOKEN=0 ;;
    E) FRAME=rms;     DATA_DIR="$RA1_DIR";       TOKEN=1 ;;
    *) echo "ERRORE: WBES_ABL_ARM=$ARM, attesi B|C|E" >&2; exit 2 ;;
esac
ARM_ARGS=()
if (( TOKEN )); then
    [[ -f "$TOKEN_JSON" ]] || { echo "ERRORE: tabella dei token assente: $TOKEN_JSON (aau/models/size_token.py)" >&2; exit 1; }
    ARM_ARGS=(--size-token-json "$TOKEN_JSON")
else
    TOKEN_JSON=""
fi
N_DATA=$(ls -1 "$DATA_DIR" | grep -c '^id.*\.npz$' || true)
N_STD=$(ls -1 "$WBES_DATA_DIR" | grep -c '^id.*\.npz$' || true)
if [[ "$DATA_DIR" != "$WBES_DATA_DIR" && "$N_DATA" != "$N_STD" ]]; then
    echo "ERRORE: $DATA_DIR ha $N_DATA npz, la cartella standard $N_STD (precompute incompleto?)" >&2
    exit 1
fi
RUN_NAME="${WBES_RUN_NAME:-abl3_${ARM}_s${SEED}}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNS_ROOT"

echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[train] arm=$ARM frame=$FRAME seed=$SEED token=${TOKEN_JSON:-no}"
echo "[train] data_dir=$DATA_DIR ($N_DATA npz)"
echo "[train] runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(
  aau/models/train_ablation.py "${ARM_ARGS[@]}"
  --cache-residency ram --cache-workers "${WBES_CACHE_WORKERS:-8}" --frame "$FRAME"
  --data_dir "$DATA_DIR"
  --dist_npz "$WBES_DIST_NPZ"
  --runs_root "$RUNS_ROOT"
  --device cuda
  --model xyz_dn
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300
  --epochs "${WBES_EPOCHS:-120}" --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0
  --train_level mixed --train_pair_mode cross_topology
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5
  --use_id_loss --lambda_id 0.25
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6
  --max_subjects_eval_train 16
  --preload_eval_workers_train "${WBES_PRELOAD_WORKERS:-2}"
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2
  --noise_modes translation,rotation,jitter
  --noise_mode_weights "translation=4,rotation=2,jitter=1"
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6
  --eval_mode average --eval_every 2 --save_every 5 --seed "$SEED"
)
printf '%q ' "${CMD[@]}" > "$RUNS_ROOT/wrapper_launch.txt"
echo >> "$RUNS_ROOT/wrapper_launch.txt"
printf 'arm=%s\nframe=%s\nops_dir_train=%s\nsize_token_json=%s\n' \
    "$ARM" "$FRAME" "$(realpath "$DATA_DIR")" "$TOKEN_JSON" > "$RUNS_ROOT/ablation_arm.txt"

exec "$AAU_DIR/run.sh" "${CMD[@]}"
