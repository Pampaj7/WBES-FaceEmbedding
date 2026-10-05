#!/usr/bin/env bash
#SBATCH --job-name=wbes-train-pilot-pot
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=40:00:00
#
# Bracci del pilota pot (aau/runs/pilot_pot/), ricetta v1 IDENTICA a train_remesh_frame.sbatch
# (stesso wrapper train_fast.py, cache in RAM, --frame current), appaiati al controllo
# aau/runs/remesh_v1recipe_current_s1234_1054511. Cambiano solo operatori/modello, attraverso
# aau/models/train_pilot.py (hook in aau/models/pilot_hooks.py):
#
#   WBES_PILOT_ARM=masked  pot_m55: --data_dir = operatori col pozzo a 0.55, pooling sulla ROI
#   WBES_PILOT_ARM=dual    due rami (aau/models/dn_dual_ops.py): --data_dir standard +
#                          --dual-ops-dir pozzo, width 103 (693140 parametri contro 691584)
#
#   WBES_PILOT_ARM=masked aau/submit.sh train_pilot_pot.sbatch --job-name=wbes-train-pot-m55
#   WBES_PILOT_ARM=dual   aau/submit.sh train_pilot_pot.sbatch --job-name=wbes-train-pot-dual --mem=200G
#
# Memoria: la cache del braccio masked e' quella del controllo (~57 GB nominali), ma due training
# del confronto frame con --mem=100G sono andati in thrashing a fine run (RES 81-93 GB in
# crescita durante le epoche, al limite del cgroup): --mem=180G. Il dual tiene in cache anche
# evecs/gradienti/massa del pozzo, 35.0 contro 19.5 MB a campione (smoke 1055011), ~102 GB
# nominali e con la stessa crescita ~165 GB: --mem=200G (nodi A10 da 239 GB; 230G non
# entrava in nessun nodo libero al momento della sottomissione).
#
# Il braccio e la cartella del pozzo finiscono in <runs_root>/pilot_arm.txt: e' da li' che
# eval_frame_topology.sbatch e ict_zeroshot_rank.sbatch capiscono come caricare il modello,
# perche' un checkpoint mascherato si carica senza errori anche nel modello NON mascherato.
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset

ARM="${WBES_PILOT_ARM:?WBES_PILOT_ARM=masked|dual}"
POT_DIR="${WBES_POT_DIR:-$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_pot055}"
SEED="${WBES_SEED:-1234}"
case "$ARM" in
    masked) TAG=m55;  DATA_DIR="$POT_DIR";       WIDTH="${WBES_WIDTH:-128}"; ARM_ARGS=(--pilot-arm masked --roi-threshold 0.5) ;;
    dual)   TAG=dual; DATA_DIR="$WBES_DATA_DIR"; WIDTH="${WBES_WIDTH:-103}"; ARM_ARGS=(--pilot-arm dual --dual-ops-dir "$POT_DIR") ;;
    *) echo "ERRORE: WBES_PILOT_ARM=$ARM, attesi masked|dual" >&2; exit 2 ;;
esac
if [[ ! -d "$POT_DIR" ]]; then
    echo "ERRORE: operatori col pozzo non trovati in $POT_DIR (aau/precompute_pot.sbatch)" >&2
    exit 1
fi
RUN_NAME="${WBES_RUN_NAME:-pilot_pot_${TAG}_s${SEED}}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNS_ROOT"

echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[train] arm=$ARM width=$WIDTH seed=$SEED"
echo "[train] data_dir=$DATA_DIR"
echo "[train] pot_dir=$POT_DIR"
echo "[train] runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(
  aau/models/train_pilot.py "${ARM_ARGS[@]}"
  --cache-residency ram --cache-workers "${WBES_CACHE_WORKERS:-8}" --frame current
  --data_dir "$DATA_DIR"
  --dist_npz "$WBES_DIST_NPZ"
  --runs_root "$RUNS_ROOT"
  --device cuda
  --model xyz_dn
  --latent_dim 256 --width "$WIDTH" --n_blocks 4 --dropout 0.1
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
printf 'arm=%s\npot_dir_train=%s\nwidth=%s\n' "$ARM" "$POT_DIR" "$WIDTH" > "$RUNS_ROOT/pilot_arm.txt"

exec "$AAU_DIR/run.sh" "${CMD[@]}"
