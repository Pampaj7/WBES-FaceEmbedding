#!/usr/bin/env bash
#SBATCH --job-name=wbes-train-frame
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=40:00:00
#
# Stessa ricetta v1 di train_remesh.sbatch (iperparametri verbatim, fingerprint incluso),
# ma lanciata dal wrapper dell'autore v2_work/fastio/train_fast.py con --frame: e' cosi' che
# erano stati addestrati i bracci pot_rms/pot_rms_s2 (v2_work/potential/_node_pot_rms.sh,
# rms_seed_chain.sh). Il trainer di ricerca sotto face_embedding/ non viene toccato: il
# wrapper ri-inquadra i vertici sul campione gia' in cache (v2_work/pointnet/frames.py).
#
#   WBES_SEED=2345 aau/submit.sh train_remesh_frame.sbatch --job-name=wbes-train-rms-s2345
#
# Variabili riconosciute: WBES_FRAME (default rms), WBES_SEED, WBES_EPOCHS, WBES_RUN_NAME,
# WBES_RUNS_ROOT, WBES_DATA_DIR, WBES_DIST_NPZ, WBES_PRELOAD_WORKERS, WBES_CACHE_WORKERS.
#
# Il frame vive SOLO nella cache: install_frame patcha CachedDataset.__getitem__, quindi
# con --no-cache il frame verrebbe ignorato senza errori. Qui la cache e' obbligatoria.
#
# Niente staging su /tmp: dopo il preload la cache tiene in RAM tutti i 3000 campioni gia'
# preparati e il training non legge piu' nulla da disco, quindi /tmp non comprerebbe niente
# e raddoppierebbe la memoria (43 GB di npz + la cache). Footprint stimato dagli header
# degli npz: 56 GB di tensori, fino a ~76 GB se l'allocatore pinned arrotonda alle potenze
# di 2; il processo di training del controllo misura ~5 GB di RSS. Da qui --mem=100G.
#
# Il frame NON entra nel fingerprint della run dir e train_fast.py toglie --frame da
# sys.argv prima di passarlo al trainer, quindi launch_command.txt non lo riporta: lo
# scriviamo noi in wrapper_launch.txt nel runs_root, e il nome del runs_root lo contiene.
# Anche l'eval post-hoc deve usare lo stesso frame (eval_by_topology.py --frame rms).
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"

aau_require_venv
aau_require_diffusion_net
aau_require_dataset

FRAME="${WBES_FRAME:-rms}"
SEED="${WBES_SEED:-1234}"
RUN_NAME="${WBES_RUN_NAME:-remesh_v1recipe_${FRAME}_s${SEED}}"
RUNS_ROOT="${WBES_RUNS_ROOT:-$AAU_DIR/runs/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
mkdir -p "$RUNS_ROOT"

echo "[train] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[train] frame=$FRAME seed=$SEED"
echo "[train] data_dir=$WBES_DATA_DIR"
echo "[train] dist_npz=$WBES_DIST_NPZ"
echo "[train] runs_root=$RUNS_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CMD=(
  v2_work/fastio/train_fast.py
  --cache-residency ram --cache-workers "${WBES_CACHE_WORKERS:-8}" --frame "$FRAME"
  --data_dir "$WBES_DATA_DIR"
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

"$AAU_DIR/run.sh" "${CMD[@]}"
