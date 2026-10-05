#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-trainrepro
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#
# Prova corta: v2_work/fastio/train_steps.py riproduce il training attuale? Dati di oggi (BFM, 500
# soggetti, operatori ad area unitaria = braccio ctrl), ricetta v1 identica a train_joint_E.sbatch
# senza le patch multi-dominio, 6 epoche da 80 passi = 480 passi. Cinque corse in sequenza su una
# sola L40S (un solo slot job: i 12 dell'utente sono condivisi):
#   guard  train_steps con heldout_frozen.json e lo split v1 seme 1234: DEVE fallire (lo split v1
#          mette in training soggetti held-out dei semi 2345/3456)
#   v1     train_fast.py --epochs 6                                     riferimento
#   v1b    idem, ripetuta                                               non-determinismo GPU
#   id     train_steps.py --total-steps 480                             atteso = v1
#   seed   train_steps.py --total-steps 480, split esplicito v1, seme 2345   rumore del seme
#   blk    train_steps.py --total-steps 480, 2 blocchi, operatori dal pre-pass sulla geometria
#          (datasets/REMESH/npz_data_topo_500), 40 passi per epoca, eval ogni 4 epoche (= 160 passi)
# Poi trainer_repro_compare.py. /tmp: pre-pass BFM ~45 GB + cache ~40 GB, contano contro --mem.
#
#   aau/submit.sh data_scale/trainer_repro.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

ROOT="$AAU_RUNS/data_scale_trainer_repro_${SLURM_JOB_ID:-manual}"
STAGE="/tmp/wbes-ds-repro-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$ROOT" "$STAGE"
VIEW="$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_areanorm"
DIST="$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
SPLIT="$AAU_DIR/data_scale/repro_split_bfm_s1234.json"
FROZEN="$AAU_DIR/data_scale/heldout_frozen.json"
echo "[repro] host=$(hostname) job=${SLURM_JOB_ID:-none} root=$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

recipe() {  # $1 seme, $2 eval_every, resto: dir e flag extra
  local seed="$1" ev="$2"; shift 2
  echo --device cuda --model xyz_dn \
    --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \
    --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \
    --epochs 6 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \
    --train_level mixed --train_pair_mode cross_topology \
    --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \
    --use_id_loss --lambda_id 0.25 \
    --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \
    --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 \
    --max_subjects_eval_train 16 --preload_eval_workers_train 2 \
    --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \
    --noise_modes translation,rotation,jitter \
    --noise_mode_weights translation=4,rotation=2,jitter=1 \
    --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \
    --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \
    --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \
    --eval_mode average --eval_every "$ev" --save_every 6 --seed "$seed" \
    --dist_npz "$DIST" "$@"
}
CACHE=(--cache-residency ram --cache-workers 16 --frame current)

echo "[repro] guard: deve fallire"
if "$AAU_DIR/run.sh" v2_work/fastio/train_steps.py --total-steps 480 --split-json "$SPLIT" \
      --frozen-heldout "$FROZEN" --no-cache $(recipe 1234 2 --data_dir "$VIEW" --runs_root "$STAGE/guard") \
      > "$ROOT/guard.log" 2>&1; then
  echo "ERRORE: la guardia held-out NON ha fermato il run" >&2; exit 1
fi
grep -m1 "GUARDIA HELD-OUT" "$ROOT/guard.log"

run() {  # $1 tag, resto: comando
  local tag="$1"; shift
  echo "[repro] $tag: $(date +%T)"
  "$AAU_DIR/run.sh" "$@" > "$ROOT/$tag.log" 2>&1
  grep -E "^\[steps\]|^Epoch|cache\] ready" "$ROOT/$tag.log" | tr '\r' '\n' | grep -v "it/s" | tail -20
}
run v1   v2_work/fastio/train_fast.py "${CACHE[@]}" $(recipe 1234 2 --data_dir "$VIEW" --runs_root "$ROOT/v1")
run v1b  v2_work/fastio/train_fast.py "${CACHE[@]}" $(recipe 1234 2 --data_dir "$VIEW" --runs_root "$ROOT/v1b")
run id   v2_work/fastio/train_steps.py --total-steps 480 "${CACHE[@]}" \
         $(recipe 1234 2 --data_dir "$VIEW" --runs_root "$ROOT/id")
run seed v2_work/fastio/train_steps.py --total-steps 480 --split-json "$SPLIT" "${CACHE[@]}" \
         $(recipe 2345 2 --data_dir "$VIEW" --runs_root "$ROOT/seed")

cat > "$STAGE/spec_blk.json" <<JSON
{"geom_dirs": ["$WBES_ROOT/datasets/REMESH/npz_data_topo_500"], "labels": null,
 "convention": "areanorm", "n_blocks": 2, "block_seed": 0}
JSON
run blk  v2_work/fastio/train_steps.py --total-steps 480 --steps-per-epoch 40 --split-json "$SPLIT" \
         --data-spec "$STAGE/spec_blk.json" --stage-root "$STAGE/stage" --prepass-proc 30 "${CACHE[@]}" \
         $(recipe 1234 4 --data_dir "$STAGE/stage" --runs_root "$ROOT/blk")
cp "$STAGE"/stage/*.prepass.log "$ROOT/" 2>/dev/null || true

"$AAU_DIR/run.sh" aau/data_scale/trainer_repro_compare.py --runs-root "$ROOT" \
    --out-json "$AAU_DIR/data_scale/trainer_repro.json"
