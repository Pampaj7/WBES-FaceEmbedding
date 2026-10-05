#!/usr/bin/env bash
#SBATCH --job-name=wbes-ds-trainrepro2
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=140G
#SBATCH --time=03:00:00
#
# Secondo giro della prova di riproduzione (il primo, trainer_repro.sbatch job 1055816, ha un solo
# campione di rumore del seme e non basta a decidere). Un job per seme (WBES_DS_SEED), tre corse con
# lo STESSO split esplicito v1 (repro_split_bfm_s1234.json), 480 passi, ricetta v1:
#   A  train_steps, un blocco, operatori in uso, S=80        (= v1: lista di soggetti identica)
#   P  train_steps, un blocco, operatori dal PRE-PASS, S=80  (isola il pre-pass e il dataset a blocchi)
#   B  train_steps, 2 blocchi dal pre-pass, S=40             (isola l'ordine a blocchi)
# Confronto su tre semi con trainer_repro_compare2.py.
#
#   WBES_DS_SEED=2345 aau/submit.sh data_scale/trainer_repro2.sbatch
set -euo pipefail

source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
aau_require_venv
aau_require_diffusion_net

SEED="${WBES_DS_SEED:?WBES_DS_SEED}"
ROOT="$AAU_RUNS/data_scale_trainer_repro2/s$SEED"
STAGE="/tmp/wbes-ds-repro2-${SLURM_JOB_ID:-manual}"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$ROOT" "$STAGE"
VIEW="$WBES_ROOT/datasets/REMESH/npz_data_topo_500_withops_areanorm"
DIST="$WBES_ROOT/face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz"
SPLIT="$AAU_DIR/data_scale/repro_split_bfm_s1234.json"
echo "[repro2] host=$(hostname) job=${SLURM_JOB_ID:-none} seed=$SEED root=$ROOT"

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
run() {  # $1 tag, resto: comando
  local tag="$1"; shift
  if ls "$ROOT/$tag"/*/checkpoints/epoch*.pth > /dev/null 2>&1; then
    echo "[repro2] $tag: gia' fatta, salto"; return 0
  fi
  echo "[repro2] $tag: $(date +%T)"
  "$AAU_DIR/run.sh" "$@" > "$ROOT/$tag.log" 2>&1
  tr '\r' '\n' < "$ROOT/$tag.log" | grep -E "^\[steps\]|^Epoch [0-9]{3}" | tail -16
}
for K in 1 2; do
cat > "$STAGE/spec_$K.json" <<JSON
{"geom_dirs": ["$WBES_ROOT/datasets/REMESH/npz_data_topo_500"], "labels": null,
 "convention": "areanorm", "n_blocks": $K, "block_seed": 0}
JSON
done
run A v2_work/fastio/train_steps.py --total-steps 480 --split-json "$SPLIT" "${CACHE[@]}" \
      $(recipe "$SEED" 2 --data_dir "$VIEW" --runs_root "$ROOT/A")
run P v2_work/fastio/train_steps.py --total-steps 480 --split-json "$SPLIT" \
      --data-spec "$STAGE/spec_1.json" --stage-root "$STAGE/stageP" --prepass-proc 30 "${CACHE[@]}" \
      $(recipe "$SEED" 2 --data_dir "$STAGE/stageP" --runs_root "$ROOT/P")
rm -rf "$STAGE/stageP"
run B v2_work/fastio/train_steps.py --total-steps 480 --steps-per-epoch 40 --split-json "$SPLIT" \
      --data-spec "$STAGE/spec_2.json" --stage-root "$STAGE/stageB" --prepass-proc 30 "${CACHE[@]}" \
      $(recipe "$SEED" 4 --data_dir "$STAGE/stageB" --runs_root "$ROOT/B")
echo "[repro2] fine $(date +%T)"
