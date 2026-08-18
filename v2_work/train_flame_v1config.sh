#!/usr/bin/env bash
# First FLAME-only training run: v1 top-model recipe, FLAME half of REMESH-2.
# Purpose: the diagonal cell of the cross-model transfer matrix (train FLAME / eval FLAME),
# and the control that says whether the v1 recipe transfers to another 3DMM at all.
#
# Launch on a GPU node:
#   bsub -I -q p1i -gpu "num=1" bash v2_work/train_flame_v1config.sh
set -euo pipefail

ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
export WBES_DIFFUSION_NET_SRC="$ROOT/diffusion-net/src"

DATA=v2_work/genflame/flame_train_ready/npz_withops
DIST=v2_work/genflame/flame_train_ready/gt_matrix.npz
OUT=v2_work/runs/flame_only_v1recipe

mkdir -p "$OUT"

# Hyperparameters copied verbatim from the v1 top model
# (dn_mixed_topology_v1/mixed_xtopo_rank0p5_id0p25_bs5_best/config.json),
# so any difference in outcome is attributable to the data, not the recipe.
exec .conda_env/bin/python \
  face_embedding/gt_encdec/remeshing/intrinsic/train_twotower_dn_spec_robust.py \
  --data_dir "$DATA" \
  --dist_npz "$DIST" \
  --runs_root "$OUT" \
  --device cuda \
  --model xyz_dn \
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \
  --epochs 120 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \
  --train_level mixed --train_pair_mode cross_topology \
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \
  --use_id_loss --lambda_id 0.25 \
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 \
  --max_subjects_eval_train 16 \
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \
  --noise_modes translation,rotation,jitter \
  --noise_mode_weights "translation=4,rotation=2,jitter=1" \
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \
  --eval_mode average --eval_every 2 --save_every 5 --seed 1234
