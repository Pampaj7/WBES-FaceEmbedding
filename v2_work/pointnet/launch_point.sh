#!/usr/bin/env bash
# The operator-free arm, against pot_plain.
#
#   pot_point   PointEncoder (DGCNN-style, area-weighted draw of 2048 points, no operators)
#
# Matched to pot_plain in everything except the encoder: same data directory (the frozen v1
# REMESH set that pot_plain itself used), same GT matrix, same losses and weights, same
# augmentation, same 60 epochs, same seed 1234, same selection rule. The bar is therefore the
# already-measured pot_plain: crop 0.7072, noisy 0.7561, resample 0.7719, all 0.7347.
#
# CAPACITY IS NOT MATCHED, and that is deliberate. PointEncoder carries 288,768 parameters
# against xyz_dn's 691,584 (0.42x); widening it to 256 only reaches 428,928, because most of
# DiffusionNet's parameters sit in its diffusion blocks and have no counterpart here. Chasing
# parity would mean inventing an architecture rather than testing one. So: if this arm wins or
# ties, it does so with fewer parameters and the conclusion holds a fortiori. If it LOSES,
# capacity becomes a live confound and the honest follow-up is a wider/deeper point arm before
# concluding anything -- that follow-up is conditional on the outcome, not run pre-emptively.
#
#   bash launch_point.sh [width] [tag] [data_dir]
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
cd "$ROOT"
W=${1:-128}
TAG=${2:-pot_point}
DATA=${3:-datasets/REMESH/npz_data_topo_500_withops}

cat > v2_work/pointnet/_node_$TAG.sh <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=\$PWD/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
echo "using GPU \$CUDA_VISIBLE_DEVICES of:"; nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency ram --cache-workers 8 \\
  --point-backbone --point-samples 2048 --point-knn 20 \\
  --data_dir $DATA --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \\
  --runs_root v2_work/runs/${TAG}_\${LSB_JOBID:-local} \\
  --device cuda --model xyz_dn \\
  --latent_dim 256 --width $W --n_blocks 4 --dropout 0.1 \\
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \\
  --epochs 60 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \\
  --train_level mixed --train_pair_mode cross_topology \\
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \\
  --use_id_loss --lambda_id 0.25 \\
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \\
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 \\
  --max_subjects_eval_train 16 \\
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \\
  --noise_modes translation,rotation,jitter \\
  --noise_mode_weights "translation=4,rotation=2,jitter=1" \\
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \\
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \\
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \\
  --eval_mode average --eval_every 2 --save_every 5 --seed 1234
NODE
chmod +x v2_work/pointnet/_node_$TAG.sh

# --model xyz_dn above is not a contradiction: the trainer still parses it (it drives the run
# directory name and the checkpoint config), but install_point_backbone() has already rebound
# build_model, so the string is never used to construct anything. Verified by test_wiring.py.

mkdir -p v2_work/logs/runs
export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
  -gpu "num=1:mode=shared" -W 720 -J $TAG \
  "bash $ROOT/v2_work/pointnet/_node_$TAG.sh" \
  > v2_work/logs/runs/$TAG.log 2>&1 < /dev/null &
echo "[$(date +%T)] $TAG sottomesso (width=$W), log: v2_work/logs/runs/$TAG.log"
