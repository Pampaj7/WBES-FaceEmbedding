#!/usr/bin/env bash
# A/B for the potential-well hypothesis: same recipe, same subjects, same GT — the only
# difference is whether the operators come from L or from L + U.
#
#   bash launch_potential_train.sh well      # trained on potential-well operators
#   bash launch_potential_train.sh plain     # control, standard operators
#
# The control is not optional: the v1 checkpoint is not a valid baseline here because the
# subject subset and the epoch budget differ, so both arms must be trained here.
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
ARM=${1:?well|plain}
QUEUE=${2:-gpua100}

case "$ARM" in
  well)  DATA=v2_work/potential/bfm_withwell ;;
  plain) DATA=datasets/REMESH/npz_data_topo_500_withops ;;
  *) echo "arm must be well|plain" >&2; exit 2 ;;
esac
TAG="pot_${ARM}"
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz

cat > "$ROOT/v2_work/potential/_node_${TAG}.sh" <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency device --cache-workers 16 \\
  --data_dir $DATA --dist_npz $DIST \\
  --runs_root v2_work/runs/${TAG} \\
  --device cuda --model xyz_dn \\
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \\
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \\
  --epochs 120 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \\
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
chmod +x "$ROOT/v2_work/potential/_node_${TAG}.sh"

bsub -q "$QUEUE" -J "$TAG" -n 4 -R "span[hosts=1] rusage[mem=64GB]" \
     -gpu "num=1:mode=exclusive_process" -W 24:00 \
     -o "$ROOT/v2_work/logs/runs/${TAG}_%J.out" \
     -e "$ROOT/v2_work/logs/runs/${TAG}_%J.err" \
     "bash $ROOT/v2_work/potential/_node_${TAG}.sh"
