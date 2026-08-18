#!/usr/bin/env bash
# Launch one FLAME+Poisson-augmented training run on a GPU node.
# Copy of v2_work/fastio/launch_flame.sh, pointed at the Poisson-augmented view
# (v2_work/poisson_aug/flame_view = the 6 REMESH-2 topologies + pois0/pois1).
#
#   bash launch_flame_poisson.sh <tag> <batch_subjects> <lr> [extra trainer flags...]
#
# GT matrix is UNCHANGED (v2_work/genflame/flame_train_ready/gt_matrix.npz): the Poisson
# variants are new meshes of the SAME identities, not new subjects, so the distance matrix
# rows/cols still apply unmodified.
#
# --max_meshes_per_subject_train raised 6 -> 8: flame_view now has 8 topology members per
# subject (original, remesh, crop, noisy, down8k, up60k, pois0, pois1); leaving the cap at 6
# would let sample_mesh_indices silently drop 2 of 8 each epoch, so the new Poisson
# realizations would only be sampled some of the time instead of every epoch.
#
# KNOWN ISSUE (pre-existing, not caused by this augmentation): on --device cpu, the
# --eval_every>0 robustness-grid eval crashes with "RuntimeError: Cannot set version_counter
# for inference tensor" (torch.inference_mode() + diffusion_net/layers.py's L.unsqueeze(0)).
# Reproduced verbatim on the UNMODIFIED flame_train_ready view (no Poisson meshes involved),
# so it is a fastio/train_runner + this torch version issue, not something introduced here.
# --eval_every 0 (the v2_work/runs/_smoke_fastio precedent) avoids it; untested whether it
# also fires on --device cuda. See v2_work/STATUS.md for the repro.
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
TAG=${1:?tag}; BS=${2:?batch_subjects}; LR=${3:?lr}; shift 3

cat > "$ROOT/v2_work/poisson_aug/_node_${TAG}.sh" <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=\$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "picked GPU \$CUDA_VISIBLE_DEVICES of:"; nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency device --cache-workers 16 \\
  --data_dir v2_work/poisson_aug/flame_view \\
  --dist_npz v2_work/genflame/flame_train_ready/gt_matrix.npz \\
  --runs_root v2_work/runs/${TAG} \\
  --device cuda --model xyz_dn \\
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \\
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \\
  --epochs 120 --batch_subjects ${BS} --lr ${LR} \\
  --weight_decay 1e-6 --grad_clip 1.0 \\
  --train_level mixed --train_pair_mode cross_topology \\
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \\
  --use_id_loss --lambda_id 0.25 \\
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \\
  --max_meshes_per_subject_train 8 --max_meshes_per_subject_eval 6 \\
  --max_subjects_eval_train 16 \\
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \\
  --noise_modes translation,rotation,jitter \\
  --noise_mode_weights "translation=4,rotation=2,jitter=1" \\
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \\
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \\
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \\
  --eval_mode average --eval_every 2 --save_every 5 --seed 1234 $*
NODE
chmod +x "$ROOT/v2_work/poisson_aug/_node_${TAG}.sh"

export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 \
     -R "span[hosts=1] rusage[mem=64GB]" -gpu "num=1:mode=shared" \
     -W 720 -J "$TAG" \
     "bash $ROOT/v2_work/poisson_aug/_node_${TAG}.sh"
