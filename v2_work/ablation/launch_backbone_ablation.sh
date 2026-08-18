#!/usr/bin/env bash
# Backbone / input ablation — the reviewers' "DiffusionNet with no justification and nothing else".
#
#   bash launch_backbone_ablation.sh <tag> <queue> [extra trainer flags...]
#
# Every arm shares the recipe of the pot_plain control (same data, 60 epochs, seed 1234, bs5),
# so each arm is directly comparable to the xyz_dn baseline already training and no separate
# control has to be trained for this table.
#
# Residency is `ram`, not `device`: the operator cache is 43 GB and a V100 has 32 GB of VRAM,
# so a device-resident cache would silently fall back or OOM on these nodes. The H100 runs can
# afford `device`; these cannot.
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
TAG=${1:?tag}; QUEUE=${2:?queue}; shift 2
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz

cat > "$ROOT/v2_work/ablation/_node_${TAG}.sh" <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DTU support asked us not to saturate the shared nodes. LSF gives this job 8 slots but does
# not pin threads, so BLAS/OpenMP/torch each default to the machine's full core count and the
# node ends up with a load average several times its core count. Cap every thread pool to the
# slots we were actually allocated.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TORCH_NUM_THREADS=8
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency ram --cache-workers 4 --cache-max-gb 60 \\
  --data_dir datasets/REMESH/npz_data_topo_500_withops --dist_npz $DIST \\
  --runs_root v2_work/runs/${TAG}_\${LSB_JOBID:-local} \\
  --device cuda \\
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \\
  --pool_mode meanmax --eig_k 300 \\
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
  --eval_mode average --eval_every 2 --save_every 5 --seed 1234 $*
NODE
chmod +x "$ROOT/v2_work/ablation/_node_${TAG}.sh"

# rusage[mem=] is PER SLOT on this cluster, not per job: -n 8 with mem=90GB asks for 720 GB and
# pends forever on nodes that have 377 GB. The RAM-resident operator cache is ~57 GB total, so
# 10 GB x 8 slots = 80 GB is the right way to express it.
bsub -q "$QUEUE" -J "$TAG" -n 8 -R "span[hosts=1] rusage[mem=10GB]" \
     -gpu "num=1:mode=exclusive_process" -W 24:00 \
     -o "$ROOT/v2_work/logs/runs/${TAG}_%J.out" \
     -e "$ROOT/v2_work/logs/runs/${TAG}_%J.err" \
     "bash $ROOT/v2_work/ablation/_node_${TAG}.sh"
