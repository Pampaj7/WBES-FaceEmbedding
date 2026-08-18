#!/usr/bin/env bash
# The Poisson experiment and its matched control.
#
#   pois_arm   trains on flame_view      (6 real topologies + pois0/pois1)
#   pois_ctrl  trains on npz_withops     (the same 6 real topologies, no Poisson)
#
# Question: does training against a Poisson-reconstructed mate recover the FaceScape transfer
# that collapsed (0.057 as-is, 0.263 re-cropped, 0.404 with a REMESH mate)? The Poisson mates
# reproduce that corruption at the right magnitude -- Hausdorff 0.291 (BFM) / 0.322 (FLAME)
# against 0.088 for the pair the model trains on and 0.38 measured on FaceScape.
#
# DESIGN CHOICE, stated because it is not the obvious one: both arms sample the SAME number of
# meshes per subject per epoch (6). The Poisson arm draws its 6 from 8 members, the control
# from 6. Raising the Poisson arm to 8 would have given it more gradient signal per epoch, so
# a win could have been "more data" rather than "Poisson data" -- the one thing this experiment
# exists to distinguish. Equal count, different composition.
#
#   bash launch_pair.sh <queue>
set -euo pipefail
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
QUEUE=${1:-gpuv100}
GT=v2_work/genflame/flame_train_ready/gt_matrix.npz

emit () {  # $1 tag, $2 data dir
  cat > "$ROOT/v2_work/poisson_aug/_node_$1.sh" <<NODE
#!/bin/bash
set -u
cd $ROOT
export WBES_DIFFUSION_NET_SRC=\$PWD/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency ram --cache-workers 4 --cache-max-gb 60 \\
  --data_dir $2 --dist_npz $GT \\
  --runs_root v2_work/runs/$1_\${LSB_JOBID:-local} \\
  --device cuda --model xyz_dn \\
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \\
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
  chmod +x "$ROOT/v2_work/poisson_aug/_node_$1.sh"
  # mem is PER SLOT on this cluster: 8 x 10GB covers the FLAME cache comfortably.
  bsub -q "$QUEUE" -J "$1" -n 8 -R "span[hosts=1] rusage[mem=10GB]" \
       -gpu "num=1:mode=exclusive_process" -W 24:00 \
       -o "$ROOT/v2_work/logs/runs/$1_%J.out" -e "$ROOT/v2_work/logs/runs/$1_%J.err" \
       "bash $ROOT/v2_work/poisson_aug/_node_$1.sh"
}

emit pois_arm  v2_work/poisson_aug/flame_view
sleep 5
emit pois_ctrl v2_work/genflame/flame_train_ready/npz_withops
