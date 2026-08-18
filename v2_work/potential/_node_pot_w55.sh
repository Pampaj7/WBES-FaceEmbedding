#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec .conda_env/bin/python v2_work/fastio/train_fast.py \
  --cache-residency ram --cache-workers 4  \
  --data_dir v2_work/potential/bfm_well055 --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
  --runs_root v2_work/runs/pot_w55_${LSB_JOBID:-local} \
  --device cuda --model xyz_dn \
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \
  --epochs 60 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \
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
