#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec .conda_env/bin/python v2_work/potential/eval_by_topology.py \
  --checkpoint v2_work/runs/pot_point_29136537/mixed_xtopo_xyz_dn_rank0.50_id0.25_z256_w128_b4_bs5_ks0_poolmeanmax_noise60_sig5e-4-2e-2_latentnoise_seed1234__4de4aa77/checkpoints/best_by_xtopo_mesh_clean.pth \
  --data-dir datasets/REMESH/npz_data_topo_500_withops \
  --dist-npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
  --tag pot_point --use-eval-split --n-subjects 100 \
  --point-backbone --point-samples 2048 --point-knn 20
