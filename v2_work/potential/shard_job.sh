#!/bin/bash
# One p1i shard worker for the potential-well operators. The h100app profile caps a job at
# 4 cores, so parallelism comes from several of these rather than one wide job.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .conda_env/bin/python v2_work/potential/potential_operators.py \
    --input-dir datasets/REMESH/npz_data_topo_500 \
    --output-dir v2_work/potential/bfm_withwell \
    --k-eig 128 --shard "$1/$2"
