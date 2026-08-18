#!/bin/bash
# One worker for the area-normalised operators. $1=shard idx $2=shard count $3=cpu list.
# CPU list is required: unpinned workers saturated the node once and DTU support asked us
# to stop, and OMP_NUM_THREADS alone does not hold these libraries.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CPUSET=${3:?cpu list required}
exec taskset -c "$CPUSET" .conda_env/bin/python v2_work/potential/areanorm_operators.py \
  --input-dir datasets/REMESH/npz_data_topo_500 \
  --output-dir v2_work/potential/bfm_areanorm \
  --k-eig 128 --shard "$1/$2"
