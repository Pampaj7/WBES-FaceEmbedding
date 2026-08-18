#!/bin/bash
set -euo pipefail
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=/dtu/p1/leopam/WBES-FaceEmbedding/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SLICE=$(printf "/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/xdomain/bfm_in_flame_withops/.slices/task%03d" $LSB_JOBINDEX)
exec .conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
    --input-dir "$SLICE" --output-dir "/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/xdomain/bfm_in_flame_withops" --k-eig 128 --n-cores 4
