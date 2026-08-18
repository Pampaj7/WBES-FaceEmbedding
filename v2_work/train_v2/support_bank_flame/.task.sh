#!/bin/bash
set -euo pipefail
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=/dtu/p1/leopam/WBES-FaceEmbedding/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .conda_env/bin/python v2_work/train_v2/make_support_bank.py \
    --in-dir '/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/genflame/flame_train_ready/npz_withops' --out-dir '/dtu/p1/leopam/WBES-FaceEmbedding/v2_work/train_v2/support_bank_flame' --n-subjects 600 --n-variants 5 \
    --shard $((LSB_JOBINDEX-1))/20
