#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
exec .conda_env/bin/python v2_work/xdomain/gt_frames.py --n 500
