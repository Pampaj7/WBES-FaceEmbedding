#!/bin/bash
#BSUB -J potops[1-8]
#BSUB -q milan
#BSUB -n 1
#BSUB -R "span[hosts=1] rusage[mem=8GB]"
#BSUB -W 8:00
#BSUB -o /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/logs/array/potops_%J_%I.out
#BSUB -e /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/logs/array/potops_%J_%I.err
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .conda_env/bin/python v2_work/potential/potential_operators.py \
    --input-dir datasets/REMESH/npz_data_topo_500 \
    --output-dir v2_work/potential/bfm_withwell \
    --k-eig 128 --shard "$((LSB_JOBINDEX-1))/8"
