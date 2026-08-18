#!/bin/bash
# Grab one p1i allocation and use ALL of it: the CPU cores chew the operator backlog
# while the GPU share runs a training. The private queue only accepts jobs that request a
# GPU, so a CPU-only request never dispatches — we ask for a shared GPU and get the cores
# with it.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "[worker] $(date +%T) cores=$(nproc) gpu=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# 1. operators for the 5000-identity FLAME set (30k meshes) — the current bottleneck
.conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
    --input-dir v2_work/genflame/flame_topo_5000 \
    --output-dir v2_work/genflame/flame_topo_5000_withops \
    --k-eig 128 --n-cores 12 &
OPS=$!

# 2. support-variant bank for BFM, sharded across the remaining cores
for i in $(seq 0 2); do
  .conda_env/bin/python v2_work/train_v2/make_support_bank.py \
      --in-dir datasets/REMESH/npz_data_topo_500_withops \
      --out-dir v2_work/train_v2/support_bank_bfm \
      --n-subjects 500 --n-variants 5 --shard $i/3 &
done

wait $OPS
echo "[worker] $(date +%T) operators done: $(ls v2_work/genflame/flame_topo_5000_withops | wc -l) files"
wait
echo "[worker] $(date +%T) ALL DONE"
