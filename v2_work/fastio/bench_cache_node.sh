#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
COMMON="--data_dir v2_work/genflame/flame_train_ready/npz_withops \
--dist_npz v2_work/genflame/flame_train_ready/gt_matrix.npz \
--device cuda --model xyz_dn --epochs 3 --batch_subjects 5 \
--train_level mixed --train_pair_mode cross_topology --lambda_rank 0.5 \
--use_id_loss --lambda_id 0.25 --k_spec 0 --no-log_spec --eval_every 0 \
--max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 --seed 1234"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "############ WITHOUT CACHE (v1 data path)"
/usr/bin/time -f "TOTAL_NOCACHE %e s" .conda_env/bin/python v2_work/fastio/train_fast.py \
    --no-cache --runs_root v2_work/runs/_bench_nocache $COMMON 2>&1 | tr '\r' '\n' | grep -aE "s/it|TOTAL_|Epoch [0-9]+ \|" | tail -6
echo "############ WITH RAM CACHE"
/usr/bin/time -f "TOTAL_CACHE %e s" .conda_env/bin/python v2_work/fastio/train_fast.py \
    --cache-workers 16 --runs_root v2_work/runs/_bench_cache $COMMON 2>&1 | tr '\r' '\n' | grep -aE "cache\]|fastio|s/it|TOTAL_|Epoch [0-9]+ \|" | tail -10
echo "############ WITH CACHE ON DEVICE"
/usr/bin/time -f "TOTAL_DEVICE %e s" .conda_env/bin/python v2_work/fastio/train_fast.py \
    --cache-residency device --cache-workers 16 --runs_root v2_work/runs/_bench_device $COMMON 2>&1 | tr '\r' '\n' | grep -aE "cache\]|s/it|TOTAL_|Epoch [0-9]+ \|" | tail -8
