#!/usr/bin/env bash
# Large FLAME set for the scale curve and the joint training run.
# Nested subsets: prefixes flame0000..N of THIS set (the sampler draws one block, so a
# 5000-run is not a superset of the earlier 600-run — that one stays the pilot).
set -euo pipefail
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src

echo "[$(date +%T)] identities"
.conda_env/bin/python v2_work/genflame/generate_identities.py \
    --n-identities 5000 --n-shape 100 --seed 1234 \
    --out-dir v2_work/genflame/flame_identities_5000

echo "[$(date +%T)] topologies"
.conda_env/bin/python v2_work/genflame/make_flame_topologies.py \
    --in-dir v2_work/genflame/flame_identities_5000 \
    --out-dir v2_work/genflame/flame_topo_5000 --n-cores 1

echo "[$(date +%T)] gt matrix"
.conda_env/bin/python v2_work/genflame/build_flame_gt_matrix.py \
    --topo-dir v2_work/genflame/flame_topo_5000 \
    --out-dir v2_work/genflame/flame_gt_5000

echo "[$(date +%T)] operators"
.conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py \
    --input-dir v2_work/genflame/flame_topo_5000 \
    --output-dir v2_work/genflame/flame_topo_5000_withops \
    --k-eig 128 --n-cores 2

echo "[$(date +%T)] DONE"
