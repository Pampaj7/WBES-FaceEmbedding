#!/bin/bash
# One worker for the PAPER-FAITHFUL potential-well operators.
#
# Difference from shard_job.sh: --alpha-mode global. The per-mesh mode places the well at a
# quantile of each mesh's own geodesic distribution, which measured out at alpha ~0.70 while
# the nearest boundary sits at ~0.30 -- i.e. the boundary was inside the region the well is
# supposed to exclude, on 8 of 8 meshes checked, and the offset differed between topologies of
# the same identity (0.679 crop vs 0.727 original). Liu, Jacobson & Crane require one offset
# for the whole collection so every patch shares a domain; calibrate_alpha.py computes it.
#
# $1 = shard index, $2 = shard count, $3 = CPU list (required; unpinned workers saturated the
# node once already and DTU support asked us to stop).
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=$PWD/diffusion-net/src
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CPUSET=${3:?cpu list required, e.g. 24-26}
ALPHA=$(.conda_env/bin/python -c "import json;print(json.load(open('v2_work/potential/alpha_global.json'))['alpha'])")
SCALE=$(.conda_env/bin/python -c "import json;print(json.load(open('v2_work/potential/alpha_global.json'))['scale'])")
exec taskset -c "$CPUSET" .conda_env/bin/python v2_work/potential/potential_operators.py \
    --input-dir datasets/REMESH/npz_data_topo_500 \
    --output-dir v2_work/potential/bfm_well055 \
    --k-eig 128 --shard "$1/$2" \
    --alpha-mode global --alpha-value "$ALPHA" --scale-value "$SCALE"
