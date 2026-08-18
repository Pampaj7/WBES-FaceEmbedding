#!/bin/bash
# Precompute DiffusionNet operators (k-eig 128) for the Poisson variants via p1i
# (hpc/milan batch queues are blocked for this account -- see launch_p1i_shards.sh
# for the empirical evidence). Cheap (~0.4s/mesh, per precompute_array.sh's own
# estimate), so one job per domain is enough -- no need to shard.
#
# Usage: bash launch_p1i_ops.sh <domain: bfm|flame>
set -u
ROOT=/dtu/p1/leopam/WBES-FaceEmbedding
DOMAIN=${1:?domain}
case "$DOMAIN" in
  bfm)   IN=v2_work/poisson_aug/bfm_poisson;   OUT=v2_work/poisson_aug/bfm_poisson_withops ;;
  flame) IN=v2_work/poisson_aug/flame_poisson; OUT=v2_work/poisson_aug/flame_poisson_withops ;;
  *) echo "domain must be bfm or flame" >&2; exit 1 ;;
esac
cd "$ROOT"
export ESUB_BYPASS=1 ESUB_QUIET=1
mkdir -p "$OUT" v2_work/logs
bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=8GB]" \
     -gpu "num=1:mode=shared" -W 240 -J "pois_ops_${DOMAIN}" \
     "cd $ROOT && export WBES_DIFFUSION_NET_SRC=$ROOT/diffusion-net/src && export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 && .conda_env/bin/python face_embedding/gt_encdec/autoencoder/precompute_operators_npz.py --input-dir $IN --output-dir $OUT --k-eig 128 --n-cores 4"
