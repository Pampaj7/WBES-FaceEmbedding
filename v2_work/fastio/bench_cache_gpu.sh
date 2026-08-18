#!/bin/bash
# Quantify the RAM-cache gain on a real GPU: 3 epochs with and without the cache,
# same seed and config, on a second GPU so the pilot run is untouched.
set -u
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=120GB]" \
     -gpu "num=1:mode=shared" -W 90 -J cachebench \
     bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/fastio/bench_cache_node.sh
