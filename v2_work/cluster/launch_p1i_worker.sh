#!/usr/bin/env bash
set -u
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -n 16 -R "span[hosts=1] rusage[mem=180GB]" \
     -gpu "num=1:mode=shared" -W 720 -J p1i_worker \
     bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/cluster/p1i_worker.sh
