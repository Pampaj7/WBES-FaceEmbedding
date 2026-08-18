#!/bin/bash
# Wrapper: submits the FLAME training to p1i (interactive-only queue) with a real
# walltime. ESUB_BYPASS=1 is what lets -W through the site esub (see ARGOS
# scripts/run_a2_drends.sh, the established pattern on this cluster).
set -u
export ESUB_BYPASS=1 ESUB_QUIET=1
exec bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
     -gpu "num=1:mode=shared" -W 720 -J flame_v1recipe \
     bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/train_flame_v1config.sh
