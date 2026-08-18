#!/bin/bash
#BSUB -J flame_v1recipe
#BSUB -q p1i
#BSUB -n 4
#BSUB -R "span[hosts=1] rusage[mem=12GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/logs/lsf_flame_%J.out
#BSUB -e /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/logs/lsf_flame_%J.err

bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/train_flame_v1config.sh
