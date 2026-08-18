#!/usr/bin/env bash
# Trains the Weyl arm once its operators are complete, and nothing else changes.
#
#   pot_area : identical recipe to pot_plain, but the operators were built on meshes scaled to
#              unit total area. The comparison pot_plain vs pot_area therefore isolates the
#              area (Weyl) normalisation and nothing else.
#
# The wait condition is the COUNT and only the count. A stall ABORTS instead of proceeding:
# last night a launcher fell through to training on 2792 of 3000 meshes because it watched a
# job name instead, and a partial dataset produces a number that looks valid.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
OPS=v2_work/potential/bfm_areanorm
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz
TARGET=3000

stall=0; last=-1
while true; do
  n=$(ls "$OPS" 2>/dev/null | wc -l)
  echo "[$(date +%T)] operatori area-norm: $n/$TARGET"
  [ "$n" -ge "$TARGET" ] && break
  if [ "$n" -eq "$last" ]; then stall=$((stall + 1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 15 ]; then
    echo "[$(date +%T)] ABORT: fermo a $n/$TARGET da 30 min, NON allena su dati parziali."; exit 1
  fi
  sleep 120
done

cat > v2_work/potential/_node_pot_area.sh <<NODE
#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=\$PWD/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency ram --cache-workers 4 \\
  --data_dir $OPS --dist_npz $DIST \\
  --runs_root v2_work/runs/pot_area_\${LSB_JOBID:-local} \\
  --device cuda --model xyz_dn \\
  --latent_dim 256 --width 128 --n_blocks 4 --dropout 0.1 \\
  --pool_mode meanmax --k_spec 0 --no-log_spec --eig_k 300 \\
  --epochs 60 --batch_subjects 5 --lr 1e-4 --weight_decay 1e-6 --grad_clip 1.0 \\
  --train_level mixed --train_pair_mode cross_topology \\
  --lambda_subject 1.0 --lambda_mesh 1.0 --lambda_rank 0.5 \\
  --use_id_loss --lambda_id 0.25 \\
  --rank_margin 0.05 --rank_pairs 1024 --rank_tau 0.02 --rank_hard_frac 0.7 \\
  --max_meshes_per_subject_train 6 --max_meshes_per_subject_eval 6 \\
  --max_subjects_eval_train 16 \\
  --p_noise 0.6 --sigma_min 5e-4 --sigma_max 2e-2 \\
  --noise_modes translation,rotation,jitter \\
  --noise_mode_weights "translation=4,rotation=2,jitter=1" \\
  --rigid_rot_deg 12.0 --rigid_rot_deg_min 0.5 \\
  --rigid_trans_scale 0.03 --rigid_trans_scale_min 0.001 \\
  --sigma_min_eval 1e-3 --sigma_max_eval 0.1 --n_sigma_eval 6 \\
  --eval_mode average --eval_every 2 --save_every 5 --seed 1234
NODE
chmod +x v2_work/potential/_node_pot_area.sh

export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
  -gpu "num=1:mode=shared" -W 720 -J pot_area \
  "bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/_node_pot_area.sh" \
  > v2_work/logs/runs/pot_area.log 2>&1 < /dev/null &
echo "[$(date +%T)] pot_area sottomesso"
