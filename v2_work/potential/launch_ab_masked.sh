#!/usr/bin/env bash
# Waits for the alpha=0.55 operators, then starts BOTH remaining arms on the same data.
#
#   pot_w55    well only, standard pooling          -> "does the well help at a good alpha?"
#   pot_m55    well + pooling restricted to the ROI -> "does fixing the SUPPORT help further?"
#
# Same alpha, same recipe, same seed: the difference between them isolates the masking, which
# is the point. Together with the already-running pot_plain (no well) and pot_well (well at the
# per-mesh offset, i.e. misplaced) this gives four arms that separate three distinct claims:
# the well's presence, its placement, and the support it is pooled over.
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
OPS=v2_work/potential/bfm_well055
DIST=face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz
TARGET=3000

# The condition is the COUNT, never the presence of a job with a particular name. The first
# version broke out when no job matching "w55" was found; the remaining shards had by then been
# relaunched as "wf*", so it saw zero workers, fell through, and started training on 2792 of
# 3000 meshes -- an incomplete dataset silently different from the control's. Training on
# partial data is worse than not training, so there is no fallback that proceeds: a stall aborts.
stall=0; last=-1
while true; do
  n=$(ls "$OPS" 2>/dev/null | wc -l)
  echo "[$(date +%T)] operatori alpha=0.55: $n/$TARGET"
  [ "$n" -ge "$TARGET" ] && break
  if [ "$n" -eq "$last" ]; then stall=$((stall + 1)); else stall=0; fi
  last=$n
  if [ "$stall" -ge 15 ]; then
    echo "[$(date +%T)] ABORT: fermo a $n/$TARGET da 30 min. NON lancio i training su dati"
    echo "                incompleti: il confronto con pot_plain (3000) sarebbe confuso."
    exit 1
  fi
  sleep 120
done

emit () {   # $1 = tag, $2 = extra train_fast flags
  cat > "v2_work/potential/_node_$1.sh" <<NODE
#!/bin/bash
set -u
cd /dtu/p1/leopam/WBES-FaceEmbedding
export WBES_DIFFUSION_NET_SRC=\$PWD/diffusion-net/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
exec .conda_env/bin/python v2_work/fastio/train_fast.py \\
  --cache-residency ram --cache-workers 4 $2 \\
  --data_dir $OPS --dist_npz $DIST \\
  --runs_root v2_work/runs/$1_\${LSB_JOBID:-local} \\
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
  chmod +x "v2_work/potential/_node_$1.sh"
}

emit pot_w55 ""
emit pot_m55 "--masked-pooling --roi-threshold 0.5"

# Split across hardware. Both arms on p1i's two shared GPUs made three trainings on two cards
# and m55 died with CUDA OOM (GPU0 was at 63.8/79 GiB, most of it another user's). The arms must
# share data, recipe and seed -- not a GPU -- so m55 takes an exclusive card on a public queue.
# Different wall-clock, identical comparison.
export ESUB_BYPASS=1 ESUB_QUIET=1
setsid nohup bsub -I -q p1i -app h100app -n 4 -R "span[hosts=1] rusage[mem=24GB]" \
  -gpu "num=1:mode=shared" -W 720 -J pot_w55 \
  "bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/_node_pot_w55.sh" \
  > v2_work/logs/runs/pot_w55.log 2>&1 < /dev/null &
sleep 30
bsub -q gpuv100 -J pot_m55 -n 8 -R "span[hosts=1] rusage[mem=10GB]" \
  -gpu "num=1:mode=exclusive_process" -W 24:00 \
  -o v2_work/logs/runs/pot_m55_%J.out -e v2_work/logs/runs/pot_m55_%J.err \
  "bash /dtu/p1/leopam/WBES-FaceEmbedding/v2_work/potential/_node_pot_m55.sh"
echo "[$(date +%T)] pot_w55 (p1i) e pot_m55 (gpuv100) sottomessi"
