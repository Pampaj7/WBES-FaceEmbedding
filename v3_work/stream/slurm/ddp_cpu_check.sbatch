#!/usr/bin/env bash
#SBATCH --job-name=wbes-stream-ddp
#SBATCH --chdir=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding
#SBATCH --output=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.out
#SBATCH --error=/home/create.aau.dk/ga41wf/WBES-FaceEmbedding/aau/logs/%x-%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#
# train_stream.py sotto torchrun con 2 rank (gloo, CPU): i rank si dividono gli shard (seq % world), la DDP
# resta sincronizzata (check_sync di train_v3) e ogni rank scrive le sue statistiche. Pochi passi piccoli.
set -euo pipefail
export AAU_NV=
source "${WBES_ROOT:-$PWD}/aau/env.sh"
cd "$WBES_ROOT"
source "$AAU_DIR/data_scale/recipe_v1.sh"
RING="/tmp/${SLURM_JOB_ID:-manual}/stream"
E="$AAU_RUNS/evidence/stream/ddp_cpu_${SLURM_JOB_ID:-manual}"
mkdir -p "$E" "$RING"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "/tmp/${SLURM_JOB_ID:-manual}"' EXIT
"$AAU_DIR/run.sh" v3_work/stream/producer.py --ring "$RING" --ring-gb 4 --n-proc 14 --k-eig 64 \
    --domains ict,flame2020,famos --stats-every 60 > "$E/producers.log" 2>&1 &
until [[ $(ls "$RING/shards" 2>/dev/null | wc -l) -ge 12 ]]; do sleep 5; done
OMP_NUM_THREADS=2 "$AAU_DIR/run.sh" -m torch.distributed.run --standalone --nproc-per-node 2 \
  v3_work/stream/train_stream.py --stream "$RING" --device cpu --total-steps 6 --steps-per-epoch 3 --epochs 2 \
  --data_dir datasets/REMESH/npz_data_topo_500_withops_areanorm --no-cache \
  --dist_npz face_embedding/gt_encdec/autoencoder/latent_analysis/gt_distance_matrix/normalized_matrix_distances.npz \
  "${RECIPE_V1[@]}" --device cpu --batch_subjects 4 --max_meshes_per_subject_train 2 --max_subjects_eval_train 2 \
  --max_meshes_per_subject_eval 2 --eval_every 0 --save_every 1000 --train-threads 2 --seed 1234 \
  --runs_root "$E/runs" --log-every 1 2>&1 | grep -v "FloatTensor\|SparseTensor" > "$E/train.log"
tail -15 "$E/train.log"
for f in "$E"/runs/*/stream_stats_rank*.jsonl; do echo "$f"; cat "$f"; done
